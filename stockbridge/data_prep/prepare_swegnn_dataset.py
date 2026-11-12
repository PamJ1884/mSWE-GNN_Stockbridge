"""Convert Stockbridge CityCAT PT tensors to the SWE-GNN dataset layout read by
`stockbridge/dataset.py` (RainMSGNN).

    python stockbridge/data_prep/prepare_swegnn_dataset.py \
        --pt-root .../pt/stockbridge_phase2plus3_300_v01 \
        --split-manifest .../stockbridge_phase2plus3_300_manifest.csv \
        --output-root .../swegnn/stockbridge_phase2plus3_300_5m_v01 \
        --cell-size 5.0 --rainfall-scale-reference 300000.0

Origin
------
The conversion logic (static graph, slopes, per-event H / Qmag / rainfall,
manifest.csv / manifest.json) is copied from
    Stockbridge-SWEGNN/scripts/prepare_stockbridge_swegnn_full_dataset.py
    at commit 6b0b07d72a0a847d3d102f70f63c3688d377da12
    ("Add milestone 13A full SWE-GNN dataset preparation"),
which built `swegnn/stockbridge_phase2_100_5m_v01` with
    --pt-root pt/stockbridge_phase2_core_v01 --cell-size 5.0
    --rainfall-scale-reference 300000.0
(Stockbridge-SWEGNN/slurm/m16/swegnn_phase2_100_5m_convert_v01.sbatch,
commit 8665202d19dbbe54434fe47816b1a79524ce8b3d).
`build_static_graph_from_reference`, `convert_event` and the helpers are kept
byte-for-byte equivalent, so the static graph and the event files are
identical to the reference ones for the same input.

Changes from the reference script
---------------------------------
* Split sizes are read from the train/valid/test folders (no fixed 70/15/15
  and no fixed total of 100). With --split-manifest, the folder membership
  must match the manifest exactly.
* --cell-size is required (the reference defaulted to 1.0).
* No --overwrite: the output root must not exist.
* R019 in test is recorded, not required.
* Writes prepare_summary.json with provenance (source, commits, options,
  counts, date, versions).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import torch


REFERENCE_SCRIPT = "Stockbridge-SWEGNN/scripts/prepare_stockbridge_swegnn_full_dataset.py"
REFERENCE_SCRIPT_COMMIT = "6b0b07d72a0a847d3d102f70f63c3688d377da12"
REFERENCE_SBATCH = "Stockbridge-SWEGNN/slurm/m16/swegnn_phase2_100_5m_convert_v01.sbatch"
REFERENCE_SBATCH_COMMIT = "8665202d19dbbe54434fe47816b1a79524ce8b3d"

NY = 305
NX = 326
NT = 25
CHANNELS = 5
NUM_NODES = 99430
EXPECTED_EDGE_COUNT = 396458
SPLIT_NAMES = ("train", "valid", "test")
EXPECTED_TENSOR_SHAPE = (NY, NX, NT, CHANNELS)

CHANNEL_MEANING = {
    0: "H",
    1: "Vx",
    2: "Vy",
    3: "rainfall",
    4: "DEM",
}
INPUT_FEATURES = [
    "slope_x",
    "slope_y",
    "rainfall_t",
    "rainfall_t_plus_1",
    "DEM",
    "H_t",
    "Qmag_t",
]
EDGE_FEATURES = ["dx", "dy", "edge_slope"]
OUTPUT_VARIABLES = ["H", "Qmag"]

RAINFALL_UNIFORMITY_TOLERANCE = 1e-8
NONNEGATIVE_TOLERANCE = 1e-8
DEM_ATOL = 1e-6
DEM_RTOL = 1e-6

MANIFEST_FIELDNAMES = [
    "event_id",
    "split",
    "source_path",
    "dynamic_path",
    "H_max",
    "Qmag_max",
    "rainfall_raw_max",
    "rainfall_raw_sum",
    "rainfall_scaled_reference_max",
    "rainfall_scaled_reference_sum",
    "rainfall_spatial_max_abs_diff",
]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Stockbridge CityCAT PT splits to the SWE-GNN layout."
    )
    parser.add_argument("--pt-root", type=Path, required=True,
                        help="folder with train/, valid/, test/ PT tensors")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="new folder (must not exist)")
    parser.add_argument("--cell-size", type=float, required=True,
                        help="raster cell size in m (5.0 for Stockbridge)")
    parser.add_argument("--rainfall-scale-reference", type=float, default=300000.0)
    parser.add_argument("--split-manifest", type=Path, default=None,
                        help="CSV with event_id and split columns; folder membership must match it")
    return parser.parse_args(argv)


# ---- copied from the reference script (see module docstring) ----------------

def event_id_from_path(path: Path) -> str:
    """Extract a Stockbridge event ID such as R019 from a file path."""
    match = re.search(r"(R\d{3})", path.name)
    if match is None:
        raise ValueError(
            f"Could not find an event ID matching R followed by three digits "
            f"in filename: {path.name}"
        )
    return match.group(1)


def assert_finite(name: str, tensor: torch.Tensor) -> None:
    """Assert that a tensor contains neither NaN nor infinite values."""
    assert torch.isfinite(tensor).all().item(), f"{name} contains NaN/inf values"


def load_event_tensor(path: Path) -> torch.Tensor:
    """Load and validate one preprocessed Stockbridge event tensor."""
    if not path.is_file():
        raise FileNotFoundError(f"Event tensor not found: {path}")
    tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"Expected torch.Tensor in {path}, got {type(tensor).__name__}"
        )
    if tuple(tensor.shape) != EXPECTED_TENSOR_SHAPE:
        raise ValueError(
            f"Expected tensor shape {EXPECTED_TENSOR_SHAPE} in {path}, "
            f"got {tuple(tensor.shape)}"
        )
    if not tensor.is_floating_point():
        raise TypeError(
            f"Expected floating-point tensor in {path}, got {tensor.dtype}"
        )
    assert_finite(str(path), tensor)
    return tensor


def calculate_dem_slopes(
    DEM: torch.Tensor,
    cell_size: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate finite-difference DEM slopes in physical grid units."""
    dem_grid = DEM.reshape(NY, NX)
    slope_x = torch.empty_like(dem_grid)
    slope_y = torch.empty_like(dem_grid)

    slope_x[:, 1:-1] = (
        dem_grid[:, 2:] - dem_grid[:, :-2]
    ) / (2.0 * cell_size)
    slope_x[:, 0] = (dem_grid[:, 1] - dem_grid[:, 0]) / cell_size
    slope_x[:, -1] = (dem_grid[:, -1] - dem_grid[:, -2]) / cell_size

    slope_y[1:-1, :] = (
        dem_grid[2:, :] - dem_grid[:-2, :]
    ) / (2.0 * cell_size)
    slope_y[0, :] = (dem_grid[1, :] - dem_grid[0, :]) / cell_size
    slope_y[-1, :] = (dem_grid[-1, :] - dem_grid[-2, :]) / cell_size
    return slope_x.reshape(-1), slope_y.reshape(-1)


def build_static_graph_from_reference(
    reference_tensor: torch.Tensor,
    cell_size: float,
) -> dict:
    """Build the shared static directed four-neighbour grid graph."""
    DEM = (
        reference_tensor[:, :, 0, 4]
        .reshape(NUM_NODES)
        .contiguous()
        .float()
    )
    node_ids = torch.arange(NUM_NODES, dtype=torch.long).reshape(NY, NX)

    left = node_ids[:, :-1].reshape(-1)
    right = node_ids[:, 1:].reshape(-1)
    top = node_ids[:-1, :].reshape(-1)
    bottom = node_ids[1:, :].reshape(-1)
    row = torch.cat([left, right, top, bottom])
    col = torch.cat([right, left, bottom, top])
    edge_index = torch.stack([row, col], dim=0)

    yy, xx = torch.meshgrid(
        torch.arange(NY, dtype=torch.float32),
        torch.arange(NX, dtype=torch.float32),
        indexing="ij",
    )
    pos = torch.stack(
        [xx.reshape(-1) * cell_size, yy.reshape(-1) * cell_size],
        dim=1,
    )
    edge_relative_distance = pos[col] - pos[row]
    edge_distance = torch.linalg.norm(edge_relative_distance, dim=1)
    edge_slope = (DEM[col] - DEM[row]) / edge_distance
    edge_attr = torch.cat(
        [edge_relative_distance, edge_slope.view(-1, 1)],
        dim=1,
    )
    slope_x, slope_y = calculate_dem_slopes(DEM, cell_size)

    expected_shapes = {
        "DEM": (NUM_NODES,),
        "slope_x": (NUM_NODES,),
        "slope_y": (NUM_NODES,),
        "pos": (NUM_NODES, 2),
        "edge_index": (2, EXPECTED_EDGE_COUNT),
        "edge_relative_distance": (EXPECTED_EDGE_COUNT, 2),
        "edge_distance": (EXPECTED_EDGE_COUNT,),
        "edge_slope": (EXPECTED_EDGE_COUNT,),
        "edge_attr": (EXPECTED_EDGE_COUNT, 3),
    }
    tensors = {
        "DEM": DEM,
        "slope_x": slope_x,
        "slope_y": slope_y,
        "pos": pos,
        "edge_index": edge_index,
        "edge_relative_distance": edge_relative_distance,
        "edge_distance": edge_distance,
        "edge_slope": edge_slope,
        "edge_attr": edge_attr,
    }
    for name, expected_shape in expected_shapes.items():
        actual_shape = tuple(tensors[name].shape)
        assert actual_shape == expected_shape, (
            f"Expected {name} shape {expected_shape}, got {actual_shape}"
        )
        assert_finite(name, tensors[name])
    assert not (row == col).any().item(), "Static graph contains self-loops"
    assert (edge_distance > 0).all().item(), "All edge distances must be positive"

    return {
        "ny": NY,
        "nx": NX,
        "nt": NT,
        "num_nodes": NUM_NODES,
        "num_edges": EXPECTED_EDGE_COUNT,
        "cell_size": cell_size,
        **tensors,
        "input_features": INPUT_FEATURES,
        "edge_features": EDGE_FEATURES,
        "output_variables": OUTPUT_VARIABLES,
    }


def convert_event(
    tensor: torch.Tensor,
    event_id: str,
    split: str,
    source_path: Path,
    dynamic_path: Path,
    static_DEM: torch.Tensor,
    rainfall_scale_reference: float,
    warnings: list[str],
) -> dict[str, object]:
    """Convert and save one dynamic event, returning its manifest row."""
    H = tensor[:, :, :, 0].reshape(NUM_NODES, NT).contiguous().float()
    Vx = tensor[:, :, :, 1].reshape(NUM_NODES, NT).contiguous().float()
    Vy = tensor[:, :, :, 2].reshape(NUM_NODES, NT).contiguous().float()
    rainfall = tensor[:, :, :, 3].contiguous().float()
    DEM_event = (
        tensor[:, :, 0, 4].reshape(NUM_NODES).contiguous().float()
    )

    if not torch.allclose(
        DEM_event,
        static_DEM,
        atol=DEM_ATOL,
        rtol=DEM_RTOL,
    ):
        max_abs_difference = torch.max(torch.abs(DEM_event - static_DEM)).item()
        raise ValueError(
            f"{event_id} DEM does not match the shared static DEM; "
            f"max_abs_difference={max_abs_difference:.8g}, "
            f"atol={DEM_ATOL}, rtol={DEM_RTOL}"
        )
    DEM_max_abs_difference = torch.max(
        torch.abs(DEM_event - static_DEM)
    ).item()
    if DEM_max_abs_difference > 0.0:
        warnings.append(
            f"{event_id}: DEM differs from the reference within tolerance "
            f"(max_abs_difference={DEM_max_abs_difference:.8g})."
        )

    rainfall_global = rainfall[0, 0, :].clone().contiguous().float()
    rainfall_spatial_max_abs_diff = torch.max(
        torch.abs(rainfall - rainfall_global.view(1, 1, NT))
    )
    rainfall_difference = rainfall_spatial_max_abs_diff.item()
    if rainfall_difference > RAINFALL_UNIFORMITY_TOLERANCE:
        raise ValueError(
            f"{event_id} rainfall is spatially non-uniform: "
            f"max_abs_difference={rainfall_difference:.8g} exceeds "
            f"{RAINFALL_UNIFORMITY_TOLERANCE}"
        )
    if rainfall_difference > 0.0:
        warnings.append(
            f"{event_id}: rainfall has a tiny spatial difference within "
            f"tolerance (max_abs_difference={rainfall_difference:.8g})."
        )

    Qx = Vx * H
    Qy = Vy * H
    Qmag = torch.sqrt(Qx.square() + Qy.square())
    for name, value in {
        "H": H,
        "Qmag": Qmag,
        "rainfall_global": rainfall_global,
    }.items():
        assert_finite(f"{event_id} {name}", value)
    assert torch.isfinite(rainfall_spatial_max_abs_diff).item(), (
        f"{event_id} rainfall spatial difference is NaN/inf"
    )

    H_min = H.min().item()
    Qmag_min = Qmag.min().item()
    if H_min < -NONNEGATIVE_TOLERANCE:
        raise ValueError(
            f"{event_id} H contains values below the allowed tolerance: "
            f"minimum={H_min:.8g}, tolerance={NONNEGATIVE_TOLERANCE}"
        )
    if Qmag_min < -NONNEGATIVE_TOLERANCE:
        raise ValueError(
            f"{event_id} Qmag contains values below the allowed tolerance: "
            f"minimum={Qmag_min:.8g}, tolerance={NONNEGATIVE_TOLERANCE}"
        )
    if H_min < 0.0:
        warnings.append(
            f"{event_id}: H has a tiny negative minimum within tolerance "
            f"({H_min:.8g})."
        )
    if Qmag_min < 0.0:
        warnings.append(
            f"{event_id}: Qmag has a tiny negative minimum within tolerance "
            f"({Qmag_min:.8g})."
        )

    H_max = H.max()
    Qmag_max = Qmag.max()
    rainfall_raw_max = rainfall_global.max()
    rainfall_raw_sum = rainfall_global.sum()
    rainfall_scaled_reference_max = (
        rainfall_raw_max * rainfall_scale_reference
    )
    rainfall_scaled_reference_sum = (
        rainfall_raw_sum * rainfall_scale_reference
    )
    for name, value in {
        "H_max": H_max,
        "Qmag_max": Qmag_max,
        "rainfall_raw_max": rainfall_raw_max,
        "rainfall_raw_sum": rainfall_raw_sum,
        "rainfall_scaled_reference_max": rainfall_scaled_reference_max,
        "rainfall_scaled_reference_sum": rainfall_scaled_reference_sum,
    }.items():
        assert_finite(f"{event_id} {name}", value)
    dynamic_event = {
        "event_id": event_id,
        "split": split,
        "source_path": str(source_path),
        "ny": NY,
        "nx": NX,
        "nt": NT,
        "num_nodes": NUM_NODES,
        "H": H,
        "Qmag": Qmag,
        "rainfall_global": rainfall_global,
        "rainfall_spatial_max_abs_diff": rainfall_spatial_max_abs_diff,
        "H_max": H_max,
        "Qmag_max": Qmag_max,
        "rainfall_raw_max": rainfall_raw_max,
        "rainfall_raw_sum": rainfall_raw_sum,
        "rainfall_scaled_reference_max": rainfall_scaled_reference_max,
        "rainfall_scaled_reference_sum": rainfall_scaled_reference_sum,
    }

    dynamic_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dynamic_event, dynamic_path)
    if not dynamic_path.is_file():
        raise RuntimeError(f"Dynamic event file was not created: {dynamic_path}")

    return {
        "event_id": event_id,
        "split": split,
        "source_path": str(source_path),
        "dynamic_path": str(dynamic_path),
        "H_max": H_max.item(),
        "Qmag_max": Qmag_max.item(),
        "rainfall_raw_max": rainfall_raw_max.item(),
        "rainfall_raw_sum": rainfall_raw_sum.item(),
        "rainfall_scaled_reference_max": (
            rainfall_scaled_reference_max.item()
        ),
        "rainfall_scaled_reference_sum": (
            rainfall_scaled_reference_sum.item()
        ),
        "rainfall_spatial_max_abs_diff": rainfall_difference,
    }


def output_size_bytes(output_root: Path) -> int:
    """Return the total size of regular output files."""
    return sum(
        path.stat().st_size
        for path in output_root.rglob("*")
        if path.is_file()
    )


# ---- new in this script -------------------------------------------------------

def find_split_files(pt_root: Path, split: str) -> list[Path]:
    """One split's source tensors sorted by event ID (count read from the folder)."""
    split_dir = pt_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Expected split directory not found: {split_dir}")
    paths = sorted(split_dir.glob("*.pt"), key=event_id_from_path)
    if not paths:
        raise ValueError(f"No .pt files in {split_dir}")
    return paths


def read_split_manifest(path: Path) -> dict[str, list[str]]:
    """Event IDs per split from a CSV with `event_id` (e.g. R019_...) and `split` columns."""
    by_split: dict[str, list[str]] = {split: [] for split in SPLIT_NAMES}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = {"event_id", "split"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} lacks columns {sorted(missing)}")
        for row in reader:
            if row["split"] not in by_split:
                raise ValueError(f"{path}: unknown split {row['split']!r}")
            by_split[row["split"]].append(event_id_from_path(Path(row["event_id"])))
    return {split: sorted(ids) for split, ids in by_split.items()}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_info(path: Path) -> dict:
    """Commit of the repository holding `path`; git may be missing in the container."""
    repo = path.resolve().parent
    info = {"commit": None, "dirty": None, "source": None}
    try:
        info["commit"] = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--", "."],
            check=True, capture_output=True, text=True).stdout
        info["dirty"] = bool(status.strip())
        info["source"] = "git"
        return info
    except (OSError, subprocess.CalledProcessError):
        pass
    for parent in [repo, *repo.parents]:  # fallback: read .git directly
        git_dir = parent / ".git"
        if git_dir.is_dir():
            head = (git_dir / "HEAD").read_text().strip()
            if head.startswith("ref: "):
                ref = head[5:]
                ref_path = git_dir / ref
                if ref_path.is_file():
                    info["commit"] = ref_path.read_text().strip()
                elif (git_dir / "packed-refs").is_file():
                    for line in (git_dir / "packed-refs").read_text().splitlines():
                        if line.endswith(" " + ref):
                            info["commit"] = line.split()[0]
            else:
                info["commit"] = head
            info["source"] = ".git files (dirty state unknown)"
            break
    return info


def main(argv=None) -> None:
    started = time.time()
    args = parse_args(argv)
    pt_root = args.pt_root.expanduser()
    output_root = args.output_root.expanduser()

    if not math.isfinite(args.cell_size) or args.cell_size <= 0.0:
        raise ValueError("--cell-size must be finite and greater than zero")
    if not math.isfinite(args.rainfall_scale_reference) or args.rainfall_scale_reference <= 0.0:
        raise ValueError("--rainfall-scale-reference must be finite and greater than zero")
    if not pt_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {pt_root}")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Output root already exists; refusing to overwrite: {output_root}")

    print("=== Stockbridge SWE-GNN dataset preparation ===", flush=True)
    print("Input root:", pt_root)
    print("Output root:", output_root)
    print("Cell size:", args.cell_size)

    split_files = {split: find_split_files(pt_root, split) for split in SPLIT_NAMES}
    event_ids_by_split = {
        split: [event_id_from_path(path) for path in paths]
        for split, paths in split_files.items()
    }
    split_counts = {split: len(paths) for split, paths in split_files.items()}
    print("Split counts (from folders):", split_counts)

    all_event_ids = [eid for split in SPLIT_NAMES for eid in event_ids_by_split[split]]
    total_events = len(all_event_ids)
    if len(set(all_event_ids)) != total_events:
        seen, duplicates = set(), set()
        for eid in all_event_ids:
            (duplicates if eid in seen else seen).add(eid)
        raise ValueError(f"Event IDs are not unique across splits: {sorted(duplicates)}")

    split_manifest_info = None
    if args.split_manifest is not None:
        manifest_path = args.split_manifest.expanduser()
        expected = read_split_manifest(manifest_path)
        for split in SPLIT_NAMES:
            folder_ids = sorted(event_ids_by_split[split])
            if folder_ids != expected[split]:
                only_folder = sorted(set(folder_ids) - set(expected[split]))
                only_manifest = sorted(set(expected[split]) - set(folder_ids))
                raise ValueError(
                    f"{split}: folder and {manifest_path.name} differ; "
                    f"only in folder={only_folder}, only in manifest={only_manifest}")
        split_manifest_info = {"path": str(manifest_path), "sha256": sha256_file(manifest_path),
                               "membership_matches_folders": True}
        print("Split membership matches", manifest_path)
    contains_R019_in_test = "R019" in event_ids_by_split["test"]

    reference_path = split_files["train"][0]
    print("Reference file for static graph:", reference_path)
    static_graph = build_static_graph_from_reference(load_event_tensor(reference_path), args.cell_size)

    output_root.mkdir(parents=True, exist_ok=False)
    static_graph_path = output_root / "static_graph.pt"
    manifest_csv_path = output_root / "manifest.csv"
    manifest_json_path = output_root / "manifest.json"
    summary_path = output_root / "prepare_summary.json"
    torch.save(static_graph, static_graph_path)

    warnings: list[str] = []
    manifest_rows = []
    dynamic_paths = []
    for split, paths in split_files.items():
        print(f"Processing {split} split: 0/{len(paths)}", flush=True)
        for event_number, source_path in enumerate(paths, start=1):
            event_id = event_id_from_path(source_path)
            tensor = load_event_tensor(source_path)
            dynamic_path = output_root / "events" / split / f"{event_id}.pt"
            manifest_rows.append(convert_event(
                tensor=tensor,
                event_id=event_id,
                split=split,
                source_path=source_path,
                dynamic_path=dynamic_path,
                static_DEM=static_graph["DEM"],
                rainfall_scale_reference=args.rainfall_scale_reference,
                warnings=warnings,
            ))
            dynamic_paths.append(dynamic_path)
            if event_number % 10 == 0 or event_number == len(paths):
                print(f"Processing {split} split: {event_number}/{len(paths)} "
                      f"({time.time() - started:.0f} s)", flush=True)
            del tensor

    missing = [path for path in dynamic_paths if not path.is_file()]
    if missing or len(dynamic_paths) != total_events:
        raise RuntimeError(f"Dynamic event files missing or miscounted: {missing}")

    with manifest_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        writer.writerows(manifest_rows)

    created_files_summary = {
        "static_graph_files": 1,
        "dynamic_event_files": len(dynamic_paths),
        "dynamic_event_files_by_split": split_counts,
        "manifest_files": 2,
        "summary_files": 1,
        "total_files": 1 + len(dynamic_paths) + 3,
    }
    # Same structure as the reference manifest.json (dataset.py reads event_ids_by_split).
    manifest = {
        "script_name": Path(__file__).name,
        "pt_root": str(pt_root),
        "output_root": str(output_root),
        "static_graph_path": str(static_graph_path),
        "manifest_csv_path": str(manifest_csv_path),
        "manifest_json_path": str(manifest_json_path),
        "cell_size": args.cell_size,
        "rainfall_scale_reference": args.rainfall_scale_reference,
        "split_counts": split_counts,
        "expected_split_counts": split_counts,
        "total_events": total_events,
        "input_tensor_shape": list(EXPECTED_TENSOR_SHAPE),
        "channel_meaning": CHANNEL_MEANING,
        "input_features": INPUT_FEATURES,
        "edge_features": EDGE_FEATURES,
        "output_variables": OUTPUT_VARIABLES,
        "num_nodes": NUM_NODES,
        "num_edges": EXPECTED_EDGE_COUNT,
        "event_ids_by_split": event_ids_by_split,
        "contains_R019_in_test": contains_R019_in_test,
        "created_files_summary": created_files_summary,
        "warnings": warnings,
    }
    with manifest_json_path.open("w") as f:
        json.dump(manifest, f, indent=2, allow_nan=False)

    total_size = output_size_bytes(output_root)
    summary = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.time() - started, 1),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "script": str(Path(__file__).resolve()),
        "script_git": git_info(Path(__file__)),
        "logic_copied_from": {
            "script": REFERENCE_SCRIPT, "commit": REFERENCE_SCRIPT_COMMIT,
            "reference_dataset_job": REFERENCE_SBATCH, "job_commit": REFERENCE_SBATCH_COMMIT,
        },
        "command_line": sys.argv if argv is None else ["prepare_swegnn_dataset.py", *argv],
        "options": {
            "pt_root": str(pt_root),
            "output_root": str(output_root),
            "cell_size": args.cell_size,
            "rainfall_scale_reference": args.rainfall_scale_reference,
            "split_manifest": str(args.split_manifest) if args.split_manifest else None,
        },
        "source": {"pt_root": str(pt_root), "static_graph_reference_file": str(reference_path),
                   "split_manifest": split_manifest_info},
        "split_counts": split_counts,
        "total_events": total_events,
        "contains_R019_in_test": contains_R019_in_test,
        "num_warnings": len(warnings),
        "output_size_bytes_excluding_summary": total_size,
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, allow_nan=False)

    print("Manifest JSON:", manifest_json_path)
    print("Summary JSON:", summary_path)
    print("Warnings:", len(warnings))
    print(f"Total output size: {total_size} bytes ({total_size / 1024**3:.2f} GiB)")
    print(f"Elapsed: {time.time() - started:.0f} s")
    print("Stockbridge SWE-GNN dataset preparation passed.", flush=True)


if __name__ == "__main__":
    main()
