"""Stockbridge (CityCAT) samples for the original mSWE-GNN MSGNN model.

One sample = one event at one time index t, as a torch_geometric Data object
on the stacked multiscale graph of `stockbridge.multiscale_grid`.

Node features x = [static | dynamic]
------------------------------------
* static (in this order): area, DEM. DEM must be the LAST static feature:
  with `with_WL=True` MSGNN computes WL = x_s[:, -1] + H_t.
  - area: per-scale z-score (as `area_scaler: 'standard'` in config.yaml, which
    fits one scaler per scale in utils/scaling.py),
  - DEM: shifted to its minimum, not scaled (as `process_attr(..., to_min=True)`
    with `DEM_scaler: null`).
* dynamic, last 2*previous_t columns, interleaved per time step exactly as
  `utils.dataset.get_previous_steps` builds them and as `use_prediction`,
  `_add_residual_connection` and `_mask_small_WD` read them:
      [H_{t-p+1}, Q_{t-p+1}, ..., H_t, Q_t]      (p = previous_t, default 2)
  Time indices before 0 use a dry state, as `add_dry_bed_condition`.
  Q is CityCAT Qmag, H is depth in m (no scaling, as WD_scaler/V_scaler: null).
* Coarse nodes: dynamic features are the area-weighted mean of their children
  (equivalently the mean over their active scale-0 descendants). MSGNN does not
  read them: in forward, the coarse hidden states are overwritten by pooling
  the processed finer scale, so these values only matter for the residual
  connection and the output masking on coarse nodes, which the loss ignores.

Edge features: [edge_length (per-scale z-score, as edge_length_scaler: 'standard'),
edge_slope (unscaled, as edge_slope_scaler: null)].

Target y: shape [N, 2, 1] = (H, Q) at t+1. The original loss
(`training.loss.get_multiscale_loss`) and validation only use scale 0, but the
model output has N rows and `loss_function` computes `pred - y` before masking,
so y keeps all N rows; coarse rows hold the area-weighted mean (unused).

Rainfall (not in x yet; patch 2 decides how it enters the model):
  rain_a, rain_b       raw CityCAT intensity (m/s) at the start and end of the
                       predicted interval t -> t+1, "alternative" alignment:
                       rain_a = PT[t+1], rain_b = PT[t+2], with one zero
                       appended to PT (Stockbridge-SWEGNN M17K).
  rain_interval_m      trapezoid depth over the interval, 0.5*(a+b)*300 s, in m.

Boundary-condition placeholders: the original training utilities
(`adapt_batch_training`, `rollout_test`, `training_step`) expect node_BC,
type_BC, BC, edge_BC_length, temporal_res and previous_t on every sample.
A rainfall domain has no inflow boundary, so node_BC/BC/edge_BC_length are
empty tensors; type_BC = 2 only satisfies `check_type_BC`.

Event loading copies the logic of `load_dynamic_event`, `derive_split_event_ids`
and `prepare_dynamic_events` from
Stockbridge-SWEGNN/scripts/train_valid_stockbridge_full_onestep_earlystop.py
and `prepare_events`/`interval_rainfall` from
Stockbridge-SWEGNN/scripts/train_valid_stockbridge_m17k.py (M17K zip). They
are copied instead of imported because that module hard-codes checks for the
99,430-node raster graph and a different default dataset.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from stockbridge.multiscale_grid import MultiscaleGrid, build_multiscale_grid

DEFAULT_DATASET_ROOT = (
    Path.home() / "flood/02_data/CityCAT_Winchester/swegnn" / "stockbridge_phase2_100_5m_v01"
)
NY, NX, NT = 305, 326, 25
NUM_RASTER_CELLS = NY * NX
CELL_SIZE = 5.0
DT_SECONDS = 300.0
TEMPORAL_RES_MIN = 5
EXPECTED_ACTIVE_CELLS = 32249
SPLITS = ("train", "valid", "test")
STATIC_NODE_FEATURES = ["area", "DEM"]
EDGE_FEATURES = ["edge_length", "edge_slope"]
WATER_VARIABLES = ["H", "Qmag"]
RAINFALL_ALIGNMENTS = ("original", "alternative")


# ---- reading the dataset (copied logic, see module docstring) ---------------

def load_static_dem(dataset_root: Path = DEFAULT_DATASET_ROOT) -> tuple[np.ndarray, dict]:
    """Return the DEM raster (NY, NX) from static_graph.pt and some metadata."""
    path = Path(dataset_root) / "static_graph.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Static graph not found: {path}")
    static_graph = torch.load(path, map_location="cpu")
    if not isinstance(static_graph, dict) or "DEM" not in static_graph:
        raise TypeError(f"Expected a dict with 'DEM' in {path}")
    for key, expected in (("ny", NY), ("nx", NX)):
        if key in static_graph and static_graph[key] != expected:
            raise ValueError(f"static_graph {key}={static_graph[key]}, expected {expected}")
    dem = static_graph["DEM"].double().reshape(-1)
    if dem.numel() != NUM_RASTER_CELLS:
        raise ValueError(f"DEM has {dem.numel()} cells, expected {NUM_RASTER_CELLS}")
    if not torch.isfinite(dem).all():
        raise ValueError("DEM contains non-finite values")
    meta = {
        "static_graph_cell_size": static_graph.get("cell_size"),
        "cell_size_used": CELL_SIZE,
    }
    # Row-major (i * NX + j), as built by prepare_stockbridge_swegnn_full_dataset.py
    return dem.reshape(NY, NX).numpy(), meta


def _natural_event_key(event_id: str) -> tuple[int, str]:
    match = re.fullmatch(r"R(\d+)", event_id)
    if match is None:
        raise ValueError(f"Invalid Stockbridge event ID: {event_id!r}")
    return int(match.group(1)), event_id


def split_counts(dataset_root: Path, split: str) -> int:
    """Number of events of a split, read from the dataset itself.

    Taken from `split_counts` of manifest.json when it has the split, else
    counted from events/<split>/*.pt. When both exist they must agree.
    """
    dataset_root = Path(dataset_root)
    manifest_count = None
    manifest_path = dataset_root / "manifest.json"
    if manifest_path.is_file():
        counts = json.loads(manifest_path.read_text()).get("split_counts")
        if isinstance(counts, dict) and split in counts:
            manifest_count = int(counts[split])
    folder_count = None
    split_dir = dataset_root / "events" / split
    if split_dir.is_dir():
        folder_count = sum(1 for _ in split_dir.glob("*.pt"))
    if manifest_count is not None and folder_count is not None and manifest_count != folder_count:
        raise ValueError(f"{split}: manifest.json split_counts says {manifest_count} events, "
                         f"{split_dir} has {folder_count}")
    if manifest_count is None and folder_count is None:
        raise FileNotFoundError(f"No split_counts[{split!r}] in {manifest_path} and no {split_dir}")
    return manifest_count if manifest_count is not None else folder_count


def split_event_ids(dataset_root: Path, split: str,
                    expected_counts: dict[str, int] | None = None) -> list[str]:
    """Event IDs of a split, from manifest.json if it lists them, else from files.

    The expected size comes from the dataset (`split_counts`); with
    `expected_counts` (e.g. {"train": 70, "valid": 15, "test": 15}) the size
    must also equal expected_counts[split].
    """
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}, expected one of {SPLITS}")
    dataset_root = Path(dataset_root)
    event_ids = None
    manifest_path = dataset_root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        by_split = manifest.get("event_ids_by_split")
        if isinstance(by_split, dict) and split in by_split:
            event_ids = list(by_split[split])
    if event_ids is None:
        split_dir = dataset_root / "events" / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Event split directory not found: {split_dir}")
        event_ids = [p.stem for p in split_dir.glob("*.pt")]
    event_ids = sorted(event_ids, key=_natural_event_key)
    expected = split_counts(dataset_root, split)
    if len(event_ids) != expected:
        raise ValueError(f"Expected {expected} {split} events (dataset split_counts), found {len(event_ids)}")
    if expected_counts is not None and split in expected_counts and expected_counts[split] != expected:
        raise ValueError(f"{split}: dataset has {expected} events, --expected-split-counts says "
                         f"{expected_counts[split]}")
    return event_ids


def load_event(dataset_root: Path, split: str, event_id: str) -> dict:
    """Load one event on CPU: H, Qmag (NUM_RASTER_CELLS, NT) and zero-extended rainfall."""
    path = Path(dataset_root) / "events" / split / f"{event_id}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Dynamic event file not found: {path}")
    event = torch.load(path, map_location="cpu")
    for key in ("H", "Qmag", "rainfall_global"):
        if key not in event:
            raise KeyError(f"{path} is missing {key!r}")
    for key, shape in (("H", (NUM_RASTER_CELLS, NT)), ("Qmag", (NUM_RASTER_CELLS, NT)), ("rainfall_global", (NT,))):
        if tuple(event[key].shape) != shape:
            raise ValueError(f"{path}: {key} has shape {tuple(event[key].shape)}, expected {shape}")
        if not torch.isfinite(event[key]).all():
            raise ValueError(f"{path}: {key} has non-finite values")
    rain = event["rainfall_global"].double()
    return {
        "event_id": event_id,
        "split": split,
        "H": event["H"].float(),
        "Qmag": event["Qmag"].float(),
        "rainfall_global": rain,
        # PT extended with one zero (TXT at 7200 s, verified zero for Phase 2)
        "rainfall_extended": torch.cat([rain, rain.new_zeros(1)]),
    }


def interval_rainfall(event: dict, t: int, alignment: str = "alternative") -> tuple[float, float]:
    """Rain intensities (m/s) at the start and end of the interval t -> t+1."""
    rain = event["rainfall_extended"]
    if alignment == "original":
        return float(rain[t]), float(rain[t + 1])
    if alignment == "alternative":
        return float(rain[t + 1]), float(rain[t + 2])
    raise ValueError(f"Unknown rainfall alignment {alignment!r}")


def interval_rainfall_tensors(event: dict, t: int, alignment: str = "alternative"):
    """Tensor version of `interval_rainfall`: (1,) slices of rainfall_extended.

    Indexing keeps the rain on the device of `rainfall_extended`, so building a
    sample on the GPU needs neither a host sync nor a host-to-device copy.
    """
    rain = event["rainfall_extended"]
    if alignment == "original":
        return rain[t:t + 1], rain[t + 1:t + 2]
    if alignment == "alternative":
        return rain[t + 1:t + 2], rain[t + 2:t + 3]
    raise ValueError(f"Unknown rainfall alignment {alignment!r}")


def node_event_to(node_event: dict, device) -> dict:
    """Copy of a node event (output of event_to_nodes) with its tensors on `device`.

    Pool on the CPU (event_to_nodes) and then move: CUDA index_add_ sums with
    atomics, so pooling on the GPU gives float32 round-off differences
    (~1e-7 relative) on the coarse rows, while CPU pooling + this move gives
    samples bit-identical to the CPU ones.
    """
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in node_event.items()}


# ---- static features and normalisation --------------------------------------

def _zscore_stats(values: np.ndarray) -> dict:
    std = float(values.std())
    return {"mean": float(values.mean()), "std": std if std > 0 else 1.0}


def compute_static_stats(grid: MultiscaleGrid, train_event_ids: list[str] | None = None) -> dict:
    """Normalisation statistics of the static features.

    The static graph is shared by all events, so the statistics are the same
    whichever split they are computed on; `train_event_ids` is only recorded
    to document that no valid/test information is used.
    """
    stats = {
        "num_scales": grid.num_scales,
        "nodes_per_scale": grid.nodes_per_scale(),
        "edges_per_scale": grid.edges_per_scale(),
        "static_node_features": STATIC_NODE_FEATURES,
        "edge_features": EDGE_FEATURES,
        "area": [],
        "edge_length": [],
        "DEM": {"shift": float(grid.dem.min()), "scale": 1.0, "rule": "DEM - min(DEM over all scales)"},
        "edge_slope": {"rule": "unscaled"},
        "area_rule": "per-scale z-score",
        "edge_length_rule": "per-scale z-score (std 0 -> 1)",
        "computed_from": "shared static graph (identical for train/valid/test)",
        "train_event_ids": train_event_ids,
        "cell_size_m": grid.cell_size,
    }
    for s in range(grid.num_scales):
        stats["area"].append(_zscore_stats(grid.area[grid.scale_slice(s)]))
        e = slice(int(grid.edge_ptr[s]), int(grid.edge_ptr[s + 1]))
        stats["edge_length"].append(_zscore_stats(grid.edge_length[e]))
    return stats


def save_stats(stats: dict, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(stats, indent=2))


def static_features(grid: MultiscaleGrid, stats: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Return x_static (N, 2) = [area, DEM] and edge_attr (E, 2) = [length, slope]."""
    area = np.empty(grid.num_nodes)
    for s in range(grid.num_scales):
        sl = grid.scale_slice(s)
        st = stats["area"][s]
        area[sl] = (grid.area[sl] - st["mean"]) / st["std"]
    dem = (grid.dem - stats["DEM"]["shift"]) / stats["DEM"]["scale"]
    x_static = torch.tensor(np.stack([area, dem], axis=1), dtype=torch.float32)

    length = np.empty(grid.num_edges)
    for s in range(grid.num_scales):
        e = slice(int(grid.edge_ptr[s]), int(grid.edge_ptr[s + 1]))
        st = stats["edge_length"][s]
        length[e] = (grid.edge_length[e] - st["mean"]) / st["std"]
    edge_attr = torch.tensor(np.stack([length, grid.edge_slope], axis=1), dtype=torch.float32)
    return x_static, edge_attr


# ---- samples ------------------------------------------------------------------

class StockbridgeMultiscale:
    """Multiscale graph + static features, ready to turn events into samples."""

    def __init__(self, dem: np.ndarray, num_scales: int = 4, stats: dict | None = None,
                 train_event_ids: list[str] | None = None, previous_t: int = 2,
                 rainfall_alignment: str = "alternative"):
        if previous_t < 1:
            raise ValueError("previous_t must be >= 1")
        if rainfall_alignment not in RAINFALL_ALIGNMENTS:
            raise ValueError(f"rainfall_alignment must be one of {RAINFALL_ALIGNMENTS}")
        self.grid = build_multiscale_grid(dem, num_scales=num_scales, cell_size=CELL_SIZE)
        n0 = self.grid.nodes_per_scale()[0]
        if n0 != EXPECTED_ACTIVE_CELLS:
            raise ValueError(f"Found {n0} active cells, expected {EXPECTED_ACTIVE_CELLS}")
        self.stats = stats if stats is not None else compute_static_stats(self.grid, train_event_ids)
        self.x_static, self.edge_attr = static_features(self.grid, self.stats)
        self.previous_t = previous_t
        self.rainfall_alignment = rainfall_alignment
        g = self.grid
        self.edge_index = torch.from_numpy(g.edge_index).long()
        self.intra_mesh_edge_index = torch.from_numpy(g.intra_mesh_edge_index).long()
        self.node_ptr = torch.from_numpy(g.node_ptr).long()
        self.edge_ptr = torch.from_numpy(g.edge_ptr).long()
        self.intra_edge_ptr = torch.from_numpy(g.intra_edge_ptr).long()
        self.flat_index0 = torch.from_numpy(g.flat_index0).long()
        self._ancestors = [torch.from_numpy(a).long() for a in g.ancestors]
        self._counts = [torch.bincount(a).float()[:, None] for a in self._ancestors]

    _TENSOR_ATTRS = ("x_static", "edge_attr", "edge_index", "intra_mesh_edge_index", "node_ptr",
                     "edge_ptr", "intra_edge_ptr", "flat_index0")

    @property
    def device(self) -> torch.device:
        return self.x_static.device

    def to(self, device) -> "StockbridgeMultiscale":
        """Shallow copy whose static tensors (features, edges, pointers, pooling
        indices) live on `device`. The numpy grid and the stats are shared.
        Samples built by the copy are created on `device` without host copies
        (move the node events with `node_event_to`)."""
        device = torch.device(device)
        out = copy.copy(self)
        for name in self._TENSOR_ATTRS:
            setattr(out, name, getattr(self, name).to(device))
        out._ancestors = [a.to(device) for a in self._ancestors]
        out._counts = [c.to(device) for c in self._counts]
        out.__dict__.pop("_pooling_cache", None)
        return out

    @property
    def num_nodes(self) -> int:
        return self.grid.num_nodes

    def event_to_nodes(self, event: dict) -> dict:
        """H, Qmag on all N nodes: (N, NT). Coarse rows = area-weighted mean of children."""
        out = {}
        for name in WATER_VARIABLES:
            v0 = event[name][self.flat_index0]  # (n0, NT)
            blocks = [v0]
            for s in range(1, self.grid.num_scales):
                n_s = self._counts[s].shape[0]
                pooled = torch.zeros(n_s, v0.shape[1], dtype=v0.dtype, device=v0.device).index_add_(
                    0, self._ancestors[s], v0)
                blocks.append(pooled / self._counts[s])
            out[name] = torch.cat(blocks, 0)
        out["event_id"] = event["event_id"]
        out["rainfall_extended"] = event["rainfall_extended"]
        return out

    def _pooling_tensors(self, device: torch.device):
        """_ancestors / _counts on `device` (cached per device)."""
        cache = self.__dict__.setdefault("_pooling_cache", {})
        key = str(device)
        if key not in cache:
            cache[key] = ([a.to(device) for a in self._ancestors], [c.to(device) for c in self._counts])
        return cache[key]

    def fine_to_all_scales(self, state0: torch.Tensor) -> torch.Tensor:
        """Full (N, C) state from the scale-0 state (n0, C), e.g. C = 2 for [H, Q].

        Coarse rows are the area-weighted mean of the active scale-0
        descendants, exactly as in `event_to_nodes` (every scale-0 cell has
        the same area, so this is the plain mean over descendants). Pure
        torch, differentiable, on the device and dtype of `state0`. Used in a
        rollout so the coarse inputs follow the fine prediction instead of
        the model's untrained coarse outputs.
        """
        n0 = self._counts[0].shape[0]
        if state0.shape[0] != n0:
            raise ValueError(f"state0 must have {n0} rows (scale 0), got {state0.shape[0]}")
        ancestors, counts = self._pooling_tensors(state0.device)
        blocks = [state0]
        for s in range(1, self.grid.num_scales):
            n_s = counts[s].shape[0]
            pooled = torch.zeros(n_s, state0.shape[1], dtype=state0.dtype, device=state0.device)
            pooled = pooled.index_add(0, ancestors[s], state0)
            blocks.append(pooled / counts[s].to(state0.dtype))
        return torch.cat(blocks, 0)

    def build_sample(self, node_event: dict, t: int) -> Data:
        """Teacher-forced sample for the step t -> t+1 (0 <= t <= NT-2).

        Every tensor is created on the builder's device (see `to`); the node
        event must be on the same device (see `node_event_to`).
        """
        if not 0 <= t <= NT - 2:
            raise ValueError(f"t must satisfy 0 <= t <= {NT - 2}, got {t}")
        p = self.previous_t
        H, Q = node_event["H"], node_event["Qmag"]
        dyn = []
        for k in range(t - p + 1, t + 1):  # oldest first, as get_previous_steps
            if k < 0:  # dry bed before the start of the event
                zeros = torch.zeros(H.shape[0], 1, dtype=H.dtype, device=H.device)
                dyn += [zeros, zeros]
            else:
                dyn += [H[:, k:k + 1], Q[:, k:k + 1]]
        x = torch.cat([self.x_static] + dyn, dim=1)
        y = torch.stack([H[:, t + 1], Q[:, t + 1]], dim=1).unsqueeze(-1)  # (N, 2, 1)

        rain_a, rain_b = interval_rainfall_tensors(node_event, t, self.rainfall_alignment)
        data = Data(
            x=x,
            y=y,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            intra_mesh_edge_index=self.intra_mesh_edge_index,
        )
        data.num_nodes = self.num_nodes
        data.node_ptr = self.node_ptr
        data.edge_ptr = self.edge_ptr
        data.intra_edge_ptr = self.intra_edge_ptr
        # same values as float32(python float) before: the sum is done in the
        # rain dtype (float64) and cast once
        data.rain_a = rain_a.to(torch.float32)
        data.rain_b = rain_b.to(torch.float32)
        data.rain_interval_m = (0.5 * (rain_a + rain_b) * DT_SECONDS).to(torch.float32)
        data.time = t
        data.event_id = node_event["event_id"]
        data.previous_t = p
        data.temporal_res = TEMPORAL_RES_MIN
        # boundary-condition placeholders (no inflow boundary in a rainfall domain)
        dev = self.device
        data.node_BC = torch.zeros(0, dtype=torch.long, device=dev)
        data.type_BC = 2
        data.BC = torch.zeros(0, p, 2, device=dev)
        data.edge_BC_length = torch.zeros(0, device=dev)
        return data
