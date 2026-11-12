"""Checks of a converted SWE-GNN dataset; writes validation_report.json.

    python stockbridge/data_prep/validate_swegnn_dataset.py \
        --dataset .../swegnn/stockbridge_phase2plus3_300_5m_v01 \
        --pt-manifest .../pt/stockbridge_phase2plus3_300_v01/stockbridge_phase2plus3_300_manifest.csv \
        --reference-dataset .../swegnn/stockbridge_phase2_100_5m_v01

Checks (each one in the report with passed + details):
  split_membership       events/<split>/ and manifest.json == the PT manifest
                         (the splits U-RNN and DNO used); every event file
                         stores its own event_id and split.
  dem                    DEM channel of every source PT event (all time
                         steps) == static_graph DEM, bit for bit; static
                         DEM == reference (phase 2) static DEM; active mask
                         DEM != DEM.max() has 32,249 cells.
  static_graph_reference whole static_graph.pt identical to the reference.
  reference_events       events present in the reference dataset have H,
                         Qmag and rainfall_global identical to it, whatever
                         split they are in.
  source_consistency     H == PT H and Qmag == |(Vx H, Vy H)| recomputed
                         from the PT tensor, rainfall_global == PT rain[0, 0].
  finite                 no NaN / inf in H, Qmag, rainfall_global.
  qmag_nonnegative       Qmag >= 0.
  rainfall_uniform       PT rainfall identical in every cell (max |diff| <= 1e-8,
                         the conversion tolerance; the value is reported).
Per-event summary: H_max, Qmag_max, rainfall sum (m/s), depth (mm,
sum x 300 s), rainfall at the last step.

Exit code 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import platform
import re
import sys
from pathlib import Path

import torch

NY, NX, NT = 305, 326, 25
NUM_NODES = NY * NX
EXPECTED_ACTIVE_CELLS = 32249
DT_SECONDS = 300.0
RAINFALL_UNIFORMITY_TOLERANCE = 1e-8
SPLITS = ("train", "valid", "test")


def event_id(text: str) -> str:
    match = re.search(r"(R\d{3})", text)
    if match is None:
        raise ValueError(f"No event ID in {text!r}")
    return match.group(1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_dicts_equal(a: dict, b: dict) -> list[str]:
    """Keys whose values differ (tensors by torch.equal, others by ==)."""
    bad = []
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            bad.append(k)
        elif isinstance(a[k], torch.Tensor):
            if not (isinstance(b[k], torch.Tensor) and a[k].dtype == b[k].dtype
                    and a[k].shape == b[k].shape and torch.equal(a[k], b[k])):
                bad.append(k)
        elif a[k] != b[k]:
            bad.append(k)
    return bad


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--pt-manifest", type=Path, required=True,
                   help="CSV with event_id, split and source/view path columns")
    p.add_argument("--reference-dataset", type=Path, required=True,
                   help="validated SWE-GNN dataset whose events must be reproduced exactly")
    p.add_argument("--report", type=Path, default=None,
                   help="default: <dataset>/validation_report.json")
    args = p.parse_args(argv)
    root = args.dataset.expanduser()
    ref_root = args.reference_dataset.expanduser()
    report_path = args.report or root / "validation_report.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite {report_path}")

    checks: dict[str, dict] = {}

    # ---- split membership ----------------------------------------------------
    with args.pt_manifest.open(newline="") as f:
        pt_rows = list(csv.DictReader(f))
    pt_by_split = {s: sorted(event_id(r["event_id"]) for r in pt_rows if r["split"] == s) for s in SPLITS}
    pt_phase = {event_id(r["event_id"]): r.get("phase") for r in pt_rows}
    folder_by_split = {s: sorted(q.stem for q in (root / "events" / s).glob("*.pt")) for s in SPLITS}
    manifest = json.loads((root / "manifest.json").read_text())
    manifest_by_split = {s: sorted(manifest["event_ids_by_split"][s]) for s in SPLITS}
    membership = {
        "pt_manifest": str(args.pt_manifest),
        "pt_manifest_sha256": sha256_file(args.pt_manifest),
        "counts": {s: len(pt_by_split[s]) for s in SPLITS},
        "folders_equal_pt_manifest": folder_by_split == pt_by_split,
        "manifest_json_equal_pt_manifest": manifest_by_split == pt_by_split,
        "num_pt_manifest_rows": len(pt_rows),
    }
    for s in SPLITS:
        if folder_by_split[s] != pt_by_split[s]:
            membership[f"{s}_only_in_folder"] = sorted(set(folder_by_split[s]) - set(pt_by_split[s]))
            membership[f"{s}_only_in_pt_manifest"] = sorted(set(pt_by_split[s]) - set(folder_by_split[s]))

    # ---- static graph vs reference ----------------------------------------------
    static = torch.load(root / "static_graph.pt", map_location="cpu")
    ref_static = torch.load(ref_root / "static_graph.pt", map_location="cpu")
    dem = static["DEM"]
    active = int((dem != dem.max()).sum())
    static_diff = tensor_dicts_equal(static, ref_static)
    ref_static_path = ref_root / "static_graph.pt"
    checks["static_graph_reference"] = {
        "passed": not static_diff,
        "sha256_equal": sha256_file(root / "static_graph.pt") == sha256_file(ref_static_path),
        "differing_keys": static_diff,
        "reference": str(ref_static_path),
    }

    ref_event_paths = {q.stem: q for s in SPLITS for q in (ref_root / "events" / s).glob("*.pt")}

    # ---- per-event pass ----------------------------------------------------------
    source_by_id = {}
    with (root / "manifest.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            source_by_id[row["event_id"]] = Path(row["source_path"])

    events = []
    failures = {k: [] for k in ("event_fields", "dem", "reference_events", "source_consistency",
                                "finite", "qmag_nonnegative", "rainfall_uniform")}
    reference_matched = []
    reference_split_changes = []
    max_rain_spatial = 0.0
    all_ids = [(s, e) for s in SPLITS for e in folder_by_split[s]]
    for i, (split, eid) in enumerate(all_ids, 1):
        ev = torch.load(root / "events" / split / f"{eid}.pt", map_location="cpu")
        H, Q, rain = ev["H"], ev["Qmag"], ev["rainfall_global"]
        if ev.get("event_id") != eid or ev.get("split") != split:
            failures["event_fields"].append(eid)

        src = torch.load(source_by_id[eid], map_location="cpu")
        dem_src = src[:, :, :, 4].reshape(NUM_NODES, NT)
        if not torch.equal(dem_src, dem.view(NUM_NODES, 1).expand(NUM_NODES, NT)):
            failures["dem"].append(eid)
        H_src = src[..., 0].reshape(NUM_NODES, NT)
        Q_src = torch.sqrt((src[..., 1].reshape(NUM_NODES, NT) * H_src).square()
                           + (src[..., 2].reshape(NUM_NODES, NT) * H_src).square())
        rain_src = src[..., 3]
        rain_spatial = float((rain_src - rain_src[0, 0, :].view(1, 1, NT)).abs().max())
        max_rain_spatial = max(max_rain_spatial, rain_spatial)
        if rain_spatial > RAINFALL_UNIFORMITY_TOLERANCE:
            failures["rainfall_uniform"].append(eid)
        if not (torch.equal(H, H_src) and torch.equal(Q, Q_src) and torch.equal(rain, rain_src[0, 0, :])):
            failures["source_consistency"].append(eid)
        del src, dem_src, H_src, Q_src, rain_src

        if not all(bool(torch.isfinite(t).all()) for t in (H, Q, rain)):
            failures["finite"].append(eid)
        if float(Q.min()) < 0.0:
            failures["qmag_nonnegative"].append(eid)

        if eid in ref_event_paths:
            ref_ev = torch.load(ref_event_paths[eid], map_location="cpu")
            same = all(torch.equal(ev[k], ref_ev[k]) and ev[k].dtype == ref_ev[k].dtype
                       for k in ("H", "Qmag", "rainfall_global"))
            (reference_matched if same else failures["reference_events"]).append(eid)
            if ref_ev.get("split") != split:
                reference_split_changes.append({"event_id": eid, "reference": ref_ev.get("split"), "new": split})

        rain_sum = float(rain.double().sum())
        events.append({
            "event_id": eid, "split": split, "phase": pt_phase.get(eid),
            "in_reference": eid in ref_event_paths,
            "H_max": float(H.max()), "H_min": float(H.min()),
            "Qmag_max": float(Q.max()), "Qmag_min": float(Q.min()),
            "rainfall_sum_m_per_s": rain_sum,
            "rainfall_depth_mm": rain_sum * DT_SECONDS * 1000.0,
            "rainfall_max_m_per_s": float(rain.max()),
            "rainfall_last_m_per_s": float(rain[-1]),
            "rainfall_spatial_max_abs_diff": rain_spatial,
        })
        if i % 25 == 0 or i == len(all_ids):
            print(f"Validated {i}/{len(all_ids)} events", flush=True)

    checks["split_membership"] = {
        "passed": membership["folders_equal_pt_manifest"] and membership["manifest_json_equal_pt_manifest"]
                  and not failures["event_fields"],
        **membership,
        "events_with_wrong_stored_id_or_split": failures["event_fields"],
    }
    checks["dem"] = {
        "passed": not failures["dem"] and active == EXPECTED_ACTIVE_CELLS
                  and torch.equal(dem, ref_static["DEM"]),
        "events_checked": len(all_ids),
        "events_with_different_dem": failures["dem"],
        "static_dem_equal_reference_static_dem": bool(torch.equal(dem, ref_static["DEM"])),
        "active_cells": active,
        "expected_active_cells": EXPECTED_ACTIVE_CELLS,
        "mask_rule": "DEM != DEM.max()",
    }
    n_ref = len(ref_event_paths)
    checks["reference_events"] = {
        "passed": not failures["reference_events"] and len(reference_matched) == n_ref and n_ref > 0,
        "reference_dataset": str(ref_root),
        "reference_events": n_ref,
        "identical_H_Qmag_rainfall": len(reference_matched),
        "differing": failures["reference_events"],
        "missing_from_new_dataset": sorted(set(ref_event_paths) - {e for _, e in all_ids}),
        "split_changes": reference_split_changes,
    }
    checks["source_consistency"] = {"passed": not failures["source_consistency"],
                                    "differing": failures["source_consistency"]}
    checks["finite"] = {"passed": not failures["finite"], "non_finite": failures["finite"]}
    checks["qmag_nonnegative"] = {"passed": not failures["qmag_nonnegative"],
                                  "negative": failures["qmag_nonnegative"],
                                  "global_min": min(e["Qmag_min"] for e in events)}
    checks["rainfall_uniform"] = {"passed": not failures["rainfall_uniform"],
                                  "tolerance": RAINFALL_UNIFORMITY_TOLERANCE,
                                  "max_spatial_abs_diff": max_rain_spatial,
                                  "non_uniform": failures["rainfall_uniform"]}

    def stats(key, phase=None):
        vals = [e[key] for e in events if phase is None or e["phase"] == phase]
        return {"min": min(vals), "max": max(vals), "mean": sum(vals) / len(vals)} if vals else None

    passed = all(c["passed"] for c in checks.values())
    report = {
        "dataset": str(root),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "torch": torch.__version__,
        "script": str(Path(__file__).resolve()),
        "passed": passed,
        "checks": checks,
        "summary_by_phase": {
            ph: {k: stats(k, ph) for k in ("H_max", "Qmag_max", "rainfall_depth_mm", "rainfall_last_m_per_s")}
            for ph in sorted({e["phase"] for e in events}, key=str)
        },
        "nonzero_last_rainfall_events": [e["event_id"] for e in events if e["rainfall_last_m_per_s"] != 0.0],
        "events": events,
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    for name, c in checks.items():
        print(f"{name:24s} {'PASS' if c['passed'] else 'FAIL'}")
    print("Active cells:", active, "| reference events identical:", len(reference_matched), "/", n_ref,
          "| split changes:", len(reference_split_changes))
    print("Events with nonzero rainfall at the last step:", len(report["nonzero_last_rainfall_events"]))
    print("Report:", report_path)
    print("VALIDATION:", "PASS" if passed else "FAIL", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
