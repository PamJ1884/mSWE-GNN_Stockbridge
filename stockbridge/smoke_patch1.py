"""Patch 1 smoke test: Stockbridge multiscale data + ORIGINAL MSGNN forward.

No training. Run from anywhere:
    python stockbridge/smoke_patch1.py [--dataset-root ...] [--output-dir ...]

Steps
  1. Imports of the original model without meshkernel/triangle.
  2. Hierarchy with 4 and 5 scales: nodes, edges, inter-scale pairs and
     connected components per scale; one-parent check; comparison with the
     expected node counts (reported, never forced).
  3. Samples of one train event (R019 if it is in train, else the first train
     event) at t = 0 and t = 12, plus a layout check against use_prediction.
  4. Original MSGNN (hyperparameters from config.yaml, previous_t = 2) with
     random weights: forward per sample (shape, NaN, time, peak memory) for 4
     and 5 scales.
  5. Batch of 2 samples with the PyG DataLoader + adapt_batch_training, and a
     check that the batched output equals the single-sample outputs.
  6. smoke_report.json, normalisation_stats_S{4,5}.json and a node-map figure.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from stockbridge.multiscale_grid import (  # noqa: E402
    EXPECTED_NODES_PER_SCALE,
    check_hierarchy,
    connected_components_per_scale,
)
from stockbridge import dataset as sbd  # noqa: E402

DEFAULT_OUTPUT_DIR = Path.home() / "flood/03_outputs/mSWE-GNN_Stockbridge/patch1_smoke_v01"
PREVIOUS_T = 2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", type=Path, default=sbd.DEFAULT_DATASET_ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--event", default="R019", help="preferred train event")
    p.add_argument("--times", default="0,12")
    p.add_argument("--scales", default="4,5")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def rss_peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # Linux: KiB


def build_model(config_path: Path, num_node_features, num_edge_features, num_scales, device):
    from models.gnn import MSGNN
    cfg = yaml.safe_load(config_path.read_text())
    params = dict(cfg["models"])
    model_type = params.pop("model_type")
    assert model_type == "MSGNN", f"config.yaml model_type is {model_type}, expected MSGNN"
    model = MSGNN(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        num_scales=num_scales,
        previous_t=PREVIOUS_T,  # config.yaml says 3; patch 1 uses 2 (decided with the user)
        device=device,
        **params,
    ).to(device)
    model.eval()
    return model, params, cfg["temporal_dataset_parameters"]["previous_t"]


def timed_forward(model, data, device):
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(data)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    mem = (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else rss_peak_mb()
    return out, dt, mem


def plot_node_maps(grid, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S = grid.num_scales
    fig, axes = plt.subplots(1, S, figsize=(4.2 * S, 4.4), constrained_layout=True)
    ny, nx = grid.raster_shape
    extent = (0, nx * grid.cell_size, ny * grid.cell_size, 0)
    for s, ax in enumerate(np.atleast_1d(axes)):
        n_children = grid.area[grid.scale_slice(s)] / grid.cell_size ** 2
        raster = grid.to_raster(n_children, scale=s)
        im = ax.imshow(raster, extent=extent, interpolation="nearest", cmap="viridis",
                       vmin=0, vmax=4 ** s if s else 1)
        e = grid.edge_index[:, grid.edge_ptr[s]:grid.edge_ptr[s + 1]]
        if s >= 2:  # draw coarse edges (scale 0/1 too dense to read)
            c = grid.centroid
            for a, b in e.T[e[0] < e[1]]:
                ax.plot([c[a, 0], c[b, 0]], [c[a, 1], c[b, 1]], color="w", lw=0.3, alpha=0.7)
        ax.set_title(f"scale {s}: {grid.nodes_per_scale()[s]} nodes, {grid.edges_per_scale()[s]} edges", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.7, label="active scale-0 cells")
    fig.suptitle("Stockbridge multiscale hierarchy (2x2 grouping; white: coarse edges)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir: Path = args.output_dir
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        sys.exit(f"Output directory {out_dir} is not empty; use --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    times = [int(t) for t in args.times.split(",")]
    scales = [int(s) for s in args.scales.split(",")]
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")

    report = {
        "git_commit": git_commit(),
        "host": platform.node(),
        "torch": torch.__version__,
        "device": str(device),
        "dataset_root": str(args.dataset_root),
        "previous_t": PREVIOUS_T,
    }

    # 1. imports -----------------------------------------------------------------
    import torch_geometric
    report["torch_geometric"] = torch_geometric.__version__
    report["meshkernel_installed"] = importlib.util.find_spec("meshkernel") is not None
    report["triangle_installed"] = importlib.util.find_spec("triangle") is not None
    from models.gnn import MSGNN  # noqa: F401
    from utils.dataset import use_prediction
    from training.train import adapt_batch_training
    from torch_geometric.loader import DataLoader
    report["imports_ok"] = True
    print(f"imports ok | meshkernel installed: {report['meshkernel_installed']} | "
          f"triangle installed: {report['triangle_installed']}")

    # 2. hierarchy ---------------------------------------------------------------
    dem, dem_meta = sbd.load_static_dem(args.dataset_root)
    report.update(dem_meta)
    if dem_meta["static_graph_cell_size"] not in (None, sbd.CELL_SIZE):
        print(f"NOTE: static_graph.pt says cell_size={dem_meta['static_graph_cell_size']}; using {sbd.CELL_SIZE} m")

    train_ids = sbd.split_event_ids(args.dataset_root, "train")
    event_id = args.event if args.event in train_ids else train_ids[0]
    report["event_id"] = event_id
    report["event_note"] = "requested event is in train" if event_id == args.event else \
        f"{args.event} is not in train; using first train event"
    print(f"event: {event_id} ({report['event_note']})")

    builders = {}
    report["hierarchy"] = {}
    for S in scales:
        b = sbd.StockbridgeMultiscale(dem, num_scales=S, train_event_ids=train_ids, previous_t=PREVIOUS_T)
        builders[S] = b
        g = b.grid
        expected = list(EXPECTED_NODES_PER_SCALE[:S])
        h = {
            "nodes_per_scale": g.nodes_per_scale(),
            "expected_nodes_per_scale": expected,
            "matches_expected": g.nodes_per_scale() == expected,
            "edges_per_scale": g.edges_per_scale(),
            "inter_scale_pairs": np.diff(g.intra_edge_ptr).tolist(),
            "connected_components_per_scale": connected_components_per_scale(g),
            "checks": check_hierarchy(g),
            "total_nodes": g.num_nodes,
            "total_edges": g.num_edges,
        }
        report["hierarchy"][S] = h
        sbd.save_stats(b.stats, out_dir / f"normalisation_stats_S{S}.json")
        print(f"\n[S={S}] nodes {h['nodes_per_scale']} (expected {expected}) -> "
              f"{'MATCH' if h['matches_expected'] else 'DIFFERENT'}")
        print(f"[S={S}] edges {h['edges_per_scale']} | inter-scale pairs {h['inter_scale_pairs']}")
        print(f"[S={S}] connected components {h['connected_components_per_scale']}")
        print(f"[S={S}] one parent per fine node {h['checks']['exactly_one_parent_per_fine_node']}")
    plot_node_maps(builders[max(scales)].grid, out_dir / "multiscale_node_maps.png")

    # 3-5. samples, forward, batching ----------------------------------------------
    event = sbd.load_event(args.dataset_root, "train", event_id)
    report["forward"] = {}
    for S in scales:
        b = builders[S]
        node_event = b.event_to_nodes(event)
        samples = {t: b.build_sample(node_event, t) for t in sorted(set(times + [0, 1]))}

        # layout check: x(t=1) dynamics == use_prediction(x(t=0), y(t=0))
        shifted = use_prediction(samples[0].x, samples[0].y[:, :, 0], PREVIOUS_T)
        layout_ok = bool(torch.equal(shifted, samples[1].x))

        model, params, cfg_prev_t = build_model(
            args.config, samples[0].x.shape[1], samples[0].edge_attr.shape[1], S, device)
        n_params = sum(p.numel() for p in model.parameters())
        res = {"layout_matches_use_prediction": layout_ok, "num_parameters": n_params,
               "model_params": params, "config_previous_t": cfg_prev_t, "samples": {}}
        print(f"\n[S={S}] x {tuple(samples[0].x.shape)} edge_attr {tuple(samples[0].edge_attr.shape)} "
              f"y {tuple(samples[0].y.shape)} | layout == use_prediction: {layout_ok} | params {n_params}")

        singles = {}
        for t in times:
            d = samples[t].clone().to(device)  # Data.to works in place; keep the CPU samples for batching
            # warm-up so the timing does not include CUDA initialisation
            if device.type == "cuda":
                with torch.no_grad():
                    model(d)
            out, dt, mem = timed_forward(model, d, device)
            singles[t] = out.detach().cpu()
            info = {
                "output_shape": list(out.shape),
                "has_nan": bool(torch.isnan(out).any()),
                "seconds": dt,
                "peak_memory_mb": mem,
                "memory_kind": "cuda max_memory_allocated" if device.type == "cuda" else "CPU RSS peak",
                "rain_a_mps": float(d.rain_a), "rain_b_mps": float(d.rain_b),
                "rain_interval_mm": float(d.rain_interval_m) * 1000,
                "scale0_H_max_input": float(d.x[:d.node_ptr[1], -2].max()),
                "scale0_H_max_target": float(d.y[:d.node_ptr[1], 0, 0].max()),
                "scale0_H_max_pred": float(out[:d.node_ptr[1], 0].max()),
            }
            res["samples"][t] = info
            print(f"[S={S}] t={t:2d} out {tuple(out.shape)} NaN {info['has_nan']} "
                  f"{dt*1000:.1f} ms peak {mem:.0f} MB | rain {info['rain_interval_mm']:.3f} mm/interval")

        # batch of 2
        loader = DataLoader([samples[t] for t in times[:2]], batch_size=2, shuffle=False)
        batch = next(iter(loader))
        temp = adapt_batch_training(batch).to(device)
        out_b, dt_b, mem_b = timed_forward(model, temp, device)
        out_b = out_b.cpu()
        N = b.num_nodes
        diffs = [float((out_b[i * N:(i + 1) * N] - singles[t]).abs().max()) for i, t in enumerate(times[:2])]
        res["batch2"] = {
            "output_shape": list(out_b.shape),
            "has_nan": bool(torch.isnan(out_b).any()),
            "seconds": dt_b,
            "peak_memory_mb": mem_b,
            "node_ptr": temp.node_ptr.tolist(),
            "edge_ptr": temp.edge_ptr.tolist(),
            "intra_edge_ptr": temp.intra_edge_ptr.tolist(),
            "max_abs_diff_vs_single": diffs,
            "matches_single": all(d < 1e-4 for d in diffs),
        }
        print(f"[S={S}] batch2 out {tuple(out_b.shape)} NaN {res['batch2']['has_nan']} "
              f"{dt_b*1000:.1f} ms peak {mem_b:.0f} MB | max|batched-single| {diffs}")
        report["forward"][S] = res
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    (out_dir / "smoke_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {out_dir / 'smoke_report.json'}")


if __name__ == "__main__":
    main()
