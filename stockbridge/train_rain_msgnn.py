"""Patch 3: pilot training of RainMSGNN on the Stockbridge CityCAT events.

Run from anywhere:
    python stockbridge/train_rain_msgnn.py --output-dir ... [--rain-injection fine|all_scales] [--smoke]

Reference: Stockbridge-SWEGNN, scripts/train_valid_stockbridge_m17k.py (stage 1,
"M17K"; repository HEAD 9bb39d00d4d9044c67a96684bf0383faa51a4631, file last
changed in fdfb600a8d3f78a79c7e76af01db63d41595524b). Its logic is copied, not
imported.

Copied from M17K (same behaviour)
---------------------------------
* `parse_rollout_schedule` (extended, see below) and `training_starts`:
  curriculum of rollout steps per epoch; every training sample is
  (event, start t) and starts from the CityCAT state at t, with
  --max-starts-per-event random starts per event and epoch.
* `train_one_epoch`: k-step rollout per sample, loss = mean over steps,
  gradient accumulation over --accumulate-samples samples (the update uses
  the mean loss of the group), non-finite gradient check (raises), gradient
  norm clipping, Adam, StepLR with step 7 epochs and gamma 0.9 (the original
  mSWE-GNN schedule, documented as such in M17K: --lr-step-size 7
  --lr-gamma 0.9). A step of 1 epoch would bring lr 1e-3 down to about
  1.5e-5 by epoch 40, so the model would stop learning in the 6-step stage.
* Loss: M17K --loss swegnn_rmse --loss-q-weight 3 --loss-mask where_water.
* Validation: M17K `evaluate_rollout` (stockbridge/m17k_metrics.py), full
  T0 -> T24 rollout of every valid event from the dry T0.
* Early stopping "since_best": epochs without improving the selection value,
  counted only in the last curriculum stage; the counter restarts when a new
  stage begins.

What changes, and why
---------------------
* Model and data: RainMSGNN on the multiscale graph (patch 1 / 2). The
  rollout feeds back constrain_state with coarse rows rebuilt from scale 0
  (stockbridge/rollout.py). Loss and metrics use the scale-0 rows only
  (= the 32,249 active cells).
* where_water is taken on the constrained state, the loss on the raw output;
  the evaluated state is constrain_state (decisions of the user, see
  stockbridge/m17k_metrics.py).
* Selection: CSI_0p05_mean of the validation rollout (higher is better),
  ties broken by the lower H_RMSE_active_mean. Validation every
  --valid-every epochs (M17K: every epoch).
* --rollout-schedule accepts "steps:rest" as the last item: that stage
  lasts until --epochs.
* Data on the GPU: events are pooled on the CPU (event_to_nodes) and moved
  to the device once; the builder is moved with StockbridgeMultiscale.to,
  so build_sample makes GPU samples without host copies. At start-up a GPU
  sample is checked to be identical to the CPU one.
* Memory: batch 1 + accumulation; --detach-rollout (truncated gradient),
  --grad-checkpoint (torch.utils.checkpoint per rollout step). Before
  training, a memory rehearsal runs one optimisation step with the longest
  rollout of the schedule; if its peak exceeds --max-gpu-mem-gb (or it runs
  out of memory) checkpointing is switched on automatically. The rehearsal
  does not change the model or the optimiser. No mixed precision (depths are
  millimetres).
* Per-module gradient norms (encoders, rain forcing per scale, processor
  per scale, intra-scale GNNs, decoder), before clipping, averaged over the
  optimiser steps of each epoch.
* Baselines with the same metrics, computed once: dry persistence (should
  reproduce the stage-1 0.0555 m H RMSE on valid) and the rain bucket
  (H = accumulated rain, Q = 0).
* The test split is never read.

--smoke: 2 train and 2 valid events, 2 epochs, 2 starts per event, schedule
"1:1,2:rest" (so the multi-step path and a curriculum change are exercised),
and a check that the dry-persistence metrics equal an independent by-hand
computation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import platform
import random
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

from stockbridge import dataset as sbd  # noqa: E402
from stockbridge.m17k_metrics import (  # noqa: E402
    ROLLOUT_FIELDNAMES, Persistence, RainBucket, evaluate_rollout, swegnn_rmse_loss, where_water)
from stockbridge.rain_msgnn import RAIN_FORCINGS, RAIN_INJECTIONS, RainMSGNN, module_groups  # noqa: E402
from stockbridge.rollout import rollout  # noqa: E402

M17K_REFERENCE = {
    "file": "Stockbridge-SWEGNN/scripts/train_valid_stockbridge_m17k.py",
    "repo_head": "9bb39d00d4d9044c67a96684bf0383faa51a4631",
    "file_last_commit": "fdfb600a8d3f78a79c7e76af01db63d41595524b",
}
STAGE1_VALID_REFERENCE = {
    "dry_persistence_H_RMSE_active_mean": 0.0555,
    "best_swegnn_P4_H_RMSE_active_mean": 0.0506,
    "best_swegnn_P4_CSI_0p05_mean": 0.10,
    "best_swegnn_Qmag_peak_percent_of_citycat": "0.3-6",
}
DEFAULT_OUTPUT_ROOT = Path.home() / "flood/03_outputs/mSWE-GNN_Stockbridge/patch3_pilot_v01"
VALID_KEYS = [
    "H_RMSE_active_mean", "Qmag_RMSE_active_mean", "CSI_0p01_mean", "CSI_0p05_mean", "CSI_0p10_mean",
    "H_peak_ratio_mean", "Qmag_peak_ratio_mean", "final_step_pred_over_true_volume_mean",
    "steps_with_all_dry_prediction", "steps_with_zero_Qmag_prediction", "swegnn_val_loss",
    "raw_negative_H_fraction_mean", "raw_negative_Qmag_fraction_mean",
]


def parse_split_counts(text: str) -> dict[str, int]:
    """'train=210,valid=45,test=45' -> {'train': 210, 'valid': 45, 'test': 45}."""
    counts = {}
    for item in text.split(","):
        split, sep, value = item.strip().partition("=")
        if not sep or split not in sbd.SPLITS or split in counts:
            raise argparse.ArgumentTypeError(f"bad split count {item!r} (use e.g. train=210,valid=45)")
        try:
            counts[split] = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"bad split count {item!r}") from None
        if counts[split] < 1:
            raise argparse.ArgumentTypeError(f"split count must be >= 1: {item!r}")
    return counts


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset-root", type=Path, default=sbd.DEFAULT_DATASET_ROOT)
    p.add_argument("--expected-split-counts", type=parse_split_counts, default=None,
                   help="e.g. 'train=70,valid=15' or 'train=210,valid=45': fail unless the dataset "
                        "has exactly these sizes (default: sizes from the dataset manifest/folders)")
    p.add_argument("--output-dir", type=Path, default=None,
                   help=f"default: {DEFAULT_OUTPUT_ROOT}/<injection>_p<previous_t>")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    # model
    p.add_argument("--num-scales", type=int, default=4)
    p.add_argument("--previous-t", type=int, default=2)
    p.add_argument("--seed", type=int, default=None, help="default: models.seed of config.yaml")
    p.add_argument("--rain-injection", choices=RAIN_INJECTIONS, default="fine")
    p.add_argument("--rain-forcing", choices=RAIN_FORCINGS, default="multiplicative")
    p.add_argument("--rain-residual", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dynamic-encoder-bias", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--rain-scale-mm", type=float, default=10.0)
    # loss and curriculum
    p.add_argument("--loss-q-weight", type=float, default=3.0)
    p.add_argument("--rollout-schedule", default="1:5,2:5,4:10,6:rest",
                   help="'steps:epochs,...'; the last item may be 'steps:rest' (until --epochs)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max-starts-per-event", type=int, default=4, help="0 = all starts")
    p.add_argument("--start-time-index", type=int, default=0)
    p.add_argument("--end-time-index", type=int, default=23)
    p.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    # memory
    p.add_argument("--accumulate-samples", type=int, default=4)
    p.add_argument("--detach-rollout", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--max-gpu-mem-gb", type=float, default=40.0)
    # optimisation
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--lr-step-size", type=int, default=7,
                   help="StepLR step in epochs (original mSWE-GNN / M17K: 7; 0 = constant lr). "
                        "With --lr-gamma 0.9, lr 1e-3 is 4.8e-4 at epoch 50 and 2.3e-4 at epoch 100")
    p.add_argument("--lr-gamma", type=float, default=0.9)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    # validation and selection
    p.add_argument("--valid-every", type=int, default=1)
    p.add_argument("--patience", type=int, default=15, help="epochs without improvement (last stage)")
    p.add_argument("--min-delta", type=float, default=1e-8)
    p.add_argument("--cell-area", type=float, default=25.0)
    p.add_argument("--map-times", default="6,12,24", help="target times of the best-checkpoint maps")
    p.add_argument("--smoke", action="store_true", help="2 train / 2 valid events, 2 epochs, 2 starts")
    args = p.parse_args(argv)
    if args.smoke:
        args.epochs = 2
        args.max_starts_per_event = 2
        args.rollout_schedule = "1:1,2:rest"
        args.valid_every = 1
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_ROOT / f"{args.rain_injection}_p{args.previous_t}"
    return args


def parse_rollout_schedule(text: str, epochs: int) -> list[int]:
    """Number of rollout steps for every epoch.

    Copied from M17K (parse_rollout_schedule, --rollout-schedule branch) with
    one extension: the last item may be 'steps:rest', which fills the
    schedule up to `epochs`. Without 'rest' the schedule length sets the
    number of epochs, as in M17K.
    """
    if not text.strip():
        return [1] * epochs
    per_epoch = []
    items = [item.strip() for item in text.split(",")]
    for i, item in enumerate(items):
        try:
            steps_text, stage_text = item.split(":")
            steps = int(steps_text)
            if stage_text == "rest":
                if i != len(items) - 1:
                    raise ValueError("'rest' is only allowed in the last item")
                stage_epochs = epochs - len(per_epoch)
                if stage_epochs <= 0:
                    raise ValueError(f"--epochs {epochs} leaves no epoch for {item!r}")
            else:
                stage_epochs = int(stage_text)
        except ValueError as error:
            raise ValueError(f"Invalid --rollout-schedule item {item!r}: {error}") from error
        if steps <= 0 or stage_epochs <= 0:
            raise ValueError(f"--rollout-schedule values must be positive: {item!r}")
        per_epoch.extend([steps] * stage_epochs)
    return per_epoch


def training_starts(num_events, time_indices, rollout_steps, max_starts_per_event):
    """(event_index, start_t) pairs whose rollout stays inside the time range. Copied from M17K."""
    last_start = time_indices[-1] - rollout_steps + 1
    starts = [t for t in time_indices if t <= last_start]
    if not starts:
        raise ValueError(
            f"{rollout_steps} rollout steps do not fit in "
            f"T{time_indices[0]}..T{time_indices[-1] + 1}"
        )
    pairs = []
    for event_index in range(num_events):
        event_starts = starts
        if 0 < max_starts_per_event < len(starts):
            event_starts = random.sample(starts, max_starts_per_event)
        pairs.extend((event_index, t) for t in event_starts)
    return pairs


def git_commit():
    try:
        head = subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True).strip()
        return head + ("-dirty" if dirty else "")
    except Exception:
        return None


def peak_mem_mb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**20
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def reset_peak(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def json_safe(v):
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, float) and not math.isfinite(v):
        return str(v)
    return v


# ---- data -------------------------------------------------------------------------

def load_data(args):
    """Train / valid node events pooled on the CPU. The test split is never read."""
    root = args.dataset_root
    dem, dem_meta = sbd.load_static_dem(root)
    train_ids = sbd.split_event_ids(root, "train", args.expected_split_counts)
    valid_ids = sbd.split_event_ids(root, "valid", args.expected_split_counts)
    if set(train_ids) & set(valid_ids):
        raise ValueError("train / valid overlap")
    dataset_split_counts = {"train": len(train_ids), "valid": len(valid_ids)}
    if args.smoke:
        train_ids, valid_ids = train_ids[:2], valid_ids[:2]
    builder = sbd.StockbridgeMultiscale(dem, num_scales=args.num_scales, train_event_ids=train_ids,
                                        previous_t=args.previous_t)
    out = {"builder": builder, "train_ids": train_ids, "valid_ids": valid_ids, "dem_meta": dem_meta,
           "dataset_split_counts": dataset_split_counts,
           "train": [], "valid": [], "rain_total": {}}
    for split, ids in (("train", train_ids), ("valid", valid_ids)):
        for eid in ids:
            ev = sbd.load_event(root, split, eid)
            out["rain_total"][eid] = float(ev["rainfall_global"].sum())
            out[split].append(builder.event_to_nodes(ev))
    return out


def check_device_sample(builder_cpu, builder_dev, ne_cpu, ne_dev):
    """A sample built on the device must equal the CPU sample bit for bit."""
    report = {}
    for t in (0, sbd.NT // 2, sbd.NT - 2):
        a, b = builder_cpu.build_sample(ne_cpu, t), builder_dev.build_sample(ne_dev, t)
        bad = []
        for k in a.keys():
            va, vb = a[k], b[k]
            if isinstance(va, torch.Tensor):
                if vb.device != builder_dev.device or va.dtype != vb.dtype or not torch.equal(va, vb.cpu()):
                    bad.append(k)
            elif va != vb:
                bad.append(k)
        report[f"t{t}"] = bad
    report["identical"] = all(not v for v in report.values())
    return report


# ---- model, loss ---------------------------------------------------------------------

def model_kwargs(args):
    params = dict(yaml.safe_load(Path(args.config).read_text())["models"])
    if params.pop("model_type") != "MSGNN":
        raise ValueError("config.yaml models.model_type must be MSGNN")
    if args.seed is not None:
        params["seed"] = args.seed
    params.update(rain_forcing=args.rain_forcing, rain_injection=args.rain_injection,
                  rain_residual=args.rain_residual, dynamic_encoder_bias=args.dynamic_encoder_bias,
                  rain_scale_mm=args.rain_scale_mm, output_activation="raw")
    return params


def build_model(args, builder, device):
    sample = builder.build_sample(builder_event_stub(builder), 0)
    kw = model_kwargs(args)
    model = RainMSGNN(sample.x.shape[1], sample.edge_attr.shape[1], args.num_scales,
                      previous_t=args.previous_t, device=device, **kw).to(device)
    return model, kw, {"num_node_features": sample.x.shape[1], "num_edge_features": sample.edge_attr.shape[1]}


def builder_event_stub(builder):
    """Zero event on the builder's device, only used to read the feature sizes."""
    n = builder.num_nodes
    z = torch.zeros(n, sbd.NT, device=builder.device)
    return {"H": z, "Qmag": z, "event_id": "stub",
            "rainfall_extended": torch.zeros(sbd.NT + 1, dtype=torch.float64, device=builder.device)}


def rollout_loss(model, builder, node_event, start_t, k, args, use_ckpt):
    """Mean over k steps of the M17K swegnn_rmse loss on scale 0 (raw output, where_water mask)."""
    n0 = int(builder.node_ptr[1])
    ro = rollout(model, builder, node_event, t0=start_t, n_steps=k,
                 detach_between_steps=args.detach_rollout, checkpoint_steps=use_ckpt)
    active = torch.ones(n0, dtype=torch.bool, device=builder.device)
    losses = []
    for s in range(k):
        raw, target = ro["raw"][s, :n0], ro["target"][s, :n0]
        mask = where_water(ro["constrained"][s, :n0].detach(), target, active)
        losses.append(swegnn_rmse_loss(raw, target, mask, args.loss_q_weight))
    return torch.stack(losses).mean()


def group_grad_norms(groups):
    out = {}
    for name, mod in groups.items():
        gs = [p.grad.detach().flatten() for p in mod.parameters() if p.grad is not None]
        out[name] = float(torch.cat(gs).norm()) if gs else 0.0
    return out


# ---- training -----------------------------------------------------------------------

def train_one_epoch(model, optimizer, builder, events, pairs, args, rollout_steps, use_ckpt):
    """One epoch of k-step training (M17K train_one_epoch, copied logic)."""
    epoch_pairs = pairs.copy()
    if args.shuffle:
        random.shuffle(epoch_pairs)
    model.train()
    groups = module_groups(model)
    losses, grad_norms, group_sums, n_steps = [], [], {k: 0.0 for k in groups}, 0
    accumulate = args.accumulate_samples
    optimizer.zero_grad(set_to_none=True)
    for sample_number, (event_index, start_t) in enumerate(epoch_pairs, start=1):
        event = events[event_index]
        loss = rollout_loss(model, builder, event, start_t, rollout_steps, args, use_ckpt)
        if not torch.isfinite(loss):
            raise ValueError(f"{event['event_id']} T{start_t} training loss is not finite: {loss.item()}")
        losses.append(loss.item())
        # With accumulation, the update uses the mean loss of the group,
        # like a mini-batch of `accumulate` samples (M17K).
        group_size = min(accumulate, len(epoch_pairs) - (sample_number - 1) // accumulate * accumulate)
        (loss / group_size).backward()
        if sample_number % accumulate != 0 and sample_number != len(epoch_pairs):
            continue
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise ValueError(f"Non-finite gradient in {name} at {event['event_id']} T{start_t}")
        for k, v in group_grad_norms(groups).items():
            group_sums[k] += v
        n_steps += 1
        if args.grad_clip_norm > 0.0:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            if not torch.isfinite(norm).item():
                raise ValueError(f"Non-finite gradient norm at {event['event_id']} T{start_t}: {norm.item()}")
            grad_norms.append(norm.item())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {
        "train_loss_mean": sum(losses) / len(losses),
        "train_loss_min": min(losses),
        "train_loss_max": max(losses),
        "mean_grad_norm": sum(grad_norms) / len(grad_norms) if grad_norms else None,
        "optimizer_steps": n_steps,
        "grad_norm_by_module": {k: v / max(n_steps, 1) for k, v in group_sums.items()},
    }


def memory_rehearsal(model, optimizer, builder, events, args, k, use_ckpt, time_indices):
    """One optimisation step with the longest rollout; model and optimiser are restored."""
    device = builder.device
    model_state = copy.deepcopy(model.state_dict())
    opt_state = copy.deepcopy(optimizer.state_dict())
    start_t = time_indices[-1] - k + 1  # latest (wettest) start
    reset_peak(device)
    t0 = time.perf_counter()
    oom = False
    try:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = rollout_loss(model, builder, events[0], start_t, k, args, use_ckpt)
        loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    except torch.cuda.OutOfMemoryError:
        oom = True
    seconds = time.perf_counter() - t0
    peak = math.inf if oom else peak_mem_mb(device)
    optimizer.zero_grad(set_to_none=True)
    model.load_state_dict(model_state)
    optimizer.load_state_dict(opt_state)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"rollout_steps": k, "start_t": start_t, "grad_checkpoint": use_ckpt, "oom": oom,
            "peak_mem_mb": peak, "seconds": seconds,
            "memory_kind": "cuda max_memory_allocated" if device.type == "cuda" else "CPU RSS peak (process)"}


# ---- selection, checkpoints, outputs --------------------------------------------------

def selection_key(summary):
    csi = summary["CSI_0p05_mean"]
    return (-math.inf if csi is None else csi, summary["H_RMSE_active_mean"])


def improves(new, best, min_delta):
    """Higher CSI_0p05 wins; within min_delta, lower H RMSE wins."""
    if new[0] > best[0] + min_delta:
        return True
    return abs(new[0] - best[0]) <= min_delta and new[1] < best[1] - min_delta


def save_checkpoint(path, model, optimizer, scheduler, epoch, args, info):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "args": {k: json_safe(v) for k, v in vars(args).items()},
        **info,
    }, path)


def dry_persistence_by_hand(builder, node_events, time_indices, cell_area):
    """Dry persistence metrics computed directly from the CityCAT scale-0 targets.

    Independent of the model / rollout code: the prediction is 0 everywhere, so
    every M17K quantity follows from the targets alone (same float32 ops).
    """
    n0 = int(builder.node_ptr[1])
    rows, event_losses = [], []
    for ne in node_events:
        wet_sq, wet_count = torch.zeros(2, dtype=torch.float64), 0
        for t in time_indices:
            true_H, true_Q = ne["H"][:n0, t + 1], ne["Qmag"][:n0, t + 1]
            zero = torch.zeros_like(true_H)
            row = {"H_RMSE_active": (zero - true_H).square().mean().sqrt().item(),
                   "Qmag_RMSE_active": (zero - true_Q).square().mean().sqrt().item(),
                   "pred_positive_cells": 0, "pred_Qmag_max": 0.0, "t": t,
                   "true_volume_m3": true_H.sum().item() * cell_area}
            for thr, label in ((0.01, "CSI_0p01"), (0.05, "CSI_0p05"), (0.10, "CSI_0p10")):
                misses = int((true_H >= thr).sum().item())
                row[label] = None if misses == 0 else 0.0
            wet = (true_H != 0) | (true_Q != 0)
            wet_sq += torch.stack([true_H, true_Q], 1)[wet].double().square().sum(0).cpu()
            wet_count += int(wet.sum().item())
            rows.append(row)
        if wet_count > 0:
            event_losses.append((wet_sq / wet_count).sqrt().mean().item())

    def mean_defined(v):
        d = [x for x in v if x is not None]
        return sum(d) / len(d) if d else None

    out = {
        "swegnn_val_loss": mean_defined(event_losses),
        "H_RMSE_active_mean": sum(r["H_RMSE_active"] for r in rows) / len(rows),
        "Qmag_RMSE_active_mean": sum(r["Qmag_RMSE_active"] for r in rows) / len(rows),
        "steps_with_all_dry_prediction": len(rows),
        "steps_with_zero_Qmag_prediction": len(rows),
        "raw_negative_H_fraction_mean": 0.0,
        "raw_negative_Qmag_fraction_mean": 0.0,
    }
    for label in ("CSI_0p01", "CSI_0p05", "CSI_0p10"):
        out[f"{label}_mean"] = mean_defined([r[label] for r in rows])
    final = [r for r in rows if r["t"] == time_indices[-1]]
    out["final_step_pred_over_true_volume_mean"] = mean_defined(
        [0.0 / r["true_volume_m3"] if r["true_volume_m3"] > 0 else None for r in final])
    return out


def plot_maps(path, builder, node_event, ro_constrained, times, event_id):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n0 = int(builder.node_ptr[1])
    g = builder.grid
    fig, axes = plt.subplots(4, len(times), figsize=(4.0 * len(times), 13), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for j, T in enumerate(times):
        pred = ro_constrained[T - 1, :n0].detach().cpu().numpy()
        true_h = node_event["H"][:n0, T].cpu().numpy()
        true_q = node_event["Qmag"][:n0, T].cpu().numpy()
        for i, (vals, label, vmax) in enumerate((
                (true_h, "H CityCAT (m)", None), (pred[:, 0], "H predicted (m)", None),
                (true_q, "Qmag CityCAT", None), (pred[:, 1], "Qmag predicted", None))):
            ref = true_h if i < 2 else true_q
            vmax = max(float(np.nanmax(ref)), float(np.nanmax(vals)), 1e-6)
            ax = axes[i, j]
            im = ax.imshow(g.to_raster(vals, scale=0), cmap="Blues", vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_title(f"{label} T{T}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(f"{event_id}: best checkpoint rollout from the dry T0 vs CityCAT")
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---- main -----------------------------------------------------------------------------

def main(argv=None, data=None):
    """Run the training. `data` (dict like load_data's output) replaces the dataset in local tests."""
    run_start = time.perf_counter()
    args = parse_args(argv)
    out_dir: Path = args.output_dir
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        sys.exit(f"Output directory {out_dir} is not empty; use --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          or args.device == "cuda" else "cpu")
    if args.seed is None:  # same seed as the model: models.seed of config.yaml
        args.seed = int(yaml.safe_load(Path(args.config).read_text())["models"]["seed"])
    random.seed(args.seed)  # training starts and shuffling
    np.random.seed(args.seed)

    time_indices = list(range(args.start_time_index, args.end_time_index + 1))
    schedule = parse_rollout_schedule(args.rollout_schedule, args.epochs)
    if max(schedule) > len(time_indices):
        raise ValueError(f"Rollout of {max(schedule)} steps exceeds the {len(time_indices)} transitions")

    # data on the device
    t_load = time.perf_counter()
    data = data if data is not None else load_data(args)
    builder_cpu = data["builder"]
    builder = builder_cpu.to(device)
    device_check = check_device_sample(builder_cpu, builder, data["train"][0],
                                       sbd.node_event_to(data["train"][0], device))
    if not device_check["identical"]:
        raise RuntimeError(f"Sample built on {device} differs from the CPU sample: {device_check}")
    train_events = [sbd.node_event_to(ne, device) for ne in data["train"]]
    valid_events = [sbd.node_event_to(ne, device) for ne in data["valid"]]
    load_seconds = time.perf_counter() - t_load
    print(f"Device {device} | train {len(train_events)} valid {len(valid_events)} events | "
          f"device sample == CPU sample: {device_check['identical']} | load {load_seconds:.1f} s", flush=True)

    model, kw, sizes = build_model(args, builder, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
                 if args.lr_step_size else None)
    n_params = sum(p.numel() for p in model.parameters())
    groups = list(module_groups(model))
    print(f"RainMSGNN {args.rain_injection}/{args.rain_forcing} residual {args.rain_residual} | "
          f"{n_params} parameters | schedule {schedule}", flush=True)

    config = {
        "args": {k: json_safe(v) for k, v in vars(args).items()},
        "model_kwargs": kw, **sizes, "num_parameters": n_params, "schedule_per_epoch": schedule,
        "git_commit": git_commit(), "m17k_reference": M17K_REFERENCE,
        "stage1_valid_reference": STAGE1_VALID_REFERENCE, "device": str(device), "host": platform.node(),
        "torch": torch.__version__, "train_event_ids": data["train_ids"], "valid_event_ids": data["valid_ids"],
        "dataset_split_counts": data.get("dataset_split_counts"),
        "nodes_per_scale": builder_cpu.grid.nodes_per_scale(), "device_sample_check": device_check,
        "data_load_seconds": load_seconds, "test_split_used": False,
    }
    if device.type == "cuda":
        config["cuda_device_name"] = torch.cuda.get_device_name(device)

    # memory rehearsal with the longest rollout of the schedule
    use_ckpt = args.grad_checkpoint
    kmax = max(schedule)
    rehearsal = [memory_rehearsal(model, optimizer, builder, train_events, args, kmax, use_ckpt, time_indices)]
    limit_mb = args.max_gpu_mem_gb * 1024
    if device.type == "cuda" and not use_ckpt and rehearsal[0]["peak_mem_mb"] > limit_mb:
        print(f"WARNING: rehearsal peak {rehearsal[0]['peak_mem_mb']:.0f} MB (oom {rehearsal[0]['oom']}) "
              f"> --max-gpu-mem-gb {args.max_gpu_mem_gb}: gradient checkpointing switched ON", flush=True)
        use_ckpt = True
        rehearsal.append(memory_rehearsal(model, optimizer, builder, train_events, args, kmax, True, time_indices))
    for r in rehearsal:
        print(f"Memory rehearsal k={r['rollout_steps']} checkpoint {r['grad_checkpoint']}: "
              f"peak {r['peak_mem_mb']:.0f} MB ({r['memory_kind']}), {r['seconds']:.1f} s, oom {r['oom']}", flush=True)
    if device.type == "cuda" and rehearsal[-1]["peak_mem_mb"] > limit_mb:
        print("WARNING: even with checkpointing the rehearsal exceeds --max-gpu-mem-gb", flush=True)
    config["memory_rehearsal"] = rehearsal
    config["grad_checkpoint_used"] = use_ckpt
    (out_dir / "config_used.json").write_text(json.dumps(config, indent=2, default=json_safe))

    # baselines (same metrics, once)
    t_b = time.perf_counter()
    baselines = {}
    for name, bmodel in (("persistence_dry", Persistence()), ("rain_bucket", RainBucket())):
        summary, _, per_event = evaluate_rollout(bmodel, builder, valid_events, time_indices, args.cell_area)
        baselines[name] = {"summary": summary, "per_event": per_event}
        print(f"Baseline {name}: H_RMSE {summary['H_RMSE_active_mean']:.5f} CSI0.05 {summary['CSI_0p05_mean']} "
              f"Q_RMSE {summary['Qmag_RMSE_active_mean']:.5f}", flush=True)
    if args.smoke:
        by_hand = dry_persistence_by_hand(builder, valid_events, time_indices, args.cell_area)
        diffs = {k: (v, baselines["persistence_dry"]["summary"][k]) for k, v in by_hand.items()
                 if baselines["persistence_dry"]["summary"][k] != v}
        baselines["persistence_dry_by_hand_check"] = {"identical": not diffs, "differences": diffs, "by_hand": by_hand}
        print(f"Smoke check: dry persistence == by-hand computation: {not diffs} {diffs if diffs else ''}", flush=True)
    baselines["stage1_valid_reference"] = STAGE1_VALID_REFERENCE
    baselines["seconds"] = time.perf_counter() - t_b
    (out_dir / "baselines.json").write_text(json.dumps(baselines, indent=2, default=json_safe))

    # training
    fieldnames = (["epoch", "rollout_steps", "train_starts", "optimizer_steps", "lr", "train_loss_mean",
                   "train_loss_min", "train_loss_max", "mean_grad_norm", "train_seconds", "valid_seconds",
                   "epoch_seconds", "peak_mem_mb", "grad_checkpoint", "validated", "is_best", "best_epoch"]
                  + [f"grad_{g}" for g in groups] + [f"valid_{k}" for k in VALID_KEYS])
    best_key, best_epoch, best_summary = (-math.inf, math.inf), 0, None
    since_improvement, early_stopped, epochs_completed = 0, False, 0
    info = {"git_commit": config["git_commit"], "model_kwargs": kw, **sizes, "num_scales": args.num_scales,
            "previous_t": args.previous_t, "m17k_reference": M17K_REFERENCE}
    with (out_dir / "history.csv").open("w", newline="") as hf:
        writer = csv.DictWriter(hf, fieldnames=fieldnames)
        writer.writeheader()
        for epoch, k in enumerate(schedule, start=1):
            if epoch > 1 and k != schedule[epoch - 2]:
                since_improvement = 0  # new curriculum stage (M17K since_best)
                print(f"--- Curriculum stage: {k} rollout steps ---", flush=True)
            final_stage = all(s == schedule[-1] for s in schedule[epoch - 1:])
            reset_peak(device)
            t_epoch = time.perf_counter()
            pairs = training_starts(len(train_events), time_indices, k, args.max_starts_per_event)
            tr = train_one_epoch(model, optimizer, builder, train_events, pairs, args, k, use_ckpt)
            train_seconds = time.perf_counter() - t_epoch
            lr = optimizer.param_groups[0]["lr"]
            if scheduler is not None:
                scheduler.step()
            validated = epoch % args.valid_every == 0 or epoch == len(schedule)
            summary, is_best, valid_seconds = None, False, 0.0
            if validated:
                t_v = time.perf_counter()
                summary, _, _ = evaluate_rollout(model, builder, valid_events, time_indices, args.cell_area)
                valid_seconds = time.perf_counter() - t_v
                key = selection_key(summary)
                is_best = best_epoch == 0 or improves(key, best_key, args.min_delta)
                if is_best:
                    best_key, best_epoch, best_summary = key, epoch, summary
                    since_improvement = 0
                    save_checkpoint(out_dir / "best.pt", model, optimizer, scheduler, epoch, args,
                                    {**info, "valid_summary": summary})
                else:
                    since_improvement += 1 if args.valid_every == 1 else args.valid_every
            peak = peak_mem_mb(device)
            save_checkpoint(out_dir / "last.pt", model, optimizer, scheduler, epoch, args,
                            {**info, "valid_summary": summary, "best_epoch": best_epoch})
            epochs_completed = epoch
            row = {"epoch": epoch, "rollout_steps": k, "train_starts": len(pairs),
                   "optimizer_steps": tr["optimizer_steps"], "lr": lr,
                   **{key_: tr[key_] for key_ in ("train_loss_mean", "train_loss_min", "train_loss_max",
                                                   "mean_grad_norm")},
                   "train_seconds": train_seconds, "valid_seconds": valid_seconds,
                   "epoch_seconds": time.perf_counter() - t_epoch, "peak_mem_mb": peak,
                   "grad_checkpoint": use_ckpt, "validated": validated, "is_best": is_best,
                   "best_epoch": best_epoch,
                   **{f"grad_{g}": v for g, v in tr["grad_norm_by_module"].items()},
                   **({f"valid_{key_}": summary[key_] for key_ in VALID_KEYS} if summary else {})}
            writer.writerow(row)
            hf.flush()
            vtxt = (f" | valid CSI0.05 {summary['CSI_0p05_mean']} H_RMSE {summary['H_RMSE_active_mean']:.5f} "
                    f"Qpeak {summary['Qmag_peak_ratio_mean']}" if summary else "")
            print(f"Epoch {epoch}/{len(schedule)} k={k} | train {tr['train_loss_mean']:.5g} | "
                  f"{row['epoch_seconds']:.1f} s | peak {peak:.0f} MB{vtxt} | best {best_epoch}", flush=True)
            if final_stage and since_improvement >= args.patience:
                early_stopped = True
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    # best checkpoint: reload in a fresh model, evaluate, write rows, metrics and maps
    fresh, _, _ = build_model(args, builder, device)
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    fresh.load_state_dict(ckpt["model_state_dict"])
    summary, rows, per_event = evaluate_rollout(fresh, builder, valid_events, time_indices, args.cell_area)
    # bit-identical on the CPU; on CUDA, scatter sums use atomics, so a re-run can
    # differ by float32 round-off (~1e-9 relative): reported with a tolerance
    reload_diffs = {k: abs((summary[k] or 0) - (ckpt["valid_summary"][k] or 0)) for k in VALID_KEYS}
    reload_identical = all(summary[k] == ckpt["valid_summary"][k] for k in VALID_KEYS)
    reload_rel = max(d / max(abs(ckpt["valid_summary"][k] or 0), 1e-12) for k, d in reload_diffs.items())
    reload_ok = reload_rel <= 1e-5
    with (out_dir / "rollout_rows_best.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROLLOUT_FIELDNAMES)
        w.writeheader()
        w.writerows(rows)

    ids = data["valid_ids"]
    map_event = "R019" if "R019" in ids else max(ids, key=lambda e: data["rain_total"][e])
    times = [int(t) for t in args.map_times.split(",")]
    ne = valid_events[ids.index(map_event)]
    with torch.inference_mode():
        fresh.eval()
        ro = rollout(fresh, builder, ne, t0=0, n_steps=max(times))
    map_path = out_dir / f"maps_best_{map_event}.png"
    plot_maps(map_path, builder, ne, ro["constrained"], times, map_event)

    best_metrics = {
        "best_epoch": best_epoch, "epochs_completed": epochs_completed, "early_stopped": early_stopped,
        "selection": "max CSI_0p05_mean, ties -> min H_RMSE_active_mean",
        "summary": summary, "per_event": per_event,
        "reload_check": {"identical_to_training_validation": reload_identical,
                         "max_relative_diff": reload_rel, "within_1e-5": reload_ok, "abs_diff": reload_diffs},
        "baselines": {k: baselines[k]["summary"] for k in ("persistence_dry", "rain_bucket")},
        "stage1_valid_reference": STAGE1_VALID_REFERENCE,
        "map_event": map_event, "map_event_reason": "R019 in valid" if map_event == "R019" else "valid event with the most rain",
        "total_seconds": time.perf_counter() - run_start,
    }
    if device.type == "cuda":
        best_metrics["max_memory_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
    (out_dir / "best_metrics.json").write_text(json.dumps(best_metrics, indent=2, default=json_safe))
    print(f"Best epoch {best_epoch}: CSI0.05 {summary['CSI_0p05_mean']} H_RMSE {summary['H_RMSE_active_mean']:.5f} "
          f"| reload identical {reload_identical} (max rel diff {reload_rel:.1e}) | maps {map_path.name} | total "
          f"{best_metrics['total_seconds']:.0f} s", flush=True)
    return best_metrics


if __name__ == "__main__":
    main()
