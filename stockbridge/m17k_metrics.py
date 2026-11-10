"""Stage-1 (M17K) validation metrics, loss and baselines for the multiscale model.

Copied from Stockbridge-SWEGNN, scripts/train_valid_stockbridge_m17k.py
(repository HEAD 9bb39d00d4d9044c67a96684bf0383faa51a4631; last change to the
file fdfb600a8d3f78a79c7e76af01db63d41595524b, which already includes the
raw-output diagnostics). The functions marked "copied" keep the M17K code
(formulas, order of operations, None handling); they are copied, not imported,
because that script imports the single-scale SWE-GNN stack.

What changes with respect to M17K, and why
------------------------------------------
* Nodes. M17K works on the 99,430 raster cells and restricts every metric to
  the `active` mask (DEM != DEM.max(), 32,249 cells). Here the metrics are
  computed on the scale-0 rows of the multiscale graph, which are exactly
  those 32,249 active cells in the same row-major order (flat_index0 is
  ascending), so `active` is an all-True mask over scale 0 and every sum
  runs over the same cells in the same order. Coarse rows never enter a
  metric or the loss.
* Rollout state. M17K feeds back and evaluates clamp_min(0) of H and Qmag
  independently. Here the state is `constrain_state` (H >= 0, Q >= 0 and
  Q = 0 where H = 0), the same state the patch 2 rollout feeds back
  (decision of the user, patch 3). It only differs from M17K on cells with
  raw H <= 0 and raw Q > 0.
* where_water. M17K: active & ((pred != 0) | (target != 0)). In the loss,
  `pred` is the constrained state, not the raw output: with the raw output of
  RainMSGNN almost no value is exactly 0 and the mask would always be all
  active cells (decision of the user, patch 3). The loss itself is computed
  on the raw output inside that mask. In `swegnn_val_loss` the prediction is
  the constrained state, as in M17K (clamped state there).
* The rollout is `stockbridge.rollout.rollout` (coarse rows rebuilt from
  scale 0). Additions: per-event peaks of H and Qmag (pred vs CityCAT).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from stockbridge.rain_msgnn import constrain_state, node_graph_index
from stockbridge.rollout import rollout

CSI_THRESHOLDS = (0.01, 0.05, 0.10)  # copied
RMSE_EPS = 1e-12  # copied: keeps sqrt differentiable when a step is predicted exactly
ROLLOUT_FIELDNAMES = [  # copied (+ raw-output diagnostics, as in M17K)
    "event_id", "input_time_index", "target_time_index", "H_RMSE_active", "Qmag_RMSE_active",
    "true_H_max", "pred_H_max", "true_Qmag_max", "pred_Qmag_max", "true_wet_cells_0p01",
    "pred_wet_cells_0p01", "pred_positive_cells", "CSI_0p01", "CSI_0p05", "CSI_0p10",
    "true_volume_m3", "pred_volume_m3", "raw_negative_H_fraction", "raw_negative_Qmag_fraction",
]


# ---- copied from M17K -----------------------------------------------------------

def masked_mse(pred, target, mask):
    """Copied from M17K."""
    error = (pred - target).square()
    if mask is not None:
        error = error[mask]
    return error.mean()


def csi(pred_H, true_H, threshold):
    """Copied from M17K."""
    pred_wet = pred_H >= threshold
    true_wet = true_H >= threshold
    hits = (pred_wet & true_wet).sum().item()
    misses = (~pred_wet & true_wet).sum().item()
    false_alarms = (pred_wet & ~true_wet).sum().item()
    denominator = hits + misses + false_alarms
    return None if denominator == 0 else hits / denominator


def threshold_label(threshold: float) -> str:
    """Copied from M17K."""
    return f"CSI_{threshold:.2f}".replace(".", "p")


def mean_defined(values):
    """Copied from M17K."""
    defined = [value for value in values if value is not None]
    return sum(defined) / len(defined) if defined else None


def where_water(pred, target, active):
    """Copied from M17K: active cells where prediction or target is nonzero."""
    wet = ((pred != 0) | (target != 0)).any(dim=1)
    return active & wet


def swegnn_rmse_loss(raw, target, mask, q_weight):
    """M17K `make_loss_fn` with --loss swegnn_rmse (copied formula).

    raw, target: (n0, 2) scale-0 [H, Q]. mask: where_water of the constrained
    state (see module docstring). RMSE per variable over the mask, weighted
    (1, q_weight) and summed. Empty mask (nothing wet yet): zero loss with the
    graph kept, as in M17K.
    """
    if mask is not None and not bool(mask.any()):
        return raw.sum() * 0.0
    diff = raw - target
    if mask is not None:
        diff = diff[mask]
    rmse = torch.sqrt(diff.square().mean(dim=0) + RMSE_EPS)
    return rmse[0] + q_weight * rmse[1]


# ---- evaluation -------------------------------------------------------------------

def evaluate_rollout(model, builder, node_events, time_indices, cell_area=25.0):
    """M17K `evaluate_rollout` on the scale-0 rows of the multiscale rollout.

    For each event: autoregressive rollout from the CityCAT state at
    time_indices[0] for len(time_indices) steps; per step the M17K row on the
    constrained state; per event the M17K swegnn validation loss. Returns
    (summary, rows, per_event).
    """
    rows = []
    event_val_losses = []
    n0 = int(builder.node_ptr[1])
    active = torch.ones(n0, dtype=torch.bool, device=builder.device)
    model.eval()
    with torch.inference_mode():
        for node_event in node_events:
            # Original SWE-GNN validation loss: RMSE per variable over all
            # wet (pred or target nonzero) node-times of the event, then the
            # mean over variables (copied from M17K).
            wet_sq = torch.zeros(2, dtype=torch.float64)
            wet_count = 0
            ro = rollout(model, builder, node_event, t0=time_indices[0], n_steps=len(time_indices))
            for k, t in enumerate(time_indices):
                raw = ro["raw"][k, :n0]
                prediction = ro["constrained"][k, :n0]
                target = ro["target"][k, :n0]
                H, Q = prediction[:, 0], prediction[:, 1]
                true_H, true_Q = target[:, 0], target[:, 1]
                wet = where_water(prediction, target, active)
                wet_sq += (prediction - target)[wet].double().square().sum(dim=0).cpu()
                wet_count += int(wet.sum().item())
                row = {
                    "event_id": node_event["event_id"],
                    "input_time_index": t,
                    "target_time_index": t + 1,
                    "H_RMSE_active": masked_mse(H, true_H, active).sqrt().item(),
                    "Qmag_RMSE_active": masked_mse(Q, true_Q, active).sqrt().item(),
                    "true_H_max": true_H[active].max().item(),
                    "pred_H_max": H[active].max().item(),
                    "true_Qmag_max": true_Q[active].max().item(),
                    "pred_Qmag_max": Q[active].max().item(),
                    "true_wet_cells_0p01": int((true_H[active] >= 0.01).sum().item()),
                    "pred_wet_cells_0p01": int((H[active] >= 0.01).sum().item()),
                    "pred_positive_cells": int((H[active] > 0).sum().item()),
                    "true_volume_m3": true_H[active].sum().item() * cell_area,
                    "pred_volume_m3": H[active].sum().item() * cell_area,
                    "raw_negative_H_fraction": (raw[active, 0] < 0).float().mean().item(),
                    "raw_negative_Qmag_fraction": (raw[active, 1] < 0).float().mean().item(),
                }
                for threshold in CSI_THRESHOLDS:
                    row[threshold_label(threshold)] = csi(H[active], true_H[active], threshold)
                rows.append(row)
            if wet_count > 0:
                event_val_losses.append((wet_sq / wet_count).sqrt().mean().item())

    summary = {  # copied from M17K
        "swegnn_val_loss": mean_defined(event_val_losses),
        "num_events": len(node_events),
        "num_steps": len(rows),
        "H_RMSE_active_mean": sum(r["H_RMSE_active"] for r in rows) / len(rows),
        "Qmag_RMSE_active_mean": sum(r["Qmag_RMSE_active"] for r in rows) / len(rows),
        "steps_with_all_dry_prediction": sum(r["pred_positive_cells"] == 0 for r in rows),
        "steps_with_zero_Qmag_prediction": sum(r["pred_Qmag_max"] == 0.0 for r in rows),
        "raw_negative_Qmag_fraction_mean": sum(r["raw_negative_Qmag_fraction"] for r in rows) / len(rows),
        "raw_negative_H_fraction_mean": sum(r["raw_negative_H_fraction"] for r in rows) / len(rows),
    }
    for threshold in CSI_THRESHOLDS:
        label = threshold_label(threshold)
        summary[f"{label}_mean"] = mean_defined([r[label] for r in rows])
    final_rows = [r for r in rows if r["input_time_index"] == time_indices[-1]]
    summary["final_step_pred_over_true_volume_mean"] = mean_defined(
        [r["pred_volume_m3"] / r["true_volume_m3"] if r["true_volume_m3"] > 0 else None for r in final_rows])

    per_event = per_event_peaks(rows, time_indices)
    summary["H_peak_ratio_mean"] = mean_defined([e["H_peak_ratio"] for e in per_event.values()])
    summary["Qmag_peak_ratio_mean"] = mean_defined([e["Qmag_peak_ratio"] for e in per_event.values()])
    return summary, rows, per_event


def per_event_peaks(rows, time_indices):
    """Per event: peak (max over steps) of H and Qmag, predicted vs CityCAT, and final volumes."""
    out = {}
    for r in rows:
        e = out.setdefault(r["event_id"], {"true_H_peak": -math.inf, "pred_H_peak": -math.inf,
                                           "true_Qmag_peak": -math.inf, "pred_Qmag_peak": -math.inf})
        e["true_H_peak"] = max(e["true_H_peak"], r["true_H_max"])
        e["pred_H_peak"] = max(e["pred_H_peak"], r["pred_H_max"])
        e["true_Qmag_peak"] = max(e["true_Qmag_peak"], r["true_Qmag_max"])
        e["pred_Qmag_peak"] = max(e["pred_Qmag_peak"], r["pred_Qmag_max"])
        if r["input_time_index"] == time_indices[-1]:
            e["final_true_volume_m3"] = r["true_volume_m3"]
            e["final_pred_volume_m3"] = r["pred_volume_m3"]
    for e in out.values():
        e["H_peak_ratio"] = e["pred_H_peak"] / e["true_H_peak"] if e["true_H_peak"] > 0 else None
        e["Qmag_peak_ratio"] = e["pred_Qmag_peak"] / e["true_Qmag_peak"] if e["true_Qmag_peak"] > 0 else None
        tv = e.get("final_true_volume_m3")
        e["final_volume_ratio"] = e["final_pred_volume_m3"] / tv if tv else None
    return out


# ---- baselines ----------------------------------------------------------------------

class Persistence(nn.Module):
    """M17K reference model: H(t+1) = H(t), Qmag(t+1) = Qmag(t).

    Same as M17K `Persistence` (it returns the current [H, Q] input columns,
    here the last two columns of x). In a rollout from the dry T0 it predicts
    a dry domain at every step: the stage-1 "dry persistence" baseline.
    """

    def forward(self, graph):
        return graph.x[:, -2:].clone()


class RainBucket(nn.Module):
    """Bucket baseline: H(t+1) = H(t) + interval rain depth on every node, Q = 0.

    In a rollout from the dry T0 it gives H = accumulated rain everywhere: the
    water stays where it falls (no runoff, no infiltration). It measures what
    moving the water adds over only accumulating it.
    """

    def forward(self, graph):
        rain = graph.rain_interval_m.reshape(-1).to(graph.x.dtype)[node_graph_index(graph)]
        h = graph.x[:, -2] + rain
        return torch.stack((h, torch.zeros_like(h)), dim=1)

    @staticmethod
    def constrain_state(pred):
        return constrain_state(pred)
