"""Patch 2 smoke test: RainMSGNN (rain forcing inside the multiscale GNN).

No training beyond a 200-step overfit of single transitions. Run from anywhere:
    python stockbridge/smoke_patch2.py [--dataset-root ...] [--output-dir ...]

Events: the first train event with rain > 0 in T0-T2 (interval rain of the
step T0 -> T1 > 0), plus R001 from whatever split holds it. 4 scales,
previous_t = 2, hyperparameters from config.yaml.

Checks (each prints PASS / FAIL / INFO / SKIP and is stored in
smoke_report.json; a FAIL or an exception does not stop the script):
  1. Equivalence: RainMSGNN(rain_forcing="none", output_activation="relu",
     rain_residual=False, dynamic_encoder_bias=False) with the state_dict of
     an original MSGNN gives the same output (max abs diff < 1e-6), single
     samples and batch. Run on CPU (CUDA scatter uses atomics, so GPU sums
     are not bit-reproducible); the GPU difference is reported as info.
  2. Dry without rain: dry input, zero rain, multiplicative forcing, no rain
     residual -> output exactly 0 (fine and all_scales). The additive
     variant is reported as info (it is not expected to be 0).
  3. Uniformity from dry (comparison, not PASS/FAIL): spatial std of the
     scale-0 output, absolute and relative to the mean, from the CityCAT T0
     state with rain > 0, for
       - reference "static": original MSGNN, rain as an extra static
         column before DEM (exactly 0 from dry: mechanisms 1-2);
       - reference "dynamic": original MSGNN weights with the dynamic encoder
         widened by one rain column (rain enters the dynamic latent; the
         intra-scale GNN may break the uniformity, mechanism 3);
       - RainMSGNN fine and all_scales.
     Also a short rollout: std per step, the share of each model's largest
     std that is already present at T1, and std(T1) relative to the CityCAT
     std at T1.
  4. Gradient: MSE of scale-0 H against CityCAT, dry T0 -> T1 and wet
     T12 -> T13, per-module gradient norms. From a dry input the dynamic
     encoder and the learned residual weights have exactly zero gradient by
     construction (zero input, no bias): marked "expected zero". Every other
     module must be > 0 and finite; in the wet transition every module.
  5. Batch: two samples with different rain (two events at T0, or two
     times) -> batched output equals single outputs, so each graph uses its
     own rain.
  6. Rollout of 24 steps with random weights from T0: no NaN / no explosion;
     per step H max, scale-0 mean H, predicted vs CityCAT volume, and max |H|
     on the coarse rows of each step's input (they must follow the fine
     prediction, not drift).
  6b. Rollout with an oracle: a fake model returns the CityCAT scale-0
     target (and 1e3 on coarse rows). Over 24 steps the input x of every
     step must equal builder.build_sample(node_event, t).x (max abs diff
     <= 1e-6), i.e. the rollout rebuilds coarse rows from the fine state
     exactly as event_to_nodes does.
  7. Overfit: 200 Adam steps (lr 1e-3) on T0 -> T1 and on T12 -> T13, fine
     and all_scales, loss RMSE(H) + RMSE(Q) on scale 0 (raw output).
     PASS if the final loss is finite and below half the initial one. Also
     the fraction of nodes with CityCAT Q > 0 where the constrained Q > 0.
  8. Time (median of --cost-reps after one warm-up) and peak memory of
     forward and forward+backward on the wet sample, parameter count, for the
     original MSGNN and RainMSGNN fine / all_scales.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml  # noqa: E402

from stockbridge import dataset as sbd  # noqa: E402
from stockbridge.batching import adapt_batch_training  # noqa: E402
from stockbridge.rain_msgnn import (  # noqa: E402
    RainMSGNN, constrain_state, module_groups, node_graph_index, rain_features)
from stockbridge.rollout import rollout  # noqa: E402
from stockbridge.smoke_patch1 import git_commit, rss_peak_mb  # noqa: E402

DEFAULT_OUTPUT_DIR = Path.home() / "flood/03_outputs/mSWE-GNN_Stockbridge/patch2_smoke_v01"
SPLITS = ("train", "valid", "test")
RAIN_CONFIGS = {"fine": {"rain_injection": "fine"}, "all_scales": {"rain_injection": "all_scales"}}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", type=Path, default=sbd.DEFAULT_DATASET_ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--extra-event", default="R001")
    p.add_argument("--scales", type=int, default=4)
    p.add_argument("--previous-t", type=int, default=2)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    p.add_argument("--rain-scale-mm", type=float, default=10.0)
    p.add_argument("--wet-t", type=int, default=12, help="wet transition t -> t+1 (checks 4, 7, 8)")
    p.add_argument("--rollout-steps", type=int, default=24)
    p.add_argument("--uniformity-steps", type=int, default=6, help="rollout length in check 3")
    p.add_argument("--overfit-steps", type=int, default=200)
    p.add_argument("--overfit-lr", type=float, default=1e-3)
    p.add_argument("--explode-h", type=float, default=50.0, help="rollout H max (m) counted as explosion")
    p.add_argument("--cost-reps", type=int, default=3, help="timed repetitions in check 8 (median)")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


# ---- report helpers ----------------------------------------------------------

class Checks:
    """Collects PASS / FAIL / INFO / SKIP results; exceptions become FAIL."""

    def __init__(self):
        self.results = {}

    def record(self, name, status, message, details=None):
        self.results[name] = {"status": status, "message": message, "details": details or {}}
        print(f"[{status}] {name}: {message}", flush=True)

    def run(self, name, fn, *args, **kwargs):
        try:
            fn(self, name, *args, **kwargs)
        except Exception as exc:  # keep going, report the failure
            self.record(name, "FAIL", f"exception {type(exc).__name__}: {exc}",
                        {"traceback": traceback.format_exc()})

    def summary(self):
        out = {}
        for r in self.results.values():
            out[r["status"]] = out.get(r["status"], 0) + 1
        return out


def model_params(config_path: Path) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text())
    params = dict(cfg["models"])
    model_type = params.pop("model_type")
    assert model_type == "MSGNN", f"config.yaml model_type is {model_type}, expected MSGNN"
    return params


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def reset_peak(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def peak_mb(device):
    return torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else rss_peak_mb()


def to_dev(data, device):
    return data.clone().to(device)  # Data.to works in place; keep the CPU sample intact


def make_batch(samples, device):
    from torch_geometric.loader import DataLoader
    batch = next(iter(DataLoader(samples, batch_size=len(samples), shuffle=False)))
    return adapt_batch_training(batch).to(device)


def n0_of(ctx):
    return int(ctx.builder.node_ptr[1])


# ---- model factories ---------------------------------------------------------

def build_msgnn(ctx, device=None, num_node_features=None):
    from models.gnn import MSGNN
    device = device or ctx.device
    m = MSGNN(num_node_features=num_node_features or ctx.F, num_edge_features=ctx.E,
              num_scales=ctx.S, previous_t=ctx.p, device=device, **ctx.params)
    return m.to(device)


def build_rain(ctx, device=None, **overrides):
    device = device or ctx.device
    kw = {"rain_scale_mm": ctx.rain_scale_mm, **overrides}
    m = RainMSGNN(ctx.F, ctx.E, ctx.S, previous_t=ctx.p, device=device, **kw, **ctx.params)
    return m.to(device)


class RainColumnReference(nn.Module):
    """Original MSGNN fed with rain as one extra input column (reference for check 3).

    position="static": column inserted before DEM (DEM must stay the last
        static column for with_WL); the inner MSGNN is built with F + 1
        features, so the rain goes to the static encoder (with bias).
    position="dynamic": column inserted right after DEM, i.e. as the first
        column of x_d; the inner model is a copy of an MSGNN whose first
        dynamic-encoder layer gets one extra input column (the other weights
        are unchanged). The learned residual and with_WL still read the last
        [H, Q] columns, so they are unaffected.
    The column value is the interval rain depth in mm / rain_scale_mm.
    """

    def __init__(self, inner, position, n_static, rain_scale_mm):
        super().__init__()
        assert position in ("static", "dynamic")
        self.inner = inner
        self.position = position
        self.n_static = n_static
        self.rain_scale_mm = rain_scale_mm

    def forward(self, graph):
        col = rain_features(graph, self.rain_scale_mm)[:, :1][node_graph_index(graph)].to(graph.x.dtype)
        k = self.n_static - 1 if self.position == "static" else self.n_static
        g = copy.copy(graph)
        g.x = torch.cat((graph.x[:, :k], col, graph.x[:, k:]), 1)
        return self.inner(g)

    @staticmethod
    def constrain_state(pred):
        return constrain_state(pred)


def widen_dynamic_encoder(msgnn, seed=0):
    """Copy of `msgnn` whose dynamic encoder takes one extra (first) input column."""
    m = copy.deepcopy(msgnn)
    old = m.dynamic_node_encoder[0]
    assert isinstance(old, nn.Linear) and old.bias is None
    gen = torch.Generator().manual_seed(seed)
    bound = 1.0 / math.sqrt(old.in_features + 1)  # torch's default Linear init range
    new = nn.Linear(old.in_features + 1, old.out_features, bias=False, device=old.weight.device)
    with torch.no_grad():
        new.weight[:, 1:] = old.weight
        new.weight[:, :1] = (torch.rand(old.out_features, 1, generator=gen) * 2 - 1).to(new.weight) * bound
    m.dynamic_node_encoder[0] = new
    m.dynamic_node_features += 1
    return m


def reference_models(ctx):
    n_static = ctx.builder.x_static.shape[1]
    ref_static = RainColumnReference(build_msgnn(ctx, num_node_features=ctx.F + 1), "static",
                                     n_static, ctx.rain_scale_mm)
    ref_dynamic = RainColumnReference(widen_dynamic_encoder(build_msgnn(ctx)), "dynamic",
                                      n_static, ctx.rain_scale_mm)
    return {"ref_static_column": ref_static, "ref_dynamic_column": ref_dynamic}


# ---- module groups for gradient norms ---------------------------------------

def grad_norms(model):
    out = {}
    for name, mod in module_groups(model).items():
        gs = [p.grad.detach().flatten() for p in mod.parameters() if p.grad is not None]
        out[name] = float(torch.cat(gs).norm()) if gs else 0.0
    if isinstance(getattr(model, "residual_weights", None), nn.Parameter):
        g = model.residual_weights.grad
        out["residual_weights"] = float(g.norm()) if g is not None else 0.0
    return out


# ---- checks ------------------------------------------------------------------

def check_equivalence(checks, name, ctx, event_id):
    ne = ctx.events[event_id]["node_event"]
    cpu = torch.device("cpu")
    samples = [ctx.builder.build_sample(ne, t) for t in (0, ctx.wet_t)]
    orig = build_msgnn(ctx, device=cpu).eval()
    rain = build_rain(ctx, device=cpu, rain_forcing="none", output_activation="relu",
                      rain_residual=False, dynamic_encoder_bias=False).eval()
    missing = rain.load_state_dict(orig.state_dict(), strict=True)
    diffs = {}
    with torch.no_grad():
        for t, s in zip((0, ctx.wet_t), samples):
            diffs[f"t{t}"] = float((orig(s.clone()) - rain(s.clone())).abs().max())
        b = make_batch(samples, cpu)
        diffs["batch2"] = float((orig(b) - rain(b)).abs().max())
    details = {"max_abs_diff_cpu": diffs, "strict_load": str(missing), "tolerance": 1e-6}
    if ctx.device.type == "cuda":
        orig_g, rain_g = orig.to(ctx.device), rain.to(ctx.device)
        orig_g.device = rain_g.device = ctx.device
        with torch.no_grad():
            s = to_dev(samples[1], ctx.device)
            details["max_abs_diff_cuda_info"] = float((orig_g(s) - rain_g(s.clone())).abs().max())
    worst = max(diffs.values())
    ok = worst < 1e-6
    checks.record(name, "PASS" if ok else "FAIL", f"max |RainMSGNN(none) - MSGNN| = {worst:.2e} (< 1e-6, CPU)", details)


def check_dry_no_rain(checks, name, ctx, event_id):
    ne = ctx.events[event_id]["node_event"]
    s = ctx.builder.build_sample(ne, 0)
    s.x[:, ctx.builder.x_static.shape[1]:] = 0
    s.rain_a = torch.zeros_like(s.rain_a)
    s.rain_b = torch.zeros_like(s.rain_b)
    s.rain_interval_m = torch.zeros_like(s.rain_interval_m)
    details, ok = {}, True
    for cfg, kw in RAIN_CONFIGS.items():
        m = build_rain(ctx, rain_forcing="multiplicative", rain_residual=False, **kw).eval()
        with torch.no_grad():
            out = m(to_dev(s, ctx.device))
        zero = bool((out == 0).all())
        details[cfg] = {"all_exactly_zero": zero, "max_abs": float(out.abs().max())}
        ok &= zero
    m = build_rain(ctx, rain_forcing="additive", rain_residual=False).eval()
    with torch.no_grad():
        out = m(to_dev(s, ctx.device))
    details["additive_info"] = {"max_abs": float(out.abs().max()), "note": "not expected to be 0"}
    checks.record(name, "PASS" if ok else "FAIL",
                  "dry + zero rain -> exactly 0 "
                  + ", ".join(f"{k}: {v['all_exactly_zero']}" for k, v in details.items() if k in RAIN_CONFIGS)
                  + f" | additive max |out| {details['additive_info']['max_abs']:.2e} (info)", details)


def spatial_stats(v):
    mean = float(v.mean())
    std = float(v.std())
    return {"mean": mean, "std": std, "std_over_abs_mean": std / abs(mean) if mean != 0 else None}


def check_uniformity(checks, name, ctx, event_id):
    ev = ctx.events[event_id]
    if ev["rain_t0_interval_mm"] <= 0:
        checks.record(name, "SKIP", f"{event_id}: no rain in the step T0 -> T1")
        return
    n0 = n0_of(ctx)
    models = reference_models(ctx)
    for cfg, kw in RAIN_CONFIGS.items():
        models[f"rain_{cfg}"] = build_rain(ctx, **kw)
    steps = min(ctx.uniformity_steps, sbd.NT - 1)
    details = {"t0_scale0_H_max": ev["t0_scale0_H_max"], "rollout_steps": steps, "models": {},
               "note": "output = model output (references: after ReLU/_mask_small_WD); decoder = node_decoder "
                       "output before residuals and activation, which is where a uniform latent would show "
                       "(with random weights the ReLU can zero a uniform negative decoder output)"}
    lines = []
    for mname, m in models.items():
        m.eval()
        inner = m.inner if isinstance(m, RainColumnReference) else m
        dec = []
        hook = inner.node_decoder.register_forward_hook(lambda mod, i, o: dec.append(o[:n0].detach().cpu()))
        try:
            with torch.no_grad():
                ro = rollout(m, ctx.builder, ev["node_event"], t0=0, n_steps=steps, device=ctx.device)
        finally:
            hook.remove()
        raw = ro["raw"][:, :n0].cpu()
        tgt = ro["target"][:, :n0].cpu()
        citycat_std_t1 = float(tgt[0, :, 0].std())
        d = {"citycat_T1_H_std": citycat_std_t1}
        for kind, series in (("output", [raw[k] for k in range(steps)]), ("decoder", dec)):
            std_steps = [float(v[:, 0].std()) for v in series]
            max_std = max(std_steps)
            d[kind] = {
                "T1_H": spatial_stats(series[0][:, 0]), "T1_Q": spatial_stats(series[0][:, 1]),
                "H_std_per_step": std_steps,
                "share_of_max_std_at_T1": std_steps[0] / max_std if max_std > 0 else None,
                "T1_std_over_citycat_T1_std": std_steps[0] / citycat_std_t1 if citycat_std_t1 > 0 else None,
            }
        details["models"][mname] = d
        o, dd = d["output"]["T1_H"], d["decoder"]["T1_H"]
        rel = o["std_over_abs_mean"]
        lines.append(f"{mname}: output std H {o['std']:.2e} (rel {rel:.2e})" if rel is not None
                     else f"{mname}: output std H {o['std']:.2e} (mean 0)")
        lines[-1] += f", decoder std H {dd['std']:.2e}"
    checks.record(name, "INFO", f"{event_id} T1 from T0 | " + " | ".join(lines), details)


def check_gradients(checks, name, ctx, event_id):
    ev = ctx.events[event_id]
    ns = ctx.builder.x_static.shape[1]
    n0 = n0_of(ctx)
    details, ok, msgs = {}, True, []
    for cfg, kw in RAIN_CONFIGS.items():
        m = build_rain(ctx, **kw).train()
        for label, t in (("dry", 0), ("wet", ctx.wet_t)):
            s = to_dev(ctx.builder.build_sample(ev["node_event"], t), ctx.device)
            input_dry = bool((s.x[:, ns:] == 0).all())
            m.zero_grad(set_to_none=True)
            out = m(s)
            loss = ((out[:n0, 0] - s.y[:n0, 0, 0]) ** 2).mean()
            loss.backward()
            norms = grad_norms(m)
            expected_zero = {"dynamic_node_encoder", "residual_weights"} if input_dry else set()
            bad = [k for k, v in norms.items()
                   if not math.isfinite(v) or (v == 0) != (k in expected_zero)]
            ok &= not bad
            details[f"{cfg}_{label}"] = {"t": t, "input_dry": input_dry, "loss": float(loss),
                                         "grad_norms": norms, "expected_zero": sorted(expected_zero), "bad": bad}
            msgs.append(f"{cfg}/{label}: {'ok' if not bad else 'bad ' + ','.join(bad)}")
    checks.record(name, "PASS" if ok else "FAIL", f"{event_id} " + " | ".join(msgs), details)


def check_batch(checks, name, ctx):
    ids = list(ctx.events)
    pair = None
    if len(ids) >= 2:
        a, b = ctx.events[ids[0]], ctx.events[ids[1]]
        if a["rain_t0_interval_mm"] != b["rain_t0_interval_mm"]:
            pair = [(ids[0], 0), (ids[1], 0)]
    if pair is None:  # same event, two times with different rain
        ne = ctx.events[ids[0]]["node_event"]
        r0 = float(ctx.builder.build_sample(ne, 0).rain_interval_m)
        t2 = next((t for t in range(1, sbd.NT - 1)
                   if float(ctx.builder.build_sample(ne, t).rain_interval_m) != r0), None)
        if t2 is None:
            checks.record(name, "SKIP", "no two samples with different rain")
            return
        pair = [(ids[0], 0), (ids[0], t2)]
    samples = [ctx.builder.build_sample(ctx.events[e]["node_event"], t) for e, t in pair]
    rains = [float(s.rain_interval_m) * 1000 for s in samples]
    N = ctx.builder.num_nodes
    details, ok = {"samples": [f"{e}@t{t}" for e, t in pair], "rain_interval_mm": rains}, rains[0] != rains[1]
    for cfg, kw in RAIN_CONFIGS.items():
        m = build_rain(ctx, **kw).eval()
        with torch.no_grad():
            singles = [m(to_dev(s, ctx.device)).cpu() for s in samples]
            out_b = m(make_batch(samples, ctx.device)).cpu()
            swapped = to_dev(samples[0], ctx.device)
            for k in ("rain_a", "rain_b", "rain_interval_m"):
                setattr(swapped, k, getattr(samples[1], k).clone().to(ctx.device))
            rain_effect = float((m(swapped).cpu() - singles[0]).abs().max())
        diffs = [float((out_b[i * N:(i + 1) * N] - singles[i]).abs().max()) for i in range(2)]
        good = all(d < 1e-4 for d in diffs) and not torch.isnan(out_b).any()
        details[cfg] = {"max_abs_diff_vs_single": diffs, "matches_single": good,
                        "max_abs_change_when_rain_swapped": rain_effect}
        ok &= good and rain_effect > 0
    checks.record(name, "PASS" if ok else "FAIL",
                  f"{details['samples']} rain {rains[0]:.3f} / {rains[1]:.3f} mm | "
                  + " | ".join(f"{c}: diff {max(details[c]['max_abs_diff_vs_single']):.1e}, rain effect "
                               f"{details[c]['max_abs_change_when_rain_swapped']:.1e}" for c in RAIN_CONFIGS), details)


def check_rollout(checks, name, ctx, event_id):
    ev = ctx.events[event_id]
    n0 = n0_of(ctx)
    area0 = torch.as_tensor(ctx.builder.grid.area[:n0], dtype=torch.float64)
    steps = min(ctx.rollout_steps, sbd.NT - 1)
    details, ok, msgs = {}, True, []
    for cfg, kw in RAIN_CONFIGS.items():
        m = build_rain(ctx, **kw).eval()
        coarse_h = []  # per step: max |H| of the latest input step, per coarse scale
        hook = m.register_forward_pre_hook(lambda mod, args: coarse_h.append(coarse_input_h_max(ctx, args[0].x)))
        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                ro = rollout(m, ctx.builder, ev["node_event"], t0=0, n_steps=steps, device=ctx.device)
        finally:
            hook.remove()
        dt = time.perf_counter() - t0
        raw = ro["raw"].cpu()
        h = ro["constrained"][:, :n0, 0].cpu().double()
        th = ro["target"][:, :n0, 0].cpu().double()
        finite = bool(torch.isfinite(raw).all())
        hmax = h.max(1).values
        per_step = {
            "t": ro["times"],
            "H_max_m": hmax.tolist(),
            "H_mean_scale0_m": h.mean(1).tolist(),
            "volume_pred_m3": (h * area0).sum(1).tolist(),
            "volume_citycat_m3": (th * area0).sum(1).tolist(),
            "rain_cumulative_volume_m3": (torch.cumsum(ro["rain_interval_m"].cpu().double(), 0)
                                          * float(area0.sum())).tolist(),
            "input_t": [t - 1 for t in ro["times"]],
            "input_coarse_abs_H_max_per_scale_m": coarse_h,
            "input_coarse_abs_H_max_m": [max(v) for v in coarse_h],
        }
        exploded = bool(hmax.max() > ctx.explode_h)
        good = finite and not exploded
        ok &= good
        details[cfg] = {"finite": finite, "exploded": exploded, "seconds": dt, "per_step": per_step}
        msgs.append(f"{cfg}: finite {finite}, H max {float(hmax.max()):.3f} m, final volume "
                    f"{per_step['volume_pred_m3'][-1]:.0f} vs CityCAT {per_step['volume_citycat_m3'][-1]:.0f} m3, "
                    f"coarse input |H| max {max(per_step['input_coarse_abs_H_max_m']):.3f} m")
    checks.record(name, "PASS" if ok else "FAIL", f"{event_id} {steps} steps | " + " | ".join(msgs), details)


def coarse_input_h_max(ctx, x):
    """max |H| of the latest input step (column -2) on each coarse scale."""
    ptr = ctx.builder.node_ptr.tolist()
    h = x[:, -2].detach()
    return [float(h[ptr[s]:ptr[s + 1]].abs().max()) for s in range(1, len(ptr) - 1)]


class CityCATOracle(nn.Module):
    """Fake model for check 6b: returns the CityCAT target on scale 0 and garbage on coarse rows."""

    def __init__(self, n0, coarse_value=1e3):
        super().__init__()
        self.n0 = n0
        self.coarse_value = coarse_value
        self.inputs = []

    def forward(self, graph):
        self.inputs.append(graph.x.detach().clone())
        out = torch.full((graph.x.shape[0], 2), self.coarse_value, dtype=graph.x.dtype, device=graph.x.device)
        out[:self.n0] = graph.y[:self.n0, :, 0]
        return out

    @staticmethod
    def constrain_state(pred):
        return constrain_state(pred)


def check_rollout_oracle(checks, name, ctx, event_id):
    ev = ctx.events[event_id]
    n0 = n0_of(ctx)
    steps = min(ctx.rollout_steps, sbd.NT - 1)
    oracle = CityCATOracle(n0)
    with torch.no_grad():
        rollout(oracle, ctx.builder, ev["node_event"], t0=0, n_steps=steps, device=ctx.device)
    ptr = ctx.builder.node_ptr.tolist()
    diffs, diffs_fine, diffs_coarse = [], [], []
    for k, x in enumerate(oracle.inputs):
        ref = ctx.builder.build_sample(ev["node_event"], k).x
        d = (x.cpu() - ref).abs()
        diffs.append(float(d.max()))
        diffs_fine.append(float(d[:ptr[1]].max()))
        diffs_coarse.append(float(d[ptr[1]:].max()))
    # constrain_state sets Q = 0 where H = 0; CityCAT nodes with Q > 0 and H = 0 would differ for a data reason
    H, Q = ev["node_event"]["H"][:n0], ev["node_event"]["Qmag"][:n0]
    q_where_dry = int(((Q > 0) & (H <= 0)).sum())
    tol = 1e-6
    ok = max(diffs) <= tol
    worst = int(np.argmax(diffs))
    details = {"steps": steps, "tolerance": tol, "max_abs_diff_per_input_t": diffs,
               "max_abs_diff_fine_rows": diffs_fine, "max_abs_diff_coarse_rows": diffs_coarse,
               "citycat_scale0_Q_pos_where_H_zero": q_where_dry}
    msg = (f"{event_id} {steps} steps | max |x_rollout - x_build_sample| {max(diffs):.2e} (tol {tol:.0e}), "
           f"worst input t={worst} (fine {diffs_fine[worst]:.2e}, coarse {diffs_coarse[worst]:.2e})")
    if q_where_dry:
        msg += f" | NOTE: {q_where_dry} CityCAT scale-0 entries with Q > 0 and H = 0"
    checks.record(name, "PASS" if ok else "FAIL", msg, details)


def rmse_hq(out, y, n0):
    return (torch.sqrt(((out[:n0, 0] - y[:n0, 0, 0]) ** 2).mean())
            + torch.sqrt(((out[:n0, 1] - y[:n0, 1, 0]) ** 2).mean()))


def check_overfit(checks, name, ctx, event_id):
    ev = ctx.events[event_id]
    n0 = n0_of(ctx)
    details, ok, msgs = {}, True, []
    for cfg, kw in RAIN_CONFIGS.items():
        for label, t in (("dry", 0), ("wet", ctx.wet_t)):
            m = build_rain(ctx, **kw).train()
            opt = torch.optim.Adam(m.parameters(), lr=ctx.overfit_lr)
            s = to_dev(ctx.builder.build_sample(ev["node_event"], t), ctx.device)
            curve = []
            t_start = time.perf_counter()
            for step in range(ctx.overfit_steps):
                opt.zero_grad(set_to_none=True)
                loss = rmse_hq(m(s), s.y, n0)
                loss.backward()
                opt.step()
                curve.append(float(loss))
            with torch.no_grad():
                out = m(s)
                final = float(rmse_hq(out, s.y, n0))
                state = constrain_state(out)[:n0]
            q_true = s.y[:n0, 1, 0] > 0
            n_q = int(q_true.sum())
            hit = float((state[q_true, 1] > 0).float().mean()) if n_q else None
            hit_raw = float((out[:n0][q_true, 1] > 0).float().mean()) if n_q else None
            initial = curve[0] if curve else final
            good = math.isfinite(final) and final < 0.5 * initial
            ok &= good
            details[f"{cfg}_{label}"] = {
                "transition": f"T{t} -> T{t + 1}", "loss_initial": initial, "loss_final": final,
                "loss_every_20": curve[::20], "seconds": time.perf_counter() - t_start,
                "citycat_Q_pos_nodes": n_q, "Q_hit_fraction_constrained": hit, "Q_hit_fraction_raw": hit_raw,
                "pass": good}
            msgs.append(f"{cfg}/T{t}->T{t + 1}: {initial:.3e} -> {final:.3e}"
                        + (f", Q hit {hit:.2f}" if hit is not None else ""))
    checks.record(name, "PASS" if ok else "FAIL", f"{event_id} " + " | ".join(msgs), details)


def check_cost(checks, name, ctx, event_id):
    ev = ctx.events[event_id]
    n0 = n0_of(ctx)
    s = ctx.builder.build_sample(ev["node_event"], ctx.wet_t)
    models = {"msgnn_original": build_msgnn(ctx)}
    for cfg, kw in RAIN_CONFIGS.items():
        models[f"rain_{cfg}"] = build_rain(ctx, **kw)
    details = {}
    for mname, m in models.items():
        d = to_dev(s, ctx.device)
        # one warm-up, then the median of ctx.cost_reps timings (CPU timings are noisy)
        m.eval()
        times = []
        reset_peak(ctx.device)
        with torch.no_grad():
            for r in range(ctx.cost_reps + 1):
                sync(ctx.device)
                t0 = time.perf_counter()
                m(d)
                sync(ctx.device)
                times.append(time.perf_counter() - t0)
        fwd_s, fwd_mb = float(np.median(times[1:])), peak_mb(ctx.device)
        m.train()
        times = []
        reset_peak(ctx.device)
        for r in range(ctx.cost_reps + 1):
            m.zero_grad(set_to_none=True)
            sync(ctx.device)
            t0 = time.perf_counter()
            ((m(d)[:n0] - d.y[:n0, :, 0]) ** 2).mean().backward()
            sync(ctx.device)
            times.append(time.perf_counter() - t0)
        bwd_s, bwd_mb = float(np.median(times[1:])), peak_mb(ctx.device)
        details[mname] = {"num_parameters": sum(p.numel() for p in m.parameters()),
                          "forward_seconds": fwd_s, "forward_peak_mb": fwd_mb,
                          "forward_backward_seconds": bwd_s, "forward_backward_peak_mb": bwd_mb,
                          "memory_kind": "cuda max_memory_allocated" if ctx.device.type == "cuda" else "CPU RSS peak"}
        del m
        if ctx.device.type == "cuda":
            torch.cuda.empty_cache()
    checks.record(name, "INFO", " | ".join(
        f"{k}: {v['num_parameters']} params, fwd {v['forward_seconds'] * 1000:.0f} ms / {v['forward_peak_mb']:.0f} MB, "
        f"fwd+bwd {v['forward_backward_seconds'] * 1000:.0f} ms / {v['forward_backward_peak_mb']:.0f} MB"
        for k, v in details.items()), details)


def run_all_checks(checks, ctx):
    """All checks on the events in ctx.events (ctx.primary first)."""
    for e in ctx.events:
        checks.run(f"1_equivalence/{e}", check_equivalence, ctx, e)
    checks.run("2_dry_no_rain", check_dry_no_rain, ctx, ctx.primary)
    for e in ctx.events:
        checks.run(f"3_uniformity_from_dry/{e}", check_uniformity, ctx, e)
    for e in ctx.events:
        checks.run(f"4_gradients/{e}", check_gradients, ctx, e)
    checks.run("5_batch_own_rain", check_batch, ctx)
    for e in ctx.events:
        checks.run(f"6_rollout/{e}", check_rollout, ctx, e)
    for e in ctx.events:
        checks.run(f"6b_rollout_oracle/{e}", check_rollout_oracle, ctx, e)
    for e in ctx.events:
        checks.run(f"7_overfit/{e}", check_overfit, ctx, e)
    checks.run("8_cost", check_cost, ctx, ctx.primary)


# ---- data ---------------------------------------------------------------------

def event_entry(builder, event, split):
    ne = builder.event_to_nodes(event)
    s0 = builder.build_sample(ne, 0)
    n0 = int(builder.node_ptr[1])
    return {
        "node_event": ne,
        "split": split,
        "rainfall_global_T0_T2_mps": event["rainfall_global"][:3].tolist(),
        "rain_t0_interval_mm": float(s0.rain_interval_m) * 1000,
        "t0_scale0_H_max": float(ne["H"][:n0, 0].max()),
    }


def find_split(dataset_root, event_id):
    for split in SPLITS:
        if event_id in sbd.split_event_ids(dataset_root, split):
            return split
    return None


def main():
    args = parse_args()
    out_dir: Path = args.output_dir
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        sys.exit(f"Output directory {out_dir} is not empty; use --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")
    import torch_geometric

    report = {"git_commit": git_commit(), "host": platform.node(), "torch": torch.__version__,
              "torch_geometric": torch_geometric.__version__, "device": str(device),
              "dataset_root": str(args.dataset_root), "num_scales": args.scales, "previous_t": args.previous_t,
              "rain_scale_mm": args.rain_scale_mm, "args": {k: str(v) for k, v in vars(args).items()}}

    dem, dem_meta = sbd.load_static_dem(args.dataset_root)
    report.update(dem_meta)
    train_ids = sbd.split_event_ids(args.dataset_root, "train")
    builder = sbd.StockbridgeMultiscale(dem, num_scales=args.scales, train_event_ids=train_ids,
                                        previous_t=args.previous_t)
    sbd.save_stats(builder.stats, out_dir / f"normalisation_stats_S{args.scales}.json")
    report["nodes_per_scale"] = builder.grid.nodes_per_scale()

    events = {}
    primary = None
    for eid in train_ids:  # first train event with rain in the step T0 -> T1 (rain in T0-T2)
        ev = sbd.load_event(args.dataset_root, "train", eid)
        entry = event_entry(builder, ev, "train")
        if entry["rain_t0_interval_mm"] > 0 and max(entry["rainfall_global_T0_T2_mps"]) > 0:
            primary, events[eid] = eid, entry
            break
    if primary is None:
        sys.exit("No train event with rain > 0 in T0-T2")
    extra_split = find_split(args.dataset_root, args.extra_event)
    if extra_split is None:
        print(f"WARNING: {args.extra_event} not found in any split; skipped")
    elif args.extra_event != primary:
        events[args.extra_event] = event_entry(
            builder, sbd.load_event(args.dataset_root, extra_split, args.extra_event), extra_split)
    report["primary_event"] = primary
    report["events"] = {k: {kk: vv for kk, vv in v.items() if kk != "node_event"} for k, v in events.items()}
    for k, v in report["events"].items():
        print(f"event {k} ({v['split']}): rain T0-T2 {v['rainfall_global_T0_T2_mps']} m/s, "
              f"interval rain T0->T1 {v['rain_t0_interval_mm']:.3f} mm, T0 H max {v['t0_scale0_H_max']:.3g} m")

    sample = builder.build_sample(events[primary]["node_event"], 0)
    ctx = SimpleNamespace(
        builder=builder, events=events, primary=primary, device=device, params=model_params(args.config),
        S=args.scales, p=args.previous_t, F=sample.x.shape[1], E=sample.edge_attr.shape[1],
        rain_scale_mm=args.rain_scale_mm, wet_t=args.wet_t, rollout_steps=args.rollout_steps,
        uniformity_steps=args.uniformity_steps, overfit_steps=args.overfit_steps,
        overfit_lr=args.overfit_lr, explode_h=args.explode_h, cost_reps=args.cost_reps)
    report["model_params"] = ctx.params

    checks = Checks()
    run_all_checks(checks, ctx)
    report["checks"] = checks.results
    report["summary"] = checks.summary()
    (out_dir / "smoke_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nSummary {report['summary']}\nWrote {out_dir / 'smoke_report.json'}")


if __name__ == "__main__":
    main()
