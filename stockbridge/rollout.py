"""Autoregressive rollout for Stockbridge multiscale models.

The rollout starts from the CityCAT state at t0 (by default T0, a dry
domain), and at every step:
  1. builds the sample of the step t -> t+1 with
     `StockbridgeMultiscale.build_sample`, which gives that step's rainfall
     (rain_a, rain_b, rain_interval_m) and the CityCAT target;
  2. from the second step on, replaces its input `x` with the input built
     from the model's own predictions;
  3. runs the model, constrains the raw prediction with `constrain_state`
     (H >= 0, Q >= 0, Q = 0 where H = 0), rebuilds the coarse rows from the
     scale-0 rows (see below) and feeds that state to the next input with
     the original `utils.dataset.use_prediction` (drops the oldest [H, Q]
     pair and appends the prediction).

Coarse rows of the fed-back state. The model predicts [H, Q] on every node,
but the loss only uses scale 0, so the coarse predictions are never trained.
They must not be fed back: with with_WL=True, WL = DEM + H is computed on
every node and enters the static encoder, and the coarse static latent
drives the coarse fluxes and the intra-scale GNN, so uncontrolled coarse H
would contaminate the fine prediction and could drift over the rollout. In
teacher forcing the coarse inputs are the area-weighted mean of the CityCAT
scale-0 values (`StockbridgeMultiscale.event_to_nodes`); in the rollout they
are the same mean of the constrained scale-0 prediction
(`StockbridgeMultiscale.fine_to_all_scales`, differentiable).

Raw predictions are returned too, because the loss is computed on them.
The coarse rows of "raw" and "constrained" are returned as the model gave
them (after constrain_state) but are not used: neither in the next input nor,
by the convention of the smoke and training code, in the loss.

Gradients: the function does not wrap anything in `torch.no_grad`. Call it
inside `torch.no_grad()` for evaluation; call it normally to backpropagate
through the whole rollout (patch 3). `detach_between_steps=True` cuts the
graph between steps (one-step gradients only, less memory).
"""

from __future__ import annotations

import copy

import torch

from stockbridge.dataset import NT
from stockbridge.rain_msgnn import constrain_state as default_constrain_state
from utils.dataset import use_prediction


def rollout(model, builder, node_event: dict, t0: int = 0, n_steps: int | None = None,
            previous_t: int | None = None, device=None, detach_between_steps: bool = False) -> dict:
    """Roll the model out for n_steps from the CityCAT state at t0.

    model: maps a sample to a raw [H, Q] prediction (N, 2). Its
        `constrain_state` is used if it has one, else the RainMSGNN one.
    builder: StockbridgeMultiscale. If previous_t differs from
        builder.previous_t, a shallow copy with the requested previous_t is
        used (grid and statistics are shared).
    node_event: output of builder.event_to_nodes(event).
    n_steps: defaults to every step until the end of the event (NT - 1 - t0).

    Returns a dict with tensors of shape (n_steps, N, 2):
        raw, constrained (coarse rows unused, see the module docstring),
        target (CityCAT at t+1),
    and times (the predicted time indices t0+1 ... t0+n_steps) and
    rain_interval_m (n_steps,).
    """
    p = builder.previous_t if previous_t is None else previous_t
    if p != builder.previous_t:
        builder = copy.copy(builder)
        builder.previous_t = p
    if n_steps is None:
        n_steps = NT - 1 - t0
    if t0 < 0 or n_steps < 1 or t0 + n_steps > NT - 1:
        raise ValueError(f"need 0 <= t0 and t0 + n_steps <= {NT - 1}, got t0={t0}, n_steps={n_steps}")
    constrain = getattr(model, "constrain_state", default_constrain_state)
    n0 = int(builder.node_ptr[1])

    raw, constrained, target, rain = [], [], [], []
    x = None
    for k in range(n_steps):
        data = builder.build_sample(node_event, t0 + k)
        if device is not None:
            data = data.to(device)
        if x is not None:
            data.x = x
        out = model(data)
        state = constrain(out)
        raw.append(out)
        constrained.append(state)
        target.append(data.y[:, :, 0])
        rain.append(data.rain_interval_m.reshape(()))
        # coarse rows of the fed-back state = area-weighted mean of scale 0
        fed_back = builder.fine_to_all_scales(state[:n0])
        if detach_between_steps:
            x = use_prediction(data.x.detach(), fed_back.detach(), p)
        else:
            x = use_prediction(data.x, fed_back, p)

    return {
        "raw": torch.stack(raw),
        "constrained": torch.stack(constrained),
        "target": torch.stack(target),
        "times": list(range(t0 + 1, t0 + n_steps + 1)),
        "rain_interval_m": torch.stack(rain),
    }
