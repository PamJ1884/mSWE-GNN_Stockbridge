"""RainMSGNN: the original MSGNN with spatially uniform rainfall injected in latent space.

Why the original MSGNN cannot start a flood from rain
----------------------------------------------------
Stockbridge events start from a dry domain and are driven only by rainfall that
is uniform in space. Four properties of the original model (models/gnn.py,
models/models.py) work against that:

1. The dynamic node encoder is built by `make_mlp` with `bias=False` and PReLU
   activations, so a dry node (H = Q = 0 at every input step) is mapped to a
   dynamic latent of exactly 0.
2. `SWEGNN.forward` only builds messages on edges where at least one end has a
   non-zero latent (`mask = out.sum(1) != 0`), and with `with_gradient=True`
   the message is `s_ij * (h_dj - h_di)`. From a fully dry state no edge is
   active, the processors return 0, the decoder (no bias) returns 0 and the
   learned residual of a zero input is 0: the output is exactly 0 whatever the
   rain. Mechanisms 1 and 2 are what block a dry start.
3. Within one scale a spatially uniform latent is absorbing: the flux is
   proportional to h_dj - h_di, so equal neighbours exchange nothing and a
   uniform state stays uniform. If rain were just one more dynamic input
   column, every dry cell would get the same latent. In MSGNN this is not
   strictly absorbing: on the way up, the intra-scale GNN uses
   `with_gradient=False` (message `s_ij * h_d,coarse`) and `s_ij` depends on
   the static latents h_s, so a uniform coarse latent can become non-uniform on
   the finer scale. That path is weak and indirect (it needs the down pass,
   the pooling and the up pass, and it depends on the per-edge normalised
   `s_ij`), and it does nothing from a dry state because of 1 and 2.
4. The final `torch.relu` and `_mask_small_WD` (depth below 1e-4 m set to 0)
   zero the gradient of every node whose prediction is negative or below
   0.1 mm. In a dry domain under light rain that is almost every node.

What RainMSGNN changes (each change is a constructor switch)
------------------------------------------------------------
* Rain features per graph (`rain_features`): interval depth in mm and the
  start/end intensities in mm per 5 min, divided by `rain_scale_mm`. In a
  `Batch` each node takes the rain of its own graph (via `graph.batch`).
* `rain_forcing`:
    - "multiplicative" (default): h_d <- h_d + r * g(h_s, rain), where
      r = W_r rain (linear, no bias) and g is an MLP of [h_s, rain]. The term
      acts directly on the fine scale from the first step and is modulated by
      the static latent, so dry cells with different DEM/area get different
      latents and the SWE-GNN fluxes h_dj - h_di are non-zero. With zero rain
      r = 0 exactly, so a dry domain without rain still predicts exactly 0
      (no spurious water).
    - "additive": h_d <- h_d + phi([h_s, rain]) with an MLP with bias. Kept
      for ablation; it is not zero without rain.
    - "none": no forcing (used for the equivalence test with MSGNN).
* `rain_injection`:
    - "fine" (default): only on scale-0 nodes, right after the encoders.
    - "all_scales": also on every coarse scale, right after pooling and before
      that scale's processor in the down pass. Decision: one forcing module
      per scale (not one shared module with a scale embedding). There are only
      num_scales - 1 extra modules (about 13k parameters each with
      hid_features=64, mlp_layers=3), and a coarse node aggregates 4**s fine
      cells, so the mapping from rain to latent flux can differ per scale
      without a shared module having to learn the scale from the area input.
* `dynamic_encoder_bias` (default False, as in the paper). With the
  multiplicative forcing it is not needed, and with a bias every dry cell
  without rain would get the same non-zero latent and the decoder would
  produce uniform spurious water. True rebuilds the encoder with bias.
* `output_activation`: "raw" (default) removes the final ReLU and
  `_mask_small_WD` so every node keeps its gradient (mechanism 4); the loss is
  computed on the raw output. "relu" keeps the original behaviour.
* `rain_residual` (default True): adds `rain_interval_m` to the H output of
  every node ("bucket" balance: what falls stays if nothing moves or
  infiltrates); the decoder learns the correction (runoff, infiltration).
* `constrain_state(pred)`: H >= 0, Q >= 0 and Q = 0 where H = 0. Used to
  build the next input in a rollout and for metrics, not for the loss.

Input layout is the one of stockbridge/dataset.py (patch 1): static columns
first with DEM last (needed by `with_WL`), then the dynamic columns
[H, Q] per step, oldest first; multiscale pointers node_ptr / edge_ptr /
intra_edge_ptr as in MSGNN.

With rain_forcing="none", output_activation="relu", rain_residual=False and
dynamic_encoder_bias=False, RainMSGNN has exactly the parameters of MSGNN
(load_state_dict works in strict mode) and its forward is the original one.
Modules are created after MSGNN.__init__ (which seeds torch), so with the same
seed the backbone weights equal those of a plain MSGNN.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.gnn import MSGNN
from models.models import make_mlp

RAIN_FORCINGS = ("multiplicative", "additive", "none")
RAIN_INJECTIONS = ("fine", "all_scales")
OUTPUT_ACTIVATIONS = ("raw", "relu")
NUM_RAIN_FEATURES = 3  # interval depth, start intensity, end intensity
M_TO_MM = 1000.0


def rain_features(graph, rain_scale_mm: float = 10.0, dt_seconds: float = 300.0) -> torch.Tensor:
    """Per-graph rain features (B, 3): [interval mm, start mm/dt, end mm/dt] / rain_scale_mm.

    rain_a / rain_b are intensities in m/s; m/s * dt_seconds * 1000 = mm per
    step (mm per 5 min for the Stockbridge 300 s step). rain_interval_m is the
    trapezoidal interval depth in m.
    """
    depth_mm = graph.rain_interval_m.reshape(-1) * M_TO_MM
    a_mm = graph.rain_a.reshape(-1) * (dt_seconds * M_TO_MM)
    b_mm = graph.rain_b.reshape(-1) * (dt_seconds * M_TO_MM)
    return torch.stack([depth_mm, a_mm, b_mm], dim=1).float() / rain_scale_mm


def node_graph_index(graph) -> torch.Tensor:
    """Graph index of every node: graph.batch in a Batch, zeros in a single Data."""
    batch = getattr(graph, "batch", None)
    if batch is not None:
        return batch
    return torch.zeros(graph.x.shape[0], dtype=torch.long, device=graph.x.device)


def constrain_state(pred: torch.Tensor) -> torch.Tensor:
    """Physical state from a raw prediction [H, Q]: H >= 0, Q >= 0, Q = 0 where H = 0."""
    h = torch.relu(pred[:, 0])
    q = torch.relu(pred[:, 1]) * (h > 0)
    return torch.stack([h, q], dim=1)


def module_groups(model) -> dict:
    """Named sub-modules of an MSGNN / RainMSGNN for per-module gradient diagnostics."""
    S = model.num_scales
    groups = {
        "edge_encoder": model.edge_encoder,
        "static_node_encoder": model.static_node_encoder,
        "dynamic_node_encoder": model.dynamic_node_encoder,
    }
    for s, mod in enumerate(getattr(model, "rain_forcing_modules", [])):
        groups[f"rain_forcing_s{s}"] = mod
    for i, mod in enumerate(model.gnn_processor):
        groups[f"processor_down_s{i}" if i < S - 1 else f"processor_up_s{2 * S - 2 - i}"] = mod
    for i, mod in enumerate(model.intra_scale_gnn):
        groups[f"intra_scale_s{S - 1 - i}_to_s{S - 2 - i}"] = mod
    groups["node_decoder"] = model.node_decoder
    return groups


class MultiplicativeRainForcing(nn.Module):
    """r * g(h_s, rain) with r = W_r rain (no bias): exactly 0 when rain = 0."""

    def __init__(self, hid_features, n_rain, mlp_layers, mlp_activation, device="cpu"):
        super().__init__()
        self.rain_proj = nn.Linear(n_rain, hid_features, bias=False, device=device)
        self.gate = make_mlp(hid_features + n_rain, hid_features, hid_features, n_layers=mlp_layers,
                             bias=True, activation=mlp_activation, device=device)

    def forward(self, h_s, rain):
        return self.rain_proj(rain) * self.gate(torch.cat((h_s, rain), 1))


class AdditiveRainForcing(nn.Module):
    """phi([h_s, rain]) with bias (ablation): not zero when rain = 0."""

    def __init__(self, hid_features, n_rain, mlp_layers, mlp_activation, device="cpu"):
        super().__init__()
        self.mlp = make_mlp(hid_features + n_rain, hid_features, hid_features, n_layers=mlp_layers,
                            bias=True, activation=mlp_activation, device=device)

    def forward(self, h_s, rain):
        return self.mlp(torch.cat((h_s, rain), 1))


class RainMSGNN(MSGNN):
    """MSGNN with rainfall forcing in latent space; see the module docstring."""

    def __init__(self, num_node_features, num_edge_features, num_scales, *,
                 rain_forcing: str = "multiplicative", rain_injection: str = "fine",
                 dynamic_encoder_bias: bool = False, output_activation: str = "raw",
                 rain_residual: bool = True, rain_scale_mm: float = 10.0,
                 rain_dt_seconds: float = 300.0, mlp_layers: int = 2, mlp_activation: str = "prelu",
                 **msgnn_kwargs):
        if rain_forcing not in RAIN_FORCINGS:
            raise ValueError(f"rain_forcing must be one of {RAIN_FORCINGS}")
        if rain_injection not in RAIN_INJECTIONS:
            raise ValueError(f"rain_injection must be one of {RAIN_INJECTIONS}")
        if output_activation not in OUTPUT_ACTIVATIONS:
            raise ValueError(f"output_activation must be one of {OUTPUT_ACTIVATIONS}")
        super().__init__(num_node_features, num_edge_features, num_scales,
                         mlp_layers=mlp_layers, mlp_activation=mlp_activation, **msgnn_kwargs)
        self.rain_model = True
        self.rain_forcing = rain_forcing
        self.rain_injection = rain_injection
        self.dynamic_encoder_bias = dynamic_encoder_bias
        self.output_activation = output_activation
        self.rain_residual = rain_residual
        self.rain_scale_mm = float(rain_scale_mm)
        self.rain_dt_seconds = float(rain_dt_seconds)

        # rain_forcing_modules[s] acts on scale s (index 0 = finest)
        n_forcing = 0 if rain_forcing == "none" else (1 if rain_injection == "fine" else num_scales)
        forcing_cls = MultiplicativeRainForcing if rain_forcing == "multiplicative" else AdditiveRainForcing
        self.rain_forcing_modules = nn.ModuleList([
            forcing_cls(self.hid_features, NUM_RAIN_FEATURES, mlp_layers, mlp_activation, device=self.device)
            for _ in range(n_forcing)])

        if dynamic_encoder_bias:
            self.dynamic_node_encoder = make_mlp(self.dynamic_node_features, self.hid_features, self.hid_features,
                                                 n_layers=mlp_layers, bias=True, activation=mlp_activation,
                                                 device=self.device)

    # ------------------------------------------------------------------
    @staticmethod
    def constrain_state(pred: torch.Tensor) -> torch.Tensor:
        return constrain_state(pred)

    def node_rain_features(self, graph) -> torch.Tensor:
        """(N, 3) rain features; every node takes the rain of its own graph."""
        return rain_features(graph, self.rain_scale_mm, self.rain_dt_seconds)[node_graph_index(graph)]

    def _apply_forcing(self, scale, x_d, x_s, rain, mask):
        if self.rain_forcing == "none" or scale >= len(self.rain_forcing_modules):
            return x_d
        term = self.rain_forcing_modules[scale](x_s, rain)
        return x_d + term * (mask == scale)[:, None]

    def forward(self, graph):
        """MSGNN.forward (models/gnn.py) with the rain changes marked [rain]."""
        x = graph.x.clone()
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr

        edge_ptr = graph.edge_ptr
        intra_mesh_edge_index = graph.intra_mesh_edge_index
        intra_edge_ptr = graph.intra_edge_ptr

        # Create scale mask
        mask = self._create_scale_mask(graph)

        # 1. Node and edge encoder
        if self.edge_mlp:
            edge_attr = self.edge_encoder(edge_attr)

        x0 = x
        x_s = x[:, :self.static_node_features - self.with_WL]
        x_d = x[:, self.static_node_features - self.with_WL:]

        if self.with_WL:
            # Add water level as static input
            WL = x_s[:, -1] + x_d[:, -self.out_dim]
            x_s = torch.cat((x_s, WL.unsqueeze(-1)), 1)

        x_s = self.static_node_encoder(x_s)
        x_d = self.dynamic_node_encoder(x_d)

        # [rain] forcing on the finest scale, right after the encoders
        rain = self.node_rain_features(graph) if self.rain_forcing != "none" else None
        x_d = self._apply_forcing(0, x_d, x_s, rain, mask)

        # Create temporary variables to save the features after down and upgoing GNNs
        x_down = torch.zeros_like(x_d, device=self.device)
        x_up = torch.zeros_like(x_d, device=self.device)

        # 2. Processor
        # fine to coarse but skipping the coarsest scale (which is processed in the next loop)
        for i in range(self.num_scales - 1):
            # downgoing GNN pass
            x_d = self.gnn_processor[i](x_s, x_d, edge_index[:, edge_ptr[i]:edge_ptr[i + 1]],
                                        edge_attr[edge_ptr[i]:edge_ptr[i + 1]])

            # keep in memory the last operation before pooling (which would be overwritten otherwise)
            x_down = x_down + x_d * (mask == i)[:, None]

            # Pool multiscale attributes from finest to coarsest scale
            col_coarse, row_fine = intra_mesh_edge_index[:, intra_edge_ptr[i]:intra_edge_ptr[i + 1]]
            x_d = self._pooling(x_d, row_fine, col_coarse, reduce='mean', learnable=self.learned_pooling)

            # [rain] all_scales: forcing on scale i+1 after pooling, before its processor
            x_d = self._apply_forcing(i + 1, x_d, x_s, rain, mask)

        x_down = x_down + x_d

        # coarse to fine
        for i in range(self.num_scales):
            gnn_id = self.num_scales - 1 + i
            # upgoing GNN pass
            x_d = self.gnn_processor[gnn_id](x_s, x_d, edge_index[:, edge_ptr[-i - 2]:edge_ptr[-i - 1]],
                                             edge_attr[edge_ptr[-i - 2]:edge_ptr[-i - 1]])

            # save GNN output before pooling
            x_up = x_up + x_d * (mask == self.num_scales - i - 1)[:, None]

            # Un-pool multiscale attributes from coarsest to finest scale
            if i < self.num_scales - 1:
                intra_scale_edges = intra_mesh_edge_index[:, intra_edge_ptr[-i - 2]:intra_edge_ptr[-i - 1]]
                x_d = self.intra_scale_gnn[i](x_s, x_d, intra_scale_edges)

                # add skip connection of saved finer scale attributes
                if self.skip_connections:
                    x_d = x_d + x_down * (mask == self.num_scales - i - 2)[:, None]
        x = x_up

        # Add non-linearity
        if self.gnn_activation is not None:
            x = self.gnn_activation(x)

        # 3. Decoder
        x = self.node_decoder(x)

        # Add residual connections
        x = x + self._add_residual_connection(x0)

        # [rain] bucket residual: the interval rain depth is added to H on every node
        if self.rain_residual:
            rain_m = graph.rain_interval_m.reshape(-1).to(x.dtype)[node_graph_index(graph)]
            x = x + torch.stack((rain_m, torch.zeros_like(rain_m)), dim=1)

        # [rain] "relu" keeps the original output; "raw" leaves it unconstrained
        if self.output_activation == "relu":
            x = torch.relu(x)
            x = self._mask_small_WD(x, epsilon=0.0001)

        return x
