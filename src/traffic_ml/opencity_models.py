"""Deep controls, paper-derived baselines, and the final RUGD model."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.autograd import Function

from .opencity_modern_models import MODERN_MODELS, build_modern_model


class GradientReverse(Function):
    """Identity forward operation with a scaled, sign-reversed backward gradient."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor):
        return -ctx.scale * gradient, None


class NodeTemporalEncoder(nn.Module):
    """Encode masked speed histories with node/city-shared GRU parameters."""

    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(2, hidden, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, speed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, nodes, length = speed.shape
        values = torch.stack((speed, mask.float()), -1).reshape(batch * nodes, length, 2)
        _, state = self.gru(values)
        return self.drop(self.norm(state[-1])).reshape(batch, nodes, -1)


def graph_multiply(adjacency: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """Multiply a shared dense/sparse graph by batched node features."""
    batch, nodes, width = features.shape
    if adjacency.is_sparse:
        flat = features.permute(1, 0, 2).reshape(nodes, batch * width)
        return torch.sparse.mm(adjacency, flat).reshape(nodes, batch, width).permute(1, 0, 2)
    return torch.einsum("nm,bmd->bnd", adjacency, features)


class DeepTrafficModel(nn.Module):
    """GRU, graph-GRU, or proposed ShiftGuard-RUGD.

    RUGD combines a node-shared temporal encoder, transferable topology/calendar features,
    normalized graph diffusion, source-domain alignment, and local/final evidential objectives.
    The preliminary head supervises locally grounded uncertainty without modifying graph
    transport; this keeps the diffusion path simple, stable, and inductive across road networks.
    """

    def __init__(self, variant="rugd", static_dim=4, hidden_dim=96, dropout=.2,
                 diffusion_steps=2, num_cities=2):
        super().__init__()
        self.variant = variant
        proposed = variant.startswith("rugd")
        self.proposed = proposed
        self.use_graph = variant not in {"gru", "rugd_no_graph"}
        self.use_topology = not variant.endswith("_no_topology")
        self.use_domain = proposed and not variant.endswith("_no_domain")
        self.use_evidence = proposed and not variant.endswith("_no_evidence")
        self.diffusion_steps = diffusion_steps if self.use_graph else 0
        self.temporal = NodeTemporalEncoder(hidden_dim, dropout)
        self.static = nn.Sequential(nn.Linear(static_dim + 4, hidden_dim), nn.GELU(),
                                    nn.LayerNorm(hidden_dim))
        self.local_fuse = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
                                        nn.LayerNorm(hidden_dim))
        self.pre_evidence = nn.Linear(hidden_dim, 3) if proposed else None
        self.mix = nn.ModuleList([nn.Linear(hidden_dim * 2, hidden_dim)
                                  for _ in range(self.diffusion_steps)])
        self.gates = nn.ModuleList([nn.Linear(hidden_dim * 2 + 1, hidden_dim)
                                    for _ in range(self.diffusion_steps)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim)
                                    for _ in range(self.diffusion_steps)])
        self.fuse = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
                                  nn.Dropout(dropout), nn.LayerNorm(hidden_dim))
        self.head = nn.Linear(hidden_dim, 3)
        domain_hidden = max(4, hidden_dim // 2)
        self.domain = nn.Sequential(nn.Linear(hidden_dim, domain_hidden), nn.GELU(),
                                    nn.Linear(domain_hidden, num_cities))

    def forward(self, batch, adjacency, domain_scale=0.):
        temporal = self.temporal(batch["history"], batch["history_mask"])
        _batch_size, nodes, _ = temporal.shape
        calendar = batch["calendar"][:, None, :].expand(-1, nodes, -1)
        topology = self.static(torch.cat((batch["static"], calendar), -1))
        if not self.use_topology:
            topology = torch.zeros_like(topology)
        local = self.local_fuse(torch.cat((temporal, topology), -1))
        if self.pre_evidence is not None:
            pre_alpha = torch.nn.functional.softplus(self.pre_evidence(local)).clamp_max(1e4) + 1
            pre_logp = torch.log(pre_alpha) - torch.log(pre_alpha.sum(-1, keepdim=True))
        else:
            pre_alpha = torch.ones((*local.shape[:-1], 3), device=local.device)
            pre_logp = torch.full_like(pre_alpha, 1 / 3).log()

        hidden = temporal
        unit_context = temporal.new_ones(temporal.shape[:2])
        for mix, gate, norm in zip(self.mix, self.gates, self.norms):
            if self.proposed:
                numerator = graph_multiply(adjacency, hidden)
                denominator = graph_multiply(adjacency, unit_context[..., None])
                propagated = numerator / denominator.clamp_min(1e-4)
            else:
                propagated = graph_multiply(adjacency, hidden)
            pair = torch.cat((hidden, propagated), -1)
            gate_input = torch.cat((pair, unit_context[..., None]), -1)
            hidden = norm(hidden + torch.sigmoid(gate(gate_input)) * torch.tanh(mix(pair)))

        representation = self.fuse(torch.cat((hidden, topology), -1))
        raw = self.head(representation)
        if self.use_evidence:
            alpha = torch.nn.functional.softplus(raw).clamp_max(1e4) + 1
            logp = torch.log(alpha) - torch.log(alpha.sum(-1, keepdim=True))
        else:
            logp = raw.log_softmax(-1)
            alpha = torch.ones_like(raw)
        pooled = representation.mean(1)
        return {"logits": logp, "alpha": alpha, "pre_logits": pre_logp,
                "pre_alpha": pre_alpha, "embedding": representation,
                "domain_logits": self.domain(GradientReverse.apply(pooled, domain_scale))}


SUPPORTED_DEEP_MODELS = {
    "gru", "graph_gru", "rugd", "rugd_no_graph", "rugd_no_topology",
    "rugd_no_domain", "rugd_no_evidence", "rugd_no_focal",
    "rugd_no_class_balance",
} | MODERN_MODELS


def build_open_model(name, cfg, num_cities):
    """Construct a registered control, full proposed model, or one-factor ablation."""
    if name not in SUPPORTED_DEEP_MODELS:
        raise ValueError(f"Unknown deep model: {name}")
    if name in MODERN_MODELS:
        return build_modern_model(name, cfg, num_cities)
    return DeepTrafficModel(name, 4, cfg["model"]["hidden_dim"], cfg["model"]["dropout"],
                            cfg["model"]["diffusion_steps"], num_cities)
