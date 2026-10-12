"""Paper-derived modern traffic baselines adapted to the common classification protocol.

These are clean-room PyTorch implementations of the core mechanisms described by
STAEformer (CIKM 2023), UIGNN (ICDM 2023), and OpenCity (2024).  The original
methods forecast continuous traffic values on their own data protocols.  Here the
backbones terminate in the same three-class head and use the same source-only
chronological splits as every other model.  Names deliberately include ``inductive``
or ``scratch``: they are protocol adaptations, not claims of byte-identical official
reproduction or of using target-exposed released checkpoints.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def graph_multiply(adjacency: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """Apply one graph support to ``[batch, node, channel]`` features."""
    batch, nodes, width = features.shape
    if adjacency.is_sparse:
        flat = features.permute(1, 0, 2).reshape(nodes, batch * width)
        return torch.sparse.mm(adjacency, flat).reshape(nodes, batch, width).permute(1, 0, 2)
    return torch.einsum("nm,bmd->bnd", adjacency, features)


def _baseline_output(
    logits: torch.Tensor,
    representation: torch.Tensor,
    num_cities: int,
    evidence: bool = False,
    reconstruction: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return the interface consumed by the shared trainer and evaluator."""
    if evidence:
        alpha = torch.nn.functional.softplus(logits).clamp_max(1e4) + 1
        logp = torch.log(alpha) - torch.log(alpha.sum(-1, keepdim=True))
    else:
        logp = logits.log_softmax(-1)
        alpha = torch.ones_like(logits)
    pooled = representation.mean(1)
    output = {
        "logits": logp,
        "alpha": alpha,
        "pre_logits": torch.full_like(logits, 1 / 3).log(),
        "pre_alpha": torch.ones_like(logits),
        "embedding": representation,
        "domain_logits": pooled.new_zeros((pooled.shape[0], num_cities)),
    }
    if reconstruction is not None:
        output["reconstruction"] = reconstruction
    return output


class AxisTransformerLayer(nn.Module):
    """Vanilla self-attention and feed-forward block along one selected axis."""

    def __init__(self, width: int, heads: int, dropout: float, expansion: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(width, expansion * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * width, width),
        )
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, axis: int) -> torch.Tensor:

        if axis == 1:
            batch, steps, nodes, width = x.shape
            sequence = x.permute(0, 2, 1, 3).reshape(batch * nodes, steps, width)
            restore = lambda z: z.reshape(batch, nodes, steps, width).permute(0, 2, 1, 3)
        elif axis == 2:
            batch, steps, nodes, width = x.shape
            sequence = x.reshape(batch * steps, nodes, width)
            restore = lambda z: z.reshape(batch, steps, nodes, width)
        else:
            raise ValueError(f"Unsupported attention axis: {axis}")
        attended = self.attention(sequence, sequence, sequence, need_weights=False)[0]
        sequence = self.norm1(sequence + self.dropout(attended))
        sequence = self.norm2(sequence + self.dropout(self.ffn(sequence)))
        return restore(sequence)


class STAEformerInductive(nn.Module):
    """Inductive STAEformer adaptation with temporal and spatial vanilla attention.

    Official STAEformer learns an embedding indexed by each dataset's time and node.
    Such an embedding cannot address a graph with a different number of unseen nodes.
    We retain its input/TOD/DOW/adaptive embedding decomposition and sequential
    temporal/spatial Transformer blocks, while generating the adaptive part from
    transferable topology features plus a shared temporal embedding.
    """

    def __init__(self, cfg: dict, num_cities: int):
        super().__init__()
        width = int(cfg["model"]["hidden_dim"])
        heads = int(cfg["model"].get("attention_heads", 4))
        layers = int(cfg["model"].get("attention_layers", 2))
        history = int(cfg["data"]["history_steps"])
        if width % heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if width < 8:
            raise ValueError("hidden_dim must be at least 8")
        input_width = width // 4
        tod_width = width // 8
        dow_width = width // 8
        adaptive_width = width - input_width - tod_width - dow_width
        self.variant = "staeformer_inductive"
        self.use_domain = self.use_evidence = False
        self.pre_evidence = None
        self.num_cities = num_cities
        self.history = history
        self.input_projection = nn.Linear(2, input_width)
        self.tod_embedding = nn.Embedding(48, tod_width)
        self.dow_embedding = nn.Embedding(7, dow_width)
        self.topology_embedding = nn.Sequential(
            nn.Linear(4, adaptive_width), nn.GELU(), nn.LayerNorm(adaptive_width)
        )
        self.temporal_adaptive = nn.Parameter(torch.empty(history, adaptive_width))
        nn.init.xavier_uniform_(self.temporal_adaptive)
        dropout = float(cfg["model"]["dropout"])
        self.temporal_layers = nn.ModuleList(
            AxisTransformerLayer(width, heads, dropout) for _ in range(layers)
        )
        self.spatial_layers = nn.ModuleList(
            AxisTransformerLayer(width, heads, dropout) for _ in range(layers)
        )
        self.output_projection = nn.Linear(history * width, 3)

    def forward(self, batch, adjacency, domain_scale=0.0):
        del adjacency, domain_scale
        history = batch["history"].transpose(1, 2)
        mask = batch["history_mask"].float().transpose(1, 2)
        batch_size, _steps, nodes = history.shape
        value = self.input_projection(torch.stack((history, mask), -1))
        tod = self.tod_embedding(batch["history_tod"])[:, :, None, :].expand(-1, -1, nodes, -1)
        dow = self.dow_embedding(batch["history_dow"])[:, :, None, :].expand(-1, -1, nodes, -1)
        topology = self.topology_embedding(batch["static"])
        adaptive = topology[:, None] + self.temporal_adaptive[None, :, None]
        x = torch.cat((value, tod, dow, adaptive), -1)
        for layer in self.temporal_layers:
            x = layer(x, 1)
        for layer in self.spatial_layers:
            x = layer(x, 2)
        representation = x.permute(0, 2, 1, 3).reshape(batch_size, nodes, -1)
        logits = self.output_projection(representation)
        return _baseline_output(logits, representation, self.num_cities)


class DiffusionGraphConv(nn.Module):
    """Bidirectional Chebyshev diffusion convolution used by UIGNN/IGNNK."""

    def __init__(self, in_features: int, out_features: int, order: int):
        super().__init__()
        self.order = order
        self.projection = nn.Linear(in_features * (2 * order + 1), out_features)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        basis = [x]
        for support in (adjacency, adjacency.transpose(0, 1)):
            x0 = x
            x1 = graph_multiply(support, x0)
            basis.append(x1)
            for _ in range(2, self.order + 1):
                x2 = 2 * graph_multiply(support, x1) - x0
                basis.append(x2)
                x0, x1 = x1, x2
        return self.projection(torch.cat(basis, -1))


class UIGNNInductive(nn.Module):
    """UIGNN-style three-layer residual diffusion network with evidential output.

    The shared trainer adds the paper's random-location recovery objective.  The
    continuous forecasting output is replaced by a three-class evidential head so
    that its uncertainty can be evaluated under the exact same task as RUGD.
    """

    def __init__(self, cfg: dict, num_cities: int):
        super().__init__()
        history = int(cfg["data"]["history_steps"])
        hidden = int(cfg["model"]["hidden_dim"])
        order = int(cfg["model"].get("diffusion_steps", 1))
        self.variant = "uignn_inductive"
        self.use_domain = False
        self.use_evidence = True
        self.use_recovery = True
        self.pre_evidence = None
        self.num_cities = num_cities
        self.gnn1 = DiffusionGraphConv(2 * history, hidden, order)
        self.gnn2 = DiffusionGraphConv(hidden, hidden, order)
        self.gnn3 = DiffusionGraphConv(hidden, hidden, order)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(float(cfg["model"]["dropout"]))
        self.reconstruction = nn.Linear(hidden, history)
        self.head = nn.Linear(hidden, 3)

    def forward(self, batch, adjacency, domain_scale=0.0):
        del domain_scale
        x = torch.cat((batch["history"], batch["history_mask"].float()), -1)
        h1 = self.dropout(torch.nn.functional.selu(self.gnn1(x, adjacency)))
        h2 = self.norm1(h1 + self.dropout(torch.nn.functional.selu(self.gnn2(h1, adjacency))))
        h3 = self.norm2(h2 + self.dropout(torch.nn.functional.selu(self.gnn3(h2, adjacency))))
        logits = self.head(h3)
        return _baseline_output(
            logits,
            h3,
            self.num_cities,
            evidence=True,
            reconstruction=self.reconstruction(h3),
        )


class RMSNorm(nn.Module):
    """RMS normalization used by the released OpenCity implementation."""

    def __init__(self, width: int, epsilon: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.epsilon)


class OpenCityBlock(nn.Module):
    """OpenCity-style temporal context attention, graph mixing, and SwiGLU FFN."""

    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.temporal = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.graph_projection = nn.Linear(width, width)
        self.norm2 = RMSNorm(width)
        self.w1 = nn.Linear(width, 2 * width, bias=False)
        self.w2 = nn.Linear(2 * width, width, bias=False)
        self.w3 = nn.Linear(width, 2 * width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch, patches, nodes, width = x.shape
        normalized = self.norm1(x)
        sequence = normalized.permute(0, 2, 1, 3).reshape(batch * nodes, patches, width)
        temporal = self.temporal(sequence, sequence, sequence, need_weights=False)[0]
        temporal = temporal.reshape(batch, nodes, patches, width).permute(0, 2, 1, 3)
        graph_input = temporal.reshape(batch * patches, nodes, width)
        graph = self.graph_projection(graph_multiply(adjacency, graph_input))
        x = x + self.dropout(graph.reshape(batch, patches, nodes, width))
        z = self.norm2(x)
        feed_forward = self.w2(torch.nn.functional.silu(self.w1(z)) * self.w3(z))
        return x + self.dropout(feed_forward)


class OpenCityScratch(nn.Module):
    """Source-only OpenCity-mini adaptation without target-exposed checkpoints.

    It retains temporal patching, shared positional/time context, graph propagation,
    RMSNorm, and gated feed-forward blocks.  The original 288-to-288 regression
    setup is shortened to the project's 12-to-1 classification horizon and trained
    from scratch because released pretraining includes TrafficCD and TrafficHZ.
    """

    def __init__(self, cfg: dict, num_cities: int):
        super().__init__()
        history = int(cfg["data"]["history_steps"])
        patch_length = int(cfg["model"].get("patch_length", 3))
        if history % patch_length:
            raise ValueError("history_steps must be divisible by patch_length")
        width = int(cfg["model"]["hidden_dim"])
        heads = int(cfg["model"].get("attention_heads", 4))
        layers = int(cfg["model"].get("attention_layers", 2))
        if width % heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.variant = "opencity_scratch"
        self.use_domain = self.use_evidence = False
        self.pre_evidence = None
        self.num_cities = num_cities
        self.patch_length = patch_length
        self.patches = history // patch_length
        self.patch_projection = nn.Linear(2 * patch_length, width, bias=False)
        self.position = nn.Parameter(torch.empty(self.patches, width))
        nn.init.normal_(self.position, std=1 / math.sqrt(width))
        self.time_projection = nn.Linear(4, width)
        self.topology_projection = nn.Linear(4, width)
        dropout = float(cfg["model"]["dropout"])
        self.blocks = nn.ModuleList(
            OpenCityBlock(width, heads, dropout) for _ in range(layers)
        )
        self.norm = RMSNorm(width)
        self.head = nn.Linear(self.patches * width, 3)

    def forward(self, batch, adjacency, domain_scale=0.0):
        del domain_scale
        speed = batch["history"]
        mask = batch["history_mask"].float()
        batch_size, nodes, _ = speed.shape
        speed = speed.reshape(batch_size, nodes, self.patches, self.patch_length)
        mask = mask.reshape(batch_size, nodes, self.patches, self.patch_length)
        patches = self.patch_projection(torch.cat((speed, mask), -1)).permute(0, 2, 1, 3)
        tod = batch["history_tod"].reshape(batch_size, self.patches, self.patch_length)
        dow = batch["history_dow"].reshape(batch_size, self.patches, self.patch_length)
        patch_tod = tod[:, :, -1].float() / 48.0
        patch_dow = dow[:, :, -1].float() / 7.0
        time_features = torch.stack(
            (torch.sin(2 * math.pi * patch_tod), torch.cos(2 * math.pi * patch_tod),
             torch.sin(2 * math.pi * patch_dow), torch.cos(2 * math.pi * patch_dow)), -1
        )
        time_context = self.time_projection(time_features)[:, :, None]
        topology = self.topology_projection(batch["static"])[:, None]
        x = patches + self.position[None, :, None] + time_context + topology
        for block in self.blocks:
            x = block(x, adjacency)
        x = self.norm(x)
        representation = x.permute(0, 2, 1, 3).reshape(batch_size, nodes, -1)
        logits = self.head(representation)
        return _baseline_output(logits, representation, self.num_cities)


MODERN_MODELS = {"staeformer_inductive", "uignn_inductive", "opencity_scratch"}


def build_modern_model(name: str, cfg: dict, num_cities: int) -> nn.Module:
    """Build one explicitly adapted modern baseline."""
    if name == "staeformer_inductive":
        return STAEformerInductive(cfg, num_cities)
    if name == "uignn_inductive":
        return UIGNNInductive(cfg, num_cities)
    if name == "opencity_scratch":
        return OpenCityScratch(cfg, num_cities)
    raise ValueError(f"Unknown modern baseline: {name}")
