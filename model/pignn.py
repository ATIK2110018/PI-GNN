"""
Physics-Informed Graph Neural Network (PIGNN)
=============================================
Encoder → Processor (K × message-passing) → Decoder architecture.

Inputs per node:
  - Geometric features : [x_norm, y_norm, z_norm, area_norm]   (graph.x)
  - Flow state         : [h, u, v] at current time step         (HEC-RAS)

Output per node:
  - Manning's n  ∈ [n_min, n_max]   (scalar, inverse problem target)
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor

try:
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import softmax as pyg_softmax
except ImportError:
    raise ImportError("Install PyTorch Geometric: pip install torch-geometric")


# ─────────────────────────────────────────────────────────────────────────────
# USGS NLCD 2021 → Manning's n Physical Prior Lookup Table
# Source: Chow (1959), USACE HEC-RAS Hydraulic Reference Manual
# ─────────────────────────────────────────────────────────────────────────────

NLCD_N_PRIOR: dict[int, float] = {
    11: 0.025,   # Open Water         — clean channel bed
    12: 0.030,   # Perennial Ice/Snow — treated as low friction
    21: 0.040,   # Developed, Open    — grass/light impervious
    22: 0.055,   # Developed, Low     — scattered buildings
    23: 0.070,   # Developed, Medium  — mixed impervious
    24: 0.080,   # Developed, High    — dense urban
    31: 0.030,   # Barren             — rock/sand/clay
    41: 0.120,   # Deciduous Forest   — trees, leaf litter
    42: 0.130,   # Evergreen Forest   — dense canopy
    43: 0.110,   # Mixed Forest       — intermediate
    52: 0.070,   # Shrub/Scrub        — low woody vegetation
    71: 0.035,   # Herbaceous         — grasslands
    81: 0.040,   # Hay/Pasture        — managed grass
    82: 0.045,   # Cultivated Crops   — agricultural fields
    90: 0.100,   # Woody Wetlands     — flooded forest
    95: 0.060,   # Emergent Herbaceous Wetlands — marsh/reed
}


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

def _mlp(dims: list[int], act=nn.SiLU, final_act: bool = False) -> nn.Sequential:
    """Build a fully-connected MLP with Xavier init."""
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or final_act:
            layers.append(act())
    seq = nn.Sequential(*layers)
    for m in seq.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return seq


class EdgeConvLayer(MessagePassing):
    """
    Message  : MLP([h_i ‖ h_j ‖ e_ij] → hidden)
    Aggregate: sum
    Update   : MLP([h_i ‖ agg] → hidden) + residual
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__(aggr="add")
        self.msg_mlp = _mlp([node_dim * 2 + edge_dim, hidden_dim, hidden_dim])
        self.upd_mlp = _mlp([node_dim + hidden_dim, hidden_dim])
        self.norm    = nn.LayerNorm(hidden_dim)
        self.drop    = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.upd_mlp(torch.cat([x, agg], dim=-1))
        return self.drop(self.norm(out + x))

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))


class AttentionEdgeLayer(MessagePassing):
    """
    Edge-feature-conditioned graph attention layer.
    Attention: softmax(MLP([h_i ‖ h_j ‖ e_ij] → 1))
    Value    : MLP([h_j ‖ e_ij] → hidden)
    Update   : MLP([h_i ‖ weighted_sum] → hidden) + residual
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__(aggr="add")
        self.attn_mlp = _mlp([node_dim * 2 + edge_dim, hidden_dim, 1])
        self.val_mlp  = _mlp([node_dim + edge_dim, hidden_dim])
        self.upd_mlp  = _mlp([node_dim + hidden_dim, hidden_dim])
        self.norm     = nn.LayerNorm(hidden_dim)
        self.drop     = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr, num_nodes=x.size(0))
        out = self.upd_mlp(torch.cat([x, agg], dim=-1))
        return self.drop(self.norm(out + x))

    def message(self, x_i: Tensor, x_j: Tensor,
                edge_attr: Tensor, index: Tensor) -> Tensor:
        alpha = self.attn_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))  # [E,1]
        alpha = pyg_softmax(alpha, index)                                   # [E,1]
        val   = self.val_mlp(torch.cat([x_j, edge_attr], dim=-1))          # [E,H]
        return alpha * val


# ─────────────────────────────────────────────────────────────────────────────
# Main PIGNN
# ─────────────────────────────────────────────────────────────────────────────

class PIGNN(nn.Module):
    """
    Physics-Informed Graph Neural Network for inverse Manning's n estimation.

    Architecture:
      1. Node encoder : [geom_feat ‖ flow_feat] → hidden
      2. Edge encoder : [edge_feat]             → hidden
      3. Processor    : K × AttentionEdgeLayer (or EdgeConvLayer)
      4. n-Decoder    : hidden → sigmoid → n ∈ [n_min, n_max]
      5. (Optional) Flow-Decoder : hidden → (h, u, v)  [surrogate mode]
    """

    def __init__(
        self,
        node_feat_dim : int   = 4,
        edge_feat_dim : int   = 6,
        flow_feat_dim : int   = 3,
        hidden_dim    : int   = 128,
        n_layers      : int   = 6,
        dropout       : float = 0.0,
        n_min         : float = 0.025,
        n_max         : float = 0.150,
        use_attention : bool  = True,
        surrogate_mode: bool  = False,
        use_checkpoint: bool  = True,
        lulc_embed_dim: int   = 8,    # Size of the learned LULC embedding vector
    ):
        super().__init__()
        self.n_min          = n_min
        self.n_max          = n_max
        self.surrogate_mode = surrogate_mode
        self.use_checkpoint = use_checkpoint
        self.lulc_embed_dim = lulc_embed_dim

        # ── LULC Embedding ────────────────────────────────────────────────
        # Maps integer NLCD class ID (0-255) → learned vector of size lulc_embed_dim
        self.lulc_embedding = nn.Embedding(
            num_embeddings=256,
            embedding_dim=lulc_embed_dim,
            padding_idx=0
        )
        nn.init.uniform_(self.lulc_embedding.weight, -0.01, 0.01)

        # ── Physical Prior Lookup ─────────────────────────────────────────
        # Pre-build a fixed tensor: index=NLCD class → n_prior value
        prior_table = torch.full((256,), fill_value=0.045, dtype=torch.float32)
        for cls, n_val in NLCD_N_PRIOR.items():
            prior_table[cls] = n_val
        # Clamp to valid range
        prior_table.clamp_(n_min, n_max)
        self.register_buffer("n_prior_table", prior_table)  # saved in checkpoint

        # ── Encoder / Processor ───────────────────────────────────────────
        # Node encoder now also receives the LULC embedding
        in_node = node_feat_dim + flow_feat_dim + lulc_embed_dim
        self.node_enc = _mlp([in_node, hidden_dim, hidden_dim], final_act=True)
        self.edge_enc = _mlp([edge_feat_dim, hidden_dim, hidden_dim], final_act=True)

        LayerClass = AttentionEdgeLayer if use_attention else EdgeConvLayer
        self.layers = nn.ModuleList([
            LayerClass(hidden_dim, hidden_dim, hidden_dim, dropout)
            for _ in range(n_layers)
        ])

        # ── Decoders ─────────────────────────────────────────────────────
        # n_decoder now predicts a RESIDUAL (Δn) around the physical LULC prior.
        # This forces spatial heterogeneity rather than letting the net start from scratch.
        # Input: [latent ‖ raw_geometry ‖ lulc_embedding]
        self.n_decoder = _mlp([hidden_dim + node_feat_dim + lulc_embed_dim, hidden_dim // 2, 1])

        if surrogate_mode:
            self.flow_decoder = _mlp([hidden_dim, hidden_dim // 2, 3])

    # ── Internal encode / process ──────────────────────────────────────────

    def _encode_flow(self, h: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """Normalise flow state and stack as [N, 3]."""
        h_n = h.unsqueeze(-1) / (h.max().clamp(min=1e-8))
        u_n = u.unsqueeze(-1) / (u.abs().max().clamp(min=1e-8))
        v_n = v.unsqueeze(-1) / (v.abs().max().clamp(min=1e-8))
        return torch.cat([h_n, u_n, v_n], dim=-1)

    def _get_lulc_emb(self, data) -> Tensor:
        """Return LULC embedding; falls back to zeros if data.lulc not available."""
        if hasattr(data, "lulc") and data.lulc is not None:
            ids = data.lulc.to(self.lulc_embedding.weight.device).clamp(0, 255)
            return self.lulc_embedding(ids)           # [N, lulc_embed_dim]
        return torch.zeros(
            data.x.size(0), self.lulc_embed_dim,
            device=data.x.device, dtype=data.x.dtype
        )

    def encode(self, data, h: Tensor, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        flow_feat = self._encode_flow(h, u, v)
        lulc_emb  = self._get_lulc_emb(data)
        node_in   = torch.cat([data.x, flow_feat, lulc_emb], dim=-1)  # [N, 4+3+8]
        node_h    = self.node_enc(node_in)
        edge_h    = self.edge_enc(data.edge_attr)
        return node_h, edge_h, lulc_emb

    def process(self, node_h: Tensor, edge_index: Tensor, edge_h: Tensor) -> Tensor:
        if self.use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint
            for layer in self.layers:
                node_h = checkpoint(layer, node_h, edge_index, edge_h, use_reentrant=False)
        else:
            for layer in self.layers:
                node_h = layer(node_h, edge_index, edge_h)
        return node_h

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, data, h: Tensor, u: Tensor, v: Tensor) -> dict[str, Tensor]:
        """
        Returns:
            'n'           : Manning's n per cell [N]         (always)
            'h', 'u', 'v' : predicted flow fields [N]        (surrogate_mode only)
        """
        node_h, edge_h, lulc_emb = self.encode(data, h, u, v)
        node_h = self.process(node_h, data.edge_index, edge_h)

        # ── Residual Manning's n around Physical Prior ────────────────────
        # The decoder predicts Δn (a small correction), NOT n directly.
        # n = clamp(n_prior + tanh(Δn) * residual_scale)
        decoder_in = torch.cat([node_h, data.x, lulc_emb], dim=-1)
        delta_n_raw = self.n_decoder(decoder_in).squeeze(-1)   # unbounded scalar
        delta_n     = torch.tanh(delta_n_raw) * 0.05           # max ±0.05 correction

        # Get the physical prior for every cell from the LULC lookup table
        if hasattr(data, "lulc") and data.lulc is not None:
            ids     = data.lulc.to(self.n_prior_table.device).clamp(0, 255)
            n_prior = self.n_prior_table[ids]                  # [N] physical baseline
        else:
            n_prior = torch.full((data.x.size(0),), 0.045,
                                 device=data.x.device, dtype=data.x.dtype)

        n = (n_prior + delta_n).clamp(self.n_min, self.n_max)  # [N]
        out = {"n": n}

        if self.surrogate_mode:
            flow_raw = self.flow_decoder(node_h)
            out["h"] = torch.nn.functional.softplus(flow_raw[:, 0])
            out["u"] = flow_raw[:, 1]
            out["v"] = flow_raw[:, 2]

        return out
