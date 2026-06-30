# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md
"""MTGNN-inspired multi-relational PM2.5 spatio-temporal forecasting model.

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.

Novelties over the original MTGNN (Wu et al. 2020):
    - Heterogeneous graph: Type A (static), Type B (wind-aware), Type C (hotspot)
    - Wind-aware dynamic adjacency for transboundary haze from Myanmar/Laos
    - NASA FIRMS hotspot clusters as first-class bipartite source nodes

Reference:
    Wu et al. (2020). Connecting the Dots: Multivariate Time Series Forecasting
    with Graph Neural Networks. KDD 2020.
    https://dl.acm.org/doi/10.1145/3394486.3403118
"""

from __future__ import annotations

import logging
import math

import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GraphConv

from src.models.base import PM25ModelBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helper modules
# ---------------------------------------------------------------------------


class _MixHopConv(nn.Module):
    """Multi-hop graph convolution that concatenates embeddings from each hop.

    For an input of shape (N, C), performs k_hop successive GraphConv passes
    and concatenates [h_0, h_1, ..., h_k] before projecting back to out_channels.

    Note:
        GraphConv from PyG handles empty edge_index (shape (2, 0)) without
        crashing — it returns the linear transform of x only (verified PyG 2.6.1).

    Args:
        in_channels: Input feature dimension.
        out_channels: Output feature dimension after projection.
        k_hop: Number of graph convolution hops.
    """

    def __init__(self, in_channels: int, out_channels: int, k_hop: int) -> None:
        super().__init__()
        self.k_hop = k_hop
        self.gcn_layers = nn.ModuleList(
            [GraphConv(in_channels, in_channels, aggr="add") for _ in range(k_hop)]
        )
        self.proj = nn.Linear((k_hop + 1) * in_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Apply multi-hop convolution.

        Args:
            x: Node features of shape (N, in_channels).
            edge_index: Edge connectivity of shape (2, E).
            edge_weight: Scalar edge weights of shape (E,).

        Returns:
            Projected node embeddings of shape (N, out_channels).
        """
        outs = [x]
        h = x
        for gcn in self.gcn_layers:
            h = gcn(h, edge_index, edge_weight)
            outs.append(h)
        return self.proj(torch.cat(outs, dim=-1))


class _BipartiteGraphConv(nn.Module):
    """Single GraphConv layer in bipartite mode for hotspot→station messages.

    Wraps PyG's GraphConv with a (src, dst) feature tuple so that source
    (hotspot) and destination (station) node features can differ in dimension.

    Args:
        src_channels: Feature dimension of source nodes (hotspots).
        dst_channels: Feature dimension of destination nodes (stations).
        out_channels: Output feature dimension for destination nodes.
    """

    def __init__(
        self,
        src_channels: int,
        dst_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        # GraphConv in bipartite mode: takes (x_src, x_dst) tuple
        self.conv = GraphConv((src_channels, dst_channels), out_channels, aggr="add")

    def forward(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Message-pass from hotspot nodes to station nodes.

        Args:
            x_src: Hotspot node features of shape (K, src_channels).
            x_dst: Station node features of shape (N, dst_channels).
            edge_index: Edge connectivity of shape (2, E_c) where row 0 indexes
                into x_src (hotspots) and row 1 indexes into x_dst (stations).
            edge_weight: Scalar edge weights of shape (E_c,).

        Returns:
            Updated station node features of shape (N, out_channels).
        """
        return self.conv((x_src, x_dst), edge_index, edge_weight)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class MTGNNModel(PM25ModelBase):
    """Multi-relational MTGNN for PM2.5 multi-horizon forecasting.

    Architecture:
        1. ``start_conv``: 1-D conv to project features to hidden_dim.
        2. Dilated temporal conv stack (gated tanh-sigmoid activation, residual).
        3. Last-timestep pooling → spatial processing:
           a. Adaptive self-learned adjacency (Wu et al. 2020 §3.2)
           b. Type A static mix-hop convolution
           c. Type B wind-aware mix-hop convolution
           d. Type C hotspot bipartite convolution
        4. Fuse all spatial signals, LayerNorm + residual + dropout.
        5. Two-layer MLP head to n_horizons outputs.

    Args:
        n_stations: Number of station nodes (used for adaptive embedding size).
        n_features: Number of input features per node per timestep.
        horizons: Forecast horizons in hours. Defaults to [6, 12, 24, 48].
        hidden_dim: Hidden channel width throughout the network.
        n_layers: Number of dilated temporal conv layers (dilation = 2^i).
        kernel_size: Temporal conv kernel size.
        k_hop: Number of graph hops in MixHop convolution.
        dropout: Dropout probability before the output head.
        adaptive_dim: Embedding dimension for adaptive adjacency (Wu et al. 2020).
        use_type_a: Include the Type A static-distance graph in the spatial sum.
        use_type_b: Include the Type B wind-aware graph in the spatial sum.
        use_type_c: Include the Type C hotspot bipartite signal in the spatial sum.
        use_adaptive: Include the self-learned adaptive adjacency in the spatial sum.
    """

    def __init__(
        self,
        n_stations: int,
        n_features: int = 5,
        horizons: list[int] | None = None,
        hidden_dim: int = 64,
        n_layers: int = 3,
        kernel_size: int = 7,
        k_hop: int = 2,
        dropout: float = 0.1,
        adaptive_dim: int = 10,
        use_type_a: bool = True,
        use_type_b: bool = True,
        use_type_c: bool = True,
        use_adaptive: bool = True,
    ) -> None:
        horizons = horizons or [6, 12, 24, 48]
        super().__init__(n_stations, n_features, horizons)

        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1; got {n_layers}")
        if k_hop < 1:
            raise ValueError(f"k_hop must be >= 1; got {k_hop}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0); got {dropout}")

        self.hidden_dim = hidden_dim
        self.dilations = [2**i for i in range(n_layers)]

        # Spatial-channel ablation gates (Session 8). All True = full model. Disabling a
        # channel zeroes its contribution to the spatial sum (and hence its gradient), so a
        # retrained variant measures that channel's true accuracy contribution.
        self.use_type_a = use_type_a
        self.use_type_b = use_type_b
        self.use_type_c = use_type_c
        self.use_adaptive = use_adaptive
        if not (use_type_a and use_type_b and use_type_c and use_adaptive):
            logger.info(
                "MTGNN ablation: type_a=%s type_b=%s type_c=%s adaptive=%s",
                use_type_a,
                use_type_b,
                use_type_c,
                use_adaptive,
            )

        # --- Temporal stack ---
        self.start_conv = nn.Conv1d(n_features, hidden_dim, kernel_size=1)

        self.tconv_f = nn.ModuleList(
            [
                nn.Conv1d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    dilation=d,
                    padding=(kernel_size - 1) * d,
                )
                for d in self.dilations
            ]
        )
        self.tconv_g = nn.ModuleList(
            [
                nn.Conv1d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    dilation=d,
                    padding=(kernel_size - 1) * d,
                )
                for d in self.dilations
            ]
        )

        # --- Adaptive adjacency embeddings (Wu et al. 2020 §3.2) ---
        # Init: uniform(-1/sqrt(d), 1/sqrt(d)) as in the original paper.
        self.adaptive_emb1 = nn.Embedding(n_stations, adaptive_dim)
        self.adaptive_emb2 = nn.Embedding(n_stations, adaptive_dim)
        bound = 1.0 / math.sqrt(adaptive_dim)
        nn.init.uniform_(self.adaptive_emb1.weight, -bound, bound)
        nn.init.uniform_(self.adaptive_emb2.weight, -bound, bound)

        # --- Spatial modules ---
        self.type_a_conv = _MixHopConv(hidden_dim, hidden_dim, k_hop)
        self.type_b_conv = _MixHopConv(hidden_dim, hidden_dim, k_hop)
        self.adaptive_proj = nn.Linear(hidden_dim, hidden_dim)

        self.hotspot_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.type_c_conv = _BipartiteGraphConv(hidden_dim, hidden_dim, hidden_dim)

        # --- Post-spatial ---
        self.spatial_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # --- Output head ---
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_horizons),
        )

        # Non-persistent buffer: station index tensor (moves with .to(device))
        self.register_buffer(
            "_station_idx",
            torch.arange(n_stations),
            persistent=False,
        )

        logger.debug(
            "MTGNNModel: n_stations=%d n_features=%d hidden_dim=%d "
            "n_layers=%d kernel_size=%d k_hop=%d adaptive_dim=%d",
            n_stations,
            n_features,
            hidden_dim,
            n_layers,
            kernel_size,
            k_hop,
            adaptive_dim,
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        """Compute multi-horizon PM2.5 predictions.

        Args:
            data: Batched HeteroData. Requires:
                - data['station'].x: shape (B*N, T_in, F)
                - data['station', 'type_a', 'station'].edge_index: (2, E_a)
                - data['station', 'type_a', 'station'].edge_attr: (E_a, 1+)
                - data['station', 'type_b', 'station'].edge_index: (2, E_b)
                - data['station', 'type_b', 'station'].edge_attr: (E_b, 1+)
                - data['hotspot'].x: (K_total, 3) — zeros if no hotspots
                - data['hotspot', 'type_c', 'station'].edge_index: (2, E_c)
                - data['hotspot', 'type_c', 'station'].edge_attr: (E_c, 1+)

        Returns:
            Clipped predictions of shape (B*N, H), values >= 0.

        Raises:
            AssertionError: If total nodes (B*N) is not divisible by n_stations.
        """
        # --- 1. Initial projection ---
        x = data["station"].x  # (B*N, t_in, F)
        t_in = x.size(1)
        x = x.permute(0, 2, 1).contiguous()  # (B*N, F, t_in)
        h = self.start_conv(x)  # (B*N, hidden_dim, t_in)

        # --- 2. Dilated temporal conv stack with gated activation + residual ---
        for i in range(len(self.dilations)):
            gate = torch.tanh(self.tconv_f[i](h)) * torch.sigmoid(self.tconv_g[i](h))
            gate = gate[..., :t_in]  # causal trim: keep only the first t_in timesteps
            h = h + gate  # residual (out-of-place to preserve grad flow)

        # --- 3. Take last timestep for spatial processing ---
        s = h[..., -1]  # (B*N, hidden_dim)

        # --- 4. Adaptive self-learned adjacency ---
        bn = s.size(0)
        if bn % self.n_stations != 0:
            raise ValueError(f"Total nodes {bn} is not divisible by n_stations={self.n_stations}")
        batch_size = bn // self.n_stations

        e1 = self.adaptive_emb1(self._station_idx)  # (N, adaptive_dim)
        e2 = self.adaptive_emb2(self._station_idx)  # (N, adaptive_dim)
        adj = torch.softmax(torch.relu(e1 @ e2.t()), dim=-1)  # (N, N)

        s_3d = s.view(batch_size, self.n_stations, self.hidden_dim)  # (B, N, hidden_dim)
        g_3d = adj @ s_3d  # (B, N, hidden_dim)  — broadcasts over batch dim
        g = g_3d.reshape(batch_size * self.n_stations, self.hidden_dim)  # (B*N, hidden_dim)
        g = self.adaptive_proj(g)  # (B*N, hidden_dim)

        # --- 5. Type A station→station (mix-hop) ---
        ei_a = data["station", "type_a", "station"].edge_index  # (2, E_a)
        ew_a = data["station", "type_a", "station"].edge_attr[:, 0].contiguous()  # (E_a,)
        a = self.type_a_conv(s, ei_a, ew_a)  # (B*N, hidden_dim)

        # --- 6. Type B station→station wind-aware (mix-hop) ---
        ei_b = data["station", "type_b", "station"].edge_index  # (2, E_b)
        ew_b = data["station", "type_b", "station"].edge_attr[:, 0].contiguous()  # (E_b,)
        b = self.type_b_conv(s, ei_b, ew_b)  # (B*N, hidden_dim)

        # --- 7. Type C hotspot→station (bipartite) ---
        hx = data["hotspot"].x  # (K_total, 3)
        ei_c = data["hotspot", "type_c", "station"].edge_index  # (2, E_c)
        ew_c = data["hotspot", "type_c", "station"].edge_attr[:, 0].contiguous()  # (E_c,)

        if hx.size(0) == 0 or ei_c.size(1) == 0:
            c = torch.zeros(s.size(0), self.hidden_dim, dtype=s.dtype, device=s.device)
        else:
            hh = self.hotspot_encoder(hx)  # (K_total, hidden_dim)
            c = self.type_c_conv(hh, s, ei_c, ew_c)  # (B*N, hidden_dim)

        # --- 8. Fuse spatial contributions (ablation gates zero disabled channels) ---
        spatial = (
            float(self.use_type_a) * a
            + float(self.use_type_b) * b
            + float(self.use_adaptive) * g
            + float(self.use_type_c) * c
        )  # (B*N, hidden_dim)

        # --- 9. Residual + norm + dropout ---
        combined = self.spatial_norm(spatial) + s  # (B*N, hidden_dim)
        combined = self.dropout(combined)

        # --- 10. Head ---
        out = self.head(combined)  # (B*N, H)
        return self._postprocess(out)
