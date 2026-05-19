# [TODO: NSC Disclaimer — see booklet page 44]
"""A3TGCN-based PM2.5 spatio-temporal forecasting model.

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.

Uses Type A static edges only. Type B (wind-aware) and Type C (hotspot)
edges are used by MTGNNModel.

Reference:
    Bai et al. (2021). A3T-GCN: Attention Temporal Graph Convolutional
    Network for Traffic Forecasting. ISPRS Int. J. Geo-Inf. 10(7), 485.
    https://doi.org/10.3390/ijgi10070485
"""

from __future__ import annotations

import logging

import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric_temporal.nn.recurrent import A3TGCN

from src.models.base import PM25ModelBase

logger = logging.getLogger(__name__)


class A3TGCNModel(PM25ModelBase):
    """Attention Temporal GCN wrapper for PM2.5 multi-horizon forecasting.

    Encodes a T_in-step window with A3TGCN (channel-first, time-last), then
    projects the hidden state to all forecast horizons in a single linear head.

    Note:
        A3TGCN.forward expects input of shape (N, F, T). The loader produces
        ``data['station'].x`` of shape (B*N, T_in, F), so this model permutes
        before passing to the encoder.

    Attributes:
        encoder: Underlying A3TGCN recurrent module.
        act: ReLU activation.
        dropout: Dropout layer.
        head: Linear projection to n_horizons outputs.
    """

    def __init__(
        self,
        n_stations: int,
        n_features: int = 5,
        horizons: list[int] | None = None,
        hidden_dim: int = 64,
        periods: int = 24,
        dropout: float = 0.1,
    ) -> None:
        """Initialise A3TGCNModel.

        Args:
            n_stations: Number of station nodes.
            n_features: Number of input features per node per timestep.
            horizons: Forecast horizons in hours. Defaults to [6, 12, 24, 48].
            hidden_dim: Hidden dimension for A3TGCN output channels.
            periods: Number of time periods (T_in window length) passed to A3TGCN.
            dropout: Dropout probability after activation.
        """
        horizons = horizons or [6, 12, 24, 48]
        super().__init__(n_stations, n_features, horizons)

        self.encoder = A3TGCN(
            in_channels=n_features,
            out_channels=hidden_dim,
            periods=periods,
        )
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(hidden_dim, self.n_horizons)

        logger.debug(
            "A3TGCNModel: n_stations=%d n_features=%d hidden_dim=%d periods=%d",
            n_stations,
            n_features,
            hidden_dim,
            periods,
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        """Compute multi-horizon PM2.5 predictions.

        Args:
            data: Batched HeteroData. Requires:
                - data['station'].x: shape (B*N, T_in, F)
                - data['station', 'type_a', 'station'].edge_index: shape (2, E_a)
                - data['station', 'type_a', 'station'].edge_attr: shape (E_a, 1+)

        Returns:
            Clipped predictions of shape (B*N, H), values >= 0.
        """
        # --- 1. Extract inputs ---
        x = data["station"].x  # (B*N, T_in, F)

        # A3TGCN requires channel-first: (N, F, T)
        x = x.permute(0, 2, 1).contiguous()  # (B*N, F, T_in)

        ei = data["station", "type_a", "station"].edge_index  # (2, E_a)
        ew = data["station", "type_a", "station"].edge_attr[:, 0].contiguous()  # (E_a,)

        # --- 2. A3TGCN encode ---
        h = self.encoder(X=x, edge_index=ei, edge_weight=ew)  # (B*N, hidden_dim)

        # --- 3. Activation + dropout ---
        h = self.dropout(self.act(h))  # (B*N, hidden_dim)

        # --- 4. Multi-horizon head ---
        out = self.head(h)  # (B*N, H)

        return self._postprocess(out)
