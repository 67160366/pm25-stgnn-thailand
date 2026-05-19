# [TODO: NSC Disclaimer — see booklet page 44]
"""Base class for all PM2.5 STGNN models.

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from torch_geometric.data import HeteroData

if TYPE_CHECKING:
    from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)


class PM25ModelBase(nn.Module, ABC):
    """Abstract base for all PM2.5 spatio-temporal forecasting models.

    Subclasses must implement :meth:`forward` which accepts a batched
    HeteroData and returns raw (un-clipped) predictions of shape (B*N, H).

    Attributes:
        n_stations: Number of station nodes.
        n_features: Number of input features per node per timestep.
        horizons: Sorted list of forecast horizons (hours ahead).
        n_horizons: Length of horizons.
    """

    def __init__(
        self,
        n_stations: int,
        n_features: int,
        horizons: list[int],
    ) -> None:
        """Initialise shared attributes.

        Args:
            n_stations: Number of station nodes. Must be > 0.
            n_features: Number of input features. Must be > 0.
            horizons: Forecast horizon offsets in hours. Must be non-empty.

        Raises:
            ValueError: If any argument violates the constraints above.
        """
        super().__init__()
        if n_stations <= 0:
            raise ValueError(f"n_stations must be > 0; got {n_stations}")
        if n_features <= 0:
            raise ValueError(f"n_features must be > 0; got {n_features}")
        if len(horizons) == 0:
            raise ValueError("horizons must be non-empty")

        self.n_stations = n_stations
        self.n_features = n_features
        self.horizons: list[int] = sorted(list(horizons))
        self.n_horizons: int = len(self.horizons)
        self._clip_output: bool = True

    @abstractmethod
    def forward(self, data: HeteroData) -> torch.Tensor:
        """Run model forward pass.

        Args:
            data: Batched HeteroData from the PM25GraphDataset loader.

        Returns:
            Raw µg/m³ predictions of shape (B*N, H).
        """

    def _postprocess(self, raw: torch.Tensor) -> torch.Tensor:
        """Clamp output to non-negative values if clip is enabled.

        Differentiable; safe for Graph-based Integrated Gradients (GB-IG).

        Args:
            raw: Raw model output of shape (B*N, H).

        Returns:
            Clamped (or identity) tensor of same shape.
        """
        return torch.clamp(raw, min=0.0) if self._clip_output else raw

    def predict(
        self,
        loader: DataLoader,
        device: torch.device | str = "cpu",
    ) -> np.ndarray:
        """Run inference over an entire DataLoader.

        Args:
            loader: PyG DataLoader yielding HeteroData batches.
            device: Target device for inference.

        Returns:
            Float32 ndarray of shape (total_N, H) with clipped predictions.
        """
        self.eval()
        self.to(device)

        preds: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = self(batch)
                preds.append(out.detach().cpu().numpy())

        if not preds:
            return np.zeros((0, self.n_horizons), dtype=np.float32)

        return np.concatenate(preds, axis=0)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters.

        Returns:
            Count of parameters where ``requires_grad=True``.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        """Compact string representation for nn.Module printing."""
        return (
            f"n_stations={self.n_stations}, "
            f"n_features={self.n_features}, "
            f"horizons={self.horizons}"
        )
