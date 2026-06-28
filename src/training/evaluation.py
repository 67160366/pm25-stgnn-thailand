# [TODO: NSC Disclaimer — see booklet page 44]

"""Shared evaluation helpers for PM2.5 STGNN models.

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.

Reused by ``scripts/04_evaluate.py`` (per-horizon RMSE vs persistence) and
``scripts/06_nwp_sensitivity.py`` (ERA5-perturbation sensitivity) so that
denormalisation, the persistence baseline, and masking are computed identically
across experiments. Keeping this logic in one place avoids the per-station
denormalisation drift that caused earlier µg/m³ figures to be overstated
(see docs/SESSION7_NOTES.md §1).
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.data.loader import PM25GraphDataset
from src.training.metrics import compute_metrics


def station_scalers(ds: PM25GraphDataset) -> tuple[np.ndarray, np.ndarray]:
    """Per-station RobustScaler center/scale arrays in station order.

    Args:
        ds: A constructed PM25GraphDataset (provides station order + scalers).

    Returns:
        Tuple of (centers, scales), each shape (N,) float64, ordered to match
        ``ds._station_ids`` (sorted station order).
    """
    centers = np.array(
        [ds.scalers[int(sid)]["center_"] for sid in ds._station_ids], dtype=np.float64
    )
    scales = np.array([ds.scalers[int(sid)]["scale_"] for sid in ds._station_ids], dtype=np.float64)
    return centers, scales


def build_ground_truth(
    ds: PM25GraphDataset, horizons: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build targets, validity masks, and a fair persistence forecast in µg/m³.

    Everything is derived directly from the dataset's raw arrays (not the loader),
    so there is no batch-ordering fragility. Rows are flattened sample-major,
    station-minor to match the model's prediction row order (PyG concatenates
    graphs in dataset order; station nodes are contiguous within each graph).

    Persistence is the *last observed* PM2.5 carried forward (scanning back over the
    input window from the anchor), not the anchor value naïvely — this avoids
    penalising the baseline for a single missing anchor reading, which would
    otherwise inflate the model's apparent improvement. Positions where no
    observation exists anywhere in the window are excluded from the mask so that
    every method is scored on identical positions.

    Args:
        ds: Validation/test dataset.
        horizons: Forecast horizons in hours.

    Returns:
        Tuple (target_ug, mask, pers_ug):
            target_ug: (S*N, H) ground-truth PM2.5 in µg/m³.
            mask: (S*N, H) bool; True where the target is valid AND a persistence
                value is available.
            pers_ug: (S*N, H) persistence forecast in µg/m³ (same value per row
                across horizons).
    """
    anchors = ds._anchor_indices  # (S,)
    raw = ds._pm25_raw  # (T_full, N) µg/m³, NaN preserved
    s, n = len(anchors), ds.n_stations

    target = np.stack([raw[anchors + h, :] for h in horizons], axis=2).astype(np.float64)
    mask = np.stack(
        [
            (~ds._mask_in_loss[anchors + h, :])
            & (~ds._exclude[anchors + h, :])
            & np.isfinite(raw[anchors + h, :])
            for h in horizons
        ],
        axis=2,
    )

    # Forward-fill last observed PM2.5 within the input window (k=0 is the anchor).
    pers = np.full((s, n), np.nan, dtype=np.float64)
    for k in range(ds.window_in):
        cand = raw[anchors - k, :]
        fill = ~np.isfinite(pers) & np.isfinite(cand)
        pers[fill] = cand[fill]

    h_count = len(horizons)
    target_ug = target.reshape(s * n, h_count)
    mask_flat = mask.reshape(s * n, h_count)
    pers_ug = np.repeat(pers.reshape(s * n, 1), h_count, axis=1)

    # Score every method on identical positions: require both a valid target and an
    # available persistence value (a finite observation somewhere in the window).
    mask_flat = mask_flat & np.isfinite(pers_ug)
    return target_ug, mask_flat, pers_ug


def predict(model: torch.nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    """Run inference over the loader, returning (S*N, H) normalised predictions."""
    model.eval().to(device)
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds.append(model(batch).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def denorm_pred(
    pred: np.ndarray, centers: np.ndarray, scales: np.ndarray, n_stations: int
) -> np.ndarray:
    """Inverse RobustScaler for model predictions: raw = scaled * scale + center.

    Row r maps to station ``r % n_stations`` (sample-major, station-minor order).
    """
    station_local = np.arange(pred.shape[0]) % n_stations
    return pred * scales[station_local][:, None] + centers[station_local][:, None]


def rmse_block(
    pred_ug: np.ndarray,
    target_ug: np.ndarray,
    mask: np.ndarray,
    horizons: list[int],
) -> dict[str, float]:
    """Return per-horizon RMSE in µg/m³ rounded to 2 dp, keyed '6h'…'48h'."""
    metrics = compute_metrics(pred_ug, target_ug, mask, horizons=horizons)
    return {f"{h}h": round(metrics[f"rmse_{h}h"], 2) for h in horizons}
