"""
[TODO: NSC Disclaimer — see booklet page 44]

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.
"""

import json
import logging
from datetime import date as _date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr  # noqa: F401 — imported for HeteroData compat in graph_builder
from torch_geometric.data import HeteroData

from src.data.graph_builder import (
    build_graph,
    build_type_a_edges,
    build_type_b_edges,
    build_type_c_edges,
)
from src.data.scrapers.openaq import curate_training_stations, discover_locations

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_SPLIT_BOUNDS: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "train": (
        pd.Timestamp("2022-01-01", tz="UTC"),
        pd.Timestamp("2023-12-31 23:00", tz="UTC"),
    ),
    "val": (
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2024-12-31 23:00", tz="UTC"),
    ),
    "test": (
        pd.Timestamp("2025-01-01", tz="UTC"),
        pd.Timestamp("2025-12-31 23:00", tz="UTC"),
    ),
}

_FULL_INDEX: pd.DatetimeIndex = pd.date_range("2022-01-01", "2025-12-31 23:00", freq="1h", tz="UTC")

EMPTY_HOTSPOT_DF: pd.DataFrame = pd.DataFrame(
    {"centroid_lat": [], "centroid_lon": [], "total_frp": []}
)

_FEATURE_COLS: list[str] = ["pm25_scaled", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]


# ---------------------------------------------------------------------------
# Existing public API — keep as is
# ---------------------------------------------------------------------------


def build_station_metadata(
    output_path: Path = Path("data/processed/stations_metadata.parquet"),
    use_cache: bool = True,
) -> pd.DataFrame:
    """Discover + curate + cache the core station set.

    Args:
        output_path: Destination parquet file for the curated station table.
            Defaults to ``data/processed/stations_metadata.parquet``.
        use_cache: When True and ``output_path`` already exists, load and
            return the cached file without hitting the API.

    Returns:
        DataFrame with columns: location_id, name, lat, lon, provider,
        datetime_first, datetime_last, sensor_id_pm25.
    """
    if use_cache and output_path.exists():
        logger.info("Loading station metadata from cache: %s", output_path)
        return pd.read_parquet(output_path)

    logger.info("Discovering locations via OpenAQ v3 …")
    df_locations = discover_locations()

    logger.info("Curating training stations …")
    df = curate_training_stations(df_locations)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Saved station metadata to %s (%d stations)", output_path, len(df))

    return df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_scalers(path: Path) -> dict[int, dict[str, float]]:
    """Load scalers.json, converting string keys to int.

    Args:
        path: Path to data/processed/scalers.json.

    Returns:
        Dict mapping station_id (int) to {"center_": float, "scale_": float}.
    """
    with open(path) as fh:
        raw: dict[str, dict[str, float]] = json.load(fh)
    return {int(k): v for k, v in raw.items()}


def _load_hotspots(hotspots_path: Path) -> dict[_date, pd.DataFrame]:
    """Load and index hotspot parquet by date.

    Args:
        hotspots_path: Path to data/processed/hotspots.parquet.

    Returns:
        Dict keyed by Python date. Each value is a DataFrame with columns
        centroid_lat, centroid_lon, total_frp (and possibly others).
        Dates with no hotspots are absent from the dict.
    """
    df = pd.read_parquet(hotspots_path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return {d: grp.reset_index(drop=True) for d, grp in df.groupby("date")}


def _load_dataset_wide(
    dataset_path: Path,
    station_ids: np.ndarray,
    full_idx: pd.DatetimeIndex,
) -> dict[str, np.ndarray]:
    """Load and pivot dataset parquet to wide (T_full, N) arrays per feature.

    Args:
        dataset_path: Path to data/processed/dataset.parquet.
        station_ids: Sorted array of station IDs to include, shape (N,).
        full_idx: Full hourly datetime index (35064 steps), UTC.

    Returns:
        Dict mapping feature name to (T_full, N) float32/bool ndarray.
        Keys: 'pm25_raw', 'pm25_scaled', 'hour_sin', 'hour_cos',
              'doy_sin', 'doy_cos', 'mask_in_loss', 'exclude_from_training'.
    """
    df = pd.read_parquet(dataset_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[df["station_id"].isin(station_ids)].copy()

    bool_features = {"mask_in_loss", "exclude_from_training"}
    cyclic_features = {"hour_sin", "hour_cos", "doy_sin", "doy_cos"}
    all_features = [
        "pm25_raw",
        "pm25_scaled",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
        "mask_in_loss",
        "exclude_from_training",
    ]

    # Check if pm25_raw exists in the parquet
    has_pm25_raw = "pm25_raw" in df.columns
    if not has_pm25_raw:
        logger.warning(
            "pm25_raw not found in dataset parquet; will attempt inverse transform from pm25_scaled"
        )

    arrays: dict[str, np.ndarray] = {}

    for feat in all_features:
        if feat == "pm25_raw" and not has_pm25_raw:
            # Defer; computed after pm25_scaled is loaded
            continue

        if feat not in df.columns:
            logger.warning("Feature %r not found in dataset parquet; skipping", feat)
            continue

        pivot = df.pivot_table(
            index="timestamp",
            columns="station_id",
            values=feat,
            aggfunc="first",
        ).reindex(index=full_idx, columns=station_ids)

        if feat in bool_features:
            arrays[feat] = pivot.fillna(True).values.astype(bool)
        elif feat in cyclic_features:
            arrays[feat] = pivot.fillna(0.0).values.astype(np.float32)
        else:
            # pm25_raw, pm25_scaled — preserve NaN
            arrays[feat] = pivot.values.astype(np.float32)

    # Inverse-transform pm25_raw if it was absent
    if not has_pm25_raw and "pm25_scaled" in arrays:
        scalers_path = dataset_path.parent / "scalers.json"
        scalers = _load_scalers(scalers_path)
        pm25_raw = np.full_like(arrays["pm25_scaled"], np.nan, dtype=np.float32)
        for col_i, sid in enumerate(station_ids):
            params = scalers.get(int(sid))
            if params is None:
                logger.warning("No scaler found for station_id=%d; pm25_raw stays NaN", sid)
                continue
            pm25_raw[:, col_i] = (
                arrays["pm25_scaled"][:, col_i] * params["scale_"] + params["center_"]
            ).astype(np.float32)
        arrays["pm25_raw"] = pm25_raw

    return arrays


def _build_anchor_index(
    split: str,
    window_in: int,
    horizons: tuple[int, ...],
    full_idx: pd.DatetimeIndex,
    mask_in_loss: np.ndarray,
    exclude: np.ndarray,
    pm25_raw: np.ndarray,
) -> np.ndarray:
    """Compute valid window anchor offsets into full_idx.

    A window anchor t is valid iff:
      1. t - (window_in - 1) >= split_start  (full input fits in split)
      2. t + max(horizons) <= split_end        (all targets inside split)
      3. At least one (station, horizon) pair has a usable target at t:
         mask_in_loss[t+h, i] == False AND exclude[t+h, i] == False
         AND pm25_raw[t+h, i] is finite.

    Args:
        split: One of 'train', 'val', 'test'.
        window_in: Input window length in hours.
        horizons: Sorted horizon tuple (e.g. (6, 12, 24, 48)).
        full_idx: Full hourly datetime index.
        mask_in_loss: (T_full, N) bool array.
        exclude: (T_full, N) bool array.
        pm25_raw: (T_full, N) float32 array.

    Returns:
        1-D int64 ndarray of valid anchor offsets into full_idx.
    """
    split_start, split_end = _SPLIT_BOUNDS[split]

    # Map timestamps to integer offsets (searchsorted is O(log T))
    i_start = int(full_idx.searchsorted(split_start, side="left"))
    i_end = int(full_idx.searchsorted(split_end, side="right")) - 1

    max_h = int(max(horizons))
    # Earliest valid anchor: need window_in-1 steps of history inside the split
    anchor_lo = i_start + (window_in - 1)
    # Latest valid anchor: all targets must fall within split
    anchor_hi = i_end - max_h

    if anchor_lo > anchor_hi:
        return np.empty(0, dtype=np.int64)

    candidates = np.arange(anchor_lo, anchor_hi + 1, dtype=np.int64)

    # Build validity mask: at least one (station, horizon) pair is usable.
    # target_offsets shape: (n_candidates, n_horizons)
    target_offsets = candidates[:, None] + np.array(horizons, dtype=np.int64)[None, :]

    # Advanced indexing yields shape (n_candidates, n_horizons, N)
    m_loss = mask_in_loss[target_offsets, :]
    excl = exclude[target_offsets, :]
    finite = np.isfinite(pm25_raw[target_offsets, :])

    usable = (~m_loss) & (~excl) & finite
    # A candidate is valid if ANY (horizon, station) pair is usable
    valid_mask = usable.any(axis=(1, 2))

    return candidates[valid_mask]


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------


class PM25GraphDataset(torch.utils.data.Dataset):
    """Sliding-window PM2.5 graph dataset for STGNN training.

    Each sample is a HeteroData containing a T_in-step input window, multi-horizon
    targets, a validity mask, and the graph structure for the reference timestep.

    Attributes:
        split: One of 'train', 'val', 'test'.
        window_in: Number of input hours per sample.
        horizons: Forecast horizons (hours ahead).
        scalers: Per-station RobustScaler params (for metric reporting).
    """

    def __init__(
        self,
        dataset_path: Path,
        hotspots_path: Path,
        metadata_path: Path,
        split: str,
        window_in: int = 24,
        horizons: list[int] | None = None,
        graph_config: dict[str, Any] | None = None,
        exclude_stations: list[int] | None = None,
        scalers_path: Path | None = None,
    ) -> None:
        """Initialise dataset for a given split.

        Args:
            dataset_path: Path to data/processed/dataset.parquet.
            hotspots_path: Path to data/processed/hotspots.parquet.
            metadata_path: Path to data/processed/stations_metadata.parquet.
            split: One of 'train', 'val', 'test'.
            window_in: Input window length in hours.
            horizons: Forecast horizons (hours ahead). Defaults to [6, 12, 24, 48].
            graph_config: Override keys for graph construction. Unset keys use defaults.
            exclude_stations: Station IDs to drop from the dataset.
            scalers_path: Path to scalers.json. Defaults to dataset_path.parent/scalers.json.
        """
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be 'train', 'val', or 'test'; got {split!r}")

        self.split = split
        self.window_in = window_in
        self.horizons: tuple[int, ...] = tuple(sorted(horizons or [6, 12, 24, 48]))

        _exclude_set: set[int] = set(exclude_stations or [])

        # Graph config — inject wind_mode default if absent
        self.graph_config: dict[str, Any] = dict(graph_config or {})
        if "wind_mode" not in self.graph_config:
            self.graph_config["wind_mode"] = "constant_ne"
        self._wind_mode: str = self.graph_config["wind_mode"]

        # Load station metadata: rename location_id -> station_id
        meta = pd.read_parquet(metadata_path)
        meta = meta.rename(columns={"location_id": "station_id"})
        meta = meta.sort_values("station_id").reset_index(drop=True)

        if _exclude_set:
            unknown = _exclude_set - set(meta["station_id"].tolist())
            if unknown:
                logger.warning(
                    "exclude_stations contains IDs not found in metadata: %s", sorted(unknown)
                )
            meta = meta[~meta["station_id"].isin(_exclude_set)].reset_index(drop=True)

        if meta.empty:
            raise RuntimeError("No stations remain after applying exclude_stations filter.")

        self._station_ids: np.ndarray = meta["station_id"].values.astype(np.int64)

        # Keep a clean stations DataFrame with station_id, lat, lon (and name if present)
        keep_cols = ["station_id", "lat", "lon"]
        if "name" in meta.columns:
            keep_cols.append("name")
        self._stations_static: pd.DataFrame = meta[keep_cols].copy()

        # Load wide feature arrays
        wide = _load_dataset_wide(dataset_path, self._station_ids, _FULL_INDEX)
        self._pm25_raw: np.ndarray = wide["pm25_raw"]
        self._pm25_scaled: np.ndarray = wide["pm25_scaled"]
        self._hour_sin: np.ndarray = wide["hour_sin"]
        self._hour_cos: np.ndarray = wide["hour_cos"]
        self._doy_sin: np.ndarray = wide["doy_sin"]
        self._doy_cos: np.ndarray = wide["doy_cos"]
        self._mask_in_loss: np.ndarray = wide["mask_in_loss"]
        self._exclude: np.ndarray = wide["exclude_from_training"]

        self._timestamps: pd.DatetimeIndex = _FULL_INDEX

        # Scalers
        _scalers_path = scalers_path or (dataset_path.parent / "scalers.json")
        self.scalers: dict[int, dict[str, float]] = _load_scalers(_scalers_path)

        # Hotspots
        self._hotspots_by_date: dict[_date, pd.DataFrame] = _load_hotspots(hotspots_path)

        # Anchor indices
        self._anchor_indices: np.ndarray = _build_anchor_index(
            split=self.split,
            window_in=self.window_in,
            horizons=self.horizons,
            full_idx=self._timestamps,
            mask_in_loss=self._mask_in_loss,
            exclude=self._exclude,
            pm25_raw=self._pm25_raw,
        )

        if len(self._anchor_indices) == 0:
            raise RuntimeError(
                f"No valid anchor indices found for split={split!r}. "
                "Check data coverage and gap masks."
            )

        # Precompute static edges
        self._precompute_static_edges()

        logger.info(
            "PM25GraphDataset split=%s N=%d T_in=%d horizons=%s samples=%d wind_mode=%s",
            self.split,
            len(self._station_ids),
            self.window_in,
            self.horizons,
            len(self._anchor_indices),
            self._wind_mode,
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._anchor_indices)

    def __getitem__(self, idx: int) -> HeteroData:
        """Return a single sample as HeteroData.

        Args:
            idx: Sample index within this split.

        Returns:
            HeteroData with node types 'station', 'hotspot', 'meta' and
            edge types ('station','type_a','station'),
            ('station','type_b','station'), ('hotspot','type_c','station').
        """
        t_anchor_idx = int(self._anchor_indices[idx])
        t_start_idx = t_anchor_idx - (self.window_in - 1)

        # --- Build x: (N, T_in, 5) float32 ---
        # Slice wide arrays over [t_start_idx : t_anchor_idx+1] → (T_in, N)
        sl = slice(t_start_idx, t_anchor_idx + 1)
        x_features = np.stack(
            [
                self._pm25_scaled[sl],
                self._hour_sin[sl],
                self._hour_cos[sl],
                self._doy_sin[sl],
                self._doy_cos[sl],
            ],
            axis=-1,
        )  # (T_in, N, 5)
        x_ntf = np.nan_to_num(x_features.transpose(1, 0, 2), nan=0.0).astype(
            np.float32
        )  # (N, T_in, 5)

        # --- Build y and mask: (N, H) ---
        # Use pm25_scaled (normalized) as training targets for loss stability.
        # Raw µg/m³ values (10-500) produce MSE losses of order 10^3-10^5, causing
        # gradient explosion and NaN weights after ~9 epochs. The mask validity check
        # still uses pm25_raw finiteness (same NaN pattern, scaling preserves NaN).
        y_cols: list[np.ndarray] = []
        m_cols: list[np.ndarray] = []
        for h_offset in self.horizons:
            t_h = t_anchor_idx + h_offset
            y_col = self._pm25_scaled[t_h, :]  # (N,) — normalized scale
            raw_col = self._pm25_raw[t_h, :]  # (N,) — used only for validity check
            m_col = (~self._mask_in_loss[t_h, :]) & (~self._exclude[t_h, :]) & np.isfinite(raw_col)
            y_cols.append(y_col)
            m_cols.append(m_col)

        mask_nh = np.stack(m_cols, axis=1)  # (N, H) bool
        y_nh = np.where(mask_nh, np.stack(y_cols, axis=1), 0.0).astype(np.float32)  # (N, H)

        # --- Hotspot lookup ---
        ref_date: _date = self._timestamps[t_anchor_idx].date()
        df_h = self._hotspots_by_date.get(ref_date, EMPTY_HOTSPOT_DF)

        # --- Assemble graph ---
        if self._wind_mode == "constant_ne":
            data = self._assemble_graph(t_anchor_idx, df_h)
        else:
            data = self._build_graph_full(t_anchor_idx, df_h)

        # --- Attach tensors ---
        data["station"].x = torch.from_numpy(x_ntf).contiguous().float()
        data["station"].y = torch.from_numpy(y_nh).float()
        data["station"].mask = torch.from_numpy(mask_nh.astype(bool))

        data["meta"].t_anchor_idx = torch.tensor(t_anchor_idx, dtype=torch.long)
        data["meta"].station_ids = torch.from_numpy(self._station_ids)

        return data

    # ------------------------------------------------------------------
    # Graph assembly helpers
    # ------------------------------------------------------------------

    def _precompute_static_edges(self) -> None:
        """Precompute Type A and Type B edges that do not change across timesteps."""
        self._type_a_edge_index, self._type_a_edge_attr = build_type_a_edges(
            self._stations_static, self.graph_config
        )
        if self._wind_mode == "constant_ne":
            self._type_b_edge_index, self._type_b_edge_attr = build_type_b_edges(
                self._stations_static, wind_field=None, config=self.graph_config
            )
        else:
            # Dynamic wind mode: edges will be rebuilt per sample in _build_graph_full
            self._type_b_edge_index = torch.zeros((2, 0), dtype=torch.long)
            self._type_b_edge_attr = torch.zeros((0, 3), dtype=torch.float32)

    def _assemble_graph(self, t_anchor_idx: int, df_h: pd.DataFrame) -> HeteroData:
        """Assemble HeteroData using precomputed static edges.

        Args:
            t_anchor_idx: Anchor timestep offset into full_idx.
            df_h: Hotspot DataFrame for the reference date.

        Returns:
            HeteroData with static Type A + B edges and fresh Type C edges.
        """
        data = HeteroData()

        data["station", "type_a", "station"].edge_index = self._type_a_edge_index
        data["station", "type_a", "station"].edge_attr = self._type_a_edge_attr

        data["station", "type_b", "station"].edge_index = self._type_b_edge_index
        data["station", "type_b", "station"].edge_attr = self._type_b_edge_attr

        ei_c, ea_c = build_type_c_edges(
            self._stations_static, df_h, wind_field=None, config=self.graph_config
        )
        data["hotspot", "type_c", "station"].edge_index = ei_c
        data["hotspot", "type_c", "station"].edge_attr = ea_c

        if len(df_h) > 0:
            hx = torch.from_numpy(
                df_h[["total_frp", "centroid_lat", "centroid_lon"]].values.astype(np.float32)
            )
        else:
            hx = torch.zeros((0, 3), dtype=torch.float32)
        data["hotspot"].x = hx

        # Placeholder station features; overwritten by caller
        data["station"].x = torch.zeros((len(self._station_ids), 1), dtype=torch.float32)

        return data

    def _build_stations_now(self, t_anchor_idx: int) -> pd.DataFrame:
        """Copy static station table and attach per-timestep feature columns.

        Args:
            t_anchor_idx: Anchor timestep offset into full_idx.

        Returns:
            DataFrame with station_id, lat, lon, and feature columns.
        """
        df = self._stations_static.copy()
        for feat, arr in (
            ("pm25_scaled", self._pm25_scaled),
            ("hour_sin", self._hour_sin),
            ("hour_cos", self._hour_cos),
            ("doy_sin", self._doy_sin),
            ("doy_cos", self._doy_cos),
        ):
            df[feat] = np.nan_to_num(arr[t_anchor_idx, :], nan=0.0)
        return df

    def _build_graph_full(self, t_anchor_idx: int, df_h: pd.DataFrame) -> HeteroData:
        """Build a fresh graph via build_graph() for non-constant wind modes.

        Args:
            t_anchor_idx: Anchor timestep offset into full_idx.
            df_h: Hotspot DataFrame for the reference date.

        Returns:
            HeteroData from build_graph().
        """
        df_stations_now = self._build_stations_now(t_anchor_idx)
        data = build_graph(
            df_stations=df_stations_now,
            df_hotspots=df_h,
            wind_field=None,
            config=self.graph_config,
        )
        return data
