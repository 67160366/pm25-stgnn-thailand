# [TODO: NSC Disclaimer — see booklet page 44]

"""Streamlit dashboard for PM2.5 STGNN — NSC 2026 Category 14 demo.

Usage:
    uv run streamlit run app/streamlit_app.py

Tabs:
    1. Live Forecast  — current PM2.5 predictions for all 18 stations
    2. Haze History   — heatmap and time-series browser
    3. Source Attribution — GB-IG occlusion + IG feature importance
    4. About          — project description and NSC Disclaimer
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from src.data.loader import PM25GraphDataset
from src.explain.attribution import station_source_report
from src.viz.maps import attribution_bar_map, feature_importance_bar, station_map
from src.viz.timeseries import forecast_vs_actual, multi_station_heatmap

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data" / "processed"
_CONFIGS_DIR = _PROJECT_ROOT / "configs"
_HORIZONS = [6, 12, 24, 48]
_HORIZON_LABELS = {6: "6h", 12: "12h", 24: "24h", 48: "48h"}

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading dataset…")
def _load_dataset(split: str = "val") -> PM25GraphDataset:
    return PM25GraphDataset(
        split=split,
        dataset_path=_DATA_DIR / "dataset.parquet",
        hotspots_path=_DATA_DIR / "hotspots.parquet",
        metadata_path=_DATA_DIR / "stations_metadata.parquet",
        scalers_path=_DATA_DIR / "scalers.json",
        window_in=24,
        horizons=_HORIZONS,
        graph_config={"wind_mode": "from_field"},
    )


@st.cache_resource(show_spinner="Loading scalers…")
def _load_scalers() -> dict:
    with open(_DATA_DIR / "scalers.json") as f:
        return json.load(f)


@st.cache_resource(show_spinner="Loading station metadata…")
def _load_metadata() -> pd.DataFrame:
    return pd.read_parquet(_DATA_DIR / "stations_metadata.parquet")


@st.cache_resource(show_spinner="Loading model…")
def _load_model(checkpoint_path: Path) -> torch.nn.Module | None:
    """Load the best MTGNN checkpoint if it exists, else return None."""
    if not checkpoint_path.exists():
        return None
    try:
        with initialize_config_dir(config_dir=str(_CONFIGS_DIR), version_base="1.3"):
            cfg = compose(config_name="config", overrides=["model=mtgnn"])
        model = instantiate(cfg.model, n_stations=18, horizons=_HORIZONS)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        # Checkpoint is a dict with "model_state_dict" key (see Trainer._save_checkpoint)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as exc:  # broad catch — graceful degradation in Streamlit UI
        logger.warning("Could not load checkpoint %s: %s", checkpoint_path, exc)
        return None


def _denorm_pm25(scaled: float, scalers: dict) -> float:
    """Convert normalized pm25_scaled back to µg/m³."""
    pm25_scaler = scalers.get("pm25", {})
    mean = pm25_scaler.get("mean", 0.0)
    std = pm25_scaler.get("std", 1.0)
    return float(scaled * std + mean)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PM2.5 STGNN — Northern Thailand",
    page_icon=":foggy:",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("PM2.5 STGNN")
    st.caption("NSC 2026 | Category 14")

    ckpt_dir = st.text_input(
        "Checkpoint path",
        value=str(_PROJECT_ROOT / "checkpoints" / "best_model.pt"),
        help="Path to best_model.pt saved by scripts/03_train.py",
    )

    horizon_label = st.selectbox(
        "Forecast horizon",
        options=list(_HORIZON_LABELS.values()),
        index=1,
    )
    horizon_h = {v: k for k, v in _HORIZON_LABELS.items()}[horizon_label]
    horizon_idx = _HORIZONS.index(horizon_h)

    st.divider()
    st.caption("Data: Air4Thai / FIRMS / ERA5")

# ---------------------------------------------------------------------------
# Load shared resources
# ---------------------------------------------------------------------------
meta = _load_metadata()
scalers = _load_scalers()
model = _load_model(Path(ckpt_dir))  # ckpt_dir is now a full file path

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_forecast, tab_history, tab_attr, tab_about = st.tabs(
    ["Live Forecast", "Haze History", "Source Attribution", "About"]
)

# ── Tab 1: Live Forecast ─────────────────────────────────────────────────────
with tab_forecast:
    st.header("Live PM2.5 Forecast")

    if model is None:
        st.warning(
            f"No trained checkpoint found at `{ckpt_dir}`. "
            "Train the model first:\n\n"
            "```\nuv run python scripts/03_train.py model=mtgnn\n```"
        )
    else:
        ds = _load_dataset("val")
        sample_idx = st.slider("Sample index (validation set)", 0, len(ds) - 1, 0)
        sample = ds[sample_idx]

        with torch.no_grad():
            pred = model(sample)  # (N, H)

        pred_np = pred.detach().cpu().numpy()
        station_ids = ds._station_ids  # type: ignore[attr-defined]

        pm25_at_horizon = {
            sid: _denorm_pm25(float(pred_np[i, horizon_idx]), scalers)
            for i, sid in enumerate(station_ids)
        }
        pm25_series = pd.Series(pm25_at_horizon)

        col_map, col_table = st.columns([2, 1])
        with col_map:
            fig_map = station_map(
                meta,
                pm25_values=pm25_series,
                title=f"Predicted PM2.5 at +{horizon_h}h",
            )
            st.plotly_chart(fig_map, use_container_width=True)

        with col_table:
            st.subheader(f"Station predictions (+{horizon_h}h)")
            tbl = meta[["station_id", "name", "province"]].copy()
            tbl["PM2.5 (µg/m³)"] = tbl["station_id"].map(
                lambda sid: f"{pm25_at_horizon.get(sid, float('nan')):.1f}"
            )
            st.dataframe(tbl[["name", "province", "PM2.5 (µg/m³)"]], use_container_width=True)

# ── Tab 2: Haze History ───────────────────────────────────────────────────────
with tab_history:
    st.header("Haze History")

    ds_val = _load_dataset("val")
    dataset_df = pd.read_parquet(_DATA_DIR / "dataset.parquet")

    has_name_col = "name" in meta.columns
    station_options = meta["name"].tolist() if has_name_col else meta["station_id"].tolist()
    selected_name = st.selectbox("Station", options=station_options)
    selected_row = meta[meta["name"] == selected_name].iloc[0] if has_name_col else meta.iloc[0]
    selected_sid = selected_row["station_id"]

    st.subheader("Multi-station PM2.5 Heatmap (last 7 days sample)")
    cutoff = dataset_df["timestamp"].max() - pd.Timedelta("7d")
    sample_df = dataset_df[dataset_df["timestamp"] >= cutoff]
    if len(sample_df) > 0:
        fig_heat = multi_station_heatmap(sample_df, value_col="pm25")
        st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader(f"Time series — {selected_name}")
    station_df = dataset_df[dataset_df["station_id"] == selected_sid].sort_values("timestamp")
    if len(station_df) > 0:
        recent = station_df.tail(168)  # last 7 days hourly
        fig_ts = forecast_vs_actual(
            timestamps=recent["timestamp"].tolist(),
            actual=recent["pm25"].tolist(),
            forecasts={},
            station_name=selected_name,
        )
        st.plotly_chart(fig_ts, use_container_width=True)

# ── Tab 3: Source Attribution ─────────────────────────────────────────────────
with tab_attr:
    st.header("Source Attribution")

    if model is None:
        st.warning("Train the model first to enable source attribution.")
    else:
        ds_attr = _load_dataset("val")
        if "name" in meta.columns:
            station_names = meta["name"].tolist()
        else:
            station_names = [str(i) for i in range(ds_attr.n_stations)]

        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            attr_station_name = st.selectbox(
                "Station to explain", options=station_names, key="attr_st"
            )
        with col_sel2:
            attr_sample_idx = st.slider(
                "Sample index", 0, min(len(ds_attr) - 1, 500), 0, key="attr_si"
            )

        attr_station_idx = station_names.index(attr_station_name)
        attr_sample = ds_attr[attr_sample_idx]

        with st.spinner("Running Integrated Gradients + Occlusion Attribution…"):
            report = station_source_report(
                model=model,
                data=attr_sample,
                station_idx=attr_station_idx,
                horizon_idx=horizon_idx,
                station_name=attr_station_name,
                n_ig_steps=30,
                device="cpu",
            )

        col_ig, col_occ = st.columns(2)

        with col_ig:
            st.subheader("Feature Importance (Integrated Gradients)")
            fig_ig = feature_importance_bar(report["ig_feature_importance"])
            st.plotly_chart(fig_ig, use_container_width=True)

        with col_occ:
            st.subheader("Country Source Attribution (Occlusion)")
            fig_occ = attribution_bar_map(
                stations_meta=meta,
                attribution_scores=report["country_attribution"],
                station_id=str(attr_station_idx),
            )
            st.plotly_chart(fig_occ, use_container_width=True)

        with st.expander("Raw attribution data"):
            st.json(report)

# ── Tab 4: About ──────────────────────────────────────────────────────────────
with tab_about:
    st.header("About this project")
    st.markdown("""
        ## Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand

        **NSC 2026 — National Software Contest (28th edition)**
        Category 14: โปรแกรมเพื่องานการพัฒนาด้านวิทยาศาสตร์และเทคโนโลยี (University level)

        ### What this system does
        - Forecasts hourly PM2.5 concentrations at 18 Air4Thai stations in Northern Thailand
          at horizons of +6h, +12h, +24h, and +48h
        - Uses a heterogeneous Spatio-Temporal Graph Neural Network (STGNN) with:
          - **Type A edges**: static spatial proximity between stations
          - **Type B edges**: wind-aware dynamic edges (ERA5 u/v wind fields)
          - **Type C edges**: FIRMS fire hotspot bipartite connections
        - Explains predictions via **Integrated Gradients** (feature importance)
          and **Occlusion Attribution** (country-level source scores)

        ### Three novelties (NSC submission)
        1. Wind-aware dynamic adjacency for transboundary haze transport
        2. FIRMS hotspot clusters as first-class heterogeneous graph nodes
        3. Graph-based Integrated Gradients for country source attribution

        ### Data sources
        | Source | Usage |
        |---|---|
        | Air4Thai (PCD) | Station PM2.5 measurements |
        | NASA FIRMS VIIRS | Fire radiative power hotspots |
        | ERA5 (ECMWF) | Wind (u10/v10), temperature (t2m), dewpoint (d2m), BLH |

        ---

        ### [TODO: NSC Disclaimer — see booklet page 44]

        *This software was developed as an entry to the National Software Contest (NSC) 2026.
        The disclaimer text from the NSC booklet page 44 will be inserted here before submission.*

        ---
        Repository: https://github.com/67160366/pm25-stgnn-thailand
        """)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "PM2.5 STGNN Thailand | NSC 2026 Category 14 | " "[TODO: NSC Disclaimer — see booklet page 44]"
)
