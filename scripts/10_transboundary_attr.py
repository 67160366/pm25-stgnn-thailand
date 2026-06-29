# [TODO: NSC Disclaimer - see booklet page 44]

"""Transboundary attribution hunt: does the model attribute PM2.5 to foreign fires?

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.

The motivating use case is Myanmar/Laos haze, yet the only attribution case study so far
(March 2024, Chiang Mai) was Thailand-100% - because Chiang Mai is interior and few foreign
hotspots connect to it (critical review finding #6). This script tests the model's
cross-border attribution capability *where it should appear*: at a border station, on the
days when foreign (Myanmar/Laos) fires near that station are strongest.

Procedure:
  1. Rank dates in the chosen split by foreign (Myanmar+Laos) FRP.
  2. For each candidate date, take the peak-PM2.5 anchor at the target border station and
     read which hotspots actually CONNECT to it (the type_c edges the model sees), summing
     connected FRP per country.
  3. Rank candidate events by connected-foreign FRP; for the top events run occlusion
     country attribution + IG (reusing src/explain) and compare the model's foreign
     attribution to the connected-foreign FRP fraction it was given.

Reading: if foreign attribution stays ~0 even when foreign fires dominate near the border
station, the model does NOT capture transboundary transport (honest negative). If foreign
attribution tracks the connected-foreign fraction, it is at least responsive to it.

Default station is Mae Hong Son (on the Myanmar border). Default split is `test` (2025),
which is held out for the pitch checkpoint (also addresses finding #8).

Usage:
    uv run python scripts/10_transboundary_attr.py
    uv run python scripts/10_transboundary_attr.py split=train station_id=225648
    uv run python scripts/10_transboundary_attr.py station_id=225626 output=outputs/tb_maesot.json
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from src.data.loader import _SPLIT_BOUNDS, PM25GraphDataset
from src.explain.attribution import station_source_report

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIGS_DIR = _PROJECT_ROOT / "configs"
_MTGNN_CKPT = "checkpoints/mtgnn/best_model.pt"
_LOCAL_KEYS = {"output", "split", "station_id", "top_k", "n_candidates", "horizon_idx", "ckpt"}
_FOREIGN = ("Myanmar", "Laos")
_DEFAULT_STATION_ID = 225648  # Mae Hong Son - deepest on the Myanmar border


def _split_cli_args(argv: list[str]) -> tuple[dict[str, str], list[str]]:
    """Separate script-local keys from Hydra overrides."""
    local: dict[str, str] = {}
    overrides: list[str] = []
    for arg in argv:
        if "=" not in arg:
            continue
        key, _, val = arg.partition("=")
        if key in _LOCAL_KEYS:
            local[key] = val
        else:
            overrides.append(arg)
    return local, overrides


def _load_mtgnn(
    n_stations: int, horizons: list[int], overrides: list[str], ckpt: str
) -> torch.nn.Module:
    """Instantiate MTGNN from Hydra config and load the trained checkpoint (CPU)."""
    with initialize_config_dir(config_dir=str(_CONFIGS_DIR), version_base="1.3"):
        mcfg = compose(config_name="config", overrides=["model=mtgnn", *overrides])
    model = instantiate(mcfg.model, n_stations=n_stations, horizons=horizons)
    state = torch.load(_PROJECT_ROOT / ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) else state)
    return model.eval()


def _candidate_foreign_dates(hotspots_path: Path, split: str, top_n: int) -> list[_date]:
    """Dates in the split ranked by total Myanmar+Laos FRP (descending)."""
    start_d = _SPLIT_BOUNDS[split][0].date()
    end_d = _SPLIT_BOUNDS[split][1].date()
    h = pd.read_parquet(hotspots_path)
    h["d"] = pd.to_datetime(h["date"]).dt.date
    h = h[(h["d"] >= start_d) & (h["d"] <= end_d) & (h["country"].isin(_FOREIGN))]
    if h.empty:
        return []
    by_date = h.groupby("d")["total_frp"].sum().sort_values(ascending=False)
    return list(by_date.head(top_n).index)


def _peak_anchor_for_date(
    ds: PM25GraphDataset, target_date: _date, station_idx: int
) -> tuple[int, float] | None:
    """Return (anchor position, peak PM2.5) for the highest-PM2.5 anchor on a date."""
    anchor_dates = pd.DatetimeIndex(ds._timestamps[ds._anchor_indices]).date
    on_date = np.where(anchor_dates == target_date)[0]
    if on_date.size == 0:
        return None
    pm = ds._pm25_raw[ds._anchor_indices[on_date], station_idx]
    finite = np.isfinite(pm)
    if not finite.any():
        return None
    best_pos = int(on_date[finite][int(np.argmax(pm[finite]))])
    return best_pos, float(np.max(pm[finite]))


def _connected_frp_by_country(sample: object, station_idx: int) -> dict[str, float]:
    """Sum FRP per country over hotspots whose type_c edge reaches the target station."""
    edge_index = sample["hotspot", "type_c", "station"].edge_index
    countries = list(getattr(sample["hotspot"], "country", []))
    hx = sample["hotspot"].x
    if edge_index.numel() == 0 or hx.numel() == 0 or not countries:
        return {}
    reaches = edge_index[1] == station_idx
    acc: dict[str, float] = {}
    for h_idx in sorted(set(edge_index[0][reaches].tolist())):
        acc[countries[h_idx]] = acc.get(countries[h_idx], 0.0) + float(hx[h_idx, 0])
    return acc


def _summarize_event(
    model: torch.nn.Module,
    sample: object,
    station_idx: int,
    station_name: str,
    horizon_idx: int,
    target_date: _date,
    peak_pm25: float,
) -> dict[str, object]:
    """Attribute one event and compare model foreign attribution to connected foreign FRP."""
    conn = _connected_frp_by_country(sample, station_idx)
    foreign_frp = sum(v for k, v in conn.items() if k in _FOREIGN)
    thai_frp = conn.get("Thailand", 0.0)
    total_frp = foreign_frp + thai_frp
    report = station_source_report(
        model, sample, station_idx, horizon_idx, station_name=station_name, device="cpu"
    )
    attr = report["country_attribution"]
    foreign_attr = sum(v for k, v in attr.items() if k in _FOREIGN)
    top_ig = dict(sorted(report["ig_feature_importance"].items(), key=lambda kv: -kv[1])[:3])
    return {
        "date": str(target_date),
        "peak_pm25_ug_m3": round(peak_pm25, 1),
        "connected_frp_by_country": {k: round(v, 1) for k, v in conn.items()},
        "connected_foreign_fraction": round(foreign_frp / total_frp, 3) if total_frp > 0 else 0.0,
        "country_attribution": {k: round(v, 3) for k, v in attr.items()},
        "foreign_attribution": round(foreign_attr, 3),
        "top_ig_features": {k: round(v, 4) for k, v in top_ig.items()},
    }


def main() -> None:
    """Hunt transboundary events at a border station and report cross-border attribution."""
    local_args, overrides = _split_cli_args(sys.argv[1:])
    with initialize_config_dir(config_dir=str(_CONFIGS_DIR), version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)

    split = local_args.get("split", "test")
    station_id = int(local_args.get("station_id", _DEFAULT_STATION_ID))
    n_candidates = int(local_args.get("n_candidates", 15))
    top_k = int(local_args.get("top_k", 5))
    horizon_idx = int(local_args.get("horizon_idx", 2))
    ckpt = local_args.get("ckpt", _MTGNN_CKPT)
    output_path = Path(local_args.get("output", f"outputs/transboundary_attr_{split}.json"))
    horizons = list(cfg.data.horizons)
    wind_mode = getattr(cfg.data, "wind_mode", "constant_ne")

    ds = PM25GraphDataset(
        dataset_path=Path(cfg.data.dataset_path),
        hotspots_path=Path(cfg.data.hotspots_path),
        metadata_path=Path(cfg.data.metadata_path),
        scalers_path=Path(cfg.data.scalers_path),
        split=split,
        window_in=cfg.data.window_in,
        horizons=horizons,
        graph_config={"wind_mode": wind_mode},
    )
    if station_id not in ds._station_ids:
        raise SystemExit(f"station_id {station_id} not in dataset stations {list(ds._station_ids)}")
    station_idx = int(np.where(ds._station_ids == station_id)[0][0])
    meta = pd.read_parquet(cfg.data.metadata_path).rename(columns={"location_id": "station_id"})
    name_row = meta.loc[meta["station_id"] == station_id, "name"]
    station_name = str(name_row.iloc[0]) if len(name_row) else f"station_{station_id}"
    logger.info("Target: %s (id=%d idx=%d) split=%s", station_name, station_id, station_idx, split)

    cand_dates = _candidate_foreign_dates(Path(cfg.data.hotspots_path), split, n_candidates)
    logger.info("Foreign-FRP candidate dates in %s: %d", split, len(cand_dates))

    # Build peak sample per candidate date; keep those with foreign hotspots connected.
    events: list[tuple[float, int, _date, float]] = []  # (conn_foreign_frp, pos, date, peak)
    for d in cand_dates:
        found = _peak_anchor_for_date(ds, d, station_idx)
        if found is None:
            continue
        pos, peak = found
        conn = _connected_frp_by_country(ds[pos], station_idx)
        foreign = sum(v for k, v in conn.items() if k in _FOREIGN)
        events.append((foreign, pos, d, peak))

    events.sort(key=lambda e: -e[0])
    model = _load_mtgnn(ds.n_stations, horizons, overrides, ckpt)

    summaries = [
        _summarize_event(model, ds[pos], station_idx, station_name, horizon_idx, d, peak)
        for (_foreign, pos, d, peak) in events[:top_k]
    ]

    attributed = [s for s in summaries if s["foreign_attribution"] > 0.0]
    results: dict[str, object] = {
        "station": station_name,
        "station_id": station_id,
        "split": split,
        "horizon_h": horizons[horizon_idx],
        "checkpoint": _MTGNN_CKPT,
        "method": (
            "Rank split dates by Myanmar+Laos FRP; at the border station take the peak-PM2.5 "
            "anchor, read hotspots connected via type_c edges, then occlusion country "
            "attribution + IG. Compares model foreign attribution to connected-foreign FRP."
        ),
        "n_events_with_connected_foreign_fire": sum(1 for _f, *_ in events if _f > 0),
        "events": summaries,
        "interpretation": (
            f"{len(attributed)}/{len(summaries)} top events show non-zero foreign (Myanmar/"
            "Laos) attribution. Compare foreign_attribution vs connected_foreign_fraction per "
            "event: attribution ~0 despite a high connected-foreign fraction => the model does "
            "not capture transboundary transport at this station; attribution tracking the "
            "fraction => it is responsive to cross-border fire input."
        ),
    }

    print("\n" + "=" * 84)
    print(
        f"Transboundary attribution @ {station_name}"
        f"  split={split}  horizon={horizons[horizon_idx]}h"
    )
    print("-" * 84)
    print(f"{'date':<12}{'peakPM':>8}{'connForeign%':>14}{'foreignAttr':>13}  country_attribution")
    for s in summaries:
        print(
            f"{s['date']:<12}{s['peak_pm25_ug_m3']:>8.1f}"
            f"{s['connected_foreign_fraction']*100:>13.1f}%{s['foreign_attribution']:>13.3f}"
            f"  {s['country_attribution']}"
        )
    print("=" * 84 + "\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    logger.info("Saved transboundary attribution to %s", output_path)


if __name__ == "__main__":
    main()
