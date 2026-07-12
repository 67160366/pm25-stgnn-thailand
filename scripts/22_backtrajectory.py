# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md

"""ERA5 kinematic back-trajectory: an independent witness for transboundary attribution
(roadmap item 11).

``scripts/10_transboundary_attr.py`` produced the frozen
``outputs/transboundary_attr_test_split2.json`` -- the model's own claim that some Mae Hong
Son PM2.5 events are attributable to Myanmar/Laos fires. This script does NOT touch the model
at all: it launches a purely kinematic backward trajectory from ERA5 10 m winds (u10/v10) at
the same peak-PM2.5 anchor hour the model's event selection uses, and asks an independent
question -- "does the air that arrived here look like it came from, and passed near active
fires in, Myanmar or Laos?" -- then compares that binary answer to the model's foreign
attribution.

**Honest limitation:** surface (10 m) winds only, single launch time, no vertical transport,
no turbulent mixing -- this is *indicative corroboration*, NOT equivalent to a full HYSPLIT
Lagrangian dispersion run. See ``src/explain/backtrajectory.py`` module docstring and the
``caveats`` field of the output JSON for the full statement (Thai, verbatim, 5 caveats).

Event selection is model-independent (``select_connected_foreign_events``, reused from
``src.explain.transboundary_events`` -- the same helper scripts 20/21 use), so this script
never loads a model checkpoint: it is CPU-only and runs in seconds.

Usage:
    uv run python scripts/22_backtrajectory.py
    uv run python scripts/22_backtrajectory.py hours_back=72 corridor_radius_km=75
    uv run python scripts/22_backtrajectory.py station_id=225626 output=outputs/tb_maesot_traj.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir

from src.data.loader import PM25GraphDataset
from src.explain.backtrajectory import (
    ERA5_BBOX,
    assess_event,
    corridor_frp_by_country,
    integrate_backtrajectory,
    label_countries,
    load_era5_window,
)
from src.explain.transboundary_events import select_connected_foreign_events

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIGS_DIR = _PROJECT_ROOT / "configs"
_ERA5_DIR = _PROJECT_ROOT / "data" / "raw" / "era5"
_MODEL_ATTR_PATH = _PROJECT_ROOT / "outputs" / "transboundary_attr_test_split2.json"

_LOCAL_KEYS = {
    "station_id",
    "split",
    "hours_back",
    "dt_hours",
    "corridor_radius_km",
    "foreign_hours_threshold",
    "n_candidates",
    "top_k",
    "output",
}

_DEFAULT_STATION_ID = 225648  # Mae Hong Son - matches the frozen split2 report protocol.

_CAVEATS: list[str] = [
    "วิถีลมนี้ใช้ลม 10 เมตร (u10/v10) ของ ERA5 เท่านั้น — เป็นการเคลื่อนที่ในแนวราบ ไม่รวมการยกตัว/"
    "จมตัวในแนวดิ่งและชั้นบรรยากาศบน จึงเป็นหลักฐานสนับสนุนเชิงบ่งชี้ ไม่เทียบเท่า HYSPLIT เต็มรูปแบบ",
    "โดเมน ERA5 ครอบคลุมแค่ lon 97.0–101.5, lat 16.0–21.0 — วิถีที่ย้อนออกนอกกรอบจะถูกหยุดและบันทึกเหตุผล "  # noqa: RUF001
    "(terminated_reason) สัดส่วนประเทศคำนวณเฉพาะช่วงที่อยู่ในโดเมน",
    "การจับคู่วิถีลมกับจุดความร้อน FIRMS ใช้รัศมีคงที่และวันที่ (UTC) ของแต่ละชั่วโมงวิถี — เป็นการชี้ 'ผ่านใกล้"
    "ไฟที่ยังคุอยู่' ไม่ใช่การพิสูจน์การขนส่งมวลฝุ่นเชิงปริมาณ",
    "เวลาปล่อยวิถีคือชั่วโมงที่ PM2.5 สูงสุดของวันนั้นที่สถานี (anchor เดียวกับที่โมเดลใช้ระบุแหล่ง) — วิถีเดียวต่อ"
    "เหตุการณ์ ไม่ใช่ ensemble หลายเวลาหรือหลายระดับความสูง",
    "การ 'เห็นตรงกัน' (agrees_with_model) เป็นการยืนยันทิศทางแบบไบนารี (ต่างชาติ/ในประเทศ) ระหว่างพยานอิสระ"
    "กับโมเดล ไม่ได้ยืนยันขนาดของ attribution",
]


def _split_cli_args(argv: list[str]) -> tuple[dict[str, str], list[str]]:
    """Separate this script's local keys (``_LOCAL_KEYS``) from Hydra overrides."""
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


def build_agreement_summary(events: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate per-event ``agrees_with_model``/``trajectory_foreign_plausible`` (pure).

    Events whose ``agrees_with_model`` is ``None`` (model attribution unavailable for that
    date) are excluded from ``n_agree``/``agreement_fraction``'s denominator.

    Args:
        events: Fully-assembled per-event dicts (schema section 5.2).

    Returns:
        Dict with ``n_events``, ``n_model_foreign``, ``n_trajectory_foreign_plausible``,
        ``n_agree``, ``n_events_with_model``, ``agreement_fraction``.
    """
    n_events = len(events)
    n_model_foreign = sum(
        1
        for e in events
        if e["model_foreign_attribution"] is not None and e["model_foreign_attribution"] >= 0.05
    )
    n_traj_foreign = sum(1 for e in events if e["trajectory_foreign_plausible"])
    with_model = [e for e in events if e["agrees_with_model"] is not None]
    n_agree = sum(1 for e in with_model if e["agrees_with_model"])
    agreement_fraction = n_agree / len(with_model) if with_model else 0.0

    return {
        "n_events": n_events,
        "n_model_foreign": n_model_foreign,
        "n_trajectory_foreign_plausible": n_traj_foreign,
        "n_agree": n_agree,
        "n_events_with_model": len(with_model),
        "agreement_fraction": round(agreement_fraction, 4),
    }


def main() -> None:
    """Run the ERA5 back-trajectory witness for the 5 frozen MHS report-protocol events."""
    local_args, overrides = _split_cli_args(sys.argv[1:])
    with initialize_config_dir(config_dir=str(_CONFIGS_DIR), version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)

    station_id = int(local_args.get("station_id", _DEFAULT_STATION_ID))
    split = local_args.get("split", "test")
    hours_back = int(local_args.get("hours_back", 48))
    dt_hours = float(local_args.get("dt_hours", 1.0))
    corridor_radius_km = float(local_args.get("corridor_radius_km", 50.0))
    foreign_hours_threshold = float(local_args.get("foreign_hours_threshold", 0.15))
    n_candidates = int(local_args.get("n_candidates", 15))
    top_k = int(local_args.get("top_k", 5))
    output_path = Path(local_args.get("output", "outputs/backtrajectory_test2025.json"))
    horizons = list(cfg.data.horizons)
    wind_mode = getattr(cfg.data, "wind_mode", "constant_ne")
    hotspots_path = Path(cfg.data.hotspots_path)

    ds = PM25GraphDataset(
        dataset_path=Path(cfg.data.dataset_path),
        hotspots_path=hotspots_path,
        metadata_path=Path(cfg.data.metadata_path),
        scalers_path=Path(cfg.data.scalers_path),
        split=split,
        window_in=cfg.data.window_in,
        horizons=horizons,
        graph_config={"wind_mode": wind_mode},
    )

    matches = np.where(ds._station_ids == station_id)[0]
    if matches.size == 0:
        raise SystemExit(f"station_id {station_id} not in dataset stations {list(ds._station_ids)}")
    station_idx = int(matches[0])

    meta = pd.read_parquet(cfg.data.metadata_path).rename(columns={"location_id": "station_id"})
    name_row = meta.loc[meta["station_id"] == station_id, "name"]
    station_name = str(name_row.iloc[0]) if len(name_row) else f"station_{station_id}"
    station_row = meta.loc[meta["station_id"] == station_id].iloc[0]
    station_lat, station_lon = float(station_row["lat"]), float(station_row["lon"])
    logger.info(
        "Target: %s (id=%d idx=%d lat=%.5f lon=%.5f) split=%s",
        station_name,
        station_id,
        station_idx,
        station_lat,
        station_lon,
        split,
    )

    events = select_connected_foreign_events(
        ds, station_idx, hotspots_path, split, n_candidates=n_candidates, top_k=top_k
    )
    if not events:
        raise SystemExit(
            f"No connected-foreign events found for station_id={station_id} split={split}"
        )
    logger.info(
        "Selected %d events (model-independent, shared event-selection helper)", len(events)
    )

    if not _MODEL_ATTR_PATH.exists():
        raise SystemExit(
            f"Frozen model attribution file not found: {_MODEL_ATTR_PATH}. This script reads "
            "it read-only; it does not regenerate it (run scripts/10_transboundary_attr.py "
            "first if genuinely missing)."
        )
    with open(_MODEL_ATTR_PATH, encoding="utf-8") as fh:
        model_json = json.load(fh)
    model_by_date = {e["date"]: e for e in model_json["events"]}

    hotspots = pd.read_parquet(hotspots_path)

    out_events: list[dict[str, object]] = []
    for foreign_frp, anchor_pos, ev_date, peak in events:
        launch = pd.Timestamp(ds._timestamps[ds._anchor_indices[anchor_pos]])
        wind = load_era5_window(_ERA5_DIR, launch, hours_back, bbox=ERA5_BBOX)
        launch_naive = launch.tz_localize(None) if launch.tzinfo is not None else launch

        traj = integrate_backtrajectory(
            wind,
            start_lat=station_lat,
            start_lon=station_lon,
            start_time=np.datetime64(launch_naive, "ns"),
            hours_back=hours_back,
            dt_hours=dt_hours,
        )
        countries = label_countries(traj.lats, traj.lons)
        corridor = corridor_frp_by_country(traj, hotspots, radius_km=corridor_radius_km)

        date_str = str(ev_date)
        m = model_by_date.get(date_str)
        if m is None:
            logger.warning(
                "Event date %s (connected_foreign_frp=%.1f) not found in frozen model "
                "attribution %s -- emitting trajectory with model fields set to null.",
                date_str,
                foreign_frp,
                _MODEL_ATTR_PATH,
            )

        assessed = assess_event(
            traj,
            countries,
            m.get("foreign_attribution") if m else None,
            m.get("country_attribution") if m else None,
            corridor,
            foreign_hours_threshold=foreign_hours_threshold,
        )
        event_out = {
            "date": date_str,
            "launch_time_utc": pd.Timestamp(traj.times[0]).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "peak_pm25_ug_m3": round(peak, 1),
            **assessed,
        }
        out_events.append(event_out)
        logger.info(
            "event=%s hours_integrated=%d/%d terminated=%s foreign_hours_frac=%.3f "
            "trajectory_foreign_plausible=%s model_foreign_attr=%s agrees=%s",
            date_str,
            traj.hours_integrated,
            hours_back,
            traj.terminated_reason,
            event_out["foreign_hours_fraction"],
            event_out["trajectory_foreign_plausible"],
            event_out["model_foreign_attribution"],
            event_out["agrees_with_model"],
        )

    agreement_summary = build_agreement_summary(out_events)

    payload: dict[str, object] = {
        "station": station_name,
        "station_id": station_id,
        "split": split,
        "method": (
            "Kinematic backward trajectory from ERA5 10m winds (u10/v10), bilinear space + "
            "linear time, RK2 backward integration, launched at the model's peak-PM2.5 "
            "anchor hour. Compares a binary 'trajectory foreign-influence plausible' witness "
            "to the model's foreign attribution."
        ),
        "trajectory_params": {
            "hours_back": hours_back,
            "dt_hours": dt_hours,
            "integrator": "rk2",
            "corridor_radius_km": corridor_radius_km,
            "foreign_hours_threshold": foreign_hours_threshold,
            "model_foreign_tau": 0.05,
            "era5_bbox": list(ERA5_BBOX),
        },
        "model_attribution_source": str(_MODEL_ATTR_PATH.relative_to(_PROJECT_ROOT)).replace(
            "\\", "/"
        ),
        "n_events": len(out_events),
        "events": out_events,
        "agreement_summary": agreement_summary,
        "caveats": _CAVEATS,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    logger.info(
        "Saved back-trajectory analysis (%d events) to %s",
        len(out_events),
        output_path,
    )


if __name__ == "__main__":
    main()
