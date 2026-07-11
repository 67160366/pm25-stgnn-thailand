# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md

"""Transboundary attribution multi-seed uncertainty (demo roadmap item 4).

``scripts/10_transboundary_attr.py`` reports transboundary attribution at Mae Hong Son
from a SINGLE checkpoint, so a viewer cannot tell how much the reported attribution
*magnitude* would move if the model had been trained with a different random seed. This
script reruns the exact same report-protocol events (station 225648, split test,
horizon_idx 2, n_candidates 15, top_k 5) across 3 fixed checkpoints of the ``full``
variant -- the original report checkpoint plus two additional seeds -- and reports
mean +/- spread (std ddof=1 and min-max range) per event.

Event selection is run ONCE via
``src.explain.transboundary_events.select_connected_foreign_events`` (model-independent:
depends only on FIRMS FRP ranking, per-station peak PM2.5, and the wind-deterministic
type_c graph), so the identical 5 events are reused across all 3 seeds -- seed-to-seed
variation can only come from attribution magnitude, never from a different event set.

All heavy per-event logic (occlusion + IG summarisation, model loading) is reused
verbatim from ``scripts/10_transboundary_attr.py`` via
``src.explain.transboundary_events.load_script10`` -- see that module's docstring for why
script 10 is loaded by path instead of edited or copy-pasted.

Usage:
    uv run python scripts/21_transboundary_uncertainty.py
    uv run python scripts/21_transboundary_uncertainty.py station_id=225626 top_k=3
"""

from __future__ import annotations

import json
import logging
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir

from src.data.loader import PM25GraphDataset
from src.explain.transboundary_events import load_script10, select_connected_foreign_events

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIGS_DIR = _PROJECT_ROOT / "configs"

# Seeds are fixed (not CLI) so the artifact is canonical and reproducible.
_SEEDS: list[dict[str, str]] = [
    {"label": "report_orig", "path": "checkpoints_split2/mtgnn/best_model.pt"},
    {"label": "full_s0", "path": "checkpoints_split2_seeds/full_s0/best_model.pt"},
    {"label": "full_s1", "path": "checkpoints_split2_seeds/full_s1/best_model.pt"},
]

_LOCAL_KEYS = {
    "split",
    "station_id",
    "n_candidates",
    "top_k",
    "horizon_idx",
    "n_ig_steps",
    "output",
}

_DEFAULT_STATION_ID = 225648  # Mae Hong Son - matches the frozen split2 report protocol.

_CAVEATS: list[str] = [
    "n=3 seeds เป็นตัวอย่างเล็ก — รายงานช่วง min-max คู่กับ mean±std ไม่ตีความ std เกินตัว",
    "spread นี้คือความไวของ 'ขนาด' attribution ต่อ seed — เปลี่ยน caveat เดิม 'ขนาดขึ้นกับโมเดล' "
    "ให้เป็นตัวเลขที่วัดจริง",
    "การเลือกเหตุการณ์เหมือนกันทุก seed (ขับด้วยข้อมูล) ต่างเฉพาะตัวเลข attribution",
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


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Sample mean and sample std (ddof=1); std is 0.0 for fewer than 2 values."""
    mean_v = statistics.fmean(values)
    std_v = statistics.stdev(values) if len(values) >= 2 else 0.0
    return mean_v, std_v


def aggregate_event_across_seeds(
    seed_labels: list[str], per_seed_reports: dict[str, dict[str, object]]
) -> dict[str, object]:
    """Aggregate one event's country/foreign attribution across model seeds (pure, no I/O).

    Args:
        seed_labels: Ordered seed labels, e.g. ``["report_orig", "full_s0", "full_s1"]``.
        per_seed_reports: Mapping ``label -> report`` where each report has
            ``country_attribution`` (``dict[str, float]``) and ``foreign_attribution``
            (``float``), as produced by the reused ``_summarize_event``.

    Returns:
        Dict with ``foreign_attribution_mean/std/min/max`` (std is sample std, ddof=1,
        0.0 if fewer than 2 seeds) and ``country_attribution_mean/std`` computed over the
        union of countries present in any seed's ``country_attribution`` (a country
        missing from a given seed contributes 0.0 for that seed).
    """
    foreign_vals = [float(per_seed_reports[label]["foreign_attribution"]) for label in seed_labels]
    mean_f, std_f = _mean_std(foreign_vals)

    countries = sorted(
        {c for label in seed_labels for c in per_seed_reports[label]["country_attribution"]}
    )
    country_mean: dict[str, float] = {}
    country_std: dict[str, float] = {}
    for country in countries:
        vals = [
            float(per_seed_reports[label]["country_attribution"].get(country, 0.0))
            for label in seed_labels
        ]
        m, s = _mean_std(vals)
        country_mean[country] = round(m, 3)
        country_std[country] = round(s, 3)

    return {
        "foreign_attribution_mean": round(mean_f, 3),
        "foreign_attribution_std": round(std_f, 3),
        "foreign_attribution_min": round(min(foreign_vals), 3),
        "foreign_attribution_max": round(max(foreign_vals), 3),
        "country_attribution_mean": country_mean,
        "country_attribution_std": country_std,
    }


def build_uncertainty_json(
    station_id: int,
    station_name: str,
    events_agg: list[dict[str, object]],
    seeds: list[dict[str, str]],
    meta_params: dict[str, object],
    caveats: list[str],
) -> dict[str, object]:
    """Assemble the ``outputs/transboundary_uncertainty.json`` payload (pure, no I/O).

    Args:
        station_id: Target station id (default Mae Hong Son, 225648).
        station_name: Human-readable station name.
        events_agg: One dict per event; each already carries ``per_seed`` plus the
            ``aggregate_event_across_seeds`` output (``foreign_attribution_mean`` etc.).
        seeds: The 3 fixed seed specs (``{"label": ..., "path": ...}``).
        meta_params: Scalar fields (``method``, ``split``, ``horizon_h``, ``horizon_idx``,
            ``n_ig_steps``, ``n_candidates``, ``top_k``, ``wind_mode``) spliced verbatim.
        caveats: Honest-thesis caveat strings for the dashboard.

    Returns:
        JSON-ready dict matching the schema in the SPEC (section 1.6), including a
        ``summary`` block with ``n_seeds``, ``n_events``, ``mean_foreign_attribution`` and
        ``mean_event_spread_minmax`` (mean over events of each event's max-min range).
    """
    if events_agg:
        mean_foreign = statistics.fmean(float(e["foreign_attribution_mean"]) for e in events_agg)
        mean_spread = statistics.fmean(
            float(e["foreign_attribution_max"]) - float(e["foreign_attribution_min"])
            for e in events_agg
        )
    else:
        mean_foreign = 0.0
        mean_spread = 0.0

    summary = {
        "n_seeds": len(seeds),
        "n_events": len(events_agg),
        "mean_foreign_attribution": round(mean_foreign, 3),
        "mean_event_spread_minmax": round(mean_spread, 3),
    }
    return {
        "generated_by": "scripts/21_transboundary_uncertainty.py",
        "station_id": station_id,
        "station": station_name,
        **meta_params,
        "seeds": seeds,
        "events": events_agg,
        "summary": summary,
        "caveats": caveats,
    }


def main() -> None:
    """Rerun the report-protocol events across 3 seeds and save mean +/- spread."""
    local_args, overrides = _split_cli_args(sys.argv[1:])
    with initialize_config_dir(config_dir=str(_CONFIGS_DIR), version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)

    split = local_args.get("split", "test")
    station_id = int(local_args.get("station_id", _DEFAULT_STATION_ID))
    n_candidates = int(local_args.get("n_candidates", 15))
    top_k = int(local_args.get("top_k", 5))
    horizon_idx = int(local_args.get("horizon_idx", 2))
    n_ig_steps = int(local_args.get("n_ig_steps", 50))
    output_path = Path(local_args.get("output", "outputs/transboundary_uncertainty.json"))
    horizons = list(cfg.data.horizons)
    wind_mode = getattr(cfg.data, "wind_mode", "constant_ne")
    hotspots_path = Path(cfg.data.hotspots_path)

    if n_ig_steps != 50:
        logger.warning(
            "n_ig_steps=%d requested, but the reused station_source_report() default (50) "
            "is used unconditionally by script 10's _summarize_event -- this value is "
            "recorded in the output metadata only and does not change the computation.",
            n_ig_steps,
        )

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

    meta = pd.read_parquet(cfg.data.metadata_path).rename(columns={"location_id": "station_id"})
    name_row = meta.loc[meta["station_id"] == station_id, "name"]
    station_name = str(name_row.iloc[0]) if len(name_row) else f"station_{station_id}"

    matches = np.where(ds._station_ids == station_id)[0]
    if matches.size == 0:
        raise SystemExit(f"station_id {station_id} not in dataset stations {list(ds._station_ids)}")
    station_idx = int(matches[0])
    logger.info("Target: %s (id=%d idx=%d) split=%s", station_name, station_id, station_idx, split)

    mod = load_script10()
    # Event selection is model-independent -- run ONCE, reused identically across all seeds.
    events = select_connected_foreign_events(
        ds, station_idx, hotspots_path, split, n_candidates=n_candidates, top_k=top_k
    )
    if not events:
        raise SystemExit(
            f"No connected-foreign events found for station_id={station_id} split={split}"
        )
    logger.info("Selected %d events (model-independent, shared across all seeds)", len(events))

    seed_labels = [s["label"] for s in _SEEDS]
    per_event_seed_reports: dict[int, dict[str, dict[str, object]]] = {
        pos: {} for (_foreign, pos, _d, _peak) in events
    }

    for seed in _SEEDS:
        logger.info("Loading seed %s (%s)", seed["label"], seed["path"])
        try:
            model = mod._load_mtgnn(ds.n_stations, horizons, overrides, seed["path"])
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            # Fail loudly -- never silently drop to fewer seeds while still labeling the
            # output "3 seeds" (SPEC risk #4).
            raise SystemExit(
                f"Failed to load checkpoint for seed '{seed['label']}' ({seed['path']}): "
                f"{exc}. Aborting -- refusing to silently continue with fewer seeds."
            ) from exc

        for _foreign, pos, d, peak in events:
            summary = mod._summarize_event(
                model, ds[pos], station_idx, station_name, horizon_idx, d, peak
            )
            per_event_seed_reports[pos][seed["label"]] = summary
        logger.info("seed=%s: summarized %d events", seed["label"], len(events))

    events_agg: list[dict[str, object]] = []
    for _foreign, pos, _d, _peak in events:
        seed_reports = per_event_seed_reports[pos]
        # Common fields (date, peak PM2.5, connected FRP) are model-independent, so they
        # are identical across seeds -- take them from the first seed's report.
        canonical = seed_reports[seed_labels[0]]
        agg = aggregate_event_across_seeds(seed_labels, seed_reports)
        events_agg.append(
            {
                "date": canonical["date"],
                "peak_pm25_ug_m3": canonical["peak_pm25_ug_m3"],
                "connected_foreign_fraction": canonical["connected_foreign_fraction"],
                "connected_frp_by_country": canonical["connected_frp_by_country"],
                "per_seed": {
                    label: {
                        "country_attribution": seed_reports[label]["country_attribution"],
                        "foreign_attribution": seed_reports[label]["foreign_attribution"],
                    }
                    for label in seed_labels
                },
                **agg,
            }
        )

    meta_params: dict[str, object] = {
        "method": (
            "Same report-protocol events as scripts/10_transboundary_attr.py (rank split "
            "dates by Myanmar+Laos FRP, take the peak-PM2.5 anchor at the border station, "
            "read hotspots connected via type_c edges, then occlusion country attribution "
            "+ IG), rerun across 3 fixed checkpoints of the full variant to report "
            "attribution mean +/- spread (std ddof=1 and min-max range) per event."
        ),
        "split": split,
        "horizon_h": horizons[horizon_idx],
        "horizon_idx": horizon_idx,
        "n_ig_steps": n_ig_steps,
        "n_candidates": n_candidates,
        "top_k": top_k,
        "wind_mode": wind_mode,
    }
    payload = build_uncertainty_json(
        station_id, station_name, events_agg, _SEEDS, meta_params, _CAVEATS
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    logger.info(
        "Saved transboundary uncertainty (%d events x %d seeds) to %s",
        len(events_agg),
        len(_SEEDS),
        output_path,
    )


if __name__ == "__main__":
    main()
