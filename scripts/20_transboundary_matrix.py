# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md

"""Transboundary attribution matrix: border stations x checkpoints (demo roadmap item 2).

``scripts/10_transboundary_attr.py`` hunts transboundary events at a single station
(Mae Hong Son) with a single checkpoint. This script broadens that to every station near
the Myanmar/Laos border and BOTH checkpoints in active use (the live-demo model and the
report/held-out-test model), so the dashboard can show that attribution *magnitude* is
model-dependent -- without cherry-picking the checkpoint that looks best.

All heavy per-event logic (event ranking, occlusion + IG summarisation, model loading) is
reused verbatim from ``scripts/10_transboundary_attr.py`` via
``src.explain.transboundary_events.load_script10`` -- see that module's docstring for why
script 10 is loaded by path instead of edited or copy-pasted.

Usage:
    uv run python scripts/20_transboundary_matrix.py
    uv run python scripts/20_transboundary_matrix.py max_dist_km=70 top_k=3
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
from src.explain.transboundary_events import (
    load_script10,
    select_border_stations,
    select_connected_foreign_events,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIGS_DIR = _PROJECT_ROOT / "configs"

# Checkpoints are fixed (not CLI) so the artifact is canonical and reproducible.
_CHECKPOINTS: list[dict[str, str]] = [
    {"label": "demo", "path": "checkpoints/mtgnn/best_model.pt"},
    {"label": "report", "path": "checkpoints_split2/mtgnn/best_model.pt"},
]

_LOCAL_KEYS = {
    "split",
    "n_candidates",
    "top_k",
    "horizon_idx",
    "max_dist_km",
    "output",
    "n_ig_steps",
}

_CAVEATS: list[str] = [
    "ขนาดของสัดส่วน attribution ขึ้นกับโมเดล — แสดงทั้ง checkpoint (demo และ report/split2) "
    "โดยไม่เลือกเฉพาะอันที่สวย",
    "foreign_attribution ≈ 0 ที่สถานี/เหตุการณ์ซึ่งไฟต่างชาติเชื่อมถึงน้อย = ผลลบเชิงซื่อสัตย์ "
    "(โมเดลไล่ตามไฟจริง ไม่ตั้งธงการเมือง) ไม่ใช่ความล้มเหลว",
    "เป็นการประมาณจากโมเดล (occlusion/IG) เทียบกับไฟจริง FIRMS — ไม่ใช่การวัดในอากาศ "
    "ห้ามใช้กล่าวโทษเชิงการทูต",
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


def build_matrix_json(
    stations: list[dict],
    results: list[dict],
    meta_params: dict[str, object],
    caveats: list[str],
) -> dict[str, object]:
    """Assemble the ``outputs/transboundary_matrix.json`` payload (pure, no I/O).

    Args:
        stations: Border stations, as returned by ``select_border_stations``.
        results: One record per (station, checkpoint) pair; each entry's ``events`` list
            is the unmodified output of the reused ``_summarize_event``.
        meta_params: Ordered top-level scalar fields (``method``, ``split``, ``horizon_h``,
            ``horizon_idx``, ``n_ig_steps``, ``n_candidates``, ``top_k``, ``wind_mode``,
            ``max_dist_km``) to splice into the payload verbatim.
        caveats: Honest-thesis caveat strings for the dashboard.

    Returns:
        JSON-ready dict matching the schema in the SPEC (section 1.6).
    """
    return {
        "generated_by": "scripts/20_transboundary_matrix.py",
        **meta_params,
        "checkpoints": _CHECKPOINTS,
        "stations": stations,
        "results": results,
        "caveats": caveats,
    }


def main() -> None:
    """Build the station x checkpoint transboundary attribution matrix and save it."""
    local_args, overrides = _split_cli_args(sys.argv[1:])
    with initialize_config_dir(config_dir=str(_CONFIGS_DIR), version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)

    split = local_args.get("split", "test")
    n_candidates = int(local_args.get("n_candidates", 15))
    top_k = int(local_args.get("top_k", 5))
    horizon_idx = int(local_args.get("horizon_idx", 2))
    max_dist_km = float(local_args.get("max_dist_km", 50.0))
    n_ig_steps = int(local_args.get("n_ig_steps", 50))
    output_path = Path(local_args.get("output", "outputs/transboundary_matrix.json"))
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

    meta = pd.read_parquet(cfg.data.metadata_path)
    stations = select_border_stations(meta, max_dist_km=max_dist_km)
    logger.info("Selected %d border stations at max_dist_km=%.1f", len(stations), max_dist_km)
    if not stations:
        raise SystemExit(
            f"No stations within max_dist_km={max_dist_km} of the Myanmar/Laos border."
        )

    mod = load_script10()
    results: list[dict[str, object]] = []
    for ckpt in _CHECKPOINTS:
        logger.info("Loading checkpoint %s (%s)", ckpt["label"], ckpt["path"])
        model = mod._load_mtgnn(ds.n_stations, horizons, overrides, ckpt["path"])

        for station in stations:
            station_id = station["station_id"]
            station_name = station["name"]
            matches = np.where(ds._station_ids == station_id)[0]
            if matches.size == 0:
                logger.warning("station_id %d not in dataset stations; skipping", station_id)
                continue
            station_idx = int(matches[0])

            events = select_connected_foreign_events(
                ds, station_idx, hotspots_path, split, n_candidates=n_candidates, top_k=top_k
            )
            summaries = [
                mod._summarize_event(
                    model, ds[pos], station_idx, station_name, horizon_idx, d, peak
                )
                for (_foreign, pos, d, peak) in events
            ]
            logger.info(
                "station=%s (%d) ckpt=%s: %d events",
                station_name,
                station_id,
                ckpt["label"],
                len(summaries),
            )
            results.append(
                {
                    "station_id": station_id,
                    "station_name": station_name,
                    "checkpoint_label": ckpt["label"],
                    "checkpoint": ckpt["path"],
                    "n_events_with_connected_foreign_fire": sum(1 for f, *_ in events if f > 0),
                    "events": summaries,
                }
            )

    meta_params: dict[str, object] = {
        "method": (
            "Rank split dates by Myanmar+Laos FRP; at each border station take the "
            "peak-PM2.5 anchor, read hotspots connected via type_c edges, then occlusion "
            "country attribution + IG. Compares model foreign attribution to "
            "connected-foreign FRP, across the demo and report checkpoints."
        ),
        "split": split,
        "horizon_h": horizons[horizon_idx],
        "horizon_idx": horizon_idx,
        "n_ig_steps": n_ig_steps,
        "n_candidates": n_candidates,
        "top_k": top_k,
        "wind_mode": wind_mode,
        "max_dist_km": max_dist_km,
    }
    payload = build_matrix_json(stations, results, meta_params, _CAVEATS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    logger.info("Saved transboundary matrix (%d results) to %s", len(results), output_path)


if __name__ == "__main__":
    main()
