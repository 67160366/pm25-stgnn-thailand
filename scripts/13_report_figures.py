# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md
"""Generate static result figures (PNG) for the NSC final report from the result JSONs.

Reads ``outputs/*.json`` (the source of truth used by the report and pitch deck) and renders
print-ready matplotlib figures to ``outputs/figures/report/``. No model inference here — figures
are a faithful visual of the already-verified numbers, so they cannot drift from the report.

Run: ``uv run python scripts/13_report_figures.py``
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "outputs"
_FIG = _OUT / "figures" / "report"
_HORIZONS = ["6h", "12h", "24h", "48h"]
_AQI = "PM2.5 RMSE (ug/m3)"  # ASCII axis label (Thai captions live in the report text)

# Thai PCD country colours, consistent with the dashboard/deck.
_C_TH, _C_MM, _C_LA = "#2196F3", "#FF5722", "#4CAF50"


def _load(name: str) -> dict:
    with (_OUT / name).open(encoding="utf-8") as fh:
        return json.load(fh)


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(_FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def fig_rmse_by_horizon() -> None:
    """Grouped bar of held-out-test RMSE per horizon for each method."""
    ev = _load("evaluation_test2025.json")
    gbm = _load("baseline_ml_test.json")
    series = {
        "Persistence": ev["persistence_rmse_ug_m3"],
        "MTGNN (graph)": ev["mtgnn"]["rmse_ug_m3"],
        "A3TGCN": ev["a3tgcn"]["rmse_ug_m3"],
        "GBM (no graph)": gbm["baseline_ml_rmse_ug_m3"],
    }
    colours = {
        "Persistence": "#888",
        "MTGNN (graph)": "#15616D",
        "A3TGCN": "#9E9E9E",
        "GBM (no graph)": "#E8743B",
    }
    x = range(len(_HORIZONS))
    w = 0.2
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, (label, d) in enumerate(series.items()):
        ax.bar(
            [xi + (i - 1.5) * w for xi in x],
            [d[h] for h in _HORIZONS],
            w,
            label=label,
            color=colours[label],
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(_HORIZONS)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel(_AQI)
    ax.set_title("RMSE by horizon — held-out test 2025 (lower is better)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "rmse_by_horizon.png")


def fig_significance() -> None:
    """48h percentage improvement vs persistence with 95% CI (val + test); both cross 0."""
    sv = _load("significance_val2025.json")["per_horizon"]["48h"]
    stt = _load("significance_test2025.json")["per_horizon"]["48h"]
    rows = [("Held-out test 2025", stt), ("Main model on 2025", sv)]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for i, (_label, d) in enumerate(rows):
        pt = d["pct_improvement_point"]
        lo, hi = d["pct_improvement_ci95"]
        ax.errorbar(pt, i, xerr=[[pt - lo], [hi - pt]], fmt="o", color="#15616D", capsize=5, lw=2)
    ax.axvline(0, color="#E8743B", ls="--", lw=1.5, label="no improvement")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("48h improvement vs persistence (%)  —  95% CI")
    ax.set_title("Forecast edge at 48h is positive but NOT significant")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    _save(fig, "significance_48h.png")


def fig_ablation() -> None:
    """Per-variant RMSE delta vs full with combined-seed-std error bars (all within noise)."""
    ab = _load("ablation_multiseed.json")["deltas_vs_full"]
    variants = list(ab.keys())
    fig, axes = plt.subplots(1, 4, figsize=(10, 3.4), sharey=True)
    for j, h in enumerate(_HORIZONS):
        ax = axes[j]
        deltas = [ab[v]["delta_vs_full"][j] for v in variants]
        errs = [ab[v]["combined_std"][j] for v in variants]
        ax.barh(variants, deltas, xerr=errs, color="#15616D", capsize=3)
        ax.axvline(0, color="#888", lw=1)
        ax.set_title(h)
        ax.grid(axis="x", alpha=0.3)
        if j == 0:
            ax.set_ylabel("variant removed")
    fig.suptitle(
        "Multi-seed ablation: every channel's effect is within seed noise "
        "(delta vs full +/- combined std)"
    )
    _save(fig, "ablation_multiseed.png")


def fig_nwp() -> None:
    """RMSE vs ERA5 noise fraction for 24h/48h, with persistence references."""
    nwp = _load("nwp_sensitivity.json")
    res = nwp["results"]
    xs = [r["noise_fraction"] for r in res]
    pers = nwp["persistence_rmse_ug_m3"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(xs, [r["rmse_ug_m3"]["24h"] for r in res], "o-", color="#15616D", label="MTGNN 24h")
    ax.plot(xs, [r["rmse_ug_m3"]["48h"] for r in res], "s-", color="#E8743B", label="MTGNN 48h")
    ax.axhline(pers["24h"], color="#15616D", ls=":", lw=1.2, label="persistence 24h")
    ax.axhline(pers["48h"], color="#E8743B", ls=":", lw=1.2, label="persistence 48h")
    ax.set_xlabel("ERA5 noise (x natural std)")
    ax.set_ylabel(_AQI)
    ax.set_title("NWP sensitivity: 24h edge lost at 0.25x, 48h at 0.5x")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, "nwp_sensitivity.png")


def fig_transboundary_events() -> None:
    """Per event: connected-foreign FRP % vs model foreign-attribution %."""
    tb = _load("transboundary_attr_test_split2.json")["events"]
    dates = [e["date"] for e in tb]
    frp = [e["connected_foreign_fraction"] * 100 for e in tb]
    attr = [e["foreign_attribution"] * 100 for e in tb]
    x = range(len(dates))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar([xi - w / 2 for xi in x], frp, w, label="foreign fire (FRP) %", color="#FF9800")
    ax.bar([xi + w / 2 for xi in x], attr, w, label="model attributes foreign %", color="#FF5722")
    ax.set_xticks(list(x))
    ax.set_xticklabels(dates, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("percent (%)")
    ax.set_title("Mae Hong Son (held-out 2025): foreign fire share vs model attribution")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "transboundary_events.png")


def fig_transboundary_map() -> None:
    """Scatter of FIRMS hotspots coloured by country + the border station.

    Uses the flagship event: the test-split event with the highest
    connected-foreign FRP fraction (per transboundary_attr_test_split2.json).
    """
    events = _load("transboundary_attr_test_split2.json")["events"]
    flagship = max(events, key=lambda e: e["connected_foreign_fraction"])["date"]
    hs = pd.read_parquet(_ROOT / "data" / "processed" / "hotspots.parquet")
    day = hs[hs["date"].astype(str) == flagship]  # 'date' holds datetime.date objects
    meta = pd.read_parquet(_ROOT / "data" / "processed" / "stations_metadata.parquet")
    st = meta[meta["location_id"] == 225648]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for country, col in (("Thailand", _C_TH), ("Myanmar", _C_MM), ("Laos", _C_LA)):
        sub = day[day["country"] == country]
        if len(sub):
            ax.scatter(
                sub["centroid_lon"],
                sub["centroid_lat"],
                s=sub["total_frp"].clip(10, 300),
                c=col,
                alpha=0.5,
                label=f"{country} fire",
                edgecolors="none",
            )
    if len(st):
        ax.scatter(st["lon"], st["lat"], marker="*", s=320, c="black", label="Mae Hong Son station")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"Fire hotspots near Mae Hong Son ({flagship})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, "transboundary_map.png")


def fig_ig() -> None:
    """Horizontal bar of IG feature importance for the Chiang Mai March-2024 case."""
    ig = _load("attribution_march2024.json")["ig_feature_importance"]
    items = sorted(ig.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh([k for k, _ in items], [v for _, v in items], color="#15616D")
    ax.set_xlabel("mean |IG attribution|")
    ax.set_title("Feature importance (Integrated Gradients) — Chiang Mai Mar-2024")
    ax.grid(axis="x", alpha=0.3)
    _save(fig, "ig_feature_importance.png")


def main() -> None:
    _FIG.mkdir(parents=True, exist_ok=True)
    print(f"Writing report figures to {_FIG}")
    fig_rmse_by_horizon()
    fig_significance()
    fig_ablation()
    fig_nwp()
    fig_transboundary_events()
    fig_transboundary_map()
    fig_ig()
    print("done.")


if __name__ == "__main__":
    main()
