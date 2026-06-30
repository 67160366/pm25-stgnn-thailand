# [TODO: NSC Disclaimer - see booklet page 44]
"""Model performance & honesty: RMSE vs baselines, significance, ablation, NWP curve.

Reads precomputed result JSONs (the source of truth that matches the pitch deck). Both result
sets cover the 2025 calendar year but differ by model/training setup, and are labelled honestly:
- evaluation_val2025 / significance_val2025: the pitch model (trained through 2024; 2025 was its
  validation / model-selection year) — the same model the live forecast pages use.
- evaluation_test2025 / significance_test2025: the report model (retrained on 2022-2023; 2025 is a
  genuine held-out test).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.lib import data_access as da
from app.lib import ui

_HK = ["6h", "12h", "24h", "48h"]


def _rmse_bar(persistence: dict, methods: dict[str, dict]) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=_HK, y=[persistence.get(h) for h in _HK], name="Persistence", marker_color="#888")
    for label, d in methods.items():
        fig.add_bar(x=_HK, y=[d.get(h) for h in _HK], name=label)
    fig.update_layout(
        barmode="group",
        title="RMSE ต่อช่วงเวลา (ยิ่งต่ำยิ่งดี)",
        yaxis_title="RMSE (µg/m³)",
        xaxis_title="ช่วงเวลาพยากรณ์",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.04),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def render() -> None:
    """Render the performance & honesty page."""
    ui.page_title(
        "ประสิทธิภาพและความซื่อสัตย์ของโมเดล",
        "รายงานผลตามจริง รวมผลที่ไม่โดดเด่น เพื่อความโปร่งใส (ธรรมาภิบาล AI)",
    )
    setups = {
        "โมเดลรายงาน — ปี 2025 (held-out test)": (
            "evaluation_test2025.json",
            "significance_test2025.json",
            True,
        ),
        "โมเดลหลัก (pitch) — ปี 2025 (validation)": (
            "evaluation_val2025.json",
            "significance_val2025.json",
            False,
        ),
    }
    choice = st.radio("ชุดผลที่ต้องการดู", list(setups), horizontal=True)
    ev_file, sig_file, is_heldout = setups[choice]
    ev = da.load_output_json(ev_file)
    sig = da.load_output_json(sig_file)
    if not ev:
        st.warning("ไม่พบไฟล์ผลการประเมิน")
        return

    persistence = ev["persistence_rmse_ug_m3"]
    methods: dict[str, dict] = {
        "MTGNN (กราฟ)": ev["mtgnn"]["rmse_ug_m3"],
        "A3TGCN": ev["a3tgcn"]["rmse_ug_m3"],
    }
    if is_heldout:
        gbm = da.load_output_json("baseline_ml_test.json")
        if gbm:
            methods["GBM (ไม่ใช้กราฟ)"] = gbm["baseline_ml_rmse_ug_m3"]
    methods["Hybrid (ใช้งานจริง)"] = ev["hybrid_operational"]["rmse_ug_m3"]

    st.subheader("ความแม่นยำเทียบ baseline")
    st.caption(
        "วัดบนปี 2025 — โมเดลรายงานเทรนใหม่บน 2022–2023 (2025 เป็น held-out ไม่เคยเห็นตอนเทรน)"
        if is_heldout
        else "วัดบนปี 2025 ที่เป็นชุด validation ของโมเดลหลัก (เทรนถึงปี 2024) — โมเดลเดียวกับหน้าพยากรณ์"
    )
    st.plotly_chart(_rmse_bar(persistence, methods), width="stretch")

    if sig:
        h48 = sig["per_horizon"]["48h"]
        ci = h48["pct_improvement_ci95"]
        st.subheader("นัยสำคัญทางสถิติ (ที่ 48 ชม.)")
        st.markdown(
            f"MTGNN ดีกว่า persistence ที่ 48 ชม. **{h48['pct_improvement_point']:+.1f}%** "
            f"แต่ช่วงความเชื่อมั่น 95% = **[{ci[0]:.1f}%, {ci[1]:.1f}%]** ซึ่ง**คร่อม 0 → ยังไม่ significant**"
        )
        st.caption("เราจึงไม่เคลมว่าชนะอย่างมีนัยสำคัญในปีเดียว และนำเสนอด้วย XAI เป็นหลัก")

    ab = da.load_output_json("ablation_multiseed.json")
    if ab:
        st.subheader("Ablation ของ 3 จุดใหม่ (multi-seed, held-out test)")
        full = ab["variants_mean_std_ug_m3"]["full"]
        rows = []
        for name, d in ab["deltas_vs_full"].items():
            row = {"ถอดส่วนออก": name}
            row.update({f"Δ{_HK[i]}": round(d["delta_vs_full"][i], 2) for i in range(4)})
            row["เกิน noise?"] = "ใช่" if any(d["robust_beyond_noise"]) else "ไม่"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        mean_std = " · ".join(
            f"{_HK[i]} {full['mean'][i]:.2f}±{full['std'][i]:.2f}" for i in range(4)
        )
        st.caption(
            f"full RMSE: {mean_std} µg/m³ · ไม่มีส่วนใดต่างจาก full เกิน training-seed noise → "
            "กราฟไม่ได้ช่วยความแม่นยำอย่างมีนัย (คุณค่าที่พิสูจน์ได้คือ attribution ไม่ใช่ RMSE)"
        )

    nwp = da.load_output_json("nwp_sensitivity.json")
    if nwp:
        st.subheader("ความทนต่อความคลาดเคลื่อนของพยากรณ์อากาศ (NWP)")
        res = nwp["results"]
        xs = [r["noise_fraction"] for r in res]
        fig = go.Figure()
        for h in _HK:
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=[r["rmse_ug_m3"][h] for r in res],
                    mode="lines+markers",
                    name=f"MTGNN {h}",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=[nwp["persistence_rmse_ug_m3"]["48h"]] * len(xs),
                mode="lines",
                name="persistence 48h",
                line=dict(dash="dot", color="#888"),
            )
        )
        fig.update_layout(
            xaxis_title="ระดับ noise ของ ERA5 (× ความผันผวนธรรมชาติ)",
            yaxis_title="RMSE (µg/m³)",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.04),
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig, width="stretch")
        cf = nwp["crossover_fraction"]
        st.caption(
            f"โมเดลเริ่มแพ้ persistence: 24h ที่ ~{cf['24h']}×, 48h ที่ ~{cf['48h']}× ของความผันผวนธรรมชาติ "
            "→ ระบบจริงควรใช้ NWP คุณภาพสูงและพึ่งช่วง 48h"
        )

    st.info(
        "🏛️ **สรุปตามจริง:** ความได้เปรียบเชิงพยากรณ์มีจำกัด (เด่นที่ 48h และยังไม่ significant) "
        "และกราฟไม่ได้ช่วยความแม่นยำเหนือ training-seed noise — คุณค่าที่พิสูจน์ได้คือ "
        "**ความสามารถอธิบายและระบุแหล่งกำเนิด** การรายงานผลตามจริงนี้คือจุดแข็งด้านธรรมาภิบาล AI"
    )
