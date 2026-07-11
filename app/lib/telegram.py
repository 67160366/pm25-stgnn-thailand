# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md
"""Telegram Bot API alerts: format a Thai forecast message and push it.

Turns a live forecast (``app.lib.inference.live_forecast``) into a Thai-language
alert — current reading, 6/12/24/48 h forecast, Thai PCD AQI band + emoji, the
90% split-conformal interval when calibrated, and risk-group health advice —
and sends it via the Telegram Bot API
(``https://api.telegram.org/bot<token>/sendMessage``).

``format_alert`` and ``should_alert`` are pure and unit-tested without network
access; ``send_message`` is the only function that touches the network, wrapped
in :class:`TelegramSendError` so callers get one specific exception to handle.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import pandas as pd
import requests

from app.lib import aqi

logger = logging.getLogger(__name__)

TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"

# Thailand PM2.5 24-hr AQI cut-in used for public advisories: the boundary in
# ``app.lib.aqi._BANDS`` between "ปานกลาง" (moderate) and "เริ่มมีผลต่อกลุ่มเสี่ยง"
# (starts affecting risk groups) — the Thai PCD standard's "เริ่มมีผลกระทบต่อสุขภาพ"
# threshold (ประกาศคณะกรรมการสิ่งแวดล้อมแห่งชาติ ฉบับที่ 43, พ.ศ. 2565). Chosen so the
# alert trigger matches the same band boundary the dashboard already displays,
# rather than introducing a second, inconsistent number.
DEFAULT_ALERT_THRESHOLD_UG: float = 37.5

_TIMEOUT_S: float = 15.0


class TelegramSendError(RuntimeError):
    """Raised when the Telegram Bot API call fails (network, timeout, or API error)."""


def _is_nan(value: float | None) -> bool:
    """True if ``value`` is None or a float NaN."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def _fmt_ug(value: float | None) -> str:
    """Format a µg/m³ value to 1 decimal, or 'N/A' if missing."""
    return "N/A" if _is_nan(value) else f"{value:.1f}"


def should_alert(
    pred_ug: Sequence[float],
    threshold: float = DEFAULT_ALERT_THRESHOLD_UG,
    *,
    force: bool = False,
) -> bool:
    """True if any finite forecast horizon meets/exceeds ``threshold``, or ``force``.

    Args:
        pred_ug: Forecast PM2.5 (µg/m³) for one station, one value per horizon.
            NaN/None entries are ignored (treated as "no decisive signal", not
            as an automatic trigger).
        threshold: Alert cut-in in µg/m³ (see ``DEFAULT_ALERT_THRESHOLD_UG``).
        force: Always return True (demo override), regardless of forecast values.

    Returns:
        True if ``force`` is set, or any finite value in ``pred_ug`` is >= threshold.
        False for an empty ``pred_ug`` (unless ``force``).
    """
    if force:
        return True
    return any(v >= threshold for v in pred_ug if not _is_nan(v))


def format_alert(
    station_name: str,
    province: str | None,
    observed: float | None,
    horizons: Sequence[int],
    pred_ug: Sequence[float],
    now: pd.Timestamp | str,
    *,
    halfwidths: Sequence[float] | None = None,
    live_mode: bool = True,
) -> str:
    """Build the Thai-language Telegram alert message for one station.

    Args:
        station_name: Station display name (e.g. "โรงเรียนยุพราชวิทยาลัย, เชียงใหม่").
        province: Province parsed from the station name
            (``app.lib.data_access._province_from_name``), or None/"" to omit.
        observed: Most recent observed PM2.5 (µg/m³), or None/NaN if unavailable.
        horizons: Forecast horizons in hours, e.g. ``[6, 12, 24, 48]``.
        pred_ug: Forecast PM2.5 (µg/m³), aligned 1:1 with ``horizons``. NaN
            entries are shown as "N/A" and excluded from the overall advice.
        now: Forecast-issue timestamp (any tz, naive treated as UTC, or an ISO
            string — parsed via ``pd.Timestamp``); rendered in Asia/Bangkok
            local time.
        halfwidths: 90% split-conformal half-widths (µg/m³) aligned with
            ``horizons`` (see ``app.lib.inference.conformal_halfwidths``), or
            None to omit the interval (e.g. calibration not yet run).
        live_mode: True if this forecast used the live NWP path — adds a
            caveat line about NWP-driven uncertainty (vs. the hindcast/ERA5 path).

    Returns:
        Plain-text (no HTML/Markdown) Thai message, ready for ``send_message``.

    Raises:
        ValueError: If ``pred_ug`` (or ``halfwidths``) does not have exactly
            one entry per horizon.
    """
    horizons = list(horizons)
    pred_ug = list(pred_ug)
    if len(pred_ug) != len(horizons):
        raise ValueError(f"pred_ug has {len(pred_ug)} values, expected {len(horizons)} horizons")
    if halfwidths is not None and len(halfwidths) != len(horizons):
        raise ValueError(
            f"halfwidths has {len(halfwidths)} values, expected {len(horizons)} horizons"
        )

    now_th = pd.Timestamp(now)
    now_th = now_th.tz_localize("UTC") if now_th.tzinfo is None else now_th.tz_convert("UTC")
    now_th = now_th.tz_convert("Asia/Bangkok")
    ts_label = now_th.strftime("%d/%m/%Y %H:%M น.")

    lines: list[str] = ["🚨 แจ้งเตือนคุณภาพอากาศ PM2.5 (STGNN)"]
    header = station_name if not province else f"{station_name} ({province})"
    lines.append(header)
    lines.append(f"เวลาออกพยากรณ์: {ts_label} (เวลาไทย)")
    lines.append("")

    if _is_nan(observed):
        lines.append("ค่าปัจจุบัน: ไม่มีข้อมูล")
    else:
        lines.append(
            f"ค่าปัจจุบัน: {_fmt_ug(observed)} µg/m³ {aqi.emoji(observed)} {aqi.category(observed)}"
        )
    lines.append("")

    lines.append("พยากรณ์ล่วงหน้า:")
    finite_preds: list[float] = []
    for i, h in enumerate(horizons):
        p = pred_ug[i]
        if _is_nan(p):
            lines.append(f"  {h} ชม. — ไม่มีข้อมูล")
            continue
        finite_preds.append(p)
        row = f"  {h} ชม. — {_fmt_ug(p)} µg/m³ {aqi.emoji(p)} {aqi.category(p)}"
        if halfwidths is not None and not _is_nan(halfwidths[i]):
            row += f" (ช่วง 90%: ±{halfwidths[i]:.1f})"
        lines.append(row)
    lines.append("")

    if finite_preds:
        worst = max(finite_preds)
        lines.append(f"คำแนะนำ ({aqi.category(worst)}): {aqi.health_advice(worst)}")
        lines.append("")

    caveat = "ข้อความนี้เป็นผลพยากรณ์จากโมเดล MTGNN (Explainable STGNN) ไม่ใช่ประกาศทางราชการ"
    if live_mode:
        caveat += (
            " — โหมดสดใช้ข้อมูลพยากรณ์อากาศ (NWP) แทนข้อมูลย้อนหลัง ERA5 "
            "จึงอาจคลาดเคลื่อนมากกว่าผลประเมินย้อนหลังของระบบ"
        )
    lines.append(caveat)
    return "\n".join(lines)


def send_message(
    token: str,
    chat_id: str,
    text: str,
    *,
    parse_mode: str | None = None,
    timeout: float = _TIMEOUT_S,
) -> dict:
    """POST a message to a Telegram chat via the Bot API.

    Args:
        token: Bot token issued by @BotFather.
        chat_id: Target chat id (numeric string/int) or "@channelusername".
        text: Message body. ``format_alert`` emits plain text with no
            HTML/Markdown control characters, which is the stable default
            (``parse_mode=None``) — pass ``parse_mode="HTML"`` only if the
            caller has escaped ``text`` for Telegram's HTML subset first.
        parse_mode: Telegram parse mode ("HTML", "MarkdownV2") or None for
            plain text (default; no escaping required).
        timeout: Request timeout in seconds.

    Returns:
        The parsed JSON response body from Telegram (includes ``result``).

    Raises:
        TelegramSendError: On a network failure, timeout, non-JSON body, or a
            non-2xx / ``"ok": false`` response from the Telegram API.
    """
    url = TELEGRAM_API_URL.format(token=token)
    payload: dict[str, str] = {"chat_id": chat_id, "text": text}
    if parse_mode is not None:
        payload["parse_mode"] = parse_mode

    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.Timeout:
        # `from None`: requests exception text embeds the URL, which contains the
        # bot token — suppress the chain so the token never reaches logs.
        raise TelegramSendError(f"Telegram API timed out after {timeout}s") from None
    except requests.RequestException as exc:
        detail = str(exc).replace(token, "<redacted-token>")
        raise TelegramSendError(f"Telegram API request failed: {detail}") from None

    try:
        body = response.json()
    except ValueError as exc:
        raise TelegramSendError(
            f"Telegram API returned a non-JSON body (HTTP {response.status_code})"
        ) from exc

    if not response.ok or not body.get("ok", False):
        desc = body.get("description", response.text)
        raise TelegramSendError(f"Telegram API error (HTTP {response.status_code}): {desc}")

    result = body.get("result", {})
    logger.info("Telegram alert sent: chat_id=%s message_id=%s", chat_id, result.get("message_id"))
    return body
