# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md

"""Push a 48 h PM2.5 forecast alert to Telegram (live demo).

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.

Runs the live NWP forecast (``app.lib.inference.live_forecast``, air4thai +
Open-Meteo — see docstring there for the hindcast-vs-live distinction), formats
a Thai-language alert per station (``app.lib.telegram.format_alert``), and
sends it to a Telegram chat via the Bot API. Intended for a live demo: run it
in front of judges and a phone in the room gets the push notification.

Requires ``TELEGRAM_BOT_TOKEN`` and ``TELEGRAM_CHAT_ID`` in the environment or
a project-root ``.env`` file (see ``docs/INSTALL.md`` for how to create a bot
via @BotFather and find your chat id). Not required for ``--dry-run``.

Usage:
    uv run python scripts/16_telegram_alert.py --dry-run
    uv run python scripts/16_telegram_alert.py --station all --dry-run
    uv run python scripts/16_telegram_alert.py --station 225579 --force --dry-run
    uv run python scripts/16_telegram_alert.py --station all --threshold 37.5

Exit codes:
    0 — every requested station either sent an alert or had nothing to alert on.
    1 — a live-data, formatting, or Telegram API error occurred for any station.
    2 — configuration error (missing token/chat id when not in --dry-run, or
        an unknown --station id).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd
from dotenv import load_dotenv

from app.lib import data_access as da
from app.lib import telegram
from app.lib.inference import HORIZONS, LiveDataError, conformal_halfwidths, live_forecast

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--station",
        default="all",
        help="OpenAQ station_id to alert on, or 'all' for every curated station (default: all).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=telegram.DEFAULT_ALERT_THRESHOLD_UG,
        help=f"PM2.5 trigger threshold in µg/m³ (default: {telegram.DEFAULT_ALERT_THRESHOLD_UG}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Send an alert for every selected station regardless of the threshold (demo mode).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the message(s) instead of calling the Telegram API. No token/chat id needed.",
    )
    return parser.parse_args(argv)


def _select_station_ids(requested: str, available_ids: list[int]) -> list[int] | None:
    """Resolve ``--station`` to a list of station ids, or None if unknown."""
    if requested == "all":
        return list(available_ids)
    try:
        sid = int(requested)
    except ValueError:
        return None
    return [sid] if sid in available_ids else None


def _process_station(
    sid: int,
    result: dict,
    meta: pd.DataFrame,
    conf: dict,
    args: argparse.Namespace,
    token: str,
    chat_id: str,
) -> str:
    """Check, format, and (dry-run or real) send the alert for one station.

    Returns one of "skipped" (below threshold), "sent" (printed/sent OK), or
    "error" (formatting or Telegram API failure — already logged).
    """
    idx = result["station_ids"].index(sid)
    pred_row = result["pred_ug"][idx, :].tolist()
    observed = float(result["observed"][idx])

    if not telegram.should_alert(pred_row, args.threshold, force=args.force):
        logger.info("station_id=%d: below threshold (%.1f µg/m³) — no alert.", sid, args.threshold)
        return "skipped"

    row = meta.loc[sid]
    halfwidths = conformal_halfwidths(conf, sid, HORIZONS) if conf else None

    station_name = str(row["name"])
    province = str(row["province"])

    try:
        text = telegram.format_alert(
            station_name=station_name,
            province=province,
            observed=observed,
            horizons=HORIZONS,
            pred_ug=pred_row,
            now=result["anchor_iso"],
            halfwidths=halfwidths,
            live_mode=True,
        )
    except ValueError:
        logger.exception("station_id=%d: failed to format alert message.", sid)
        return "error"

    if args.dry_run:
        print(f"\n----- [DRY RUN] station_id={sid} -----")
        print(text)
        return "sent"

    try:
        telegram.send_message(token, chat_id, text)
        logger.info("station_id=%d: alert sent.", sid)
        return "sent"
    except telegram.TelegramSendError:
        logger.exception("station_id=%d: Telegram send failed.", sid)
        return "error"


def main(argv: list[str] | None = None) -> int:
    """Run the live-forecast Telegram alert flow. Returns a process exit code."""
    args = _parse_args(argv)

    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not args.dry_run and (not token or not chat_id):
        logger.error(
            "TELEGRAM_BOT_TOKEN and/or TELEGRAM_CHAT_ID missing from the environment/.env — "
            "set both (see docs/INSTALL.md) or pass --dry-run to preview without sending."
        )
        return 2

    try:
        result = live_forecast()
    except LiveDataError:
        logger.exception("Live forecast unavailable — cannot build alerts.")
        return 1

    station_ids = _select_station_ids(args.station, result["station_ids"])
    if station_ids is None:
        logger.error(
            "--station %r is not a recognised station_id (or 'all'). Known ids: %s",
            args.station,
            result["station_ids"],
        )
        return 2

    meta = da.load_stations_meta().set_index("station_id")
    conf = da.load_conformal()
    if not conf:
        logger.warning(
            "No conformal_intervals.json found — alerts will omit the 90%% interval "
            "(run scripts/15_conformal_calibrate.py to enable it)."
        )

    outcomes = [
        _process_station(sid, result, meta, conf, args, token, chat_id) for sid in station_ids
    ]
    n_sent = outcomes.count("sent")
    had_error = "error" in outcomes

    logger.info(
        "Done: %d station(s) checked, %d alert(s) %s, errors=%s",
        len(station_ids),
        n_sent,
        "printed (dry-run)" if args.dry_run else "sent",
        had_error,
    )
    return 1 if had_error else 0


if __name__ == "__main__":
    sys.exit(main())
