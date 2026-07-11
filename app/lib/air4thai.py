# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md
"""Live PM2.5 access from the air4thai public endpoints (dashboard-only).

The air4thai history endpoint has a ~90-day rolling window — enough to fill a
24 h input window for live forecasting, but NOT usable for model training (see
CLAUDE.md data-source notes). Station codes come from the offline-built map in
``configs/air4thai_station_codes.json`` (see ``scripts/14_air4thai_station_map.py``);
our OpenAQ ``location_id`` is the key, the air4thai ``stationID`` the value.

``DATETIMEDATA`` from the history endpoint is Asia/Bangkok local time; every
timestamp here is converted to tz-aware UTC to match the model's hourly grid.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATION_CODE_MAP_PATH = _PROJECT_ROOT / "configs" / "air4thai_station_codes.json"

HISTORY_URL = "http://air4thai.com/forweb/getHistoryData.php"
_HEADERS = {"User-Agent": "Mozilla/5.0"}
_BANGKOK_TZ = "Asia/Bangkok"


def load_station_code_map(path: Path = STATION_CODE_MAP_PATH) -> dict[int, str]:
    """Load the OpenAQ ``location_id`` -> air4thai ``stationID`` mapping.

    Args:
        path: Path to ``configs/air4thai_station_codes.json``.

    Returns:
        Dict mapping our integer station_id to the air4thai code (e.g. ``"36t"``).
        Empty if the file is absent (live mode then degrades gracefully).
    """
    if not path.exists():
        logger.warning("air4thai station code map not found at %s", path)
        return {}
    with path.open(encoding="utf-8") as fh:
        raw: dict[str, dict] = json.load(fh)
    return {int(k): v["air4thai_id"] for k, v in raw.items()}


def _get_json(url: str, params: dict) -> dict:
    """GET ``url`` and parse JSON, tolerating a BOM and a broken TLS chain.

    air4thai hosts redirect http->https but some clients cannot verify their
    certificate chain; on an SSL error we retry once without verification (a
    read-only public endpoint, no credentials sent). The response may carry a
    UTF-8 BOM, so it is decoded with ``utf-8-sig``.

    Args:
        url: Endpoint URL.
        params: Query parameters.

    Returns:
        Parsed JSON as a dict.

    Raises:
        requests.HTTPError: On a non-2xx response.
    """
    try:
        resp = requests.get(url, headers=_HEADERS, params=params, timeout=30)
    except requests.exceptions.SSLError:
        import urllib3

        urllib3.disable_warnings()
        logger.warning("TLS verification failed for %s; retrying without verify.", url)
        # Read-only public endpoint, no credentials sent; documented in docs/api_quirks.md.
        resp = requests.get(
            url, headers=_HEADERS, params=params, timeout=30, verify=False  # noqa: S501
        )
    resp.raise_for_status()
    return json.loads(resp.content.decode("utf-8-sig"))


def fetch_history(air4thai_id: str, start: date, end: date) -> pd.DataFrame:
    """Fetch hourly PM2.5 history for one air4thai station.

    Args:
        air4thai_id: air4thai ``stationID`` (e.g. ``"36t"``).
        start: First calendar day (Asia/Bangkok) to request.
        end: Last calendar day (Asia/Bangkok) to request.

    Returns:
        DataFrame with columns ``time`` (tz-aware UTC) and ``pm25`` (float,
        negative/sentinel values become NaN). Empty if the station returns no
        rows.

    Raises:
        requests.HTTPError: On a non-2xx response.
    """
    params = {
        "stationID": air4thai_id,
        "param": "PM25",
        "type": "hr",
        "sdate": start.isoformat(),
        "edate": end.isoformat(),
        "stime": "00",
        "etime": "23",
    }
    payload = _get_json(HISTORY_URL, params)
    stations = payload.get("stations") or []
    if not stations:
        logger.warning("air4thai history: no station block for %s", air4thai_id)
        return pd.DataFrame({"time": pd.to_datetime([], utc=True), "pm25": []})

    rows = stations[0].get("data") or []
    if not rows:
        return pd.DataFrame({"time": pd.to_datetime([], utc=True), "pm25": []})

    df = pd.DataFrame(rows)
    local = pd.to_datetime(df["DATETIMEDATA"]).dt.tz_localize(
        _BANGKOK_TZ, ambiguous="NaT", nonexistent="NaT"
    )
    pm25 = pd.to_numeric(df["PM25"], errors="coerce")
    pm25 = pm25.where(pm25 >= 0)  # -1 / negative sentinels -> NaN
    out = pd.DataFrame({"time": local.dt.tz_convert("UTC"), "pm25": pm25})
    return out.dropna(subset=["time"]).reset_index(drop=True)


def fetch_history_stations(code_map: dict[int, str], start: date, end: date) -> pd.DataFrame:
    """Fetch PM2.5 history for every mapped station.

    Stations whose individual call fails are logged and skipped so one dead
    station does not sink the whole live view.

    Args:
        code_map: OpenAQ station_id -> air4thai code (from
            :func:`load_station_code_map`).
        start: First calendar day (Asia/Bangkok).
        end: Last calendar day (Asia/Bangkok).

    Returns:
        Long DataFrame with columns ``station_id``, ``time`` (UTC), ``pm25``.
    """
    frames: list[pd.DataFrame] = []
    for station_id, code in code_map.items():
        try:
            df = fetch_history(code, start, end)
        except requests.RequestException as exc:
            logger.warning("air4thai history failed for %s (%s): %s", station_id, code, exc)
            continue
        if df.empty:
            continue
        df.insert(0, "station_id", station_id)
        frames.append(df)

    if not frames:
        return pd.DataFrame({"station_id": [], "time": pd.to_datetime([], utc=True), "pm25": []})
    return pd.concat(frames, ignore_index=True)
