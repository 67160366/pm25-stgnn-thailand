# NSC 2026 หมวด 14 - ระบบพยากรณ์และวิเคราะห์แหล่งกำเนิด PM2.5 (Explainable STGNN)
# พัฒนาโดย นายรณชัย ขาวสะอาด ม.บูรพา; สนับสนุนโดย สวทช.
# เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ข้อตกลงฉบับเต็ม (ไทย/อังกฤษ) ดู README.md

"""One-time builder for the OpenAQ location_id -> air4thai stationID mapping.

This module is part of the NSC 2026 Category 14 entry:
Explainable Spatio-Temporal GNN for PM2.5 in Northern Thailand.

Our 18 curated stations are identified by OpenAQ ``location_id`` /
``sensor_id_pm25`` (see ``data/processed/stations_metadata.parquet``); the
air4thai realtime + history endpoints key on their own ``stationID`` (e.g.
``"36t"``). No column links the two. All 18 are ``provider == "Air4Thai"``,
so this script recovers the link deterministically: it fetches the air4thai
realtime station list (authoritative ``stationID`` + ``lat``/``long`` +
Thai/English name) and, for each of our stations, picks the nearest air4thai
station by great-circle distance, then cross-checks the name.

Match policy (per coordinator decision, Session 13):
  * distance < 500 m AND name plausibly matches  -> auto-accept
  * 500 m <= distance <= 2 km, OR dubious name    -> emit but flag NEEDS VERIFICATION
  * distance > 2 km                                -> no match, fail loudly

The result is written to ``configs/air4thai_station_codes.json`` keyed by our
``location_id`` (int as string), keeping audit fields so the file is
self-documenting. Stations with no confident match are omitted so live mode
can degrade gracefully.

This makes ONE real network call and is meant to be run interactively by a
human (the CLAUDE.md no-network rule applies to tests, not to scripts/ tools).

Usage:
    uv run python scripts/14_air4thai_station_map.py
    uv run python scripts/14_air4thai_station_map.py --insecure   # skip TLS verify
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = _PROJECT_ROOT / "data" / "processed" / "stations_metadata.parquet"
OUTPUT_PATH = _PROJECT_ROOT / "configs" / "air4thai_station_codes.json"

REALTIME_URL = "http://air4thai.pcd.go.th/services/getNewAQI_JSON.php"
_HEADERS = {"User-Agent": "Mozilla/5.0"}

# Distance thresholds (metres).
_AUTO_ACCEPT_M = 500.0
_MAX_MATCH_M = 2000.0

# Tokens dropped before comparing air4thai vs OpenAQ station names.
_STOPWORDS = frozenset(
    {
        "the",
        "of",
        "office",
        "hospital",
        "school",
        "public",
        "park",
        "station",
        "stations",
        "staions",  # air4thai/OpenAQ mis-spelling seen in our metadata
        "center",
        "centre",
        "provincial",
        "municipality",
        "health",
        "promotion",
        "meteorological",
        "meteorology",
        "and",
        "environment",
        "natural",
        "resources",
    }
)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two WGS84 points."""
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _tokens(name: str) -> set[str]:
    """Lowercase alphanumeric content tokens of a name, minus generic stopwords."""
    cleaned = "".join(c if c.isalnum() else " " for c in (name or "").lower())
    return {t for t in cleaned.split() if len(t) > 2 and t not in _STOPWORDS}


def _name_plausible(openaq_name: str, air4thai_en: str, air4thai_area: str) -> bool:
    """True if the OpenAQ name shares a content token with the air4thai name/area."""
    ours = _tokens(openaq_name)
    theirs = _tokens(air4thai_en) | _tokens(air4thai_area)
    return bool(ours & theirs)


def fetch_realtime_stations(*, insecure: bool = False, timeout: int = 30) -> list[dict]:
    """Fetch the air4thai realtime station list.

    Args:
        insecure: Skip TLS certificate verification (some hosts cannot verify
            air4thai's chain). Off by default.
        timeout: Request timeout in seconds.

    Returns:
        List of station dicts (keys: ``stationID``, ``nameEN``, ``areaEN``,
        ``lat``, ``long``, ...).

    Raises:
        requests.HTTPError: On a non-2xx response.
        KeyError: If the payload lacks a ``stations`` array.
    """
    verify = not insecure
    if insecure:
        import urllib3

        urllib3.disable_warnings()
        logger.warning("TLS verification disabled (--insecure).")
    resp = requests.get(REALTIME_URL, headers=_HEADERS, timeout=timeout, verify=verify)
    resp.raise_for_status()
    return resp.json()["stations"]


def build_mapping(meta: pd.DataFrame, stations: list[dict]) -> tuple[dict, list[dict]]:
    """Match each metadata row to its nearest air4thai station.

    Args:
        meta: stations_metadata frame (columns location_id, name, lat, lon).
        stations: air4thai realtime station dicts.

    Returns:
        (mapping, rows) where ``mapping`` is the confident subset keyed by
        ``str(location_id)`` and ``rows`` is the full audit list (one per
        metadata station, including rejects) for printing.
    """
    coords = [(s, float(s["lat"]), float(s["long"])) for s in stations]
    mapping: dict[str, dict] = {}
    rows: list[dict] = []

    for r in meta.itertuples(index=False):
        best_s, best_d = None, math.inf
        for s, slat, slon in coords:
            d = _haversine_m(r.lat, r.lon, slat, slon)
            if d < best_d:
                best_s, best_d = s, d

        plausible = _name_plausible(r.name, best_s["nameEN"], best_s.get("areaEN", ""))
        if best_d > _MAX_MATCH_M:
            status = "NO_MATCH"
        elif best_d < _AUTO_ACCEPT_M and plausible:
            status = "AUTO"
        else:
            status = "NEEDS VERIFICATION"

        row = {
            "location_id": int(r.location_id),
            "openaq_name": r.name,
            "air4thai_id": best_s["stationID"],
            "air4thai_name": best_s["nameEN"],
            "air4thai_area": best_s.get("areaEN", ""),
            "distance_m": round(best_d, 1),
            "name_plausible": plausible,
            "status": status,
        }
        rows.append(row)

        if status != "NO_MATCH":
            mapping[str(int(r.location_id))] = {
                "air4thai_id": best_s["stationID"],
                "air4thai_name": best_s["nameEN"],
                "openaq_name": r.name,
                "distance_m": round(best_d, 1),
                "needs_verification": status == "NEEDS VERIFICATION",
            }

    return mapping, rows


def _print_table(rows: list[dict]) -> None:
    """Print the 18-row audit table to stdout for a human to eyeball."""
    header = (
        f"{'loc_id':>7}  {'a4t':>5}  {'dist_m':>8}  {'nm?':>4}  "
        f"{'status':<18}  openaq_name  |  air4thai_name"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['location_id']:>7}  {r['air4thai_id']:>5}  {r['distance_m']:>8.1f}  "
            f"{('Y' if r['name_plausible'] else 'n'):>4}  {r['status']:<18}  "
            f"{r['openaq_name']}  |  {r['air4thai_name']}"
        )


def main() -> None:
    """Build and write the air4thai station-code mapping."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--insecure", action="store_true", help="Skip TLS certificate verification."
    )
    args = parser.parse_args()

    meta = pd.read_parquet(METADATA_PATH)
    logger.info("Loaded %d stations from %s", len(meta), METADATA_PATH)

    stations = fetch_realtime_stations(insecure=args.insecure)
    logger.info("air4thai realtime returned %d stations", len(stations))

    mapping, rows = build_mapping(meta, stations)
    _print_table(rows)

    n_auto = sum(r["status"] == "AUTO" for r in rows)
    n_flag = sum(r["status"] == "NEEDS VERIFICATION" for r in rows)
    n_none = sum(r["status"] == "NO_MATCH" for r in rows)
    print(f"\nAUTO={n_auto}  NEEDS_VERIFICATION={n_flag}  NO_MATCH={n_none}  (of {len(rows)})")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(mapping, fh, ensure_ascii=False, indent=2)
    logger.info("Wrote %d confident mappings to %s", len(mapping), OUTPUT_PATH)

    if n_none:
        logger.warning("%d station(s) had NO match >2 km — excluded, live mode degrades.", n_none)


if __name__ == "__main__":
    main()
