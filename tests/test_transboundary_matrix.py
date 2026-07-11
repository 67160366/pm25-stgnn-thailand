"""
ข้อตกลงการใช้ซอฟต์แวร์ (NSC/สวทช.): เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ฉบับเต็มดู README.md

Unit tests for ``src/explain/transboundary_events.py`` and the pure ``build_matrix_json``
in ``scripts/20_transboundary_matrix.py`` (digit-prefixed -> loaded by file path, mirroring
``tests/test_eval_2026.py::_load_eval_module``).

All tests use synthetic in-memory / tmp_path data and a duck-typed fake dataset -- no real
checkpoints, no real dataset/hotspots parquet from data/processed, no network.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from src.explain.transboundary_events import (
    nearest_foreign_distance_km,
    select_border_stations,
    select_connected_foreign_events,
)

_ROOT = Path(__file__).resolve().parent.parent


def _load_script20():
    """Import the digit-prefixed scripts/20_transboundary_matrix.py by file path."""
    spec = importlib.util.spec_from_file_location(
        "script20_transboundary_matrix", _ROOT / "scripts" / "20_transboundary_matrix.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


script20 = _load_script20()


# ---------------------------------------------------------------------------
# Synthetic borders geojson
# ---------------------------------------------------------------------------


def _write_synthetic_borders(tmp_path: Path) -> Path:
    """A minimal geojson with one Myanmar box (west) and one Laos box (east)."""
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "id": "THA",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[98.0, 18.0], [98.0, 18.5], [100.5, 18.5], [100.5, 18.0], [98.0, 18.0]]
                    ],
                },
            },
            {
                "id": "MMR",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[97.0, 18.0], [97.0, 18.5], [97.5, 18.5], [97.5, 18.0], [97.0, 18.0]]
                    ],
                },
            },
            {
                "id": "LAO",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[101.0, 18.0], [101.0, 18.5], [101.5, 18.5], [101.5, 18.0], [101.0, 18.0]]
                    ],
                },
            },
        ],
    }
    path = tmp_path / "synthetic_borders.geojson"
    path.write_text(json.dumps(geojson), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# nearest_foreign_distance_km
# ---------------------------------------------------------------------------


class TestNearestForeignDistanceKm:
    def test_point_near_myanmar_is_closest_to_myanmar(self, tmp_path: Path) -> None:
        borders_path = _write_synthetic_borders(tmp_path)
        # Close to the (97.5, 18.0) corner vertex -> small distance, dominated by the lon gap.
        dist_km, country = nearest_foreign_distance_km(18.0, 97.55, borders_path=borders_path)
        assert country == "Myanmar"
        assert dist_km < 10.0

    def test_point_near_laos_is_closest_to_laos(self, tmp_path: Path) -> None:
        borders_path = _write_synthetic_borders(tmp_path)
        # Close to the (101.0, 18.0) corner vertex.
        dist_km, country = nearest_foreign_distance_km(18.0, 100.95, borders_path=borders_path)
        assert country == "Laos"
        assert dist_km < 10.0

    def test_monotonic_distance_ordering(self, tmp_path: Path) -> None:
        """A point farther from the Myanmar box reports a larger distance than a near one."""
        borders_path = _write_synthetic_borders(tmp_path)
        near_dist, near_country = nearest_foreign_distance_km(
            18.25, 97.6, borders_path=borders_path
        )
        far_dist, _far_country = nearest_foreign_distance_km(18.25, 98.5, borders_path=borders_path)
        assert near_country == "Myanmar"
        assert far_dist > near_dist

    def test_ignores_thailand_polygon(self, tmp_path: Path) -> None:
        """A point deep inside the Thailand box is still measured to Myanmar/Laos, not 0."""
        borders_path = _write_synthetic_borders(tmp_path)
        dist_km, country = nearest_foreign_distance_km(18.25, 99.25, borders_path=borders_path)
        assert country in {"Myanmar", "Laos"}
        assert dist_km > 50.0


# ---------------------------------------------------------------------------
# select_border_stations
# ---------------------------------------------------------------------------


class TestSelectBorderStations:
    def _metadata(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "location_id": [1001, 1002],
                "name": ["Border Station", "Interior Station"],
                "lat": [18.25, 18.25],
                "lon": [97.55, 99.25],  # near Myanmar box vs. deep interior
            }
        )

    def test_only_border_station_selected_at_small_threshold(self, tmp_path: Path) -> None:
        borders_path = _write_synthetic_borders(tmp_path)
        rows = select_border_stations(self._metadata(), max_dist_km=50.0, borders_path=borders_path)
        assert [r["station_id"] for r in rows] == [1001]
        assert rows[0]["nearest_country"] == "Myanmar"

    def test_both_stations_selected_at_large_threshold_sorted_by_distance(
        self, tmp_path: Path
    ) -> None:
        borders_path = _write_synthetic_borders(tmp_path)
        rows = select_border_stations(
            self._metadata(), max_dist_km=1000.0, borders_path=borders_path
        )
        assert [r["station_id"] for r in rows] == [1001, 1002]
        assert rows[0]["dist_km_nearest_foreign"] <= rows[1]["dist_km_nearest_foreign"]

    def test_accepts_station_id_column_name_too(self, tmp_path: Path) -> None:
        """location_id vs station_id naming (SPEC risk #8) -- both must work."""
        borders_path = _write_synthetic_borders(tmp_path)
        meta = self._metadata().rename(columns={"location_id": "station_id"})
        rows = select_border_stations(meta, max_dist_km=50.0, borders_path=borders_path)
        assert [r["station_id"] for r in rows] == [1001]

    def test_result_schema(self, tmp_path: Path) -> None:
        borders_path = _write_synthetic_borders(tmp_path)
        rows = select_border_stations(
            self._metadata(), max_dist_km=1000.0, borders_path=borders_path
        )
        for row in rows:
            assert set(row.keys()) == {
                "station_id",
                "name",
                "lat",
                "lon",
                "dist_km_nearest_foreign",
                "nearest_country",
            }


# ---------------------------------------------------------------------------
# select_connected_foreign_events
# ---------------------------------------------------------------------------


def _make_event_sample(
    frp_countries: list[tuple[float, str]], connected_idxs: list[int]
) -> HeteroData:
    """Build a single-station HeteroData sample with hotspot type_c edges to station 0.

    Args:
        frp_countries: One (frp, country) pair per hotspot node.
        connected_idxs: Hotspot indices whose type_c edge reaches station 0.
    """
    data = HeteroData()
    data["station"].x = torch.zeros(1, 2, 3)
    data["hotspot"].x = torch.tensor(
        [[frp, 0.0, 0.0] for frp, _c in frp_countries], dtype=torch.float32
    )
    data["hotspot"].country = [c for _f, c in frp_countries]
    src = torch.tensor(connected_idxs, dtype=torch.long)
    dst = torch.zeros(len(connected_idxs), dtype=torch.long)
    data["hotspot", "type_c", "station"].edge_index = (
        torch.stack([src, dst]) if connected_idxs else (torch.zeros(2, 0, dtype=torch.long))
    )
    data["station", "spatial", "station"].edge_index = torch.zeros(2, 0, dtype=torch.long)
    return data


class _FakeDataset:
    """Duck-typed PM25GraphDataset stand-in exposing only what script 10's helpers touch."""

    def __init__(
        self,
        timestamps: pd.DatetimeIndex,
        anchor_indices: np.ndarray,
        pm25_raw: np.ndarray,
        station_ids: np.ndarray,
        samples: dict[int, HeteroData],
    ) -> None:
        self._timestamps = timestamps
        self._anchor_indices = anchor_indices
        self._pm25_raw = pm25_raw
        self._station_ids = station_ids
        self._samples = samples

    def __getitem__(self, pos: int) -> HeteroData:
        return self._samples[pos]


def _build_fixture() -> tuple[_FakeDataset, pd.DataFrame]:
    """3 candidate dates where total-parquet-FRP ranking differs from connected-FRP ranking.

    date 2025-03-01 ("A"): parquet foreign total = 90 + 10 = 100, but only the 10-FRP
      hotspot is type_c-connected to the station -> connected foreign FRP = 10.
    date 2025-03-02 ("B"): parquet foreign total = 50, fully connected -> connected = 50.
    date 2025-03-03 ("C"): parquet foreign total = 80, fully connected -> connected = 80.

    Expected connected-FRP ranking (descending): C (80) > B (50) > A (10) -- the OPPOSITE
    order from the raw parquet total (A=100 > C=80 > B=50), so a test asserting the
    connected ordering also proves the code ranks by connected FRP, not raw FRP.
    """
    timestamps = pd.date_range("2025-03-01", periods=3, freq="1D", tz="UTC")
    anchor_indices = np.array([0, 1, 2])
    pm25_raw = np.array([[20.0], [15.0], [30.0]])  # peak per date, station idx 0
    station_ids = np.array([500])

    samples = {
        0: _make_event_sample([(90.0, "Myanmar"), (10.0, "Myanmar")], connected_idxs=[1]),
        1: _make_event_sample([(50.0, "Myanmar")], connected_idxs=[0]),
        2: _make_event_sample([(80.0, "Laos")], connected_idxs=[0]),
    }
    ds = _FakeDataset(timestamps, anchor_indices, pm25_raw, station_ids, samples)

    hotspots = pd.DataFrame(
        {
            "date": [
                "2025-03-01",
                "2025-03-01",
                "2025-03-02",
                "2025-03-03",
            ],
            "total_frp": [90.0, 10.0, 50.0, 80.0],
            "country": ["Myanmar", "Myanmar", "Myanmar", "Laos"],
        }
    )
    return ds, hotspots


class TestSelectConnectedForeignEvents:
    def test_ranks_by_connected_foreign_frp_not_raw_frp(self, tmp_path: Path) -> None:
        ds, hotspots = _build_fixture()
        hotspots_path = tmp_path / "hotspots.parquet"
        hotspots.to_parquet(hotspots_path, index=False)

        events = select_connected_foreign_events(
            ds, station_idx=0, hotspots_path=hotspots_path, split="test", n_candidates=15, top_k=5
        )

        assert [round(e[0], 1) for e in events] == [80.0, 50.0, 10.0]
        assert [e[2] for e in events] == [date(2025, 3, 3), date(2025, 3, 2), date(2025, 3, 1)]

    def test_anchor_positions_and_peak_match_injected_values(self, tmp_path: Path) -> None:
        ds, hotspots = _build_fixture()
        hotspots_path = tmp_path / "hotspots.parquet"
        hotspots.to_parquet(hotspots_path, index=False)

        events = select_connected_foreign_events(
            ds, station_idx=0, hotspots_path=hotspots_path, split="test", n_candidates=15, top_k=5
        )
        by_date = {e[2]: (e[1], e[3]) for e in events}
        assert by_date[date(2025, 3, 1)] == (0, 20.0)
        assert by_date[date(2025, 3, 2)] == (1, 15.0)
        assert by_date[date(2025, 3, 3)] == (2, 30.0)

    def test_top_k_truncates(self, tmp_path: Path) -> None:
        ds, hotspots = _build_fixture()
        hotspots_path = tmp_path / "hotspots.parquet"
        hotspots.to_parquet(hotspots_path, index=False)

        events = select_connected_foreign_events(
            ds, station_idx=0, hotspots_path=hotspots_path, split="test", n_candidates=15, top_k=2
        )
        assert len(events) == 2

    def test_determinism_same_inputs_same_events(self, tmp_path: Path) -> None:
        """Model-independence guard: identical inputs must yield identical event lists."""
        ds, hotspots = _build_fixture()
        hotspots_path = tmp_path / "hotspots.parquet"
        hotspots.to_parquet(hotspots_path, index=False)

        events_1 = select_connected_foreign_events(
            ds, station_idx=0, hotspots_path=hotspots_path, split="test", n_candidates=15, top_k=5
        )
        events_2 = select_connected_foreign_events(
            ds, station_idx=0, hotspots_path=hotspots_path, split="test", n_candidates=15, top_k=5
        )
        assert events_1 == events_2


# ---------------------------------------------------------------------------
# build_matrix_json (script 20, pure function)
# ---------------------------------------------------------------------------


class TestBuildMatrixJson:
    def test_required_top_level_keys_present(self) -> None:
        stations = [
            {
                "station_id": 1,
                "name": "S1",
                "lat": 18.0,
                "lon": 99.0,
                "dist_km_nearest_foreign": 5.0,
                "nearest_country": "Myanmar",
            }
        ]
        results = [
            {
                "station_id": 1,
                "station_name": "S1",
                "checkpoint_label": "demo",
                "checkpoint": "checkpoints/mtgnn/best_model.pt",
                "n_events_with_connected_foreign_fire": 1,
                "events": [{"date": "2025-03-01"}],
            },
        ]
        meta_params = {
            "method": "m",
            "split": "test",
            "horizon_h": 24,
            "horizon_idx": 2,
            "n_ig_steps": 50,
            "n_candidates": 15,
            "top_k": 5,
            "wind_mode": "constant_ne",
            "max_dist_km": 50.0,
        }
        payload = script20.build_matrix_json(stations, results, meta_params, ["caveat 1"])

        for key in (
            "generated_by",
            "method",
            "split",
            "horizon_h",
            "horizon_idx",
            "n_ig_steps",
            "n_candidates",
            "top_k",
            "wind_mode",
            "max_dist_km",
            "checkpoints",
            "stations",
            "results",
            "caveats",
        ):
            assert key in payload

    def test_results_length_equals_n_stations_times_n_checkpoints(self) -> None:
        stations = [
            {
                "station_id": i,
                "name": f"S{i}",
                "lat": 18.0,
                "lon": 99.0,
                "dist_km_nearest_foreign": 5.0,
                "nearest_country": "Myanmar",
            }
            for i in range(3)
        ]
        results = [
            {
                "station_id": s["station_id"],
                "station_name": s["name"],
                "checkpoint_label": ckpt,
                "checkpoint": f"checkpoints/{ckpt}.pt",
                "n_events_with_connected_foreign_fire": 0,
                "events": [],
            }
            for s in stations
            for ckpt in ("demo", "report")
        ]
        meta_params = {
            "method": "m",
            "split": "test",
            "horizon_h": 24,
            "horizon_idx": 2,
            "n_ig_steps": 50,
            "n_candidates": 15,
            "top_k": 5,
            "wind_mode": "constant_ne",
            "max_dist_km": 50.0,
        }
        payload = script20.build_matrix_json(stations, results, meta_params, [])

        assert len(payload["results"]) == len(stations) * 2

    def test_events_passed_through_unchanged(self) -> None:
        events = [{"date": "2025-03-18", "foreign_attribution": 0.627}]
        stations: list[dict] = []
        results = [
            {
                "station_id": 225648,
                "station_name": "Mae Hong Son",
                "checkpoint_label": "report",
                "checkpoint": "checkpoints_split2/mtgnn/best_model.pt",
                "n_events_with_connected_foreign_fire": 1,
                "events": events,
            },
        ]
        meta_params = {
            "method": "m",
            "split": "test",
            "horizon_h": 24,
            "horizon_idx": 2,
            "n_ig_steps": 50,
            "n_candidates": 15,
            "top_k": 5,
            "wind_mode": "constant_ne",
            "max_dist_km": 50.0,
        }
        payload = script20.build_matrix_json(stations, results, meta_params, [])

        assert payload["results"][0]["events"] == events

    def test_caveats_present_and_preserved(self) -> None:
        caveats = ["caveat one", "caveat two"]
        payload = script20.build_matrix_json(
            [],
            [],
            {
                "method": "m",
                "split": "test",
                "horizon_h": 24,
                "horizon_idx": 2,
                "n_ig_steps": 50,
                "n_candidates": 15,
                "top_k": 5,
                "wind_mode": "constant_ne",
                "max_dist_km": 50.0,
            },
            caveats,
        )
        assert payload["caveats"] == caveats

    def test_checkpoints_are_the_two_fixed_labels(self) -> None:
        payload = script20.build_matrix_json(
            [],
            [],
            {
                "method": "m",
                "split": "test",
                "horizon_h": 24,
                "horizon_idx": 2,
                "n_ig_steps": 50,
                "n_candidates": 15,
                "top_k": 5,
                "wind_mode": "constant_ne",
                "max_dist_km": 50.0,
            },
            [],
        )
        labels = {c["label"] for c in payload["checkpoints"]}
        assert labels == {"demo", "report"}


# ---------------------------------------------------------------------------
# CLI arg splitting
# ---------------------------------------------------------------------------


class TestSplitCliArgs:
    def test_local_keys_separated_from_overrides(self) -> None:
        argv = ["max_dist_km=70", "top_k=3", "model.hidden_dim=64"]
        local, overrides = script20._split_cli_args(argv)
        assert local == {"max_dist_km": "70", "top_k": "3"}
        assert overrides == ["model.hidden_dim=64"]

    def test_args_without_equals_are_ignored(self) -> None:
        local, overrides = script20._split_cli_args(["--help", "split=test"])
        assert local == {"split": "test"}
        assert overrides == []
