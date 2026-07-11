"""
ข้อตกลงการใช้ซอฟต์แวร์ (NSC/สวทช.): เผยแพร่ตามต้นฉบับ ไม่รับประกันความเสียหาย; ฉบับเต็มดู README.md

Unit tests for the pure ``aggregate_event_across_seeds`` / ``build_uncertainty_json``
functions in ``scripts/21_transboundary_uncertainty.py`` (digit-prefixed -> loaded by
file path, mirroring ``tests/test_eval_2026.py::_load_eval_module``).

All tests use synthetic in-memory data and hand-written per-seed reports -- no real
checkpoints, no real dataset/hotspots parquet from data/processed, no network.
"""

from __future__ import annotations

import importlib.util
import statistics
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent


def _load_script21():
    """Import the digit-prefixed scripts/21_transboundary_uncertainty.py by file path."""
    spec = importlib.util.spec_from_file_location(
        "script21_transboundary_uncertainty",
        _ROOT / "scripts" / "21_transboundary_uncertainty.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


script21 = _load_script21()


# ---------------------------------------------------------------------------
# aggregate_event_across_seeds
# ---------------------------------------------------------------------------


class TestAggregateEventAcrossSeeds:
    def _reports(self, foreign_vals: list[float], country_attrs: list[dict[str, float]]):
        labels = ["report_orig", "full_s0", "full_s1"]
        return labels, {
            label: {"foreign_attribution": f, "country_attribution": c}
            for label, f, c in zip(labels, foreign_vals, country_attrs, strict=True)
        }

    def test_mean_min_max_range(self) -> None:
        labels, reports = self._reports(
            [0.6, 0.0, 0.3],
            [
                {"Myanmar": 0.6, "Thailand": 0.4},
                {"Thailand": 1.0},
                {"Myanmar": 0.3, "Thailand": 0.7},
            ],
        )
        agg = script21.aggregate_event_across_seeds(labels, reports)
        assert agg["foreign_attribution_mean"] == round(statistics.fmean([0.6, 0.0, 0.3]), 3)
        assert agg["foreign_attribution_min"] == 0.0
        assert agg["foreign_attribution_max"] == 0.6
        assert round(agg["foreign_attribution_max"] - agg["foreign_attribution_min"], 3) == 0.6

    def test_std_matches_statistics_stdev_ddof1(self) -> None:
        labels, reports = self._reports(
            [0.6, 0.0, 0.3],
            [{"Thailand": 1.0}, {"Thailand": 1.0}, {"Thailand": 1.0}],
        )
        agg = script21.aggregate_event_across_seeds(labels, reports)
        expected_std = statistics.stdev([0.6, 0.0, 0.3])
        assert agg["foreign_attribution_std"] == round(expected_std, 3)

    def test_std_matches_numpy_ddof1(self) -> None:
        labels, reports = self._reports(
            [0.1, 0.5, 0.9],
            [{"Thailand": 1.0}, {"Thailand": 1.0}, {"Thailand": 1.0}],
        )
        agg = script21.aggregate_event_across_seeds(labels, reports)
        expected_std = float(np.std([0.1, 0.5, 0.9], ddof=1))
        assert agg["foreign_attribution_std"] == round(expected_std, 3)

    def test_country_present_in_only_one_seed_treated_as_zero_elsewhere(self) -> None:
        labels, reports = self._reports(
            [0.5, 0.2, 0.3],
            [
                {"Myanmar": 0.5},  # Laos absent here -> 0.0
                {"Myanmar": 0.1, "Laos": 0.1},
                {"Myanmar": 0.2, "Laos": 0.1},
            ],
        )
        agg = script21.aggregate_event_across_seeds(labels, reports)
        assert set(agg["country_attribution_mean"].keys()) == {"Myanmar", "Laos"}
        expected_laos_mean = statistics.fmean([0.0, 0.1, 0.1])
        assert agg["country_attribution_mean"]["Laos"] == round(expected_laos_mean, 3)
        expected_laos_std = statistics.stdev([0.0, 0.1, 0.1])
        assert agg["country_attribution_std"]["Laos"] == round(expected_laos_std, 3)

    def test_all_zero_seeds_no_div_by_zero(self) -> None:
        labels, reports = self._reports(
            [0.0, 0.0, 0.0],
            [{"Thailand": 1.0}, {"Thailand": 1.0}, {"Thailand": 1.0}],
        )
        agg = script21.aggregate_event_across_seeds(labels, reports)
        assert agg["foreign_attribution_mean"] == 0.0
        assert agg["foreign_attribution_std"] == 0.0
        assert agg["foreign_attribution_min"] == 0.0
        assert agg["foreign_attribution_max"] == 0.0
        assert agg["country_attribution_mean"]["Thailand"] == 1.0
        assert agg["country_attribution_std"]["Thailand"] == 0.0

    def test_single_seed_std_is_zero_no_error(self) -> None:
        """Fewer than 2 values must not raise statistics.StatisticsError."""
        agg = script21.aggregate_event_across_seeds(
            ["only"], {"only": {"foreign_attribution": 0.4, "country_attribution": {"Laos": 0.4}}}
        )
        assert agg["foreign_attribution_mean"] == 0.4
        assert agg["foreign_attribution_std"] == 0.0
        assert agg["foreign_attribution_min"] == 0.4
        assert agg["foreign_attribution_max"] == 0.4


# ---------------------------------------------------------------------------
# build_uncertainty_json
# ---------------------------------------------------------------------------


_SEEDS = [
    {"label": "report_orig", "path": "checkpoints_split2/mtgnn/best_model.pt"},
    {"label": "full_s0", "path": "checkpoints_split2_seeds/full_s0/best_model.pt"},
    {"label": "full_s1", "path": "checkpoints_split2_seeds/full_s1/best_model.pt"},
]

_META_PARAMS = {
    "method": "m",
    "split": "test",
    "horizon_h": 24,
    "horizon_idx": 2,
    "n_ig_steps": 50,
    "n_candidates": 15,
    "top_k": 5,
    "wind_mode": "constant_ne",
}


def _fake_event_agg(
    date: str, foreign_min: float, foreign_max: float, foreign_mean: float | None = None
) -> dict[str, object]:
    mean_v = foreign_mean if foreign_mean is not None else (foreign_min + foreign_max) / 2
    return {
        "date": date,
        "peak_pm25_ug_m3": 50.0,
        "connected_foreign_fraction": 0.5,
        "connected_frp_by_country": {"Myanmar": 100.0, "Thailand": 100.0},
        "per_seed": {
            "report_orig": {"country_attribution": {"Thailand": 1.0}, "foreign_attribution": 0.0},
            "full_s0": {"country_attribution": {"Thailand": 1.0}, "foreign_attribution": 0.0},
            "full_s1": {"country_attribution": {"Thailand": 1.0}, "foreign_attribution": 0.0},
        },
        "foreign_attribution_mean": mean_v,
        "foreign_attribution_std": 0.1,
        "foreign_attribution_min": foreign_min,
        "foreign_attribution_max": foreign_max,
        "country_attribution_mean": {"Thailand": 1.0},
        "country_attribution_std": {"Thailand": 0.0},
    }


class TestBuildUncertaintyJson:
    def test_required_top_level_keys_present(self) -> None:
        events_agg = [_fake_event_agg("2025-03-18", 0.4, 0.6)]
        payload = script21.build_uncertainty_json(
            225648, "Mae Hong Son", events_agg, _SEEDS, _META_PARAMS, ["caveat"]
        )
        for key in (
            "generated_by",
            "station_id",
            "station",
            "method",
            "split",
            "horizon_h",
            "horizon_idx",
            "n_ig_steps",
            "n_candidates",
            "top_k",
            "wind_mode",
            "seeds",
            "events",
            "summary",
            "caveats",
        ):
            assert key in payload

    def test_summary_n_seeds_and_n_events(self) -> None:
        events_agg = [
            _fake_event_agg("2025-03-18", 0.4, 0.6),
            _fake_event_agg("2025-03-13", 0.1, 0.5),
        ]
        payload = script21.build_uncertainty_json(
            225648, "Mae Hong Son", events_agg, _SEEDS, _META_PARAMS, []
        )
        assert payload["summary"]["n_seeds"] == 3
        assert payload["summary"]["n_events"] == 2

    def test_mean_event_spread_minmax_equals_mean_of_per_event_ranges(self) -> None:
        events_agg = [
            _fake_event_agg("2025-03-18", 0.4, 0.6),  # range 0.2
            _fake_event_agg("2025-03-13", 0.0, 1.0),  # range 1.0
        ]
        payload = script21.build_uncertainty_json(
            225648, "Mae Hong Son", events_agg, _SEEDS, _META_PARAMS, []
        )
        expected = round(statistics.fmean([0.2, 1.0]), 3)
        assert payload["summary"]["mean_event_spread_minmax"] == expected

    def test_events_and_seeds_passed_through(self) -> None:
        events_agg = [_fake_event_agg("2025-03-18", 0.4, 0.6)]
        payload = script21.build_uncertainty_json(
            225648, "Mae Hong Son", events_agg, _SEEDS, _META_PARAMS, []
        )
        assert payload["events"] == events_agg
        assert payload["seeds"] == _SEEDS

    def test_caveats_present_and_preserved(self) -> None:
        caveats = ["caveat one", "caveat two"]
        payload = script21.build_uncertainty_json(
            225648, "Mae Hong Son", [], _SEEDS, _META_PARAMS, caveats
        )
        assert payload["caveats"] == caveats

    def test_no_events_does_not_crash_summary(self) -> None:
        payload = script21.build_uncertainty_json(
            225648, "Mae Hong Son", [], _SEEDS, _META_PARAMS, []
        )
        assert payload["summary"]["n_events"] == 0
        assert payload["summary"]["mean_foreign_attribution"] == 0.0
        assert payload["summary"]["mean_event_spread_minmax"] == 0.0


# ---------------------------------------------------------------------------
# CLI arg splitting
# ---------------------------------------------------------------------------


class TestSplitCliArgs:
    def test_local_keys_separated_from_overrides(self) -> None:
        argv = ["station_id=225626", "top_k=3", "model.hidden_dim=64"]
        local, overrides = script21._split_cli_args(argv)
        assert local == {"station_id": "225626", "top_k": "3"}
        assert overrides == ["model.hidden_dim=64"]

    def test_args_without_equals_are_ignored(self) -> None:
        local, overrides = script21._split_cli_args(["--help", "split=test"])
        assert local == {"split": "test"}
        assert overrides == []


# ---------------------------------------------------------------------------
# Event-count fixture (mirrors Package A's fake-ds idea, duplicated locally per SPEC 3.2 --
# fixtures are test scaffolding, not production logic, so duplication is fine here).
# ---------------------------------------------------------------------------


def _make_event_sample(frp_countries: list[tuple[float, str]], connected_idxs: list[int]):
    """Build a single-station HeteroData sample with hotspot type_c edges to station 0."""
    import torch
    from torch_geometric.data import HeteroData

    data = HeteroData()
    data["station"].x = torch.zeros(1, 2, 3)
    data["hotspot"].x = torch.tensor(
        [[frp, 0.0, 0.0] for frp, _c in frp_countries], dtype=torch.float32
    )
    data["hotspot"].country = [c for _f, c in frp_countries]
    src = torch.tensor(connected_idxs, dtype=torch.long)
    dst = torch.zeros(len(connected_idxs), dtype=torch.long)
    data["hotspot", "type_c", "station"].edge_index = (
        torch.stack([src, dst]) if connected_idxs else torch.zeros(2, 0, dtype=torch.long)
    )
    data["station", "spatial", "station"].edge_index = torch.zeros(2, 0, dtype=torch.long)
    return data


class _FakeDataset:
    """Duck-typed PM25GraphDataset stand-in exposing only what script 10's helpers touch."""

    def __init__(self, timestamps, anchor_indices, pm25_raw, station_ids, samples) -> None:
        self._timestamps = timestamps
        self._anchor_indices = anchor_indices
        self._pm25_raw = pm25_raw
        self._station_ids = station_ids
        self._samples = samples

    def __getitem__(self, pos: int):
        return self._samples[pos]


def _build_fixture():
    """4 candidate dates, each with a connected foreign hotspot, so top_k can truncate."""
    import numpy as np_
    import pandas as pd

    timestamps = pd.date_range("2025-03-01", periods=4, freq="1D", tz="UTC")
    anchor_indices = np_.array([0, 1, 2, 3])
    pm25_raw = np_.array([[20.0], [15.0], [30.0], [25.0]])
    station_ids = np_.array([500])

    samples = {
        i: _make_event_sample([(float(frp), "Myanmar")], connected_idxs=[0])
        for i, frp in enumerate([10.0, 50.0, 80.0, 40.0])
    }
    ds = _FakeDataset(timestamps, anchor_indices, pm25_raw, station_ids, samples)

    hotspots = pd.DataFrame(
        {
            "date": ["2025-03-01", "2025-03-02", "2025-03-03", "2025-03-04"],
            "total_frp": [10.0, 50.0, 80.0, 40.0],
            "country": ["Myanmar", "Myanmar", "Myanmar", "Myanmar"],
        }
    )
    return ds, hotspots


def test_select_connected_foreign_events_top_k_matches_via_shared_helper(tmp_path: Path) -> None:
    """script 21 reuses A's shared select_connected_foreign_events -- event count == top_k."""
    ds, hotspots = _build_fixture()
    hotspots_path = tmp_path / "hotspots.parquet"
    hotspots.to_parquet(hotspots_path, index=False)

    events = script21.select_connected_foreign_events(
        ds, station_idx=0, hotspots_path=hotspots_path, split="test", n_candidates=15, top_k=2
    )
    assert len(events) == 2
    # Ranked by connected foreign FRP descending: 80.0 then 50.0.
    assert [round(e[0], 1) for e in events] == [80.0, 50.0]
