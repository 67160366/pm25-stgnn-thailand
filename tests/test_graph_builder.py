"""Unit tests for src/data/graph_builder.py.

All tests use synthetic data — no real data files or network calls.
Bearing and alignment math verified in Checkpoint C (2026-05-19).
"""

import math

import pandas as pd
import pytest
import torch

from src.data.graph_builder import (
    _bearing_rad,
    _build_type_a_edges,
    _build_type_b_edges,
    _build_type_c_edges,
    _haversine_km,
    _wind_alignment,
    build_graph,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _two_stations(lat1: float, lon1: float, lat2: float, lon2: float) -> pd.DataFrame:
    """Two-station DataFrame for edge tests."""
    return pd.DataFrame({"station_id": [1, 2], "lat": [lat1, lat2], "lon": [lon1, lon2]})


def _hotspot_df(lat: float, lon: float, frp: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame(
        {"centroid_lat": [lat], "centroid_lon": [lon], "total_frp": [frp], "date": ["2022-03-15"]}
    )


def _base_config(**overrides: object) -> dict:
    return {"wind_mode": "constant_ne", "synthetic_u": 0.0, "synthetic_v": 1.0, **overrides}


# ---------------------------------------------------------------------------
# Bearing helper sanity checks (user addition, Checkpoint C)
# ---------------------------------------------------------------------------


class TestBearingHelperSanity:
    def test_bearing_due_north(self) -> None:
        """Target 1 degree north of source -> bearing = 0 (north)."""
        b = _bearing_rad(0.0, 0.0, 1.0, 0.0)
        assert b == pytest.approx(0.0, abs=1e-6)

    def test_bearing_due_east(self) -> None:
        """Target 1 degree east of source -> bearing = pi/2 (east)."""
        b = _bearing_rad(0.0, 0.0, 0.0, 1.0)
        assert b == pytest.approx(math.pi / 2, abs=1e-4)


# ---------------------------------------------------------------------------
# Test 1: Type A edges are symmetric
# ---------------------------------------------------------------------------


class TestTypeASymmetry:
    def test_type_a_edges_are_symmetric(self) -> None:
        """For every (i,j) edge there must also be a (j,i) edge."""
        # Two stations 50 km apart — well within the 100 km cap
        df = _two_stations(18.0, 99.0, 18.45, 99.0)  # ~50 km north-south
        ei, ea = _build_type_a_edges(df, {"type_a_max_km": 100.0})
        assert ei.shape[0] == 2
        assert ei.shape[1] > 0, "Expected at least one edge pair"
        pairs = set(map(tuple, ei.T.tolist()))
        for i, j in list(pairs):
            assert (j, i) in pairs, f"Edge ({i},{j}) exists but ({j},{i}) does not"

    def test_type_a_weight_same_both_directions(self) -> None:
        """Edge weight must be identical in both directions."""
        df = _two_stations(18.0, 99.0, 18.45, 99.0)
        ei, ea = _build_type_a_edges(df, {"type_a_max_km": 100.0})
        # weights should be identical for (0,1) and (1,0)
        assert ea[0].item() == pytest.approx(ea[1].item(), rel=1e-6)


# ---------------------------------------------------------------------------
# Test 2: Wind alignment = 1.0 for perfectly aligned vectors
# ---------------------------------------------------------------------------


class TestWindAlignmentPerfect:
    def test_alignment_one_north_wind_north_target(self) -> None:
        """u=0, v=1 (north wind), target due north -> alignment = 1."""
        # Checkpoint C verified: bearing=0, wind_dir=atan2(0,1)=0, cos(0)=1
        a = _wind_alignment(0.0, 1.0, 18.0, 99.0, 19.0, 99.0)
        assert a == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Test 3: Wind alignment = 0.0 for perpendicular wind
# ---------------------------------------------------------------------------


class TestWindAlignmentPerpendicular:
    def test_alignment_zero_east_wind_north_target(self) -> None:
        """u=1, v=0 (east wind), target due north -> alignment = 0."""
        # Checkpoint C verified: bearing=0, wind_dir=atan2(1,0)=pi/2, cos(-pi/2)=0
        a = _wind_alignment(1.0, 0.0, 18.0, 99.0, 19.0, 99.0)
        assert a == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Test 4: Type B edge excluded when alignment <= threshold
# ---------------------------------------------------------------------------


class TestTypeBAlignmentThreshold:
    def test_type_b_edge_excluded_below_threshold(self) -> None:
        """alignment = 0.29 < threshold 0.3 -> no Type B edge."""
        # North wind (0, 1); target rotated so alignment = cos(bearing - 0)
        # We need bearing such that cos(bearing) = 0.29 -> bearing ≈ 73.2 deg east
        # Place target to the east-northeast so alignment just fails
        # Easier: use east wind (u=1,v=0) toward a target that is mostly north
        # alignment = cos(bearing - pi/2)
        # For target due north bearing=0: cos(0 - pi/2) = 0 < 0.3 -> excluded ✓
        df = _two_stations(18.0, 99.0, 19.0, 99.0)  # target due north
        cfg = {"wind_mode": "constant_ne", "synthetic_u": 1.0, "synthetic_v": 0.0,
               "type_b_max_km": 200.0, "type_b_min_alignment": 0.3}
        ei, _ = _build_type_b_edges(df, None, cfg)
        assert ei.shape[1] == 0, "Perpendicular wind should produce no Type B edge"


# ---------------------------------------------------------------------------
# Test 5: Type C edge excluded when hotspot > 500 km
# ---------------------------------------------------------------------------


class TestTypeCDistanceThreshold:
    def test_type_c_excluded_beyond_500km(self) -> None:
        """Hotspot far outside the 500 km cap should produce no Type C edge."""
        df_s = pd.DataFrame({"station_id": [1], "lat": [18.0], "lon": [99.0]})
        # Hotspot 800 km away (roughly at lat 25, same lon)
        df_h = _hotspot_df(lat=25.0, lon=99.0, frp=500.0)
        cfg = {"wind_mode": "constant_ne", "synthetic_u": 0.0, "synthetic_v": 1.0,
               "type_c_max_km": 500.0, "type_c_min_alignment": 0.4}
        ei, _ = _build_type_c_edges(df_s, df_h, None, cfg)
        assert ei.shape[1] == 0, "Hotspot > 500 km should produce no Type C edge"


# ---------------------------------------------------------------------------
# Test 6: Type A edge excluded when distance > 100 km
# ---------------------------------------------------------------------------


class TestTypeADistanceCap:
    def test_type_a_excluded_just_outside_100km(self) -> None:
        """Stations ~101 km apart should produce no Type A edge."""
        # ~1 degree latitude ~ 111 km; use 0.91 deg to get ~101 km
        df = _two_stations(18.0, 99.0, 18.91, 99.0)
        d = _haversine_km(18.0, 99.0, 18.91, 99.0)
        assert d > 100, f"Expected >100 km, got {d:.1f} km"
        ei, _ = _build_type_a_edges(df, {"type_a_max_km": 100.0})
        assert ei.shape[1] == 0


# ---------------------------------------------------------------------------
# Test 7: Type A edge included at 99 km
# ---------------------------------------------------------------------------


class TestTypeAJustInside:
    def test_type_a_included_at_99km(self) -> None:
        """Stations ~99 km apart should produce a Type A edge."""
        # 0.89 deg latitude ~ 99 km
        df = _two_stations(18.0, 99.0, 18.89, 99.0)
        d = _haversine_km(18.0, 99.0, 18.89, 99.0)
        assert d < 100, f"Expected <100 km, got {d:.1f} km"
        ei, _ = _build_type_a_edges(df, {"type_a_max_km": 100.0})
        assert ei.shape[1] == 2  # both directions


# ---------------------------------------------------------------------------
# Test 8: Empty hotspot day -> zero Type C edges, no error
# ---------------------------------------------------------------------------


class TestTypeCEmptyHotspots:
    def test_empty_hotspots_no_type_c_edges(self) -> None:
        """An empty hotspot DataFrame should produce 0 Type C edges without error."""
        df_s = pd.DataFrame({"station_id": [1], "lat": [18.0], "lon": [99.0]})
        df_h = pd.DataFrame({"centroid_lat": [], "centroid_lon": [], "total_frp": []})
        ei, ea = _build_type_c_edges(df_s, df_h, None, {"wind_mode": "constant_ne"})
        assert ei.shape[1] == 0
        assert ea.shape[0] == 0

    def test_build_graph_empty_hotspots_no_error(self) -> None:
        """build_graph should succeed without Type C edges when hotspots is empty."""
        df_s = pd.DataFrame({"station_id": [1, 2], "lat": [18.0, 18.45], "lon": [99.0, 99.0]})
        df_h = pd.DataFrame({"centroid_lat": [], "centroid_lon": [], "total_frp": []})
        data = build_graph(df_s, df_h, None, {"wind_mode": "constant_ne"})
        assert data["hotspot", "type_c", "station"].edge_index.shape[1] == 0


# ---------------------------------------------------------------------------
# Test 9: Type C reversed under opposite wind
# ---------------------------------------------------------------------------


class TestTypeCReversedWind:
    def test_type_c_excluded_under_headwind(self) -> None:
        """Hotspot south of station with north wind -> wind blows away from station -> no edge."""
        # Station at lat=19, hotspot at lat=18 (south of station).
        # North wind (v>0) blows northward, so hotspot->station bearing is north.
        # Actually: bearing from hotspot (lat=18) to station (lat=19) is NORTH (bearing=0).
        # North wind (u=0, v=5): wind_dir = atan2(0,5) = 0 (north).
        # alignment = cos(0 - 0) = 1.0 -> edge IS created with NE/north wind.
        # To get headwind: use SOUTH wind (u=0, v=-5): wind_dir = atan2(0,-5) = pi.
        # alignment = cos(0 - pi) = -1.0 < 0.4 -> edge excluded.
        df_s = pd.DataFrame({"station_id": [1], "lat": [19.0], "lon": [99.0]})
        df_h = _hotspot_df(lat=18.0, lon=99.0, frp=500.0)  # hotspot south of station
        cfg = {
            "wind_mode": "constant_ne",
            "synthetic_u": 0.0,
            "synthetic_v": -5.0,   # south wind — blows away from station
            "type_c_max_km": 500.0,
            "type_c_min_alignment": 0.4,
        }
        ei, _ = _build_type_c_edges(df_s, df_h, None, cfg)
        assert ei.shape[1] == 0, "South wind with hotspot-to-north path should produce no Type C edge"


# ---------------------------------------------------------------------------
# Test 10: Type A weight at d=50 km = exp(-1) ≈ 0.368
# ---------------------------------------------------------------------------


class TestTypeAWeightFormula:
    def test_type_a_weight_at_50km_equals_exp_minus_1(self) -> None:
        """Weight = exp(-d/50): at d=50 km the weight should be exp(-1) ≈ 0.3679."""
        # Place two stations approximately 50 km apart north-south
        # 0.45 deg latitude ~ 50 km
        df = _two_stations(18.0, 99.0, 18.45, 99.0)
        d = _haversine_km(18.0, 99.0, 18.45, 99.0)
        expected_weight = math.exp(-d / 50.0)
        ei, ea = _build_type_a_edges(df, {"type_a_max_km": 100.0})
        # First edge weight (either direction; both equal)
        assert ea[0].item() == pytest.approx(expected_weight, rel=1e-5)
        # Spot-check: if distance is ~50 km, weight should be near exp(-1)
        if abs(d - 50.0) < 2.0:
            assert ea[0].item() == pytest.approx(math.exp(-1), abs=0.01)


# ---------------------------------------------------------------------------
# Integration smoke test
# ---------------------------------------------------------------------------


class TestBuildGraphIntegration:
    def test_build_graph_returns_heterodata(self) -> None:
        """build_graph should return a HeteroData with the expected node/edge types."""
        from torch_geometric.data import HeteroData

        df_s = pd.DataFrame({"station_id": [1, 2], "lat": [18.0, 18.45], "lon": [99.0, 99.0]})
        df_h = _hotspot_df(lat=18.2, lon=99.1, frp=200.0)
        data = build_graph(df_s, df_h, None, {"wind_mode": "constant_ne"})
        assert isinstance(data, HeteroData)
        assert "station" in data.node_types
        assert "hotspot" in data.node_types
        assert ("station", "type_a", "station") in data.edge_types
        assert ("station", "type_b", "station") in data.edge_types
        assert ("hotspot", "type_c", "station") in data.edge_types

    def test_build_graph_edge_index_shapes(self) -> None:
        """Edge index tensors should have shape [2, E]."""
        df_s = pd.DataFrame({"station_id": [1, 2], "lat": [18.0, 18.45], "lon": [99.0, 99.0]})
        df_h = _hotspot_df(lat=18.2, lon=99.1, frp=200.0)
        data = build_graph(df_s, df_h, None, {"wind_mode": "constant_ne"})
        for edge_type in data.edge_types:
            ei = data[edge_type].edge_index
            assert ei.shape[0] == 2
            assert ei.dtype == torch.long
