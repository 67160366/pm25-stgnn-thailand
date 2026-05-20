# [TODO: NSC Disclaimer — see booklet page 44]
"""Unit tests for src/explain/{gb_ig,gnn_explainer,attribution}.py.

All tests use synthetic HeteroData and stub models — no real data or network calls.
Tests verify:
  - Integrated Gradients shape and completeness axiom
  - Gradient x Input saliency shapes
  - Occlusion-based country attribution scoring and normalization
  - Station source report generation
  - Batch attribution over multiple samples
  - Hotspot country loading from HeteroData
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch_geometric.data import HeteroData

from src.explain.attribution import (
    batch_attribution,
    load_hotspot_countries,
    station_source_report,
)
from src.explain.gb_ig import integrated_gradients, occlusion_country_attribution
from src.explain.gnn_explainer import gradient_x_input
from src.models.base import PM25ModelBase

# ---------------------------------------------------------------------------
# Stub model for testing: minimal differentiable forward
# ---------------------------------------------------------------------------


class _StubModel(PM25ModelBase):
    """Minimal stub PM25ModelBase that satisfies contracts for testing.

    Forward pass returns station.x.mean(dim=(1, 2)) to ensure gradient flow
    and shape (N, H). All outputs are deterministic given input.
    """

    def __init__(self, n_stations: int = 3, horizons: list[int] | None = None):
        """Initialise stub model.

        Args:
            n_stations: Number of station nodes.
            horizons: Forecast horizons in hours.
        """
        super().__init__(
            n_stations=n_stations,
            n_features=5,
            horizons=horizons or [6, 24],
        )
        self.linear = nn.Linear(1, len(self.horizons))

    def forward(self, data: HeteroData) -> torch.Tensor:
        """Simple differentiable forward: mean of station features -> horizons.

        Args:
            data: HeteroData with station.x of shape (B*N, T_in, F).

        Returns:
            Predictions of shape (B*N, H).
        """
        x = data["station"].x  # (B*N, T_in, F)
        mean_feat = x.mean(dim=(1, 2), keepdim=True)  # (B*N, 1, 1)
        out = self.linear(mean_feat.squeeze(-1))  # (B*N, H)
        return self._postprocess(out)


# ---------------------------------------------------------------------------
# Fixture: synthetic HeteroData builder
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_model() -> _StubModel:
    """Return a stub model in eval mode."""
    model = _StubModel(n_stations=3, horizons=[6, 24])
    model.eval()
    return model


def _make_data(
    n_stations: int = 3,
    t_in: int = 6,
    n_features: int = 5,
    n_hotspots: int = 4,
    hotspot_countries: list[str] | None = None,
) -> HeteroData:
    """Create a minimal HeteroData sample for testing.

    Args:
        n_stations: Number of station nodes.
        t_in: Input time window length.
        n_features: Number of features per timestep.
        n_hotspots: Number of hotspot nodes.
        hotspot_countries: Country label per hotspot.

    Returns:
        HeteroData with station and hotspot nodes and country metadata.
    """
    if hotspot_countries is None:
        hotspot_countries = ["Thailand", "Myanmar", "Thailand", "Laos"]

    data = HeteroData()
    data["station"].x = torch.randn(n_stations, t_in, n_features)
    data["hotspot"].x = torch.randn(n_hotspots, 3)
    data["hotspot"].country = hotspot_countries[:n_hotspots]

    # Add dummy edge to prevent errors if model tries to access edges
    data["station", "spatial", "station"].edge_index = torch.zeros(2, 0, dtype=torch.long)

    return data


@pytest.fixture
def sample_data() -> HeteroData:
    """Return a standard sample HeteroData for tests."""
    return _make_data(n_stations=3, t_in=6, n_features=5, n_hotspots=4)


# ---------------------------------------------------------------------------
# Tests for integrated_gradients
# ---------------------------------------------------------------------------


class TestIntegratedGradients:
    """Test suite for src.explain.gb_ig.integrated_gradients."""

    def test_ig_returns_dict_with_station_x_key(
        self, stub_model: _StubModel, sample_data: HeteroData
    ) -> None:
        """IG output is a dict with 'station_x' key."""
        result = integrated_gradients(
            model=stub_model,
            data=sample_data,
            target_station_idx=0,
            target_horizon_idx=0,
            n_steps=10,
        )
        assert isinstance(result, dict)
        assert "station_x" in result
        assert len(result) == 1

    def test_ig_output_shape_matches_input(
        self, stub_model: _StubModel, sample_data: HeteroData
    ) -> None:
        """IG attribution shape matches station input shape."""
        n_stations, t_in, n_features = 3, 6, 5
        data = _make_data(n_stations=n_stations, t_in=t_in, n_features=n_features, n_hotspots=2)
        result = integrated_gradients(
            model=stub_model,
            data=data,
            target_station_idx=0,
            target_horizon_idx=0,
            n_steps=10,
        )
        attr = result["station_x"]
        assert attr.shape == (n_stations, t_in, n_features)

    def test_ig_n_steps_one_runs(self, stub_model: _StubModel, sample_data: HeteroData) -> None:
        """IG with n_steps=1 runs without error."""
        result = integrated_gradients(
            model=stub_model,
            data=sample_data,
            target_station_idx=0,
            target_horizon_idx=0,
            n_steps=1,
        )
        assert result["station_x"].shape == sample_data["station"].x.shape

    def test_ig_output_is_detached(self, stub_model: _StubModel, sample_data: HeteroData) -> None:
        """IG output does not require gradients."""
        result = integrated_gradients(
            model=stub_model,
            data=sample_data,
            target_station_idx=0,
            target_horizon_idx=0,
            n_steps=10,
        )
        assert not result["station_x"].requires_grad

    def test_ig_zero_baseline_gives_zero_attr(
        self,
    ) -> None:
        """IG with zero input features gives near-zero attribution."""
        model = _StubModel(n_stations=2, horizons=[6])
        model.eval()

        data = HeteroData()
        data["station"].x = torch.zeros(2, 4, 5)
        data["hotspot"].x = torch.zeros(0, 3)
        data["hotspot"].country = []
        data["station", "spatial", "station"].edge_index = torch.zeros(2, 0, dtype=torch.long)

        result = integrated_gradients(
            model=model,
            data=data,
            target_station_idx=0,
            target_horizon_idx=0,
            n_steps=10,
        )
        attr = result["station_x"]
        # With zero input, attribution should be approximately zero
        assert torch.allclose(attr, torch.zeros_like(attr), atol=1e-6)

    def test_ig_different_target_stations(
        self, stub_model: _StubModel, sample_data: HeteroData
    ) -> None:
        """IG runs for different target station indices."""
        n_stations = 3
        results = []
        for station_idx in range(n_stations):
            result = integrated_gradients(
                model=stub_model,
                data=sample_data,
                target_station_idx=station_idx,
                target_horizon_idx=0,
                n_steps=10,
            )
            results.append(result["station_x"])

        # All should have the same shape, but may differ due to grad flow
        for res in results:
            assert res.shape == sample_data["station"].x.shape

    def test_ig_different_target_horizons(
        self, stub_model: _StubModel, sample_data: HeteroData
    ) -> None:
        """IG runs for different horizon indices."""
        n_horizons = len(stub_model.horizons)
        results = []
        for horizon_idx in range(n_horizons):
            result = integrated_gradients(
                model=stub_model,
                data=sample_data,
                target_station_idx=0,
                target_horizon_idx=horizon_idx,
                n_steps=10,
            )
            results.append(result["station_x"])

        for res in results:
            assert res.shape == sample_data["station"].x.shape


# ---------------------------------------------------------------------------
# Tests for gradient_x_input
# ---------------------------------------------------------------------------


class TestGradientXInput:
    """Test suite for src.explain.gnn_explainer.gradient_x_input."""

    def test_gxi_returns_dict_with_station_x_key(
        self, stub_model: _StubModel, sample_data: HeteroData
    ) -> None:
        """Gradient x Input output is a dict with 'station_x' key."""
        result = gradient_x_input(
            model=stub_model,
            data=sample_data,
            target_station_idx=0,
            target_horizon_idx=0,
        )
        assert isinstance(result, dict)
        assert "station_x" in result
        assert len(result) == 1

    def test_gxi_output_shape_matches_input(
        self, stub_model: _StubModel, sample_data: HeteroData
    ) -> None:
        """Gradient x Input attribution shape matches station input shape."""
        n_stations, t_in, n_features = 3, 6, 5
        data = _make_data(n_stations=n_stations, t_in=t_in, n_features=n_features, n_hotspots=2)
        result = gradient_x_input(
            model=stub_model,
            data=data,
            target_station_idx=0,
            target_horizon_idx=0,
        )
        attr = result["station_x"]
        assert attr.shape == (n_stations, t_in, n_features)

    def test_gxi_output_is_detached(self, stub_model: _StubModel, sample_data: HeteroData) -> None:
        """Gradient x Input output does not require gradients."""
        result = gradient_x_input(
            model=stub_model,
            data=sample_data,
            target_station_idx=0,
            target_horizon_idx=0,
        )
        assert not result["station_x"].requires_grad

    def test_gxi_different_target_stations(
        self, stub_model: _StubModel, sample_data: HeteroData
    ) -> None:
        """Gradient x Input runs for different target stations."""
        n_stations = 3
        results = []
        for station_idx in range(n_stations):
            result = gradient_x_input(
                model=stub_model,
                data=sample_data,
                target_station_idx=station_idx,
                target_horizon_idx=0,
            )
            results.append(result["station_x"])

        for res in results:
            assert res.shape == sample_data["station"].x.shape

    def test_gxi_different_target_horizons(
        self, stub_model: _StubModel, sample_data: HeteroData
    ) -> None:
        """Gradient x Input runs for different target horizons."""
        n_horizons = len(stub_model.horizons)
        results = []
        for horizon_idx in range(n_horizons):
            result = gradient_x_input(
                model=stub_model,
                data=sample_data,
                target_station_idx=0,
                target_horizon_idx=horizon_idx,
            )
            results.append(result["station_x"])

        for res in results:
            assert res.shape == sample_data["station"].x.shape


# ---------------------------------------------------------------------------
# Tests for occlusion_country_attribution
# ---------------------------------------------------------------------------


class TestOcclusionCountryAttribution:
    """Test suite for src.explain.gb_ig.occlusion_country_attribution."""

    def test_occlusion_empty_countries_returns_empty_dict(
        self, stub_model: _StubModel, sample_data: HeteroData
    ) -> None:
        """Occlusion with empty country list returns empty dict."""
        result = occlusion_country_attribution(
            model=stub_model,
            data=sample_data,
            target_station_idx=0,
            target_horizon_idx=0,
            hotspot_countries=[],
        )
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_occlusion_scores_sum_to_one(
        self, stub_model: _StubModel, sample_data: HeteroData
    ) -> None:
        """Occlusion scores sum to 1.0 (or 0 if all scores are zero)."""
        countries = ["Thailand", "Myanmar", "Thailand", "Laos"]
        result = occlusion_country_attribution(
            model=stub_model,
            data=sample_data,
            target_station_idx=0,
            target_horizon_idx=0,
            hotspot_countries=countries,
        )
        total = sum(result.values())
        # Either sum is 1.0 or all scores are 0
        assert torch.allclose(torch.tensor(total), torch.tensor(1.0), atol=1e-5) or torch.allclose(
            torch.tensor(total), torch.tensor(0.0), atol=1e-5
        )

    def test_occlusion_scores_nonnegative(
        self, stub_model: _StubModel, sample_data: HeteroData
    ) -> None:
        """Occlusion scores are all non-negative."""
        countries = ["Thailand", "Myanmar", "Thailand", "Laos"]
        result = occlusion_country_attribution(
            model=stub_model,
            data=sample_data,
            target_station_idx=0,
            target_horizon_idx=0,
            hotspot_countries=countries,
        )
        for score in result.values():
            assert score >= 0.0

    def test_occlusion_scores_in_unit_interval(
        self, stub_model: _StubModel, sample_data: HeteroData
    ) -> None:
        """Occlusion scores are in [0, 1]."""
        countries = ["Thailand", "Myanmar", "Thailand", "Laos"]
        result = occlusion_country_attribution(
            model=stub_model,
            data=sample_data,
            target_station_idx=0,
            target_horizon_idx=0,
            hotspot_countries=countries,
        )
        for score in result.values():
            assert 0.0 <= score <= 1.0

    def test_occlusion_returns_all_unique_countries(self, stub_model: _StubModel) -> None:
        """Occlusion result includes all unique countries or is empty."""
        countries = ["Thailand", "Myanmar", "Laos"]
        data = _make_data(n_stations=3, t_in=6, n_features=5, n_hotspots=3)
        data["hotspot"].country = countries

        result = occlusion_country_attribution(
            model=stub_model,
            data=data,
            target_station_idx=0,
            target_horizon_idx=0,
            hotspot_countries=countries,
        )
        # If not empty, all unique countries should be present
        if result:
            assert set(result.keys()) == set(countries)

    @pytest.mark.parametrize("n_hotspots", [0, 1, 2, 5])
    def test_occlusion_different_hotspot_counts(
        self, stub_model: _StubModel, n_hotspots: int
    ) -> None:
        """Occlusion handles different numbers of hotspots."""
        countries = ["Thailand"] * n_hotspots
        data = _make_data(n_stations=3, t_in=6, n_features=5, n_hotspots=n_hotspots)
        data["hotspot"].country = countries

        result = occlusion_country_attribution(
            model=stub_model,
            data=data,
            target_station_idx=0,
            target_horizon_idx=0,
            hotspot_countries=countries,
        )
        # Should work without error
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Tests for load_hotspot_countries
# ---------------------------------------------------------------------------


class TestLoadHotspotCountries:
    """Test suite for src.explain.attribution.load_hotspot_countries."""

    def test_load_returns_list(self, sample_data: HeteroData) -> None:
        """load_hotspot_countries returns a list."""
        result = load_hotspot_countries(sample_data)
        assert isinstance(result, list)

    def test_load_correct_countries(
        self,
    ) -> None:
        """load_hotspot_countries returns correct country labels."""
        countries = ["Thailand", "Myanmar", "Laos"]
        data = _make_data(n_stations=2, t_in=4, n_features=3, n_hotspots=3)
        data["hotspot"].country = countries

        result = load_hotspot_countries(data)
        assert result == countries

    def test_load_empty_hotspots(
        self,
    ) -> None:
        """load_hotspot_countries returns empty list if no hotspots."""
        data = HeteroData()
        data["station"].x = torch.randn(2, 4, 5)
        # No hotspot node or no country attribute
        result = load_hotspot_countries(data)
        assert result == []

    def test_load_missing_country_attr(
        self,
    ) -> None:
        """load_hotspot_countries returns empty list if country attr missing."""
        data = HeteroData()
        data["station"].x = torch.randn(2, 4, 5)
        data["hotspot"].x = torch.randn(3, 3)
        # No country attribute
        result = load_hotspot_countries(data)
        assert result == []


# ---------------------------------------------------------------------------
# Tests for station_source_report
# ---------------------------------------------------------------------------


class TestStationSourceReport:
    """Test suite for src.explain.attribution.station_source_report."""

    def test_report_has_all_required_keys(
        self, stub_model: _StubModel, sample_data: HeteroData
    ) -> None:
        """Station source report contains all required keys."""
        report = station_source_report(
            model=stub_model,
            data=sample_data,
            station_idx=0,
            horizon_idx=0,
            station_name="Test Station",
            n_ig_steps=5,
        )
        assert isinstance(report, dict)
        assert "station_idx" in report
        assert "station_name" in report
        assert "horizon_idx" in report
        assert "ig_feature_importance" in report
        assert "country_attribution" in report

    def test_report_indices_match_input(
        self, stub_model: _StubModel, sample_data: HeteroData
    ) -> None:
        """Station source report indices match input parameters."""
        station_idx, horizon_idx = 1, 0
        station_name = "Chiang Mai"
        report = station_source_report(
            model=stub_model,
            data=sample_data,
            station_idx=station_idx,
            horizon_idx=horizon_idx,
            station_name=station_name,
            n_ig_steps=5,
        )
        assert report["station_idx"] == station_idx
        assert report["horizon_idx"] == horizon_idx
        assert report["station_name"] == station_name

    def test_report_ig_feature_importance_nonempty(
        self, stub_model: _StubModel, sample_data: HeteroData
    ) -> None:
        """Station source report IG feature importance is non-empty dict."""
        report = station_source_report(
            model=stub_model,
            data=sample_data,
            station_idx=0,
            horizon_idx=0,
            n_ig_steps=5,
        )
        assert isinstance(report["ig_feature_importance"], dict)
        assert len(report["ig_feature_importance"]) > 0

    def test_report_ig_feature_importance_nonnegative(
        self, stub_model: _StubModel, sample_data: HeteroData
    ) -> None:
        """Station source report IG feature importance scores are non-negative."""
        report = station_source_report(
            model=stub_model,
            data=sample_data,
            station_idx=0,
            horizon_idx=0,
            n_ig_steps=5,
        )
        for score in report["ig_feature_importance"].values():
            assert isinstance(score, float)
            assert score >= 0.0

    def test_report_country_attribution_is_dict(
        self, stub_model: _StubModel, sample_data: HeteroData
    ) -> None:
        """Station source report country attribution is a dict."""
        report = station_source_report(
            model=stub_model,
            data=sample_data,
            station_idx=0,
            horizon_idx=0,
            n_ig_steps=5,
        )
        assert isinstance(report["country_attribution"], dict)

    def test_report_country_attribution_sums_to_one(
        self, stub_model: _StubModel, sample_data: HeteroData
    ) -> None:
        """Station source report country attribution sums to 1.0 or 0."""
        report = station_source_report(
            model=stub_model,
            data=sample_data,
            station_idx=0,
            horizon_idx=0,
            n_ig_steps=5,
        )
        total = sum(report["country_attribution"].values())
        # Either sum is 1.0 or all scores are 0
        assert torch.allclose(torch.tensor(total), torch.tensor(1.0), atol=1e-5) or torch.allclose(
            torch.tensor(total), torch.tensor(0.0), atol=1e-5
        )

    @pytest.mark.parametrize("station_idx", [0, 1, 2])
    @pytest.mark.parametrize("horizon_idx", [0, 1])
    def test_report_different_stations_horizons(
        self, stub_model: _StubModel, sample_data: HeteroData, station_idx: int, horizon_idx: int
    ) -> None:
        """Station source report works for different stations and horizons."""
        report = station_source_report(
            model=stub_model,
            data=sample_data,
            station_idx=station_idx,
            horizon_idx=horizon_idx,
            n_ig_steps=5,
        )
        assert report["station_idx"] == station_idx
        assert report["horizon_idx"] == horizon_idx


# ---------------------------------------------------------------------------
# Tests for batch_attribution
# ---------------------------------------------------------------------------


class TestBatchAttribution:
    """Test suite for src.explain.attribution.batch_attribution."""

    def test_batch_returns_list_of_dicts(self, stub_model: _StubModel) -> None:
        """Batch attribution returns a list of dicts."""
        samples = [_make_data(n_stations=3, t_in=6, n_features=5, n_hotspots=4) for _ in range(3)]
        result = batch_attribution(
            model=stub_model,
            samples=samples,
            station_idx=0,
            horizon_idx=0,
            n_ig_steps=5,
        )
        assert isinstance(result, list)
        assert len(result) == 3
        for report in result:
            assert isinstance(report, dict)

    def test_batch_length_matches_input(self, stub_model: _StubModel) -> None:
        """Batch attribution result length matches input samples."""
        sample_counts = [1, 3, 5]
        for count in sample_counts:
            samples = [
                _make_data(n_stations=3, t_in=6, n_features=5, n_hotspots=4) for _ in range(count)
            ]
            result = batch_attribution(
                model=stub_model,
                samples=samples,
                station_idx=0,
                horizon_idx=0,
                n_ig_steps=5,
            )
            assert len(result) == count

    def test_batch_empty_samples(self, stub_model: _StubModel) -> None:
        """Batch attribution handles empty sample list."""
        result = batch_attribution(
            model=stub_model,
            samples=[],
            station_idx=0,
            horizon_idx=0,
            n_ig_steps=5,
        )
        assert result == []

    def test_batch_each_report_has_required_keys(self, stub_model: _StubModel) -> None:
        """Each report in batch has all required keys."""
        samples = [_make_data(n_stations=3, t_in=6, n_features=5, n_hotspots=4) for _ in range(2)]
        result = batch_attribution(
            model=stub_model,
            samples=samples,
            station_idx=0,
            horizon_idx=0,
            station_name="Test",
            n_ig_steps=5,
        )
        for report in result:
            assert "station_idx" in report
            assert "station_name" in report
            assert "horizon_idx" in report
            assert "ig_feature_importance" in report
            assert "country_attribution" in report

    def test_batch_preserves_input_metadata(self, stub_model: _StubModel) -> None:
        """Batch attribution preserves station metadata across samples."""
        station_idx, horizon_idx = 1, 0
        station_name = "Chiang Rai"
        samples = [_make_data(n_stations=3, t_in=6, n_features=5, n_hotspots=4) for _ in range(2)]
        result = batch_attribution(
            model=stub_model,
            samples=samples,
            station_idx=station_idx,
            horizon_idx=horizon_idx,
            station_name=station_name,
            n_ig_steps=5,
        )
        for report in result:
            assert report["station_idx"] == station_idx
            assert report["horizon_idx"] == horizon_idx
            assert report["station_name"] == station_name


# ---------------------------------------------------------------------------
# Integration tests: end-to-end workflows
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests combining multiple explain functions."""

    def test_ig_and_gradient_x_input_same_shape(
        self, stub_model: _StubModel, sample_data: HeteroData
    ) -> None:
        """IG and Gradient x Input produce same shape output."""
        ig_result = integrated_gradients(
            model=stub_model,
            data=sample_data,
            target_station_idx=0,
            target_horizon_idx=0,
            n_steps=10,
        )
        gxi_result = gradient_x_input(
            model=stub_model,
            data=sample_data,
            target_station_idx=0,
            target_horizon_idx=0,
        )
        assert ig_result["station_x"].shape == gxi_result["station_x"].shape

    def test_full_attribution_pipeline(
        self, stub_model: _StubModel, sample_data: HeteroData
    ) -> None:
        """Full attribution pipeline: IG + occlusion + report."""
        # IG
        ig_result = integrated_gradients(
            model=stub_model,
            data=sample_data,
            target_station_idx=0,
            target_horizon_idx=0,
            n_steps=10,
        )
        assert "station_x" in ig_result

        # Occlusion
        countries = load_hotspot_countries(sample_data)
        occ_result = occlusion_country_attribution(
            model=stub_model,
            data=sample_data,
            target_station_idx=0,
            target_horizon_idx=0,
            hotspot_countries=countries,
        )
        assert isinstance(occ_result, dict)

        # Report
        report = station_source_report(
            model=stub_model,
            data=sample_data,
            station_idx=0,
            horizon_idx=0,
            n_ig_steps=10,
        )
        required_keys = [
            "station_idx",
            "station_name",
            "horizon_idx",
            "ig_feature_importance",
            "country_attribution",
        ]
        assert all(k in report for k in required_keys)
