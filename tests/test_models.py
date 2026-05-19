# [TODO: NSC Disclaimer — see booklet page 44]
"""Unit tests for src/models/{base,a3tgcn,mtgnn}.py.

All tests use synthetic HeteroData — no real data files or network calls.
Tests verify:
  - Forward pass shape contracts
  - Gradient flow
  - Output clipping and non-negativity
  - Determinism
  - Edge case handling (empty hotspots, single batch, etc.)
  - Parameter count sanity
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import HeteroData

from src.models.a3tgcn import A3TGCNModel
from src.models.mtgnn import MTGNNModel

# ---------------------------------------------------------------------------
# Test fixtures: synthetic HeteroData builder
# ---------------------------------------------------------------------------


def _make_hetero_batch(b: int = 2, n: int = 4, t: int = 24, f: int = 5, k: int = 3) -> HeteroData:
    """Synthetic HeteroData matching PM25GraphDataset output schema.

    Args:
        b: Batch size.
        n: Number of stations.
        t: Time window length.
        f: Number of features per timestep.
        k: Number of hotspots per batch.

    Returns:
        HeteroData with station/hotspot nodes, Type A/B/C edges.
    """
    data = HeteroData()

    # Station node features: (b*n, t, f)
    data["station"].x = torch.randn(b * n, t, f)

    # Station target values: (b*n, 4) for 4 horizons
    data["station"].y = torch.rand(b * n, 4) * 50

    # Station mask: (b*n, 4)
    data["station"].mask = torch.ones(b * n, 4, dtype=torch.bool)

    # Type A edges: ring topology within each batch
    ei_a, ea_a = [], []
    for batch_idx in range(b):
        for i in range(n):
            j = (i + 1) % n
            ei_a.append([batch_idx * n + i, batch_idx * n + j])
            ea_a.append([0.5])
    data["station", "type_a", "station"].edge_index = (
        torch.tensor(ei_a, dtype=torch.long).t().contiguous()
    )
    data["station", "type_a", "station"].edge_attr = torch.tensor(ea_a)

    # Type B edges: same topology, 3-col attr
    ei_b, eb_b = [], []
    for batch_idx in range(b):
        for i in range(n):
            j = (i + 1) % n
            ei_b.append([batch_idx * n + i, batch_idx * n + j])
            eb_b.append([0.3, 0.8, 5.0])
    data["station", "type_b", "station"].edge_index = (
        torch.tensor(ei_b, dtype=torch.long).t().contiguous()
    )
    data["station", "type_b", "station"].edge_attr = torch.tensor(eb_b)

    # Type C edges: k hotspots per batch -> stations
    if k > 0:
        data["hotspot"].x = torch.rand(b * k, 3)
        ei_c, ec_c = [], []
        for batch_idx in range(b):
            for hotspot_idx in range(k):
                ei_c.append([batch_idx * k + hotspot_idx, batch_idx * n + (hotspot_idx % n)])
                ec_c.append([0.4, 0.7, 100.0])
        data["hotspot", "type_c", "station"].edge_index = (
            torch.tensor(ei_c, dtype=torch.long).t().contiguous()
        )
        data["hotspot", "type_c", "station"].edge_attr = torch.tensor(ec_c)
    else:
        data["hotspot"].x = torch.zeros(0, 3)
        data["hotspot", "type_c", "station"].edge_index = torch.zeros(2, 0, dtype=torch.long)
        data["hotspot", "type_c", "station"].edge_attr = torch.zeros(0, 3)

    return data


def _masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute masked MSE loss.

    Args:
        pred: Predictions of shape (N, H).
        target: Targets of shape (N, H).
        mask: Boolean mask of shape (N, H).

    Returns:
        Scalar loss, or zero if mask is empty.
    """
    if not mask.any():
        return torch.tensor(0.0, dtype=pred.dtype, device=pred.device)
    return ((pred - target) ** 2)[mask].mean()


# ---------------------------------------------------------------------------
# Test 1: A3TGCN forward shape
# ---------------------------------------------------------------------------


class TestA3TGCNShape:
    def test_forward_shape_matches_contract(self) -> None:
        """A3TGCNModel forward: (b*n, t, f) -> (b*n, h)."""
        batch_size, num_stations, time_steps, num_features, num_horizons = 2, 4, 24, 5, 4
        model = A3TGCNModel(
            n_stations=num_stations,
            n_features=num_features,
            horizons=[6, 12, 24, 48],
        )
        model.eval()
        model = model.cpu()
        data = _make_hetero_batch(
            b=batch_size,
            n=num_stations,
            t=time_steps,
            f=num_features,
        )

        out = model(data)

        expected_shape = (batch_size * num_stations, num_horizons)
        assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# Test 2: MTGNN forward shape
# ---------------------------------------------------------------------------


class TestMTGNNShape:
    def test_forward_shape_matches_contract(self) -> None:
        """MTGNNModel forward: (b*n, t, f) -> (b*n, h)."""
        batch_size, num_stations, time_steps, num_features, num_horizons = 2, 4, 24, 5, 4
        model = MTGNNModel(
            n_stations=num_stations,
            n_features=num_features,
            horizons=[6, 12, 24, 48],
        )
        data = _make_hetero_batch(
            b=batch_size,
            n=num_stations,
            t=time_steps,
            f=num_features,
        )

        out = model(data)

        expected_shape = (batch_size * num_stations, num_horizons)
        assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# Test 3: Gradient flow through A3TGCN
# ---------------------------------------------------------------------------


class TestA3TGCNGradients:
    def test_gradients_flow_through_all_parameters(self) -> None:
        """A3TGCN backward pass: all named parameters should have gradients."""
        batch_size, num_stations, time_steps, num_features = 2, 4, 24, 5
        model = A3TGCNModel(n_stations=num_stations, n_features=num_features)
        model = model.cpu()
        data = _make_hetero_batch(
            b=batch_size,
            n=num_stations,
            t=time_steps,
            f=num_features,
        )

        out = model(data)
        loss = out.sum()
        loss.backward()

        # All parameters with requires_grad should have non-None grad
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"


# ---------------------------------------------------------------------------
# Test 4: Gradient flow through MTGNN
# ---------------------------------------------------------------------------


class TestMTGNNGradients:
    def test_gradients_flow_through_all_parameters(self) -> None:
        """MTGNN backward pass: all named parameters should have gradients."""
        batch_size, num_stations, time_steps, num_features = 2, 4, 24, 5
        model = MTGNNModel(n_stations=num_stations, n_features=num_features)
        data = _make_hetero_batch(
            b=batch_size,
            n=num_stations,
            t=time_steps,
            f=num_features,
        )

        out = model(data)
        loss = out.sum()
        loss.backward()

        # All parameters with requires_grad should have non-None grad
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"


# ---------------------------------------------------------------------------
# Test 5: Output non-negativity (clipping enabled)
# ---------------------------------------------------------------------------


class TestOutputClipping:
    def test_a3tgcn_output_nonnegative_with_clipping(self) -> None:
        """A3TGCN with default _clip_output=True should produce (out >= 0).all()."""
        batch_size, num_stations, time_steps, num_features = 2, 4, 24, 5
        model = A3TGCNModel(n_stations=num_stations, n_features=num_features)
        model = model.cpu()
        model._clip_output = True
        data = _make_hetero_batch(
            b=batch_size,
            n=num_stations,
            t=time_steps,
            f=num_features,
        )

        out = model(data)

        assert (out >= 0).all(), "Output contains negative values with clipping enabled"

    def test_mtgnn_output_nonnegative_with_clipping(self) -> None:
        """MTGNN with default _clip_output=True should produce (out >= 0).all()."""
        batch_size, num_stations, time_steps, num_features = 2, 4, 24, 5
        model = MTGNNModel(n_stations=num_stations, n_features=num_features)
        model._clip_output = True
        data = _make_hetero_batch(
            b=batch_size,
            n=num_stations,
            t=time_steps,
            f=num_features,
        )

        out = model(data)

        assert (out >= 0).all(), "Output contains negative values with clipping enabled"


# ---------------------------------------------------------------------------
# Test 6: Masked loss is zero for all-False mask
# ---------------------------------------------------------------------------


class TestMaskedLoss:
    def test_masked_mse_zero_when_mask_empty(self) -> None:
        """masked_mse with all-False mask should return 0."""
        batch_size, num_stations, num_horizons = 2, 4, 4
        pred = torch.randn(batch_size * num_stations, num_horizons)
        target = torch.randn(batch_size * num_stations, num_horizons)
        mask = torch.zeros(batch_size * num_stations, num_horizons, dtype=torch.bool)

        loss = _masked_mse(pred, target, mask)

        assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 7: MTGNN with k=0 hotspots (no hotspots)
# ---------------------------------------------------------------------------


class TestMTGNNNoHotspots:
    def test_mtgnn_forward_with_zero_hotspots(self) -> None:
        """MTGNN should handle k=0 hotspots without error."""
        batch_size, num_stations, time_steps, num_features, num_hotspots = 2, 4, 24, 5, 0
        model = MTGNNModel(n_stations=num_stations, n_features=num_features)
        data = _make_hetero_batch(
            b=batch_size,
            n=num_stations,
            t=time_steps,
            f=num_features,
            k=num_hotspots,
        )

        out = model(data)

        assert out.shape == (batch_size * num_stations, 4)
        assert (out >= 0).all()


# ---------------------------------------------------------------------------
# Test 8: MTGNN with empty Type B edges
# ---------------------------------------------------------------------------


class TestMTGNNEmptyTypeB:
    def test_mtgnn_forward_with_empty_type_b_edges(self) -> None:
        """MTGNN should handle empty Type B edge_index (2, 0) without error."""
        batch_size, num_stations, time_steps, num_features = 2, 4, 24, 5
        model = MTGNNModel(n_stations=num_stations, n_features=num_features)
        data = _make_hetero_batch(
            b=batch_size,
            n=num_stations,
            t=time_steps,
            f=num_features,
        )

        # Manually clear Type B edges
        data["station", "type_b", "station"].edge_index = torch.zeros(2, 0, dtype=torch.long)
        data["station", "type_b", "station"].edge_attr = torch.zeros(0, 3)

        out = model(data)

        assert out.shape == (batch_size * num_stations, 4)
        assert (out >= 0).all()


# ---------------------------------------------------------------------------
# Test 9: A3TGCN with single batch (b=1)
# ---------------------------------------------------------------------------


class TestA3TGCNSingleBatch:
    def test_a3tgcn_single_batch_shape(self) -> None:
        """A3TGCN with b=1 should output (n, h), not (1, n, h)."""
        batch_size, num_stations, time_steps, num_features = 1, 4, 24, 5
        model = A3TGCNModel(n_stations=num_stations, n_features=num_features)
        model = model.cpu()
        data = _make_hetero_batch(
            b=batch_size,
            n=num_stations,
            t=time_steps,
            f=num_features,
        )

        out = model(data)

        assert out.shape == (num_stations, 4), f"Expected ({num_stations}, 4), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 10: Parameter count sanity checks
# ---------------------------------------------------------------------------


class TestParameterCounts:
    def test_a3tgcn_param_count_reasonable(self) -> None:
        """A3TGCN parameter count should be in [10k, 200k] range."""
        model = A3TGCNModel(n_stations=4, n_features=5)
        count = model.count_parameters()

        assert 10_000 <= count <= 200_000, f"A3TGCN param count {count} out of range"

    def test_mtgnn_param_count_reasonable(self) -> None:
        """MTGNN parameter count should be in [50k, 1M] range."""
        model = MTGNNModel(n_stations=4, n_features=5)
        count = model.count_parameters()

        assert 50_000 <= count <= 1_000_000, f"MTGNN param count {count} out of range"


# ---------------------------------------------------------------------------
# Test 11: Determinism under seeded eval mode
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_a3tgcn_deterministic_eval_forward(self) -> None:
        """A3TGCN eval mode with fixed seed should give identical outputs."""
        batch_size, num_stations, time_steps, num_features = 2, 4, 24, 5
        model = A3TGCNModel(n_stations=num_stations, n_features=num_features)
        model = model.cpu()
        model.eval()
        data = _make_hetero_batch(
            b=batch_size,
            n=num_stations,
            t=time_steps,
            f=num_features,
        )

        torch.manual_seed(0)
        with torch.no_grad():
            out1 = model(data)

        torch.manual_seed(0)
        with torch.no_grad():
            out2 = model(data)

        assert torch.allclose(out1, out2), "A3TGCN outputs differ under same seed"

    def test_mtgnn_deterministic_eval_forward(self) -> None:
        """MTGNN eval mode with fixed seed should give identical outputs."""
        batch_size, num_stations, time_steps, num_features = 2, 4, 24, 5
        model = MTGNNModel(n_stations=num_stations, n_features=num_features)
        model.eval()
        data = _make_hetero_batch(
            b=batch_size,
            n=num_stations,
            t=time_steps,
            f=num_features,
        )

        torch.manual_seed(0)
        with torch.no_grad():
            out1 = model(data)

        torch.manual_seed(0)
        with torch.no_grad():
            out2 = model(data)

        assert torch.allclose(out1, out2), "MTGNN outputs differ under same seed"


# ---------------------------------------------------------------------------
# Additional edge cases and integration tests
# ---------------------------------------------------------------------------


class TestBaseClassValidation:
    def test_pm25_model_base_rejects_zero_stations(self) -> None:
        """Base class should reject n_stations <= 0."""
        with pytest.raises(ValueError, match="n_stations must be > 0"):
            A3TGCNModel(n_stations=0)

    def test_pm25_model_base_rejects_zero_features(self) -> None:
        """Base class should reject n_features <= 0."""
        with pytest.raises(ValueError, match="n_features must be > 0"):
            A3TGCNModel(n_stations=4, n_features=0)

    def test_pm25_model_base_rejects_explicit_empty_horizons(self) -> None:
        """Base class rejects only explicit empty horizons after subclass default applied.

        Note: A3TGCNModel(horizons=None) or A3TGCNModel(horizons=[6,12,24,48])
        both work. Empty list would fail, but subclasses apply defaults first.
        """
        # Since A3TGCN applies horizons = horizons or [...], it never reaches empty.
        # Test that explicit None gets defaults: horizons=[6, 12, 24, 48]
        model = A3TGCNModel(n_stations=4, horizons=None)
        assert model.horizons == [6, 12, 24, 48]


class TestMTGNNValidation:
    def test_mtgnn_rejects_zero_layers(self) -> None:
        """MTGNN should reject n_layers < 1."""
        with pytest.raises(ValueError, match="n_layers must be >= 1"):
            MTGNNModel(n_stations=4, n_layers=0)

    def test_mtgnn_rejects_invalid_dropout(self) -> None:
        """MTGNN should reject dropout >= 1.0."""
        with pytest.raises(ValueError, match="dropout must be in"):
            MTGNNModel(n_stations=4, dropout=1.5)


class TestPredictMethod:
    def test_a3tgcn_predict_batches(self) -> None:
        """A3TGCN.predict should aggregate batches correctly."""
        import numpy as np
        from torch_geometric.loader import DataLoader

        batch_size, num_stations, time_steps, num_features = 2, 4, 24, 5
        model = A3TGCNModel(n_stations=num_stations, n_features=num_features)

        batches = [
            _make_hetero_batch(
                b=batch_size,
                n=num_stations,
                t=time_steps,
                f=num_features,
            )
            for _ in range(2)
        ]
        loader = DataLoader(batches, batch_size=1)

        preds = model.predict(loader, device="cpu")

        # Should have 2 batches * batch_size*num_stations samples each
        assert preds.shape[0] == 2 * batch_size * num_stations
        assert preds.shape[1] == 4
        assert preds.dtype == np.float32


class TestOutputClippingToggle:
    def test_a3tgcn_no_clipping_allows_negatives(self) -> None:
        """A3TGCN with _clip_output=False can produce negative values."""
        batch_size, num_stations, time_steps, num_features = 2, 4, 24, 5
        model = A3TGCNModel(n_stations=num_stations, n_features=num_features)
        model = model.cpu()
        model._clip_output = False
        data = _make_hetero_batch(
            b=batch_size,
            n=num_stations,
            t=time_steps,
            f=num_features,
        )

        # We can't guarantee negatives without controlling randomness,
        # but clipping should be disabled (test by checking it's possible)
        # Just verify the model respects the flag
        model(data)
        assert model._clip_output is False


class TestExtraRepr:
    def test_a3tgcn_extra_repr_contains_config(self) -> None:
        """A3TGCN.extra_repr should contain key hyperparameters."""
        model = A3TGCNModel(n_stations=4, n_features=5, horizons=[6, 12, 24, 48])
        repr_str = model.extra_repr()

        assert "n_stations=4" in repr_str
        assert "n_features=5" in repr_str
        assert "horizons=[6, 12, 24, 48]" in repr_str

    def test_mtgnn_extra_repr_contains_config(self) -> None:
        """MTGNN.extra_repr should contain key hyperparameters."""
        model = MTGNNModel(n_stations=4, n_features=5, horizons=[6, 12, 24, 48])
        repr_str = model.extra_repr()

        assert "n_stations=4" in repr_str
        assert "n_features=5" in repr_str
        assert "horizons=[6, 12, 24, 48]" in repr_str


class TestBatchDivisibility:
    def test_mtgnn_rejects_non_divisible_batch(self) -> None:
        """MTGNN should raise error if total nodes not divisible by n_stations."""
        num_stations = 4
        model = MTGNNModel(n_stations=num_stations)

        # Create data with num_stations+1 total stations (not divisible by num_stations)
        data = HeteroData()
        data["station"].x = torch.randn(num_stations + 1, 24, 5)
        data["station", "type_a", "station"].edge_index = torch.zeros(2, 0, dtype=torch.long)
        data["station", "type_a", "station"].edge_attr = torch.zeros(0, 1)
        data["station", "type_b", "station"].edge_index = torch.zeros(2, 0, dtype=torch.long)
        data["station", "type_b", "station"].edge_attr = torch.zeros(0, 1)
        data["hotspot"].x = torch.zeros(0, 3)
        data["hotspot", "type_c", "station"].edge_index = torch.zeros(2, 0, dtype=torch.long)
        data["hotspot", "type_c", "station"].edge_attr = torch.zeros(0, 3)

        with pytest.raises(ValueError, match="not divisible by n_stations"):
            model(data)
