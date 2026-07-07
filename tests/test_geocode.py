"""Unit tests for src/data/geocode.py (polygon country geocoder).

Offline — reads only the committed geojson asset, no network calls.
"""

import numpy as np
import pytest

from src.data.geocode import DEFAULT_BORDERS_PATH, _point_in_ring, country_of
from src.data.hotspot_clustering import _country_from_centroid


class TestCountryOf:
    """Known locations resolve to the right country."""

    @pytest.mark.parametrize(
        ("lon", "lat", "expected"),
        [
            (98.98, 18.79, "Thailand"),  # Chiang Mai city
            (100.53, 13.75, "Thailand"),  # Bangkok
            (96.16, 16.85, "Myanmar"),  # Yangon
            (97.60, 19.60, "Myanmar"),  # just across the border west of Mae Hong Son
            (102.63, 17.96, "Laos"),  # Vientiane
            (101.30, 19.90, "Laos"),  # across the Nan border
            (90.00, 5.00, "other"),  # open sea (Indian Ocean)
            (105.85, 21.03, "other"),  # Hanoi — outside the three countries
        ],
    )
    def test_known_locations(self, lon: float, lat: float, expected: str) -> None:
        assert country_of(lon, lat) == expected

    def test_asset_exists(self) -> None:
        assert DEFAULT_BORDERS_PATH.exists(), "border geojson asset missing from app/assets"

    def test_hotspot_clustering_delegates(self) -> None:
        """_country_from_centroid must agree with the polygon geocoder."""
        for lon, lat in [(98.98, 18.79), (97.60, 19.60), (102.63, 17.96), (90.0, 5.0)]:
            assert _country_from_centroid(lon, lat) == country_of(lon, lat)


class TestPointInRing:
    """Vectorized ray-casting primitive on a simple unit square."""

    def _square(self) -> tuple[np.ndarray, np.ndarray]:
        xs = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64)
        ys = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        return xs, ys

    def test_inside(self) -> None:
        xs, ys = self._square()
        assert _point_in_ring(0.5, 0.5, xs, ys)

    def test_outside(self) -> None:
        xs, ys = self._square()
        assert not _point_in_ring(1.5, 0.5, xs, ys)
        assert not _point_in_ring(0.5, -0.5, xs, ys)
