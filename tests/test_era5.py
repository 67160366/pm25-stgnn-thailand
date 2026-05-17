"""
[TODO: NSC Disclaimer — see booklet page 44]

Tests for src/data/scrapers/era5.py.
"""

import pytest

from src.data.scrapers import era5


class TestFetchEra5:
    """Tests for fetch_era5() function."""

    def test_fetch_era5_raises_not_implemented(self):
        """fetch_era5() should raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            era5.fetch_era5()

    def test_fetch_era5_message_mentions_notes(self):
        """NotImplementedError message should mention SESSION1_NOTES.md."""
        with pytest.raises(NotImplementedError) as exc_info:
            era5.fetch_era5()

        assert "SESSION1_NOTES.md" in str(exc_info.value)

    def test_fetch_era5_message_mentions_implementation(self):
        """NotImplementedError message should mention implementation plan."""
        with pytest.raises(NotImplementedError) as exc_info:
            era5.fetch_era5()

        error_msg = str(exc_info.value).lower()
        assert "not yet implemented" in error_msg
