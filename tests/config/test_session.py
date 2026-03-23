"""
Test suite for session configuration.
"""

from datetime import time

import pytest

from config.schemas.session import Session, VN30SessionConfig


class TestVN30SessionConfig:
    """Test VN30SessionConfig session time logic."""

    @pytest.fixture
    def session_config(self):
        """Create VN30SessionConfig instance."""
        return VN30SessionConfig()

    def test_morning_session(self, session_config):
        """Test morning session time range."""
        assert session_config.get_session(time(9, 0)) == Session.MORNING
        assert session_config.get_session(time(10, 30)) == Session.MORNING
        assert session_config.get_session(time(11, 29)) == Session.MORNING

    def test_morning_end_is_exclusive(self, session_config):
        """Test morning session end time is exclusive."""
        assert session_config.get_session(time(11, 30)) == Session.CLOSED

    def test_afternoon_session(self, session_config):
        """Test afternoon session time range."""
        assert session_config.get_session(time(13, 0)) == Session.AFTERNOON
        assert session_config.get_session(time(14, 0)) == Session.AFTERNOON
        assert session_config.get_session(time(14, 29)) == Session.AFTERNOON

    def test_atc_session(self, session_config):
        """Test ATC session time range."""
        assert session_config.get_session(time(14, 30)) == Session.ATC
        assert session_config.get_session(time(14, 44)) == Session.ATC

    def test_closed_session(self, session_config):
        """Test closed session times."""
        assert session_config.get_session(time(8, 59)) == Session.CLOSED
        assert session_config.get_session(time(12, 0)) == Session.CLOSED
        assert session_config.get_session(time(14, 45)) == Session.CLOSED
        assert session_config.get_session(time(18, 0)) == Session.CLOSED

    def test_signal_not_allowed_during_atc(self, session_config):
        """Test signals are not allowed during ATC."""
        assert not session_config.is_signal_allowed(time(14, 35))

    def test_signal_allowed_during_morning_afternoon(self, session_config):
        """Test signals are allowed during morning and afternoon."""
        assert session_config.is_signal_allowed(time(9, 30))
        assert session_config.is_signal_allowed(time(13, 30))

    def test_bars_per_year_5min(self, session_config):
        """Test bars_per_year calculation for 5-minute frequency."""
        # 255 minutes / 5 = 51 bars/day × 252 = 12,852
        assert session_config.bars_per_year(5) == 12_852

    def test_bars_per_year_1min(self, session_config):
        """Test bars_per_year calculation for 1-minute frequency."""
        assert session_config.bars_per_year(1) == 64_260

    def test_bars_per_year_non_divisible_minutes(self, session_config):
        """Test non-divisible minute frequency uses fractional bars/day."""
        assert session_config.bars_per_year(8) == pytest.approx(8_032.5)

    def test_bars_per_year_1h(self, session_config):
        """Test hourly timeframe string support."""
        assert session_config.bars_per_year("1H") == pytest.approx(1_071.0)

    def test_bars_per_year_1d(self, session_config):
        """Test daily timeframe string support."""
        assert session_config.bars_per_year("1D") == pytest.approx(252.0)

    def test_bars_per_year_invalid_freq(self, session_config):
        """Test bars_per_year raises error for invalid frequency format."""
        with pytest.raises(ValueError, match="must be int minutes"):
            session_config.bars_per_year("abc")
