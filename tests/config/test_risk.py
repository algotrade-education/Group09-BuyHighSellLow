"""
Test suite for RiskConfig.
"""

import pytest
from pydantic import ValidationError

from config.schemas.base import RiskConfig


class TestRiskConfig:
    """Test RiskConfig validation logic."""

    def test_valid_config(self):
        """Test valid RiskConfig creation."""
        cfg = RiskConfig(min_position_size=1, max_position_size=5)
        assert cfg.min_position_size == 1
        assert cfg.max_position_size == 5

    def test_max_less_than_min_raises(self):
        """Test max_position_size < min_position_size raises error."""
        with pytest.raises(ValidationError, match="max_position_size"):
            RiskConfig(min_position_size=5, max_position_size=3)

    def test_max_daily_loss_upper_bound(self):
        """Test max_daily_loss upper bound validation."""
        with pytest.raises(ValidationError):
            RiskConfig(max_daily_loss=0.5)  # > 0.20

    def test_max_daily_loss_lower_bound(self):
        """Test max_daily_loss lower bound validation."""
        with pytest.raises(ValidationError):
            RiskConfig(max_daily_loss=0.0)  # < 0.001

    def test_max_daily_loss_valid_range(self):
        """Test max_daily_loss accepts valid values."""
        cfg = RiskConfig(max_daily_loss=0.02)
        assert cfg.max_daily_loss == 0.02

    def test_unknown_field_rejected(self):
        """Test unknown fields are rejected (V1 bug fix)."""
        # V1 bug: silently ignore unknown fields
        # V2: extra="forbid" raises ValidationError
        with pytest.raises(ValidationError, match="Extra inputs"):
            RiskConfig(unknown_field="value")

    def test_string_int_coercion_rejected(self):
        """Test string to int coercion is rejected (V1 bug fix)."""
        # V1 bug: "5" was accepted silently
        # V2: strict type - int field must be int
        with pytest.raises(ValidationError):
            RiskConfig(min_position_size="five")

    def test_default_values(self):
        """Test default values are set correctly."""
        cfg = RiskConfig()

        assert cfg.min_position_size == 1
        assert cfg.max_position_size == 10
        assert cfg.risk_per_trade_pct == 1.0
        assert cfg.max_daily_loss == 0.02
        assert cfg.use_trailing_stop is False

    def test_trailing_atr_multiplier_bounds(self):
        """Test trailing_atr_multiplier validation."""
        cfg = RiskConfig(trailing_atr_multiplier=2.5)
        assert cfg.trailing_atr_multiplier == 2.5

        with pytest.raises(ValidationError):
            RiskConfig(trailing_atr_multiplier=0.0)  # Should be > 0

    def test_entry_cutoff_seconds_bounds(self):
        """Test entry_cutoff_seconds validation."""
        cfg = RiskConfig(entry_cutoff_seconds=120)
        assert cfg.entry_cutoff_seconds == 120

        with pytest.raises(ValidationError):
            RiskConfig(entry_cutoff_seconds=-1)  # Should be >= 0
