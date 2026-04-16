"""
Test suite for ORB strategy configuration.
"""

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from config.schemas.orb import ORBConfig, ORBStrategyConfig


class TestORBStrategyConfig:
    """Test ORBStrategyConfig validation logic."""

    @pytest.fixture
    def base_kwargs(self):
        """Base kwargs for ORBStrategyConfig."""
        return {}

    def test_valid_config(self, base_kwargs):
        """Test valid ORBStrategyConfig creation."""
        cfg = ORBStrategyConfig(**base_kwargs, resample_freq="5min")
        assert cfg.orb_minutes == 20  # default
        assert cfg.atr_period == 14
        assert cfg.use_range_sl is True

    def test_orb_minutes_lower_bound(self, base_kwargs):
        """Test orb_minutes lower bound validation."""
        with pytest.raises(ValidationError):
            ORBStrategyConfig(**base_kwargs, resample_freq="5min", orb_minutes=0)

    def test_orb_minutes_upper_bound(self, base_kwargs):
        """Test orb_minutes upper bound validation."""
        with pytest.raises(ValidationError):
            ORBStrategyConfig(**base_kwargs, resample_freq="5min", orb_minutes=121)

    def test_orb_minutes_valid_range(self, base_kwargs):
        """Test orb_minutes accepts valid values."""
        cfg = ORBStrategyConfig(**base_kwargs, resample_freq="5min", orb_minutes=30)
        assert cfg.orb_minutes == 30

    def test_atr_period_lower_bound(self, base_kwargs):
        """Test atr_period lower bound validation."""
        with pytest.raises(ValidationError):
            ORBStrategyConfig(**base_kwargs, resample_freq="5min", atr_period=4)  # < 5

    def test_atr_period_upper_bound(self, base_kwargs):
        """Test atr_period upper bound validation."""
        with pytest.raises(ValidationError):
            ORBStrategyConfig(**base_kwargs, resample_freq="5min", atr_period=51)  # > 50

    def test_atr_period_valid_range(self, base_kwargs):
        """Test atr_period accepts valid values."""
        cfg = ORBStrategyConfig(**base_kwargs, resample_freq="5min", atr_period=20)
        assert cfg.atr_period == 20

    def test_range_atr_order_validation(self, base_kwargs):
        """Test min_range_atr must be less than max_range_atr."""
        with pytest.raises(ValidationError, match="min_range_atr"):
            ORBStrategyConfig(
                **base_kwargs,
                resample_freq="5min",
                min_range_atr=3.0,
                max_range_atr=2.0,
            )

    def test_range_atr_equal_raises(self, base_kwargs):
        """Test min_range_atr cannot equal max_range_atr."""
        with pytest.raises(ValidationError):
            ORBStrategyConfig(
                **base_kwargs,
                resample_freq="5min",
                min_range_atr=2.0,
                max_range_atr=2.0,
            )

    def test_range_atr_valid(self, base_kwargs):
        """Test valid range_atr configuration."""
        cfg = ORBStrategyConfig(
            **base_kwargs,
            resample_freq="5min",
            min_range_atr=0.5,
            max_range_atr=3.0,
        )
        assert cfg.min_range_atr == 0.5
        assert cfg.max_range_atr == 3.0

    def test_tp_sl_ratio_when_no_range_sl(self, base_kwargs):
        """Test TP must be > SL when not using range SL."""
        with pytest.raises(ValidationError, match="atr_tp_multiplier"):
            ORBStrategyConfig(
                **base_kwargs,
                resample_freq="5min",
                use_range_sl=False,
                atr_tp_multiplier=1.0,
                atr_sl_multiplier=1.5,
            )

    def test_tp_sl_ratio_skipped_when_using_range_sl(self, base_kwargs):
        """Test TP/SL ratio check is skipped when using range SL."""
        cfg = ORBStrategyConfig(
            **base_kwargs,
            resample_freq="5min",
            use_range_sl=True,
            atr_tp_multiplier=1.0,
            atr_sl_multiplier=1.5,
        )
        assert cfg.use_range_sl is True

    def test_invalid_resample_freq(self, base_kwargs):
        """Test invalid resample_freq raises error."""
        with pytest.raises(ValidationError):
            ORBStrategyConfig(**base_kwargs, resample_freq="2min")

    def test_valid_resample_freq_options(self, base_kwargs):
        """Test all valid resample_freq options."""
        valid_freqs = ["1min", "5min", "15min", "30min", "1H"]
        for freq in valid_freqs:
            cfg = ORBStrategyConfig(**base_kwargs, resample_freq=freq)
            assert cfg.resample_freq == freq

    def test_unknown_field_rejected(self, base_kwargs):
        """Test unknown fields are rejected."""
        with pytest.raises(ValidationError):
            ORBStrategyConfig(**base_kwargs, resample_freq="5min", nonexistent_param=123)

    def test_default_values(self, base_kwargs):
        """Test default values are set correctly."""
        cfg = ORBStrategyConfig(**base_kwargs, resample_freq="5min")

        assert cfg.orb_minutes == 20
        assert cfg.atr_period == 14
        assert cfg.atr_tp_multiplier == 2.0
        assert cfg.atr_sl_multiplier == 1.0
        assert cfg.breakout_buffer == 0.0
        assert cfg.use_range_sl is True


class TestORBConfig:
    """Test ORBConfig top-level configuration."""

    @pytest.fixture
    def valid_dict(self):
        """Valid ORBConfig dictionary."""
        return {
            "name": "ORB",
            "version": "2.0.0",
            "strategy": {
                "resample_freq": "5min",
                "orb_minutes": 20,
                "atr_period": 14,
                "atr_tp_multiplier": 2.0,
                "atr_sl_multiplier": 1.0,
                "breakout_buffer": 0.0,
                "use_range_sl": True,
                "min_range_atr": 0.5,
                "max_range_atr": 3.0,
                "long_only": False,
                "use_volume_filter": False,
                "volume_filter_threshold": 0.5,
                "volume_ma_period": 20,
                "use_adx_filter": False,
                "adx_period": 14,
                "adx_min": 20.0,
                "require_close_confirmation": False,
                "max_trades_per_session": 2,
            },
            "risk": {
                "min_position_size": 1,
                "max_position_size": 10,
                "risk_per_trade_pct": 1.0,
                "max_daily_loss": 2.0,
                "use_trailing_stop": True,
                "trailing_atr_multiplier": 2.0,
                "entry_cutoff_seconds": 60,
                "allow_late_entry": False,
                "force_flat_on_session_close": False,
                "defer_exit_outside_session": True,
            },
        }

    def test_from_dict(self, valid_dict):
        """Test creating ORBConfig from dictionary."""
        cfg = ORBConfig.from_dict(valid_dict)
        assert cfg.strategy.orb_minutes == 20
        assert cfg.risk.max_daily_loss == 2.0

    def test_from_json_file(self, valid_dict):
        """Test loading ORBConfig from JSON file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(valid_dict, f)
            tmp_path = f.name

        cfg = ORBConfig.from_json(tmp_path)
        assert cfg.strategy.orb_minutes == 20
        Path(tmp_path).unlink()

    def test_to_json_roundtrip(self, valid_dict):
        """Test saving and loading ORBConfig maintains data."""
        cfg = ORBConfig.from_dict(valid_dict)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name

        cfg.to_json(tmp_path)
        reloaded = ORBConfig.from_json(tmp_path)
        assert reloaded.strategy.orb_minutes == cfg.strategy.orb_minutes
        assert reloaded.risk.max_daily_loss == cfg.risk.max_daily_loss
        Path(tmp_path).unlink()

    def test_missing_required_field_uses_default(self, valid_dict):
        """Test missing fields use default values."""
        data = valid_dict.copy()
        del data["strategy"]["orb_minutes"]
        # orb_minutes has default so no error
        cfg = ORBConfig.from_dict(data)
        assert cfg.strategy.orb_minutes == 20  # default value

    def test_default_json_file_is_valid(self):
        """Test orb_default.json is valid with current schema."""
        cfg = ORBConfig.from_json("config/strategy_params/orb_default.json")
        assert cfg.strategy.orb_minutes > 0
        assert cfg.risk.max_position_size >= cfg.risk.min_position_size

    def test_name_and_version(self, valid_dict):
        """Test name and version fields."""
        cfg = ORBConfig.from_dict(valid_dict)
        assert cfg.name == "ORB"
        assert cfg.version == "2.0.0"
