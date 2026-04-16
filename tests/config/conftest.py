"""
Pytest fixtures for config tests.
"""

import pytest


@pytest.fixture
def valid_risk_config_dict():
    """Valid RiskConfig dictionary."""
    return {
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
    }


@pytest.fixture
def valid_orb_strategy_dict():
    """Valid ORBStrategyConfig dictionary."""
    return {
        "resample_freq": "5min",
        "orb_minutes": 20,
        "atr_period": 14,
        "atr_tp_multiplier": 2.0,
        "atr_sl_multiplier": 1.5,
        "breakout_buffer": 0.1,
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
    }
