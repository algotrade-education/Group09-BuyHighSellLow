"""Shared fixtures for strategy tests."""

from datetime import datetime

import pytest


@pytest.fixture
def sample_bar():
    """Create a sample OHLCV bar."""
    return {
        "datetime": datetime(2024, 1, 1, 9, 30),
        "open": 100.0,
        "high": 105.0,
        "low": 99.0,
        "close": 103.0,
        "volume": 1000,
    }


@pytest.fixture
def sample_bar_with_indicators(sample_bar):
    """Create a sample bar with indicators."""
    return {
        **sample_bar,
        "atr_14": 2.0,
        "adx_14": 25.0,
        "volume_ma_20": 1000.0,
    }
