"""Shared fixtures for metrics tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def daily_returns():
    """Sample daily returns (252 trading days)."""
    np.random.seed(42)
    return pd.Series(np.random.normal(0.001, 0.02, 252))


@pytest.fixture
def intraday_15m_returns():
    """Sample 15-minute returns for VN30 futures.

    VN30 has 2 sessions:
    - Morning: 9:00-11:30 (10 bars of 15min)
    - Afternoon: 13:00-15:00 (8 bars of 15min)
    Total: 18 bars/day × 240 days = 4320 bars/year
    """
    np.random.seed(42)
    n_bars = 4320  # 1 year of 15-minute data
    return pd.Series(np.random.normal(0.0001, 0.005, n_bars))


@pytest.fixture
def positive_returns():
    """Consistently positive returns."""
    return pd.Series([0.01, 0.02, 0.015, 0.01, 0.025, 0.02])


@pytest.fixture
def negative_returns():
    """Consistently negative returns."""
    return pd.Series([-0.01, -0.02, -0.015, -0.01, -0.025, -0.02])


@pytest.fixture
def zero_returns():
    """Zero returns (no volatility)."""
    return pd.Series([0.0] * 100)


@pytest.fixture
def mixed_returns():
    """Mixed positive and negative returns."""
    return pd.Series([0.02, -0.01, 0.03, -0.02, 0.01, -0.015, 0.025, -0.005])


@pytest.fixture
def equity_curve():
    """Sample equity curve starting at 100."""
    returns = pd.Series([0.02, -0.01, 0.03, -0.05, 0.04, -0.02, 0.01, 0.02])
    return (1 + returns).cumprod() * 100


@pytest.fixture
def equity_with_drawdown():
    """Equity curve with significant drawdown."""
    values = [100, 110, 120, 115, 100, 90, 85, 95, 105, 115, 125]
    return pd.Series(values)


@pytest.fixture
def equity_monotonic_up():
    """Monotonically increasing equity (no drawdown)."""
    return pd.Series([100, 105, 110, 115, 120, 125, 130])


@pytest.fixture
def benchmark_returns():
    """Benchmark returns for Information Ratio tests."""
    np.random.seed(123)
    return pd.Series(np.random.normal(0.0005, 0.015, 252))
