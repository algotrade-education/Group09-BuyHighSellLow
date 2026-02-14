"""
Pytest fixtures and test utilities.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_prices() -> pd.Series:
    """Generate sample price series for testing."""
    np.random.seed(42)
    n = 252  # One trading year

    # Generate random walk with drift
    returns = np.random.normal(0.0005, 0.02, n)
    prices = 1000 * np.exp(np.cumsum(returns))

    return pd.Series(prices)


@pytest.fixture
def sample_returns() -> pd.Series:
    """Generate sample returns series for testing."""
    np.random.seed(42)
    n = 252

    returns = np.random.normal(0.0005, 0.02, n)
    return pd.Series(returns)


@pytest.fixture
def sample_equity_curve() -> pd.Series:
    """Generate sample equity curve for testing."""
    np.random.seed(42)
    n = 252

    returns = np.random.normal(0.001, 0.02, n)
    equity = 100000 * np.exp(np.cumsum(returns))

    return pd.Series(equity)


@pytest.fixture
def sample_ohlc_data() -> pd.DataFrame:
    """Generate sample OHLC data for testing."""
    np.random.seed(42)
    n = 100

    # Generate base prices
    base = 1000
    returns = np.random.normal(0.0005, 0.015, n)
    closes = base * np.exp(np.cumsum(returns))

    # Generate OHLC around close
    volatility = 0.01
    data = []

    start_date = datetime(2024, 1, 1, 9, 0, 0)

    for i, close in enumerate(closes):
        high = close * (1 + abs(np.random.normal(0, volatility)))
        low = close * (1 - abs(np.random.normal(0, volatility)))
        open_price = (high + low) / 2 + np.random.normal(0, volatility * close)

        # Ensure OHLC relationships
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        data.append(
            {
                "datetime": start_date + timedelta(minutes=i),
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
            }
        )

    return pd.DataFrame(data)


@pytest.fixture
def sample_ohlc_with_indicators(sample_ohlc_data: pd.DataFrame) -> pd.DataFrame:
    """Generate OHLC data with pre-calculated indicators."""
    df = sample_ohlc_data.copy()

    # Calculate SMA(20)
    df["sma_20"] = df["close"].rolling(window=20).mean()

    # Calculate SMA slope
    df["sma_20_slope"] = df["sma_20"] - df["sma_20"].shift(1)

    # Calculate Bollinger Bands
    rolling_std = df["close"].rolling(window=20).std()
    df["bb_middle"] = df["sma_20"]
    df["bb_upper"] = df["sma_20"] + (2 * rolling_std)
    df["bb_lower"] = df["sma_20"] - (2 * rolling_std)

    # Drop NaN rows
    df = df.dropna().reset_index(drop=True)

    return df


@pytest.fixture
def uptrend_bar() -> dict:
    """Sample bar in uptrend with long signal conditions."""
    return {
        "datetime": datetime(2024, 1, 15, 10, 30, 0),
        "open": 1010.0,
        "high": 1015.0,
        "low": 1000.0,  # Touches SMA
        "close": 1012.0,  # Closes above SMA
        "sma_20": 1005.0,
        "sma_20_slope": 2.5,  # Positive slope (uptrend)
        "bb_upper": 1030.0,
        "bb_middle": 1005.0,
        "bb_lower": 980.0,
    }


@pytest.fixture
def downtrend_bar() -> dict:
    """Sample bar in downtrend with short signal conditions."""
    return {
        "datetime": datetime(2024, 1, 15, 10, 30, 0),
        "open": 990.0,
        "high": 1000.0,  # Touches SMA
        "low": 985.0,
        "close": 988.0,  # Closes below SMA
        "sma_20": 995.0,
        "sma_20_slope": -2.5,  # Negative slope (downtrend)
        "bb_upper": 1020.0,
        "bb_middle": 995.0,
        "bb_lower": 970.0,
    }


@pytest.fixture
def neutral_bar() -> dict:
    """Sample bar with no signal conditions."""
    return {
        "datetime": datetime(2024, 1, 15, 10, 30, 0),
        "open": 1000.0,
        "high": 1010.0,
        "low": 990.0,
        "close": 1005.0,
        "sma_20": 1000.0,
        "sma_20_slope": 0.0,  # Flat
        "bb_upper": 1020.0,
        "bb_middle": 1000.0,
        "bb_lower": 980.0,
    }


@pytest.fixture
def long_exit_bar() -> dict:
    """Sample bar that triggers long exit (take profit at upper band)."""
    return {
        "datetime": datetime(2024, 1, 16, 14, 0, 0),
        "open": 1025.0,
        "high": 1035.0,
        "low": 1023.0,
        "close": 1032.0,  # At or above upper band
        "sma_20": 1005.0,
        "sma_20_slope": 2.0,
        "bb_upper": 1030.0,
        "bb_middle": 1005.0,
        "bb_lower": 980.0,
    }


@pytest.fixture
def long_stop_bar() -> dict:
    """Sample bar that triggers long stop loss (close below SMA)."""
    return {
        "datetime": datetime(2024, 1, 16, 14, 0, 0),
        "open": 1005.0,
        "high": 1008.0,
        "low": 998.0,
        "close": 1000.0,  # Below SMA
        "sma_20": 1005.0,
        "sma_20_slope": 2.0,
        "bb_upper": 1030.0,
        "bb_middle": 1005.0,
        "bb_lower": 980.0,
    }
