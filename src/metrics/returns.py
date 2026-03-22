"""
Return calculation utilities.
"""

import numpy as np
import pandas as pd


def _to_numeric_series(values: pd.Series | np.ndarray | list[float]) -> pd.Series:
    """Normalize input values to a clean float Series."""
    if not isinstance(values, pd.Series):
        values = pd.Series(values)

    return pd.to_numeric(values, errors="coerce").dropna().astype(float)


def calculate_returns(
    prices: pd.Series | np.ndarray | list[float],
    method: str = "simple",
) -> pd.Series:
    """
    Calculate returns from price series.

    Args:
        prices: Price series
        method: 'simple' for arithmetic, 'log' for logarithmic

    Returns:
        Returns series
    """
    prices = _to_numeric_series(prices)

    if len(prices) == 0:
        return pd.Series(dtype=float)

    if method == "log":
        if (prices <= 0).any():
            raise ValueError("Log returns require strictly positive prices.")
        log_ret = np.log(prices / prices.shift(1))
        return pd.Series(log_ret, index=prices.index)
    if method == "simple":
        return prices.pct_change(fill_method=None)

    raise ValueError("Invalid method. Use 'simple' or 'log'.")


def calculate_cumulative_returns(
    returns: pd.Series | np.ndarray | list[float],
    starting_value: float = 1.0,
) -> pd.Series:
    """
    Calculate cumulative returns.

    Args:
        returns: Returns series
        starting_value: Starting value (default 1.0 for percentage)

    Returns:
        Cumulative returns series
    """
    returns = _to_numeric_series(returns)

    return (1.0 + returns).cumprod() * float(starting_value)


def calculate_total_return(
    equity: pd.Series | np.ndarray | list[float],
    as_percentage: bool = True,
) -> float:
    """
    Calculate total return from equity curve.

    Args:
        equity: Equity curve values
        as_percentage: Return as percentage

    Returns:
        Total return
    """
    equity = _to_numeric_series(equity)

    if len(equity) < 2:
        return 0.0

    start_value = float(equity.iloc[0])
    end_value = float(equity.iloc[-1])

    if start_value == 0.0:
        return 0.0

    total_return = (end_value - start_value) / start_value

    if as_percentage:
        total_return *= 100

    return total_return


def calculate_annualized_return(
    returns: pd.Series | np.ndarray | list[float],
    periods_per_year: float = 252.0,
) -> float:
    """
    Calculate annualized return.

    Args:
        returns: Returns series
        periods_per_year: Number of periods per year

    Returns:
        Annualized return
    """
    returns = _to_numeric_series(returns)

    if len(returns) == 0:
        return 0.0

    if periods_per_year <= 0:
        return 0.0

    # Calculate compound return
    gross_returns = (1.0 + returns).to_numpy(dtype=np.float64)
    total_return = float(np.prod(gross_returns, dtype=np.float64))

    # Annualize
    n_periods = len(returns)
    years = n_periods / periods_per_year

    if years <= 0:
        return 0.0

    if total_return <= 0.0:
        return -100.0

    annualized = (total_return ** (1.0 / float(years))) - 1.0

    return float(annualized or 0.0) * 100


def calculate_volatility(
    returns: pd.Series | np.ndarray | list[float],
    annualized: bool = True,
    periods_per_year: float = 252.0,
) -> float:
    """
    Calculate volatility (standard deviation of returns).

    Args:
        returns: Returns series
        annualized: Whether to annualize
        periods_per_year: Periods per year for annualization

    Returns:
        Volatility value
    """
    returns = _to_numeric_series(returns)

    if len(returns) == 0:
        return 0.0

    volatility = float(returns.std())

    if annualized and periods_per_year > 0:
        volatility *= np.sqrt(periods_per_year)

    return volatility * 100


def calculate_cagr(
    equity: pd.Series | np.ndarray | list[float],
    periods_per_year: float = 252.0,
) -> float:
    """
    Calculate Compound Annual Growth Rate.

    Args:
        equity: Equity curve values
        periods_per_year: Trading periods per year

    Returns:
        CAGR as percentage
    """
    equity = _to_numeric_series(equity)

    if len(equity) < 2:
        return 0.0

    start_value = float(equity.iloc[0])
    end_value = float(equity.iloc[-1])
    n_periods = len(equity) - 1

    if start_value <= 0 or n_periods <= 0 or periods_per_year <= 0:
        return 0.0

    years = n_periods / periods_per_year
    if years <= 0:
        return 0.0

    ratio = end_value / start_value
    if ratio <= 0:
        return -100.0

    cagr = (ratio ** (1.0 / float(years))) - 1.0

    return float(cagr or 0.0) * 100
