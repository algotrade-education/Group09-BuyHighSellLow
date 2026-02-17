"""
Return calculation utilities.
"""

from typing import List, Union

import numpy as np
import pandas as pd


def calculate_returns(
    prices: Union[pd.Series, np.ndarray, List[float]],
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
    if isinstance(prices, (list, np.ndarray)):
        prices = pd.Series(prices)

    if method == "log":
        return np.log(prices / prices.shift(1))
    else:  # simple
        return prices.pct_change()


def calculate_cumulative_returns(
    returns: Union[pd.Series, np.ndarray, List[float]],
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
    if isinstance(returns, (list, np.ndarray)):
        returns = pd.Series(returns)

    return (1 + returns).cumprod() * starting_value


def calculate_total_return(
    equity: Union[pd.Series, np.ndarray, List[float]],
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
    if isinstance(equity, (list, np.ndarray)):
        equity = pd.Series(equity)

    if len(equity) < 2:
        return 0.0

    total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]

    if as_percentage:
        total_return *= 100

    return total_return


def calculate_annualized_return(
    returns: Union[pd.Series, np.ndarray, List[float]],
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
    if isinstance(returns, (list, np.ndarray)):
        returns = pd.Series(returns)

    returns = returns.dropna()

    if len(returns) == 0:
        return 0.0

    # Calculate compound return
    total_return = (1 + returns).prod()

    # Annualize
    n_periods = len(returns)
    years = n_periods / periods_per_year

    if years <= 0:
        return 0.0

    annualized = (total_return ** (1 / years)) - 1

    return annualized * 100


def calculate_volatility(
    returns: Union[pd.Series, np.ndarray, List[float]],
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
    if isinstance(returns, (list, np.ndarray)):
        returns = pd.Series(returns)

    volatility = returns.std()

    if annualized:
        volatility *= np.sqrt(periods_per_year)

    return volatility * 100


def calculate_cagr(
    equity: Union[pd.Series, np.ndarray, List[float]],
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
    if isinstance(equity, (list, np.ndarray)):
        equity = pd.Series(equity)

    if len(equity) < 2:
        return 0.0

    start_value = equity.iloc[0]
    end_value = equity.iloc[-1]
    n_periods = len(equity) - 1

    if start_value <= 0 or n_periods <= 0:
        return 0.0

    years = n_periods / periods_per_year
    cagr = ((end_value / start_value) ** (1 / years)) - 1

    return cagr * 100
