"""
Backtest results class
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd


@dataclass
class BacktestResult:
    """
    Container for backtest results

    Attributes:
        trades (List[Trade]): list of executed trades
        equity_curve (pd.DataFrame): equity curve over time
        signals (List[Dict[str, any]]): list of generated trading signals
        parameters (Dict[str, any]): parameters used for the backtest
    """

    trades: List
    equity_curve: pd.DataFrame
    signals: List[Dict[str, Any]] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
