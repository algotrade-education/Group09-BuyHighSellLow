import logging
from typing import Callable, Optional

import pandas as pd

from src.engine.equity_tracker import EquityTracker, SimpleEquityTracker
from src.engine.result import BacktestResult
from src.engine.session_manager import SessionManager, VN30Session

logger = logging.getLogger(__name__)

class Backtester:
    """
    Backtester for evaluating trading strategies on historical data.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.00015,
        slippage_points: float = 0.5,
        contract_multiplier: float = 1.0,
        margin_rate: float = 0.18,
        order_ttl: int = 0,

        equity_tracker: Optional[EquityTracker] = None,
        session_manager: Optional[SessionManager] = None,
    ) -> None:
        """
        Initialize the backtester.

        Args:
            initial_capital: Starting capital for backtesting
            commission_rate: Commission rate per trade (e.g., 0.00015 for 0.015%)
            slippage_points: Slippage in price points (e.g., 0.5 for half a point)
            contract_multiplier: Multiplier for contract size (e.g., 1 for index futures)
            margin_rate: Margin requirement as a percentage (e.g., 0.18 for 18%)
            order_ttl: Time-to-live for orders in minutes (0 means no expiration)
        """

        self.order_ttl = order_ttl # 0 = no expiration
        
        # Session manager
        if session_manager is None:
            self.session_manager = VN30Session()
        else:
            self.session_manager = session_manager

        # Equity tracker
        if equity_tracker is None:
            self.equity_tracker = SimpleEquityTracker()
        else:
            self.equity_tracker = equity_tracker

        logger.info(
            "Backtester initialized with %s and %s",
            self.equity_tracker.__class__.__name__,
            {
                "initial_capital": initial_capital,
                "commission_rate": commission_rate,
                "slippage_points": slippage_points,
                "contract_multiplier": contract_multiplier,
                "margin_rate": margin_rate,
                "order_ttl": order_ttl,
            },
        )
    
    def run(
        self,
        data: pd.DataFrame,
        datetime_column: str = "datetime",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BacktestResult:
        """
        Run the backtest on the provided historical data.

        Args:
            data: Historical price data as a DataFrame
            datetime_column: Name of the column containing datetime information
            progress_callback: Optional callback function for progress updates (current, total)

        Returns:
            BacktestResult: The result of the backtest
        """
        logger.info("Starting backtest with %d data points", len(data))

        # Reset state
        self.equity_tracker.reset()

        total_bars = len(data)

        # Pre-convert for fast iteration
        bars = data.to_dict("records")
        timestamps = pd.to_datetime(data[datetime_column]).tolist()

        # Main Event Loop
        # For each bar, process event in order:
        # 1. Execute pending orders
        # 2. Manage open positions (e.g., check for stop-loss, take-profit, EOD close)
        # 3. Generate new signals (if not skipped by session manager)
        # 4. Update equity curve

        return BacktestResult(
            trades=[],
            equity_curve=pd.DataFrame(),
            signals=[],
            parameters={},
        )