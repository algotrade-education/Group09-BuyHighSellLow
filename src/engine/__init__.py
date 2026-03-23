"""
Trading engine for backtesting and paper trading.

This package provides a comprehensive trading engine with:
- Backtesting: Bar-by-bar execution with realistic order fills
- Account management: Position tracking, margin, commission
- Order execution: Market/limit orders with slippage models
- Session management: Trading hours, EOD close, entry cutoffs
- Position sizing: Fixed, risk-based, Kelly, volatility-adjusted
- Risk management: Stop loss, take profit, trailing stops, daily loss limits
- Performance tracking: Equity curve, trade metrics, MAE/MFE

Main components:
- Backtester: Main backtesting engine
- AccountState: Account and position management
- Order: Order lifecycle and execution
- Position: Position tracking with MAE/MFE
- SessionManager: Trading session rules
- EquityTracker: Equity curve tracking
- BacktestResult: Result container and serialization

Quick start:
    ```python
    from src.engine import Backtester, BacktestResult
    from src.strategy import ORBStrategy

    strategy = ORBStrategy(config)
    backtester = Backtester(strategy, initial_capital=500_000_000)
    result = backtester.run(data)

    print(f"Total trades: {result.total_trades}")
    print(f"Win rate: {result.win_rate:.2f}%")
    print(f"Profit factor: {result.profit_factor:.2f}")
    ```
"""

# Core backtesting
# Account management
from src.engine.account import (
    AccountState,
    FixedSizer,
    KellySizer,
    PercentEquitySizer,
    PercentRiskSizer,
    Position,
    PositionSide,
    PositionSizer,
    VolatilityAdjustedSizer,
)
from src.engine.backtester import Backtester

# Equity tracking
from src.engine.equity_tracker import EquityTracker, SimpleEquityTracker

# Order execution
from src.engine.execution import (
    FixedSlippage,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    PercentageSlippage,
    SlippageModel,
    VolatilityAdjustedSlippage,
    VolumeBasedSlippage,
    ZeroSlippage,
)
from src.engine.result import BacktestResult

# Session management
from src.engine.session import AlwaysOpenSession, SessionManager, VN30Session

__all__ = [
    # Core
    "Backtester",
    "BacktestResult",
    # Account
    "AccountState",
    "Position",
    "PositionSide",
    "PositionSizer",
    "FixedSizer",
    "PercentRiskSizer",
    "PercentEquitySizer",
    "KellySizer",
    "VolatilityAdjustedSizer",
    # Execution
    "Order",
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "SlippageModel",
    "FixedSlippage",
    "PercentageSlippage",
    "VolatilityAdjustedSlippage",
    "VolumeBasedSlippage",
    "ZeroSlippage",
    # Session
    "SessionManager",
    "AlwaysOpenSession",
    "VN30Session",
    # Tracking
    "EquityTracker",
    "SimpleEquityTracker",
]
