"""
Account state management for backtesting and paper trading.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from config.constants import VN30F_CONTRACT_MULTIPLIER
from src.engine.account.portfolio import PortfolioState
from src.engine.account.position import Position
from src.engine.account.risk_manager import RiskManager
from src.engine.account.sizer import FixedSizer, PositionSizer
from src.engine.account.trade_recorder import TradeRecorder
from src.engine.execution.order import Order, OrderSide, OrderType
from src.engine.execution.slippage import FixedSlippage, SlippageModel
from src.metrics.trade_metrics import Trade
from src.strategy.base import PositionSnapshot
from src.strategy.signal import Signal, TradeSignal

logger = logging.getLogger(__name__)


class AccountState:
    """
    Account state for backtesting and paper trading.

    Usage (backtester):
        ```python
        account = AccountState(initial_capital=500_000_000)
        account.reset()

        # Each bar:
        account.update_daily(timestamp)
        account.check_sl_tp(bar, timestamp)
        if should_close_eod:
            account.close_position(next_bar_open, timestamp, "EOD")
        signal = strategy.generate_signal(bar, account.position_snapshot)
        order  = account.create_order(signal, bar, timestamp)
        if order:
            account.execute_order(order, bar, timestamp)
        account.update_equity(bar["close"])
        ```
    """

    def __init__(
        self,
        initial_capital: float = 500_000_000.0,
        commission_rate: float = 0.00015,
        contract_multiplier: float = VN30F_CONTRACT_MULTIPLIER,
        margin_rate: float = 0.18,
        position_size: int = 1,
        position_sizer: PositionSizer | None = None,
        slippage_model: SlippageModel | None = None,
        use_trailing_stop: bool = False,
        trailing_atr_multiplier: float = 2.0,
        max_daily_loss_pct: float = 0.0,
        enable_async_safety: bool = False,
    ) -> None:
        """
        Args:
            initial_capital: Starting capital
            commission_rate: Commission as fraction of notional (e.g., 0.00015 = 0.015%)
            contract_multiplier: Contract multiplier for futures
            margin_rate: Margin requirement as fraction (e.g., 0.18 = 18%)
            position_size: Default position size (used if no sizer provided)
            position_sizer: Position sizing strategy (default: FixedSizer)
            slippage_model: Slippage model (default: FixedSlippage(0.5))
            use_trailing_stop: Enable trailing stop based on ATR
            trailing_atr_multiplier: ATR multiplier for trailing stop
            max_daily_loss_pct: Max daily loss as % of equity (e.g. 2.0 = 2%, 0 = disabled)
            enable_async_safety: Enable asyncio.Lock for paper trading (default: False for backtesting)
        """
        if initial_capital <= 0:
            raise ValueError("initial_capital must be > 0")
        if not 0 <= commission_rate < 1:
            raise ValueError("commission_rate must be in [0, 1)")
        if position_size <= 0:
            raise ValueError("position_size must be > 0")

        # Configuration
        self.commission_rate = commission_rate
        self.contract_multiplier = contract_multiplier
        self.position_sizer = position_sizer or FixedSizer(position_size)
        self.slippage_model = slippage_model or FixedSlippage(0.5)
        self.use_trailing_stop = use_trailing_stop
        self.trailing_atr_multiplier = trailing_atr_multiplier

        # Components (composition pattern)
        self.portfolio = PortfolioState(initial_capital, margin_rate, contract_multiplier)
        self.position = Position(multiplier=contract_multiplier)
        self.trade_recorder = TradeRecorder(contract_multiplier, enable_async_safety)
        self.risk_manager = RiskManager(margin_rate, contract_multiplier, max_daily_loss_pct)

        # Signal tracking (kept in AccountState for now)
        self._signals: list[dict[str, Any]] = []
        self._enable_async_safety = enable_async_safety
        self._lock: asyncio.Lock | None = asyncio.Lock() if enable_async_safety else None

    # --- Reset ---

    def reset(self) -> None:
        """
        Full reset - call before each backtest.

        Resets:
        - Portfolio (cash and equity)
        - Position to FLAT
        - Trade history
        - Risk manager (daily P&L tracking)
        - Signal history
        - Order ID counter
        """
        self.portfolio.reset()
        self.position.reset()
        self.trade_recorder.reset()
        self.risk_manager.reset()
        self._signals = []
        Order.reset_id_counter()  # Reset order IDs for new backtest

    # --- Properties (delegated to components) ---

    @property
    def cash(self) -> float:
        """Get current cash balance."""
        return self.portfolio.cash

    @property
    def equity(self) -> float:
        """Get current equity."""
        return self.portfolio.equity

    @property
    def trades(self) -> list[Trade]:
        """Return copy of trades list (thread-safe for async if enabled)."""
        return self.trade_recorder.get_trades()

    @property
    def signals(self) -> list[dict[str, Any]]:
        """Return copy of signals list (sync-only, use get_signals_async for async)."""
        return list(self._signals)

    async def get_trades_async(self) -> list[Trade]:
        """Async-safe method to get trades copy (for paper trading)."""
        return await self.trade_recorder.get_trades_async()

    async def get_signals_async(self) -> list[dict[str, Any]]:
        """Async-safe method to get signals copy (for paper trading).

        Lazy lock creation pattern:
        The lock is created inside this async method (not in __init__) to avoid
        "no running event loop" errors. Creating asyncio.Lock() requires an active
        event loop, which may not exist during object initialization in sync contexts.

        By creating the lock lazily on first async access, we ensure:
        1. Backtesting (sync) never creates unnecessary locks
        2. Paper trading (async) creates locks only when needed
        3. No event loop errors during initialization
        """
        if not self._enable_async_safety:
            return list(self._signals)

        # Lazy lock creation (inside event loop)
        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            return list(self._signals)

    @property
    def is_daily_loss_hit(self) -> bool:
        """Check if daily loss limit has been hit."""
        return self.risk_manager.is_daily_loss_hit

    @property
    def position_snapshot(self) -> PositionSnapshot:
        """
        Get position snapshot for strategy signal generation.

        Returns:
            PositionSnapshot with current position state
        """
        return PositionSnapshot(
            is_flat=self.position.is_flat,
            is_long=self.position.is_long,
            is_short=self.position.is_short,
            quantity=self.position.quantity,
            entry_price=self.position.entry_price,
            stop_loss=self.position.stop_loss or 0.0,
            take_profit=self.position.take_profit or 0.0,
        )

    # --- Daily tracking (delegated to RiskManager) ---

    def update_daily(self, timestamp: datetime) -> None:
        """
        Reset daily P&L tracking when new trading day starts.

        Args:
            timestamp: Current timestamp
        """
        self.risk_manager.update_daily(timestamp)

    # --- Order creation ---

    def create_order(
        self,
        signal: TradeSignal,
        bar: dict[str, Any],
        timestamp: datetime,
    ) -> Order | None:
        """
        Create Order from TradeSignal.

        Order creation workflow:
        1. Record signal for analysis (even if no order created)
        2. Handle EXIT signals (close position immediately)
        3. Validate entry conditions (not already in position)
        4. Calculate position size based on risk parameters
        5. Check margin availability at "check price"
        6. Create and return Order object

        Check price vs execution price:
        - Check price: Used for margin calculation (signal.entry_price or current close)
        - Execution price: Determined later in execute_order() with slippage applied

        This separation allows us to validate margin before order submission while
        still applying realistic slippage at execution time.

        Args:
            signal: TradeSignal from strategy
            bar: Current bar data
            timestamp: Current timestamp

        Returns:
            Order if created successfully, None otherwise
            (HOLD signal, already in position, insufficient margin)
        """
        # Record signal (sync-safe for backtesting)
        # For async paper trading, caller should use create_order_async()
        self._signals.append(self._build_signal_record(signal, timestamp))

        return self._create_order_core(signal, bar, timestamp)

    async def create_order_async(
        self,
        signal: TradeSignal,
        bar: dict[str, Any],
        timestamp: datetime,
    ) -> Order | None:
        """Async-safe order creation for paper trading.

        Records signal under lock when async safety is enabled, then delegates
        to the shared order-creation logic.
        """
        signal_record = self._build_signal_record(signal, timestamp)

        if self._enable_async_safety and self._lock is not None:
            async with self._lock:
                self._signals.append(signal_record)
        else:
            self._signals.append(signal_record)

        return self._create_order_core(signal, bar, timestamp)

    @staticmethod
    def _build_signal_record(signal: TradeSignal, timestamp: datetime) -> dict[str, Any]:
        return {
            "datetime": timestamp,
            "signal": signal.signal.value,
            "ord_type": signal.ord_type,
            "entry_price": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "reason": signal.reason,
        }

    def _create_order_core(
        self,
        signal: TradeSignal,
        bar: dict[str, Any],
        timestamp: datetime,
    ) -> Order | None:
        """Shared core logic for create_order/create_order_async."""

        if signal.signal == Signal.HOLD:
            return None

        # EXIT signal - close position immediately
        if signal.signal == Signal.EXIT:
            if not self.position.is_flat:
                self.close_position(bar["close"], timestamp, signal.reason or "Exit signal")
            return None

        # LONG / SHORT - validate entry conditions
        if signal.signal not in (Signal.LONG, Signal.SHORT):
            return None
        if not self.position.is_flat:
            return None

        # V2 fix: Clarified margin check price
        # Use signal entry price if specified, otherwise current close
        # This is the "check price" for margin calculation only
        # Actual execution price will be determined in execute_order with slippage
        check_price = signal.entry_price if signal.entry_price > 0 else bar["close"]

        quantity = self.position_sizer.calculate_size(
            equity=self.equity,
            entry_price=check_price,
            stop_loss=signal.stop_loss,
            contract_multiplier=self.contract_multiplier,
        )
        if quantity <= 0:
            return None

        # Check margin availability at check_price
        # Note: Slippage will be applied later in execute_order
        max_qty = self.risk_manager.max_affordable_quantity(
            check_price, self.portfolio.available_cash, self.position
        )
        if max_qty <= 0:
            return None
        quantity = min(quantity, max_qty)

        side = OrderSide.BUY if signal.is_long else OrderSide.SELL
        requested_ord_type = signal.ord_type.upper()

        if requested_ord_type == "LIMIT":
            limit_price = signal.entry_price if signal.entry_price > 0 else check_price
            return Order(
                order_type=OrderType.LIMIT,
                side=side,
                quantity=quantity,
                limit_price=limit_price,
                stop_loss=signal.stop_loss or None,
                take_profit=signal.take_profit or None,
                created_at=timestamp,
                symbol=bar.get("symbol", "VN30F1M"),
            )

        if requested_ord_type != "MARKET":
            logger.warning("Unknown ord_type '%s', fallback to MARKET", signal.ord_type)

        return Order(
            order_type=OrderType.MARKET,
            side=side,
            quantity=quantity,
            stop_loss=signal.stop_loss or None,
            take_profit=signal.take_profit or None,
            created_at=timestamp,
            symbol=bar.get("symbol", "VN30F1M"),
        )

    # --- Order execution ---

    def execute_order(
        self,
        order: Order,
        bar: dict[str, Any],
        timestamp: datetime,
    ) -> bool:
        """
        Execute pending order against current bar.

        Execution workflow:
        1. Determine fill price based on order type and bar data
        2. Apply slippage model (simulates market impact and spread)
        3. Re-check margin after slippage (price may have moved against us)
        4. Calculate commission
        5. Fill order and update account state

        Why re-check margin after slippage?
        Slippage makes entry more expensive (buy higher, sell lower), so we need
        to verify we still have enough margin after the slippage adjustment.
        Without this check, we could enter positions we can't afford.

        Args:
            order: Order to execute
            bar: Current bar data
            timestamp: Current timestamp

        Returns:
            True if order filled successfully, False otherwise
        """
        try:
            exec_price = self._determine_exec_price(order, bar)
            if exec_price is None:
                return False

            # V2 fix: Apply slippage to get actual execution price
            # This is the ONLY place where slippage is applied for entry
            exec_price, slippage_point = self.slippage_model.calculate(exec_price, order.side)

            # Re-check affordability after slippage adjustment
            # Slippage makes entry more expensive, so we need to verify margin again
            max_qty = self.risk_manager.max_affordable_quantity(
                exec_price, self.portfolio.available_cash, self.position
            )
            if max_qty <= 0:
                return False

            if order.quantity > max_qty:
                logger.info("Reducing order qty %d->%d (margin)", order.quantity, max_qty)
                order.quantity = max_qty

            commission = self._calc_commission(exec_price, order.quantity)
            order.fill(
                price=exec_price,
                timestamp=timestamp,
                commission=commission,
                slippage=slippage_point,
            )

            # Deduct entry commission from cash
            self.portfolio.deduct_cash(commission)

            # Open position + create Trade record
            self._open_position(order, timestamp)
            return True

        except KeyError as e:
            logger.error("Missing bar data for order execution: %s", e)
        except Exception as e:
            logger.error("Order execution failed: %s", e, exc_info=True)
        return False

    def _determine_exec_price(self, order: Order, bar: dict[str, Any]) -> float | None:
        """
        Determine fill price for order.

        Args:
            order: Order to fill
            bar: Current bar data

        Returns:
            Execution price if order can be filled, None if limit not reached
        """
        if order.order_type == OrderType.MARKET:
            return float(bar["open"])

        if order.limit_price is None:
            logger.error("LIMIT order missing limit_price: %s", order)
            return None

        if order.is_buy:
            if bar["low"] > order.limit_price:
                return None  # Limit not reached
            return bar["open"] if bar["open"] <= order.limit_price else order.limit_price

        # SELL limit
        if bar["high"] < order.limit_price:
            return None
        return bar["open"] if bar["open"] >= order.limit_price else order.limit_price

    # --- Position management ---

    def _open_position(self, order: Order, timestamp: datetime) -> Trade:
        """Open position and create trade record."""
        self.position.open(order, timestamp)
        return self.trade_recorder.open_trade(order, timestamp)

    def close_position(
        self,
        exit_price: float,
        timestamp: datetime,
        exit_reason: str = "",
        apply_slippage: bool = True,
    ) -> Trade | None:
        """
        Close current position.

        Args:
            exit_price: Exit price
            timestamp: Exit timestamp
            exit_reason: Reason for exit (e.g., "Stop loss", "EOD")
            apply_slippage: Whether to apply slippage model

        Returns:
            Closed Trade record, or None if no position open
        """
        if self.position.is_flat:
            return None

        # Apply slippage if requested
        exit_side = OrderSide.SELL if self.position.is_long else OrderSide.BUY
        if apply_slippage:
            exit_price, _ = self.slippage_model.calculate(exit_price, exit_side)

        # Calculate exit commission
        commission = self._calc_commission(exit_price, self.position.quantity)

        # Close trade record (calculates P&L)
        trade = self.trade_recorder.close_trade(
            self.position, exit_price, commission, timestamp, exit_reason
        )

        if trade:
            # Update portfolio cash
            self.portfolio.deduct_cash(commission)  # Exit commission
            self.portfolio.add_cash(trade.gross_pnl)  # Realized P&L

            # Record P&L for daily loss limit tracking
            self.risk_manager.record_trade_pnl(trade.pnl, self.portfolio.equity)

        # Close position
        self.position.close()
        return trade

    def check_sl_tp(self, bar: dict[str, Any], timestamp: datetime) -> None:
        """
        Check and execute stop loss / take profit.

        Logic:
        1. Check gap at open (SL first, then TP)
        2. Check intrabar range (SL first, then TP)
        3. Update trailing stop if enabled

        Note: SL is checked before TP in same bar (conservative bias)
        since we don't have tick data to know which hit first.

        Gap handling:
        When a bar opens with a gap (open != previous close), we check if the gap
        jumped over our SL/TP levels. If so, we execute at the open price (not the
        SL/TP level) since that's the first available execution price. We also skip
        slippage for gap executions since the gap itself represents the slippage.

        Args:
            bar: Current bar data (must have open, high, low, close)
            timestamp: Current timestamp
        """
        if self.position.is_flat:
            return

        open_price = float(bar.get("open", bar["close"]))

        # Gap SL check - execute at open if gap jumped over SL
        if self.position.stop_loss is not None:  # noqa: SIM102
            if (self.position.is_long and open_price <= self.position.stop_loss) or (
                self.position.is_short and open_price >= self.position.stop_loss
            ):
                # Gap execution: use open price, skip slippage (gap IS the slippage)
                self.close_position(
                    open_price, timestamp, "Stop loss (gap at open)", apply_slippage=False
                )
                return

        # Gap TP check - execute at open if gap jumped over TP
        if self.position.take_profit is not None:  # noqa: SIM102
            if (self.position.is_long and open_price >= self.position.take_profit) or (
                self.position.is_short and open_price <= self.position.take_profit
            ):
                self.close_position(
                    open_price, timestamp, "Take profit (gap at open)", apply_slippage=False
                )
                return

        # Intrabar SL (check low for long, high for short)
        # Use bar extremes to detect if SL was hit during the bar
        sl_price = bar["low"] if self.position.is_long else bar["high"]
        if self.position.check_stop_loss(sl_price):
            # Execute at SL level (not bar extreme) with slippage
            self.close_position(
                self.position.stop_loss or bar["close"],
                timestamp,
                "Stop loss",
            )
            return

        # Intrabar TP (check high for long, low for short)
        tp_price = bar["high"] if self.position.is_long else bar["low"]
        if self.position.check_take_profit(tp_price):
            self.close_position(
                self.position.take_profit or bar["close"],
                timestamp,
                "Take profit",
            )
            return

        # Trailing stop update (only if position still open)
        if self.use_trailing_stop and self.position.stop_loss is not None:
            self._update_trailing_stop(bar)

    def _update_trailing_stop(self, bar: dict[str, Any]) -> None:
        atr = next(
            (float(v) for k, v in bar.items() if str(k).startswith("atr_") and v and float(v) > 0),
            0.0,
        )
        if atr <= 0:
            return
        trail = self.trailing_atr_multiplier * atr
        close = float(bar["close"])
        if self.position.is_long:
            new_sl = close - trail
            if self.position.stop_loss is None or new_sl > self.position.stop_loss:
                self.position.stop_loss = new_sl
        elif self.position.is_short:
            new_sl = close + trail
            if self.position.stop_loss is None or new_sl < self.position.stop_loss:
                self.position.stop_loss = new_sl

    # --- Equity (delegated to PortfolioState) ---

    def update_equity(self, current_price: float) -> None:
        """
        Update equity based on current price.

        Equity = Cash + Unrealized P&L

        Args:
            current_price: Current market price
        """
        if not self.position.is_flat:
            self.position.update_unrealized_pnl(current_price)
        self.portfolio.update_equity(self.position)

    # --- Helpers ---

    def _calc_commission(self, price: float, quantity: int) -> float:
        """Calculate commission for a trade."""
        return price * quantity * self.contract_multiplier * self.commission_rate

    def calc_commission(self, price: float, quantity: int) -> float:
        """Public wrapper - calculate commission for a trade."""
        return self._calc_commission(price, quantity)

    def open_position(self, order: Order, timestamp: datetime) -> Trade:
        """Public wrapper - open position and create trade record."""
        return self._open_position(order, timestamp)
