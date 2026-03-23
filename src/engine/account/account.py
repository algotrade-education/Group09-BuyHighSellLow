"""
Account state management for backtesting and paper trading.

This module provides AccountState class that tracks:
- Cash and equity
- Position state with MAE/MFE tracking
- Trade history and signals
- Daily P&L and loss limits
- Margin and commission calculations
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from config.constants import VN30F_CONTRACT_MULTIPLIER
from src.engine.account.position import Position, PositionSide
from src.engine.account.sizer import FixedSizer, PositionSizer
from src.engine.execution.order import Order, OrderSide, OrderType
from src.engine.execution.slippage import FixedSlippage, SlippageModel
from src.metrics.trade_metrics import Trade, TradeSide
from src.strategy.base import PositionSnapshot
from src.strategy.signal import Signal, TradeSignal

logger = logging.getLogger(__name__)


class AccountState:
    """
    Unified account state for backtesting and paper trading.

    Tracks cash, position, trades, signals, and enforces risk limits.

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
            max_daily_loss_pct: Max daily loss as % of equity (0 = disabled)
        """

        if initial_capital <= 0:
            raise ValueError("initial_capital must be > 0")
        if not 0 <= commission_rate < 1:
            raise ValueError("commission_rate must be in [0, 1)")
        if position_size <= 0:
            raise ValueError("position_size must be > 0")

        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.contract_multiplier = contract_multiplier
        self.margin_rate = margin_rate
        self.position_sizer = position_sizer or FixedSizer(position_size)
        self.slippage_model = slippage_model or FixedSlippage(0.5)
        self.use_trailing_stop = use_trailing_stop
        self.trailing_atr_multiplier = trailing_atr_multiplier
        self.max_daily_loss_pct = max_daily_loss_pct

        # ── State - reset() initializes all ──────────────────────
        self.position = Position(multiplier=contract_multiplier)
        self.cash: float = initial_capital
        self.equity: float = initial_capital

        self._trades: list[Trade] = []
        self._signals: list[dict[str, Any]] = []
        self._trade_counter: int = 0
        self._current_trade: Trade | None = None

        self._daily_pnl: float = 0.0
        self._current_date: date | None = None
        self._daily_loss_hit: bool = False

    # ── Reset ─────────────────────────────────────────────────────

    def reset(self) -> None:
        """
        Full reset - call before each backtest.

        Resets:
        - Position to FLAT
        - Cash and equity to initial capital
        - Trade and signal history
        - Daily P&L tracking
        - Order ID counter
        """
        self.position.reset()
        self.cash = self.initial_capital
        self.equity = self.initial_capital
        self._trades = []
        self._signals = []
        self._trade_counter = 0
        self._current_trade = None
        self._daily_pnl = 0.0
        self._current_date = None
        self._daily_loss_hit = False
        Order.reset_id_counter()  # Reset order IDs for new backtest

    # ── Properties ────────────────────────────────────────────────

    @property
    def trades(self) -> list[Trade]:
        return list(self._trades)

    @property
    def signals(self) -> list[dict[str, Any]]:
        return list(self._signals)

    @property
    def is_daily_loss_hit(self) -> bool:
        return self._daily_loss_hit

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

    # ── Daily tracking ────────────────────────────────────────────

    def update_daily(self, timestamp: datetime) -> None:
        """
        Reset daily P&L tracking when new trading day starts.

        Args:
            timestamp: Current timestamp
        """
        trading_date = timestamp.date() if hasattr(timestamp, "date") else None
        if trading_date and trading_date != self._current_date:
            self._current_date = trading_date
            self._daily_pnl = 0.0
            self._daily_loss_hit = False

    def _record_trade_pnl(self, pnl: float) -> None:
        """
        Record trade P&L and check daily loss limit.

        Args:
            pnl: Trade P&L (net after commission)
        """
        self._daily_pnl += pnl
        if self.max_daily_loss_pct > 0:
            limit = -(self.max_daily_loss_pct * self.equity)
            if self._daily_pnl < limit:
                self._daily_loss_hit = True
                logger.info(
                    "Max daily loss hit: daily_pnl=%.0f, limit=%.0f",
                    self._daily_pnl,
                    limit,
                )

    # ── Order creation ────────────────────────────────────────────

    def create_order(
        self,
        signal: TradeSignal,
        bar: dict[str, Any],
        timestamp: datetime,
    ) -> Order | None:
        """
        Create Order from TradeSignal.

        Args:
            signal: TradeSignal from strategy
            bar: Current bar data
            timestamp: Current timestamp

        Returns:
            Order if created successfully, None otherwise
            (HOLD signal, already in position, insufficient margin)
        """
        # Record signal
        self._signals.append(
            {
                "datetime": timestamp,
                "signal": signal.signal.value,
                "entry_price": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "reason": signal.reason,
            }
        )

        if signal.signal == Signal.HOLD:
            return None

        # EXIT signal
        if signal.signal == Signal.EXIT:
            if not self.position.is_flat:
                self.close_position(bar["close"], timestamp, signal.reason or "Exit signal")
            return None

        # LONG / SHORT
        if signal.signal not in (Signal.LONG, Signal.SHORT):
            return None
        if not self.position.is_flat:
            return None

        check_price = signal.entry_price if signal.entry_price > 0 else bar["close"]
        quantity = self.position_sizer.calculate_size(
            equity=self.equity,
            entry_price=check_price,
            stop_loss=signal.stop_loss,
            contract_multiplier=self.contract_multiplier,
        )
        if quantity <= 0:
            return None

        # Check margin availability
        max_qty = self._max_affordable_quantity(check_price)
        if max_qty <= 0:
            logger.warning("Insufficient margin at price %.2f", check_price)
            return None
        quantity = min(quantity, max_qty)

        side = OrderSide.BUY if signal.is_long else OrderSide.SELL

        if signal.entry_price > 0:
            return Order(
                order_type=OrderType.LIMIT,
                side=side,
                quantity=quantity,
                limit_price=signal.entry_price,
                stop_loss=signal.stop_loss or None,
                take_profit=signal.take_profit or None,
                created_at=timestamp,
                symbol=bar.get("symbol", "VN30F1M"),
            )
        return Order(
            order_type=OrderType.MARKET,
            side=side,
            quantity=quantity,
            stop_loss=signal.stop_loss or None,
            take_profit=signal.take_profit or None,
            created_at=timestamp,
            symbol=bar.get("symbol", "VN30F1M"),
        )

    # ── Order execution ───────────────────────────────────────────

    def execute_order(
        self,
        order: Order,
        bar: dict[str, Any],
        timestamp: datetime,
    ) -> bool:
        """
        Execute pending order against current bar.

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

            # Apply slippage
            exec_price, slippage_point = self.slippage_model.calculate(exec_price, order.side)

            # Re-check affordability sau slippage
            max_qty = self._max_affordable_quantity(exec_price)
            if max_qty <= 0:
                logger.warning("No margin at execution price %.2f", exec_price)
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
            self.cash -= commission

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

    # ── Position management ───────────────────────────────────────

    def _open_position(self, order: Order, timestamp: datetime) -> Trade:
        self.position.open(order, timestamp)
        self._trade_counter += 1
        side = TradeSide.LONG if order.is_buy else TradeSide.SHORT
        trade = Trade(
            trade_id=str(self._trade_counter),
            side=side,
            entry_time=timestamp,
            entry_price=order.filled_price or 0.0,
            quantity=order.quantity,
            commission=order.commission,
            stop_loss=order.stop_loss or 0.0,
            take_profit=order.take_profit or 0.0,
        )
        self._trades.append(trade)
        self._current_trade = trade
        logger.info("Position opened: %s @ %.2f", side, order.filled_price)
        return trade

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

        exit_side = OrderSide.SELL if self.position.is_long else OrderSide.BUY
        if apply_slippage:
            exit_price, slippage_point = self.slippage_model.calculate(exit_price, exit_side)

        commission = self._calc_commission(exit_price, self.position.quantity)
        self.cash -= commission

        if self._current_trade:
            # Attach MAE/MFE from Position to Trade record
            self._current_trade.mae = self.position.mae
            self._current_trade.mfe = self.position.mfe

            # Finalize Trade
            trade = self._current_trade
            side = self.position.side
            qty = self.position.quantity
            entry_px = self.position.entry_price

            gross_pnl = (
                (exit_price - entry_px) * qty * self.contract_multiplier
                if side == PositionSide.LONG
                else (entry_px - exit_price) * qty * self.contract_multiplier
            )
            total_commission = trade.commission + commission
            net_pnl = gross_pnl - total_commission

            trade.exit_time = timestamp
            trade.exit_price = exit_price
            trade.commission = total_commission
            trade.gross_pnl = gross_pnl
            trade.pnl = net_pnl
            trade.exit_reason = exit_reason

            # Cash accounting: add back gross_pnl
            # (entry cash deducted at open via margin)
            # Commission for exit already deducted above
            self.cash += gross_pnl

            logger.info(
                "Position closed (%s): pnl=%.0f, reason=%s",
                exit_reason,
                net_pnl,
                exit_reason,
            )

            # V2 fix: always record PnL, not just on SL/TP
            self._record_trade_pnl(net_pnl)

        else:
            logger.warning(
                "Position closed but no trade record - exit_price=%.2f, reason=%s",
                exit_price,
                exit_reason,
            )
            trade = None

        self._current_trade = None
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

        Args:
            bar: Current bar data (must have open, high, low, close)
            timestamp: Current timestamp
        """
        if self.position.is_flat:
            return

        open_price = float(bar.get("open", bar["close"]))

        # Gap SL check
        if self.position.stop_loss is not None:  # noqa: SIM102
            if (self.position.is_long and open_price <= self.position.stop_loss) or (
                self.position.is_short and open_price >= self.position.stop_loss
            ):
                self.close_position(
                    open_price, timestamp, "Stop loss (gap at open)", apply_slippage=False
                )
                return

        # Gap TP check
        if self.position.take_profit is not None:  # noqa: SIM102
            if (self.position.is_long and open_price >= self.position.take_profit) or (
                self.position.is_short and open_price <= self.position.take_profit
            ):
                self.close_position(
                    open_price, timestamp, "Take profit (gap at open)", apply_slippage=False
                )
                return

        # Intrabar SL (check low for long, high for short)
        sl_price = bar["low"] if self.position.is_long else bar["high"]
        if self.position.check_stop_loss(sl_price):
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

        # Trailing stop
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

    # ── Equity ────────────────────────────────────────────────────

    def update_equity(self, current_price: float) -> None:
        """
        Update equity based on current price.

        Equity = Cash + Unrealized P&L

        Args:
            current_price: Current market price
        """
        if not self.position.is_flat:
            self.position.update_unrealized_pnl(current_price)
        self.equity = self.cash + self.position.unrealized_pnl

    # ── Helpers ───────────────────────────────────────────────────

    def _calc_commission(self, price: float, quantity: int) -> float:
        return price * quantity * self.contract_multiplier * self.commission_rate

    def _max_affordable_quantity(self, price: float) -> int:
        """
        Calculate maximum affordable quantity based on available margin.

        Args:
            price: Entry price

        Returns:
            Maximum quantity that can be afforded
        """
        if price <= 0 or self.margin_rate <= 0:
            return 0

        used_margin = 0.0
        if not self.position.is_flat:
            used_margin = (
                self.position.entry_price
                * self.position.quantity
                * self.contract_multiplier
                * self.margin_rate
            )

        available = max(0.0, self.cash - used_margin)
        margin_per = price * self.contract_multiplier * self.margin_rate

        if margin_per <= 0:
            return 0

        return int(available // margin_per)
