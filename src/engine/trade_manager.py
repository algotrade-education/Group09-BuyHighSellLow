"""
Trade Manager module responsible for managing trade signals and positions.
"""

from datetime import datetime
import logging
from typing import Any, Dict, List, Optional

from src.engine.order import Order, OrderSide, OrderType
from src.engine.position import Position, Trade
from src.engine.position_sizer import PositionSizer
from src.strategy.base import Signal, TradeSignal

logger = logging.getLogger(__name__)


class TradeManager:
    """
    Manages trade signals and positions for a trading strategy.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.00015,
        slippage_points: float = 0.5,
        contract_multiplier: float = 1.0,
        margin_rate: float = 0.18,
        position_size: int = 1,
        position_sizer: Optional["PositionSizer"] = None,
        use_trailing_stop: bool = False,
        trailing_atr_multiplier: float = 2.0,
        max_daily_loss_pct: float = 0.0,
    ) -> None:
        """
        Initialize the Trade Manager.

        Args:
            initial_capital: Starting capital for trading
            commission_rate: Commission rate per trade (e.g., 0.00015 for 0.015%)
            slippage_points: Slippage in price points (e.g., 0.5 for half a point)
            contract_multiplier: Multiplier for contract size (e.g., 1 for index futures)
            margin_rate: Margin requirement as a percentage (e.g., 0.18 for 18%)
            position_size: Number of contracts to trade per signal
            use_trailing_stop: If True, move SL to lock in profits as price moves
            trailing_atr_multiplier: Trail distance in ATR units
            max_daily_loss_pct: Max daily loss as fraction of equity (0 = disabled)
        """

        # Validate inputs
        if initial_capital <= 0:
            raise ValueError("Initial capital must be greater than zero.")
        if not (0 <= commission_rate < 1):
            raise ValueError("Commission rate must be between 0 and 1.")
        if position_size <= 0:
            raise ValueError("Position size must be greater than zero.")

        # Initialize attributes
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_points = slippage_points
        self.contract_multiplier = contract_multiplier
        self.margin_rate = margin_rate
        self.position_size = position_size
        self.position_sizer = position_sizer
        self.use_trailing_stop = use_trailing_stop
        self.trailing_atr_multiplier = trailing_atr_multiplier
        self.max_daily_loss_pct = max_daily_loss_pct

        # State
        self.position = Position(multiplier=contract_multiplier)
        self.equity = initial_capital  # Track equity separately from cash to account for unrealized PnL
        self.cash = initial_capital  # Track cash separately to account for commissions and realized PnL

        self._signals: List[Dict[str, Any]] = []  # Store trade signals for analysis
        self._trades: List[Trade] = []  # Store completed trades for analysis
        self._trade_counter: int = 0
        self._current_trade: Optional[Trade] = None  # Track the current open trade
        self._daily_pnl: float = 0.0  # Track daily P&L
        self._current_trading_date = None  # Track current day for daily reset
        self._daily_loss_hit: bool = False  # Flag when max daily loss is reached

    def reset(self) -> None:
        """Reset state for new backtest."""
        self.position.reset()
        self.equity = self.initial_capital
        self.cash = self.initial_capital
        self._signals = []
        self._trades = []
        self._trade_counter = 0
        self._current_trade = None
        self._daily_pnl = 0.0
        self._current_trading_date = None
        self._daily_loss_hit = False

    @property
    def trades(self) -> List[Trade]:
        """Get list of completed trades."""
        return self._trades

    def get_signals(self) -> List[Dict[str, Any]]:
        """Get processed signals."""
        return self._signals

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: Optional[float] = None,
        **kwargs,
    ) -> int:
        """
        Calculate position size dynamically.

        If a position_sizer is configured, uses it.
        Otherwise returns fixed position_size.

        Args:
            entry_price: Intended entry price
            stop_loss: Stop loss price if available
            **kwargs: Additional parameters (e.g., atr for volatility sizing)

        Returns:
            Number of contracts to trade
        """
        if self.position_sizer is not None:
            return self.position_sizer.calculate_size(
                equity=self.equity,
                entry_price=entry_price,
                stop_loss=stop_loss,
                contract_multiplier=self.contract_multiplier,
                **kwargs,
            )

        # Default: fixed position size
        return self.position_size

    def _check_margin(self, price: float, quantity: int) -> bool:
        """
        Check if sufficient margin is available.

        Required Margin = Price * Quantity * Multiplier * Margin Rate
        Available Margin = Equity - Used Margin

        Args:
            price: Execution price
            quantity: Number of contracts

        Returns:
            True if sufficient margin, False otherwise
        """
        required_margin = price * quantity * self.contract_multiplier * self.margin_rate

        # Calculate used margin for existing position
        used_margin = 0.0
        if not self.position.is_flat:
            # Mark-to-market margin calculation
            # Could use entry_price or current price. Using entry logic assumes initial margin locked at entry.
            used_margin = (
                self.position.entry_price
                * self.position.quantity
                * self.contract_multiplier
                * self.margin_rate
            )

        available_equity = self.equity - used_margin

        if available_equity < required_margin:
            logger.warning(
                "Insufficient margin: Required=%.2f, Available=%.2f",
                required_margin,
                available_equity,
            )
            return False

        return True

    def _calculate_commission(self, price: float, quantity: int) -> float:
        """
        Calculate commission for a trade.
        The commission is calculated on the notional value of the trade, which is:
            Notional Value = Price * Quantity * Multiplier

        Args:
            price: Execution price
            quantity: Number of contracts

        Returns:
            Commission amount
        """
        return price * quantity * self.contract_multiplier * self.commission_rate

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        """
        Apply slippage to execution price.

        For buys, slippage increases the price; for sells, it decreases the price.

        Args:
            price: Original execution price
            side: Order side (BUY or SELL)

        Returns:
            Adjusted price after slippage
        """

        if side == OrderSide.BUY:
            return price + self.slippage_points
        else:
            return price - self.slippage_points

    def execute_order(
        self,
        order: Order,
        bar: Dict[str, Any],
        timestamp: datetime,
    ) -> bool:
        """
        Execute the order based on the current bar data.

        If it's a market order, it fills at the bar's open price.
        If it's a limit order, it checks if the limit price was reached during the bar and fills accordingly.

        Args:
            order: Order to execute
            bar: Current bar data
            timestamp: Execution timestamp

        Returns:
            True if order was filled
        """
        try:
            # Determine execution price
            if order.order_type == OrderType.MARKET:
                exec_price = bar["open"]  # Market orders filled at bar open price
            else:  # LIMIT
                if order.limit_price is None:
                    logger.error("Limit order missing limit_price: %s", order)
                    return False

                if order.is_buy:
                    # Check limit if price was reached during bar
                    if bar["low"] > order.limit_price:
                        return False  # Limit not reached

                    # Limit was reached, calculate fill price
                    if bar["open"] <= order.limit_price:
                        exec_price = bar["open"]  # Filled at open if better
                    else:
                        exec_price = order.limit_price

                elif order.is_sell:
                    if bar["high"] < order.limit_price:
                        return False  # Limit not reached

                    if bar["open"] >= order.limit_price:
                        exec_price = bar["open"]
                    else:
                        exec_price = order.limit_price

                else:
                    return False

            exec_price = float(
                exec_price or 0.0
            )  # Ensure exec_price is a float and not None

            # Apply slippage
            exec_price = self._apply_slippage(exec_price, order.side)

            # Calculate commission
            commission = self._calculate_commission(exec_price, order.quantity)

            # Fill order
            order.fill(
                price=exec_price,
                timestamp=timestamp,
                commission=commission,
                slippage=self.slippage_points,
            )

            # Deduct commission from cash (Entry commission)
            self.cash -= commission

            logger.debug("Order executed: %s", order)
            return True
        except KeyError as e:
            logger.error("Missing required bar data for order execution: %s", e)
            return False
        except ValueError as e:
            logger.error("Invalid order parameters: %s", e)
            return False
        except Exception as e:
            logger.error(
                "Unexpected error executing order %s: %s", order, e, exc_info=True
            )
            return False

    def create_order_from_signal(
        self,
        signal: TradeSignal,
        bar: Dict[str, Any],
        timestamp: datetime,
    ) -> Optional[Order]:
        """
        Create an order based on the trading signal and current bar data.

        Args:
            signal: Trading signal
            bar: Current bar data
            timestamp: Current timestamp

        Returns:
            Order if created, None otherwise
        """
        # Record signal
        self._signals.append(
            {
                "datetime": timestamp,
                "signal": signal.signal.value,
                "entry_price": signal.entry_price,
                "take_profit": signal.take_profit,
                "stop_loss": signal.stop_loss,
                "reason": signal.reason,
            }
        )

        # Ignore HOLD signals
        if signal.signal == Signal.HOLD:
            return None

        # Handle close signal
        if signal.signal == Signal.CLOSE:
            # This should return Order CLOSE to match with Next Open.
            # But for simplicity, we see this as MOC (market on close) at current bar close price.
            self.close_position(
                exit_price=bar["close"],
                timestamp=timestamp,
                exit_reason=signal.reason,
            )
            return None

        # Handle entry signals
        if signal.signal in (Signal.LONG, Signal.SHORT):
            if not self.position.is_flat:
                return None  # Already in position

            # Determine entry price for check
            check_price = signal.entry_price if signal.entry_price > 0 else bar["close"]

            # Calculate position size (can be dynamic)
            quantity = self.calculate_position_size(
                entry_price=check_price,
                stop_loss=signal.stop_loss,
            )

            # Validate quantity
            if quantity <= 0:
                logger.warning("Invalid position size calculated: %d", quantity)
                return None

            # Check Margin
            if not self._check_margin(check_price, quantity):
                return None  # Insufficient margin

            # Create order
            side = OrderSide.BUY if signal.is_long else OrderSide.SELL

            # Use limit order if entry_price is provided, else Market
            if signal.entry_price > 0:
                order = Order(
                    order_type=OrderType.LIMIT,
                    side=side,
                    quantity=quantity,
                    limit_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                )
            else:
                order = Order(
                    order_type=OrderType.MARKET,
                    side=side,
                    quantity=quantity,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                )

            return order

        return None

    def open_position(self, order: Order, timestamp: datetime) -> Trade:
        """
        Open a position from a filled order and create a Trade.

        This should only be called after an order has been successfully filled.

        Args:
            order: Filled order to open position from
            timestamp: Time of position opening

        Returns:
            Trade object representing the opened position
        """
        # Update position state
        self.position.open(order, timestamp)

        # Create trade record
        self._trade_counter += 1
        trade = Trade(
            trade_id=self._trade_counter,
            side=self.position.side,
            entry_time=timestamp,
            entry_price=order.filled_price or 0.0,
            quantity=order.quantity,
            commission=order.commission,
            multiplier=self.contract_multiplier,
        )
        self._trades.append(trade)
        self._current_trade = trade

        logger.info("Position opened: %s", order)
        return trade

    def close_position(
        self,
        exit_price: float,
        timestamp: datetime,
        exit_reason: str = "",
        apply_slippage: bool = True,
    ) -> Optional[Trade]:
        """Close current position and finalize the trade."""
        if self.position.is_flat:
            return None

        # Apply slippage to exit price (closing long = sell, closing short = buy)
        if apply_slippage:
            exit_side = OrderSide.SELL if self.position.is_long else OrderSide.BUY
            exit_price = self._apply_slippage(exit_price, exit_side)

        # Calculate commission
        commission = self._calculate_commission(exit_price, self.position.quantity)
        self.cash -= commission

        # Finalize Trade
        if self._current_trade:
            self._current_trade.close(
                exit_price=exit_price,
                exit_time=timestamp,
                commission=commission,
                exit_reason=exit_reason,
            )
            self.cash += self._current_trade.gross_pnl
            logger.info("Position closed (%s): %s", exit_reason, self._current_trade)
        else:
            logger.warning(
                "Position closed but no trade record found — P&L not tracked. "
                "exit_price=%.2f, reason=%s",
                exit_price,
                exit_reason,
            )

        trade = self._current_trade
        self._current_trade = None

        # Reset position state
        self.position.close()

        return trade

    def check_sl_tp(self, bar: Dict[str, Any], timestamp: datetime) -> None:
        """
        Check and execute Stop Loss / Take Profit.

        Handles price gaps at bar open to ensure realistic fills.
        If market gaps through stop loss, fills at open price (worse fill).

        NOTE: SL/TP same-bar priority:
            When both SL and TP could trigger within the same bar, SL is
            checked first. This is a conservative assumption that introduces
            a slight systematic loss bias. Without tick data, it is impossible
            to know which level was hit first intra-bar. Prioritizing SL
            avoids overstating performance. The alternative (TP-first) would
            create a systematic profit bias, which is more dangerous for
            live trading decisions.

        Args:
            bar: Current bar data
            timestamp: Current time on the bar being checked
        """
        if self.position.is_flat:
            return

        open_price = bar.get("open", bar["close"])

        # Event Open occurred before High/Low so we have to check with Open first
        if self.position.stop_loss is not None:
            gap_triggered_sl = False
            if self.position.is_long and open_price <= self.position.stop_loss:
                gap_triggered_sl = True
            elif self.position.is_short and open_price >= self.position.stop_loss:
                gap_triggered_sl = True

            if gap_triggered_sl:
                self.close_position(
                    exit_price=open_price,
                    timestamp=timestamp,
                    exit_reason="Stop loss (gap at open)",
                    apply_slippage=False,  # Gap IS the slippage
                )
                return

        # Check Gap Take Profit
        if self.position.take_profit is not None:
            gap_triggered_tp = False
            if self.position.is_long and open_price >= self.position.take_profit:
                gap_triggered_tp = True
            elif self.position.is_short and open_price <= self.position.take_profit:
                gap_triggered_tp = True

            if gap_triggered_tp:
                self.close_position(
                    exit_price=open_price,
                    timestamp=timestamp,
                    exit_reason="Take profit (gap at open)",
                    apply_slippage=False,  # Gap IS the slippage
                )
                return

        # If Open does not trigger SL/TP, check within bar range
        if self.position.check_stop_loss(
            bar["low"] if self.position.is_long else bar["high"]
        ):
            # Fill at stop loss price (normal case)
            self.close_position(
                exit_price=self.position.stop_loss or bar["close"],
                timestamp=timestamp,
                exit_reason="Stop loss",
            )
            return

        # Check take-profit within bar range (no gap)
        if self.position.check_take_profit(
            bar["high"] if self.position.is_long else bar["low"]
        ):
            # Fill at take profit price (normal case)
            self.close_position(
                exit_price=self.position.take_profit or bar["close"],
                timestamp=timestamp,
                exit_reason="Take profit",
            )
            return

        # --- TRAILING STOP ---
        if self.use_trailing_stop and self.position.stop_loss is not None:
            # Find any ATR column in the bar
            atr = 0.0
            for key in bar:
                if str(key).startswith("atr_"):
                    val = bar[key]
                    if val and val > 0:
                        atr = val
                        break
            if atr > 0:
                trail_distance = self.trailing_atr_multiplier * atr
                close = bar["close"]

                if self.position.is_long:
                    new_sl = close - trail_distance
                    if new_sl > self.position.stop_loss:
                        self.position.stop_loss = new_sl
                elif self.position.is_short:
                    new_sl = close + trail_distance
                    if new_sl < self.position.stop_loss:
                        self.position.stop_loss = new_sl

    def update_daily_pnl(self, timestamp: datetime) -> None:
        """Track daily P&L. Reset on new trading day."""
        trading_date = timestamp.date() if hasattr(timestamp, "date") else None
        if trading_date and trading_date != self._current_trading_date:
            self._current_trading_date = trading_date
            self._daily_pnl = 0.0
            self._daily_loss_hit = False

    def record_trade_pnl(self, pnl: float) -> None:
        """Record P&L from a closed trade for daily tracking."""
        self._daily_pnl += pnl
        if self.max_daily_loss_pct > 0 and self._daily_pnl < -(
            self.max_daily_loss_pct * self.equity
        ):
            self._daily_loss_hit = True
            logger.info(
                "Max daily loss reached: daily_pnl=%.2f, limit=%.2f",
                self._daily_pnl,
                -(self.max_daily_loss_pct * self.equity),
            )

    @property
    def is_daily_loss_hit(self) -> bool:
        """Check if max daily loss has been reached."""
        return self._daily_loss_hit

    def update_equity(self, current_price: float) -> None:
        """
        Update unrealized P&L and total equity.

        Args:
            current_price: Current market price to mark position to market
        """
        if not self.position.is_flat:
            self.position.update_unrealized_pnl(current_price)
        self.equity = self.cash + self.position.unrealized_pnl
