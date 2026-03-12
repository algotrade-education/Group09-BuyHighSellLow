"""
PositionTracker - Live State and P&L Monitor.

The `PositionTracker` maintains the 'Ground Truth' of the current trading
session. It mirrors the accounting logic of the backtester while adding
support for live synchronization and incremental fills.

Accounting Principles:
- **Cash**: Deducts commissions on entry and exit; adds/subtracts gross P&L.
- **Equity**: Calculated as `Cash + Unrealized P&L`.
- **Fills**: Supports weighted-average entry prices for scaled-in positions.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from config.config import COMMISSION_RATE, CONTRACT_MULTIPLIER
from src.engine.position import Position, PositionSide, Trade

logger = logging.getLogger(__name__)


class PositionTracker:
    """
    Tracks the live paper-trading position, P&L, and closed trade history.

    Usage:
        tracker = PositionTracker(initial_capital=500_000_000)
        # On FIX fill confirmation:
        tracker.record_open(fill_price=1300.0, qty=1, side="LONG",
                            stop_loss=1280.0, take_profit=1340.0,
                            timestamp=datetime.now())
        # On each new bar:
        exit_reason = tracker.check_sl_tp(bar)  # → "STOP_LOSS" / "TAKE_PROFIT" / None
        tracker.update_unrealized(current_price=1310.0)
    """

    def __init__(
        self,
        initial_capital: float = 500_000_000,
        commission_rate: float = COMMISSION_RATE,
        contract_multiplier: float = CONTRACT_MULTIPLIER,
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.contract_multiplier = contract_multiplier

        self.cash: float = initial_capital
        self.equity: float = initial_capital

        self._position = Position(multiplier=contract_multiplier)
        self._trades: List[Trade] = []
        self._trade_counter: int = 0
        self._current_trade: Optional[Trade] = None

        # Partial Exit tracking
        self._cum_exit_qty: int = 0
        self._weighted_exit_price: float = 0.0

        # Equity curve: list of (datetime, equity) snapshots
        self._equity_snapshots: List[Tuple[datetime, float]] = []

    # ------------------------------------------------------------------
    # Position lifecycle
    # ------------------------------------------------------------------

    def record_open(
        self,
        fill_price: float,
        qty: int,
        side: str,  # "LONG" or "SHORT"
        timestamp: datetime,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> None:
        """
        Record that an entry order was filled and open the position.

        Args:
            fill_price:  Actual fill price from FIX execution report.
            qty:         Number of contracts filled.
            side:        'LONG' or 'SHORT'.
            timestamp:   Fill timestamp.
            stop_loss:   SL price from the originating TradeSignal.
            take_profit: TP price from the originating TradeSignal.
        """
        logger.info(
            "record_open called: fill_price=%.2f, qty=%d, side=%s, SL=%.2f, TP=%.2f, current_is_flat=%s",
            fill_price,
            qty,
            side,
            stop_loss or 0,
            take_profit or 0,
            self._position.is_flat,
        )

        pos_side = PositionSide.LONG if side.upper() == "LONG" else PositionSide.SHORT

        # Allow scale-ins / partial fills
        if not self._position.is_flat and self._position.side != pos_side:
            logger.warning(
                "record_open: reversing position directly is unsupported here"
            )
            return

        if self._position.is_flat:
            self._position.side = pos_side
            self._position.entry_price = fill_price
            self._position.quantity = qty
            self._position.entry_time = timestamp
        else:
            # Calculate weighted average entry price
            old_qty = self._position.quantity
            old_price = self._position.entry_price
            new_qty = old_qty + qty
            new_price = ((old_price * old_qty) + (fill_price * qty)) / new_qty

            self._position.quantity = new_qty
            self._position.entry_price = new_price
            logger.info("Position scaled: %d -> %d @ %.2f", old_qty, new_qty, new_price)

        # Update SL/TP if provided (or keep existing)
        if stop_loss is not None:
            self._position.stop_loss = stop_loss
        if take_profit is not None:
            self._position.take_profit = take_profit

        commission = self._calc_commission(fill_price, qty)
        self.cash -= commission

        if self._current_trade is None:
            # New trade
            self._trade_counter += 1
            trade = Trade(
                trade_id=self._trade_counter,
                side=pos_side,
                entry_time=timestamp,
                entry_price=fill_price,
                quantity=qty,
                commission=commission,
                multiplier=self.contract_multiplier,
            )
            self._trades.append(trade)
            self._current_trade = trade
        else:
            # Scale-in: Update existing trade record to match the tracked position
            old_qty = self._current_trade.quantity
            old_price = self._current_trade.entry_price
            new_qty = old_qty + qty
            new_price = ((old_price * old_qty) + (fill_price * qty)) / new_qty

            self._current_trade.quantity = new_qty
            self._current_trade.entry_price = new_price
            self._current_trade.commission += commission

        logger.info(
            "Position opened: %s %d @ %.2f | SL=%.2f | TP=%.2f",
            side,
            qty,
            fill_price,
            stop_loss or 0,
            take_profit or 0,
        )
        logger.info(
            "After record_open: is_flat=%s, side=%s, qty=%d, entry=%.2f, equity=%.0f, cash=%.0f",
            self._position.is_flat,
            self._position.side.value if self._position.side else "NONE",
            self._position.quantity,
            self._position.entry_price,
            self.equity,
            self.cash,
        )

    def record_close(
        self,
        fill_price: float,
        qty: int,
        timestamp: datetime,
        exit_reason: str = "",
    ) -> Optional[Trade]:
        """
        Record that an exit order was filled and close (or partially close) the position.

        Cash accounting:
          - Entry commission was already deducted in record_open.
          - Trade.close() accumulates commission and sets pnl = gross - total_commission.
          - We simply add gross P&L then subtract exit commission, which equals:
              cash += gross_pnl - exit_commission
              = (trade.pnl + total_commission) - exit_commission
              = trade.pnl + entry_commission
            Combined with record_open deduction: net effect = trade.pnl ✓
        """
        if self._position.is_flat or self._current_trade is None:
            logger.warning("record_close called but no open position - ignoring.")
            return None

        pos = self._position
        exit_qty = min(qty, pos.quantity)
        exit_commission = self._calc_commission(fill_price, exit_qty)

        # Calculate gross P&L from position prices (before any commission)
        if pos.is_long:
            gross_pnl = (
                (fill_price - pos.entry_price) * exit_qty * self.contract_multiplier
            )
        else:
            gross_pnl = (
                (pos.entry_price - fill_price) * exit_qty * self.contract_multiplier
            )

        # Update cash: gross P&L realised, minus exit commission
        # (entry commission was already subtracted in record_open)
        self.cash += gross_pnl - exit_commission

        # Partial closes: we keep the trade 'open' in _current_trade
        # but accumulate the weighted average exit price.
        old_exit_qty = self._cum_exit_qty
        old_exit_px = self._weighted_exit_price
        new_exit_qty = old_exit_qty + exit_qty
        self._weighted_exit_price = (
            (old_exit_px * old_exit_qty) + (fill_price * exit_qty)
        ) / new_exit_qty
        self._cum_exit_qty = new_exit_qty

        self._current_trade.commission += exit_commission

        trade = self._current_trade

        self._position.quantity -= exit_qty
        if self._position.quantity <= 0:
            # Full close: finalise the trade record using the blended exit price
            self._current_trade.close(
                exit_time=timestamp,
                exit_price=self._weighted_exit_price,
                commission=0,  # Already added above
                exit_reason=exit_reason,
            )
            self._current_trade = None
            self._position.close()
            # Reset exit tracking
            self._cum_exit_qty = 0
            self._weighted_exit_price = 0.0

        logger.info(
            "Position reduced by %d (%s): fill=%.2f | P&L=%.2f | Remaining: %d",
            exit_qty,
            exit_reason,
            fill_price,
            gross_pnl - exit_commission,
            self._position.quantity,
        )
        return trade

    def sync_position(self, qty: float, avg_price: float) -> None:
        """
        Synchronize tracker state with a pre-existing broker position.

        Allows the engine to restart mid-session without losing track of
        currently open trades. Sets the initial `avg_price` and `quantity`
        to match the REST API's portfolio view.

        Args:
            qty: Signed quantity (positive for Long, negative for Short).
            avg_price: The average entry price reported by the broker.
        """
        if qty == 0:
            return

        pos_side = PositionSide.LONG if qty > 0 else PositionSide.SHORT
        abs_qty = abs(int(qty))

        self._position.side = pos_side
        self._position.entry_price = avg_price
        self._position.quantity = abs_qty
        self._position.entry_time = datetime.now()  # Use current time as fallback

        # Create a dummy "current trade" so record_close can calculate PnL against it
        self._trade_counter += 1
        trade = Trade(
            trade_id=self._trade_counter,
            side=pos_side,
            entry_time=self._position.entry_time,
            entry_price=avg_price,
            quantity=abs_qty,
            commission=self._calc_commission(avg_price, abs_qty),
            multiplier=self.contract_multiplier,
        )
        self._trades.append(trade)
        self._current_trade = trade

        logger.info(
            "Resynced tracked position: %s %d @ %.2f", pos_side.name, abs_qty, avg_price
        )

    # ------------------------------------------------------------------
    # Real-time updates
    # ------------------------------------------------------------------

    def update_unrealized(self, current_price: float) -> None:
        """Mark position to market and update equity."""
        if not self._position.is_flat:
            self._position.update_unrealized_pnl(current_price)
        self.equity = self.cash + self._position.unrealized_pnl

    def equity_snapshot(self, timestamp: datetime) -> Tuple[datetime, float]:
        """Record and return the current equity snapshot."""
        snap = (timestamp, self.equity)
        self._equity_snapshots.append(snap)
        return snap

    def check_sl_tp(self, bar: Dict[str, Any]) -> Optional[str]:
        """
        Check whether SL or TP should be triggered on this bar.

        Checks the bar's low (for long SL) and high (for long TP) - same
        conservative logic as the backtester.

        Returns:
            'STOP_LOSS', 'TAKE_PROFIT', or None.
        """
        if self._position.is_flat:
            return None

        high = bar.get("high", bar.get("close", 0))
        low = bar.get("low", bar.get("close", 0))

        # SL takes priority (conservative, same as backtester)
        if self._position.is_long:
            if self._position.check_stop_loss(low):
                return "STOP_LOSS"
            if self._position.check_take_profit(high):
                return "TAKE_PROFIT"
        else:  # SHORT
            if self._position.check_stop_loss(high):
                return "STOP_LOSS"
            if self._position.check_take_profit(low):
                return "TAKE_PROFIT"

        return None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def position(self) -> Position:
        return self._position

    @property
    def trades(self) -> List[Trade]:
        return self._trades

    @property
    def equity_snapshots(self) -> List[Tuple[datetime, float]]:
        return self._equity_snapshots

    @property
    def is_flat(self) -> bool:
        return self._position.is_flat

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _calc_commission(self, price: float, qty: int) -> float:
        return price * qty * self.contract_multiplier * self.commission_rate
