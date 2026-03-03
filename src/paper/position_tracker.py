"""
PositionTracker — lightweight live position and P&L tracker for paper trading.

Mirrors the interface of the backtest TradeManager but is slimmed down
for live use: no order management, just position state + Trade history.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.engine.position import Position, PositionSide, Trade
from config.config import COMMISSION_RATE, CONTRACT_MULTIPLIER

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
        if not self._position.is_flat:
            logger.warning("record_open called but position is not flat — ignoring.")
            return

        pos_side = PositionSide.LONG if side.upper() == "LONG" else PositionSide.SHORT

        self._position.side = pos_side
        self._position.entry_price = fill_price
        self._position.quantity = qty
        self._position.entry_time = timestamp
        self._position.stop_loss = stop_loss
        self._position.take_profit = take_profit

        commission = self._calc_commission(fill_price, qty)
        self.cash -= commission

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

        logger.info(
            "Position opened: %s %d @ %.2f | SL=%.2f | TP=%.2f",
            side, qty, fill_price,
            stop_loss or 0, take_profit or 0,
        )

    def record_close(
        self,
        fill_price: float,
        timestamp: datetime,
        exit_reason: str = "",
    ) -> Optional[Trade]:
        """
        Record that an exit order was filled and close the position.

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
            logger.warning("record_close called but no open position — ignoring.")
            return None

        pos = self._position
        exit_commission = self._calc_commission(fill_price, pos.quantity)

        # Calculate gross P&L from position prices (before any commission)
        if pos.is_long:
            gross_pnl = (fill_price - pos.entry_price) * pos.quantity * self.contract_multiplier
        else:
            gross_pnl = (pos.entry_price - fill_price) * pos.quantity * self.contract_multiplier

        # Update cash: gross P&L realised, minus exit commission
        # (entry commission was already subtracted in record_open)
        self.cash += gross_pnl - exit_commission

        # Finalise trade record (Trade.close accumulates exit commission into trade.pnl)
        self._current_trade.close(
            exit_time=timestamp,
            exit_price=fill_price,
            commission=exit_commission,
            exit_reason=exit_reason,
        )

        trade = self._current_trade
        self._current_trade = None
        self._position.close()

        logger.info(
            "Position closed (%s): fill=%.2f | P&L=%.2f",
            exit_reason, fill_price, trade.pnl,
        )
        return trade


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

        Checks the bar's low (for long SL) and high (for long TP) — same
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
