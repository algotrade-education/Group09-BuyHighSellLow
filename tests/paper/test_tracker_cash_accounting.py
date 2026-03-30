"""Numerical tests for PositionTracker cash/equity accounting correctness.

These tests use real commission rates and contract multiplier to verify
that cash accounting is correct after any number of round-trip trades.

Key invariant tested:
    final_cash == initial_capital + sum(trade.pnl for all closed trades)

This test suite is designed to catch bugs like:
- Double-commission charging (commission deducted twice)
- Incorrect gross P&L calculation
- Cash/equity mismatch when position is flat
"""

from datetime import datetime

import pytest

from src.paper.account.tracker import Tracker


class TestTrackerCashAccounting:
    """
    Numerical tests for Tracker cash / equity accounting.

    These tests use real commission rates and the contract multiplier.
    They are designed to catch double-commission or incorrect gross-PnL
    accounting bugs.

    Key invariant after any number of completed round-trips:
        final_cash == initial_capital + sum(trade.pnl for trade in closed_trades)
    """

    INITIAL = 10_000_000
    RATE = 0.0003  # 0.03% per leg - realistic VN30 commission rate
    MULT = 100_000.0  # VN30 contract multiplier

    @pytest.fixture
    def tracker(self):
        """Create tracker with realistic VN30 parameters."""
        return Tracker(
            initial_capital=self.INITIAL,
            commission_rate=self.RATE,
            contract_multiplier=self.MULT,
        )

    def _calc_comm(self, price: float, qty: int = 1) -> float:
        """Calculate commission for a single leg."""
        return price * qty * self.MULT * self.RATE

    # --- Single Trade Invariants ---

    def test_long_winning_trade_cash_equals_initial_plus_pnl(self, tracker):
        """Cash after a profitable LONG round-trip must equal initial + trade.pnl."""
        ts = datetime(2025, 1, 6, 9, 15)
        tracker.record_open(1300.0, qty=1, side="LONG", timestamp=ts)
        tracker.record_close(1350.0, qty=1, timestamp=ts, exit_reason="TP")

        trade = tracker.trades[0]
        expected_cash = self.INITIAL + trade.pnl
        assert tracker.cash == pytest.approx(expected_cash, rel=1e-9), (
            f"Cash mismatch: got {tracker.cash:,.0f}, expected {expected_cash:,.0f}"
        )

    def test_long_losing_trade_cash_equals_initial_plus_pnl(self, tracker):
        """Cash after a losing LONG round-trip must equal initial + trade.pnl (negative)."""
        ts = datetime(2025, 1, 6, 9, 15)
        tracker.record_open(1300.0, qty=1, side="LONG", timestamp=ts)
        tracker.record_close(1270.0, qty=1, timestamp=ts, exit_reason="SL")

        trade = tracker.trades[0]
        expected_cash = self.INITIAL + trade.pnl
        assert tracker.cash == pytest.approx(expected_cash, rel=1e-9)

    def test_short_winning_trade_cash_invariant(self, tracker):
        """Cash after a profitable SHORT round-trip is correct."""
        ts = datetime(2025, 1, 6, 9, 15)
        tracker.record_open(1300.0, qty=1, side="SHORT", timestamp=ts)
        tracker.record_close(1250.0, qty=1, timestamp=ts, exit_reason="TP")

        trade = tracker.trades[0]
        expected_cash = self.INITIAL + trade.pnl
        assert tracker.cash == pytest.approx(expected_cash, rel=1e-9)

    def test_trade_pnl_equals_gross_minus_total_commission(self, tracker):
        """trade.pnl must equal (exit - entry) * qty * multiplier - total_commission."""
        ts = datetime(2025, 1, 6, 9, 15)
        entry, exit_ = 1300.0, 1350.0
        tracker.record_open(entry, qty=1, side="LONG", timestamp=ts)
        tracker.record_close(exit_, qty=1, timestamp=ts, exit_reason="TP")

        trade = tracker.trades[0]
        gross = (exit_ - entry) * 1 * self.MULT
        total_comm = self._calc_comm(entry) + self._calc_comm(exit_)
        assert trade.pnl == pytest.approx(gross - total_comm, rel=1e-9)

    def test_commission_charged_exactly_once_per_leg(self, tracker):
        """
        Commission for each leg (entry + exit) should be deducted exactly once.

        Regression test for the double-commission bug in record_close where
        entry commission was deducted twice (once in record_open, once baked
        into trade.pnl which was then added to cash that already had it removed).
        """
        ts = datetime(2025, 1, 6, 9, 15)
        entry, exit_ = 1300.0, 1300.0  # break-even trade (gross = 0)
        tracker.record_open(entry, qty=1, side="LONG", timestamp=ts)
        tracker.record_close(exit_, qty=1, timestamp=ts, exit_reason="EOD")

        total_comm = self._calc_comm(entry) + self._calc_comm(exit_)
        expected_cash = self.INITIAL - total_comm
        assert tracker.cash == pytest.approx(expected_cash, rel=1e-9), (
            "Break-even trade should only reduce cash by total commission (not 2x or 3x)"
        )

    def test_flat_equity_equals_cash_after_close(self, tracker):
        """When position is flat, equity should equal cash exactly."""
        ts = datetime(2025, 1, 6, 9, 15)
        tracker.record_open(1300.0, qty=1, side="LONG", timestamp=ts)
        tracker.record_close(1320.0, qty=1, timestamp=ts, exit_reason="TP")
        tracker.update_unrealized(1320.0)  # re-mark to market (position flat)

        assert tracker.equity == pytest.approx(tracker.cash, rel=1e-9)

    # --- Multi-Trade Invariants ---

    def test_multi_trade_cash_invariant(self, tracker):
        """
        After N completed round-trips:
          final_cash == initial_capital + sum(trade.pnl)

        This is the fundamental accounting identity. If it fails, there is a
        bug in how cash is updated on open or close.
        """
        ts = datetime(2025, 1, 6, 9, 0)
        trades_data = [
            ("LONG", 1300.0, 1350.0),  # +50 pts
            ("SHORT", 1350.0, 1300.0),  # +50 pts
            ("LONG", 1300.0, 1280.0),  # -20 pts (loss)
            ("SHORT", 1280.0, 1290.0),  # -10 pts (loss)
            ("LONG", 1290.0, 1310.0),  # +20 pts
        ]

        for i, (side, entry, exit_) in enumerate(trades_data):
            t_open = ts.replace(minute=i * 10)
            t_close = ts.replace(minute=i * 10 + 5)
            tracker.record_open(entry, qty=1, side=side, timestamp=t_open)
            tracker.record_close(exit_, qty=1, timestamp=t_close, exit_reason="test")

        total_pnl = sum(t.pnl for t in tracker.trades)
        expected_cash = self.INITIAL + total_pnl
        assert tracker.cash == pytest.approx(expected_cash, rel=1e-9), (
            f"After {len(tracker.trades)} trades: cash={tracker.cash:,.0f}, "
            f"expected={expected_cash:,.0f} (diff={tracker.cash - expected_cash:,.0f})"
        )

    def test_multi_trade_final_equity_consistent_with_net_pnl(self, tracker):
        """Final Equity = Initial Capital + Net P&L (when all positions closed)."""
        ts = datetime(2025, 1, 6, 9, 0)
        for i, (side, entry, exit_) in enumerate(
            [
                ("LONG", 1300.0, 1340.0),
                ("LONG", 1340.0, 1330.0),
            ]
        ):
            tracker.record_open(entry, qty=1, side=side, timestamp=ts.replace(minute=i * 10))
            tracker.record_close(
                exit_,
                qty=1,
                timestamp=ts.replace(minute=i * 10 + 5),
                exit_reason="test",
            )

        tracker.update_unrealized(0.0)  # position is flat, unrealized = 0
        net_pnl = sum(t.pnl for t in tracker.trades)
        assert tracker.equity == pytest.approx(self.INITIAL + net_pnl, rel=1e-9)

    # --- Commission Edge Cases ---

    def test_zero_commission_rate_no_deduction(self):
        """With zero commission rate, cash should only change by gross P&L."""
        tracker = Tracker(
            initial_capital=self.INITIAL,
            commission_rate=0.0,  # Zero commission
            contract_multiplier=self.MULT,
        )

        ts = datetime(2025, 1, 6, 9, 15)
        entry, exit_ = 1300.0, 1350.0
        tracker.record_open(entry, qty=1, side="LONG", timestamp=ts)
        tracker.record_close(exit_, qty=1, timestamp=ts, exit_reason="TP")

        gross_pnl = (exit_ - entry) * 1 * self.MULT
        expected_cash = self.INITIAL + gross_pnl
        assert tracker.cash == pytest.approx(expected_cash, rel=1e-9)

    def test_high_commission_rate_reduces_pnl(self):
        """High commission rate should significantly reduce net P&L."""
        tracker = Tracker(
            initial_capital=self.INITIAL,
            commission_rate=0.01,  # 1% per leg (very high)
            contract_multiplier=self.MULT,
        )

        ts = datetime(2025, 1, 6, 9, 15)
        entry, exit_ = 1300.0, 1350.0
        tracker.record_open(entry, qty=1, side="LONG", timestamp=ts)
        tracker.record_close(exit_, qty=1, timestamp=ts, exit_reason="TP")

        trade = tracker.trades[0]
        gross = (exit_ - entry) * 1 * self.MULT
        entry_comm = entry * 1 * self.MULT * 0.01
        exit_comm = exit_ * 1 * self.MULT * 0.01
        total_comm = entry_comm + exit_comm

        # Net P&L should be significantly reduced by commission
        assert trade.pnl == pytest.approx(gross - total_comm, rel=1e-9)
        assert total_comm > gross * 0.5  # Commission > 50% of gross

    # --- Multiple Quantities ---

    def test_multi_quantity_trade_cash_accounting(self):
        """Test cash accounting with quantity > 1."""
        ts = datetime(2025, 1, 6, 9, 15)
        entry, exit_ = 1300.0, 1350.0
        qty = 3

        tracker = Tracker(
            initial_capital=self.INITIAL,
            commission_rate=self.RATE,
            contract_multiplier=self.MULT,
        )

        tracker.record_open(entry, qty=qty, side="LONG", timestamp=ts)
        tracker.record_close(exit_, qty=qty, timestamp=ts, exit_reason="TP")

        trade = tracker.trades[0]
        gross = (exit_ - entry) * qty * self.MULT
        entry_comm = entry * qty * self.MULT * self.RATE
        exit_comm = exit_ * qty * self.MULT * self.RATE
        total_comm = entry_comm + exit_comm

        expected_pnl = gross - total_comm
        assert trade.pnl == pytest.approx(expected_pnl, rel=1e-9)

        expected_cash = self.INITIAL + expected_pnl
        assert tracker.cash == pytest.approx(expected_cash, rel=1e-9)

    # --- Unrealized P&L ---

    def test_unrealized_pnl_does_not_affect_cash(self, tracker):
        """Unrealized P&L should affect equity but not cash."""
        ts = datetime(2025, 1, 6, 9, 15)
        tracker.record_open(1300.0, qty=1, side="LONG", timestamp=ts)

        initial_cash = tracker.cash

        # Update unrealized P&L
        tracker.update_unrealized(current_price=1350.0)

        # Cash should not change
        assert tracker.cash == initial_cash

        # Equity should reflect unrealized P&L
        unrealized_pnl = (1350.0 - 1300.0) * 1 * self.MULT
        expected_equity = initial_cash + unrealized_pnl
        assert tracker.equity == pytest.approx(expected_equity, rel=1e-9)

    def test_unrealized_loss_reduces_equity_not_cash(self, tracker):
        """Unrealized loss should reduce equity but not cash."""
        ts = datetime(2025, 1, 6, 9, 15)
        tracker.record_open(1300.0, qty=1, side="LONG", timestamp=ts)

        initial_cash = tracker.cash

        # Update with loss
        tracker.update_unrealized(current_price=1250.0)

        # Cash unchanged
        assert tracker.cash == initial_cash

        # Equity reduced
        unrealized_pnl = (1250.0 - 1300.0) * 1 * self.MULT
        expected_equity = initial_cash + unrealized_pnl
        assert tracker.equity == pytest.approx(expected_equity, rel=1e-9)
