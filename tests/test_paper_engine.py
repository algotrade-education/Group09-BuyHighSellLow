"""
Tests for the paper trading engine modules.

Run with the project's .venv (Python 3.12 — required for quickfix):
    .venv\Scripts\python.exe -m pytest tests/test_paper_engine.py -v

Coverage:
  - BarProvider: OHLC accumulation, ATR warmup, session filtering, sim replay
  - PositionTracker: open/close lifecycle, P&L calculation, SL/TP detection
  - SessionStats: metrics computation from stub trades
"""

import asyncio
from datetime import datetime, time
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ── Helpers ────────────────────────────────────────────────────────────────

def make_quote(price: float, volume: float = 100.0, ts: float = None, dt: datetime = None):
    """Build a minimal QuoteSnapshot-like object."""
    q = MagicMock()
    q.latest_matched_price = price
    q.latest_matched_quantity = volume
    if ts is not None:
        q.timestamp = ts
    elif dt is not None:
        q.timestamp = dt.timestamp()
    else:
        q.timestamp = datetime(2025, 1, 2, 9, 5, 0).timestamp()
    return q


def make_bar(
    dt: datetime = None,
    open_: float = 1300.0,
    high: float = 1310.0,
    low: float = 1290.0,
    close: float = 1305.0,
    atr: float = 15.0,
) -> Dict[str, Any]:
    """Build a minimal bar dict matching Strategy.generate_signal() expectations."""
    return {
        "datetime": dt or datetime(2025, 1, 2, 10, 0, 0),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": 200.0,
        "atr_14": atr,
    }


def make_sim_df(n_bars: int = 20, start: datetime = None) -> pd.DataFrame:
    """Build a minimal OHLC DataFrame that covers the ATR warmup window."""
    start = start or datetime(2025, 1, 6, 9, 0, 0)
    rows = []
    price = 1300.0
    for i in range(n_bars):
        minute = i * 5
        dt = start.replace(
            hour=start.hour + minute // 60,
            minute=(start.minute + minute) % 60,
        )
        # Stay inside morning session (09:00–11:30)
        if dt.time() >= time(11, 30):
            break
        rows.append({
            "datetime": dt,
            "open": price,
            "high": price + 5,
            "low": price - 5,
            "close": price + 2,
            "volume": 100.0,
        })
        price += 1.0
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# BarProvider
# ═══════════════════════════════════════════════════════════════════════════

class TestBarProvider:
    """Tests for BarProvider — OHLC accumulation and ATR warmup."""

    def test_invalid_bar_freq_raises(self):
        from src.paper.bar_provider import BarProvider
        with pytest.raises(ValueError, match="Unsupported bar_freq"):
            BarProvider(bar_freq="10min")

    def test_single_quote_does_not_emit_bar(self):
        """A single tick doesn't complete a bar."""
        from src.paper.bar_provider import BarProvider

        emitted = []
        provider = BarProvider(bar_freq="5min", atr_period=3, on_bar=emitted.append)

        # Session: 09:05 inside morning
        dt = datetime(2025, 1, 6, 9, 5, 0)
        provider._tick(dt, price=1300.0, volume=50.0)
        assert emitted == []

    def test_new_bucket_emits_previous_bar_after_warmup(self):
        """Switching to a new 5-min bucket should emit the previous bar (after warmup)."""
        from src.paper.bar_provider import BarProvider

        emitted = []
        # atr_period=2 → warmup=3
        provider = BarProvider(bar_freq="5min", atr_period=2, on_bar=emitted.append)

        base = datetime(2025, 1, 6, 9, 0, 0)
        # Feed 10 bars across different 5-min buckets (bucket changes at :05, :10, …)
        for i in range(10):
            dt = base.replace(minute=i * 5)
            provider._tick(dt, price=1300.0 + i, volume=10.0)
            # Start next bucket to flush previous
        # Trigger flush by stepping into an 11th bucket
        provider._tick(base.replace(minute=50), price=1310.0, volume=10.0)

        assert len(emitted) > 0, "Expected at least one bar to be emitted after warmup"

    def test_emitted_bar_has_atr(self):
        """Every emitted bar should have an atr_{period} key with a positive value."""
        from src.paper.bar_provider import BarProvider

        period = 3
        emitted = []
        provider = BarProvider(bar_freq="5min", atr_period=period, on_bar=emitted.append)

        base = datetime(2025, 1, 6, 9, 0, 0)
        # Feed enough distinct 5-min buckets to clear warmup (period+1 = 4 needed before emit)
        for i in range(10):
            provider._tick(base.replace(minute=i * 5), price=1300.0 + i * 2, volume=100.0)
        # Flush last open bar
        provider._tick(base.replace(minute=50), price=1320.0, volume=10.0)

        assert len(emitted) > 0
        for bar in emitted:
            assert f"atr_{period}" in bar, f"Bar missing atr_{period}: {list(bar.keys())}"
            atr_val = bar[f"atr_{period}"]
            assert atr_val > 0 or pd.isna(atr_val), "ATR should be positive or NaN (not computed)"

    def test_bar_ohlc_values(self):
        """Open/high/low/close should be correctly accumulated within a bucket."""
        from src.paper.bar_provider import BarProvider

        provider = BarProvider(bar_freq="5min", atr_period=2, on_bar=None)
        dt = datetime(2025, 1, 6, 9, 0, 0)

        provider._tick(dt.replace(second=0), price=1300.0, volume=10.0)  # open
        provider._tick(dt.replace(second=10), price=1315.0, volume=10.0)  # high
        provider._tick(dt.replace(second=20), price=1295.0, volume=10.0)  # low
        provider._tick(dt.replace(second=50), price=1308.0, volume=10.0)  # close

        assert provider._bar_open == 1300.0
        assert provider._bar_high == 1315.0
        assert provider._bar_low == 1295.0
        assert provider._bar_close == 1308.0

    def test_out_of_session_ticks_ignored(self):
        """Ticks outside VN30 trading hours (via on_quote) should not build a bar."""
        from src.paper.bar_provider import BarProvider

        emitted = []
        provider = BarProvider(bar_freq="5min", atr_period=2, on_bar=emitted.append)

        # 12:00 is the lunch break (between morning 11:30 and afternoon 13:00)
        lunch_dt = datetime(2025, 1, 6, 12, 0, 0)
        lunch_dt2 = datetime(2025, 1, 6, 12, 5, 0)

        # on_quote is the async Redis callback — it checks session time before calling _tick
        async def run():
            await provider.on_quote("HNXDS:VN30F2601", make_quote(1300.0, dt=lunch_dt))
            await provider.on_quote("HNXDS:VN30F2601", make_quote(1305.0, dt=lunch_dt2))

        asyncio.run(run())

        assert provider._current_bucket is None, "No bucket should be started for out-of-session ticks"
        assert emitted == []

    def test_sim_replay_emits_bars(self):
        """Sim replay should emit at least some bars after the warmup window fills."""
        from src.paper.bar_provider import BarProvider

        emitted = []
        provider = BarProvider(bar_freq="5min", atr_period=3, on_bar=emitted.append)
        df = make_sim_df(n_bars=20)

        asyncio.run(provider.replay(df, speed=0.0))

        # warmup = 3+1 = 4 bars consumed; rest should be emitted
        assert len(emitted) > 0, "Sim replay should emit bars after warmup"

    def test_sim_replay_bars_have_atr(self):
        """All bars emitted in sim should carry the atr_{period} indicator."""
        from src.paper.bar_provider import BarProvider

        period = 3
        emitted = []
        provider = BarProvider(bar_freq="5min", atr_period=period, on_bar=emitted.append)
        df = make_sim_df(n_bars=20)

        asyncio.run(provider.replay(df, speed=0.0))

        assert emitted, "Expected at least one bar"
        for bar in emitted:
            assert f"atr_{period}" in bar


# ═══════════════════════════════════════════════════════════════════════════
# PositionTracker
# ═══════════════════════════════════════════════════════════════════════════

class TestPositionTracker:
    """Tests for PositionTracker — position lifecycle and P&L calculation."""

    @pytest.fixture()
    def tracker(self):
        from src.paper.position_tracker import PositionTracker
        return PositionTracker(
            initial_capital=10_000_000,
            commission_rate=0.0,       # zero commission for simple P&L checks
            contract_multiplier=1.0,
        )

    def test_initial_state_is_flat(self, tracker):
        assert tracker.is_flat
        assert tracker.equity == tracker.initial_capital
        assert tracker.trades == []

    def test_record_open_creates_trade(self, tracker):
        ts = datetime(2025, 1, 6, 9, 15, 0)
        tracker.record_open(fill_price=1300.0, qty=1, side="LONG", timestamp=ts,
                            stop_loss=1280.0, take_profit=1340.0)
        assert not tracker.is_flat
        assert len(tracker.trades) == 1
        assert tracker.trades[0].entry_price == 1300.0

    def test_long_pnl_positive_on_price_rise(self, tracker):
        ts = datetime(2025, 1, 6, 9, 15, 0)
        tracker.record_open(fill_price=1300.0, qty=1, side="LONG", timestamp=ts)
        tracker.record_close(fill_price=1350.0, timestamp=ts, exit_reason="Take Profit")

        trade = tracker.trades[0]
        assert trade.is_closed
        assert trade.pnl == pytest.approx(50.0, abs=1e-6), f"Expected 50, got {trade.pnl}"

    def test_short_pnl_positive_on_price_fall(self, tracker):
        ts = datetime(2025, 1, 6, 9, 15, 0)
        tracker.record_open(fill_price=1300.0, qty=1, side="SHORT", timestamp=ts)
        tracker.record_close(fill_price=1250.0, timestamp=ts, exit_reason="Take Profit")

        trade = tracker.trades[0]
        assert trade.pnl == pytest.approx(50.0, abs=1e-6)

    def test_pnl_negative_on_loss(self, tracker):
        ts = datetime(2025, 1, 6, 9, 15, 0)
        tracker.record_open(fill_price=1300.0, qty=1, side="LONG", timestamp=ts)
        tracker.record_close(fill_price=1270.0, timestamp=ts, exit_reason="Stop Loss")

        trade = tracker.trades[0]
        assert trade.pnl == pytest.approx(-30.0, abs=1e-6)

    def test_unrealized_pnl_updates_equity(self, tracker):
        ts = datetime(2025, 1, 6, 9, 15, 0)
        capital = tracker.initial_capital
        tracker.record_open(fill_price=1300.0, qty=1, side="LONG", timestamp=ts)

        tracker.update_unrealized(current_price=1350.0)
        assert tracker.equity == pytest.approx(capital + 50.0, abs=1e-6)

    def test_record_close_while_flat_is_noop(self, tracker):
        result = tracker.record_close(fill_price=1300.0, timestamp=datetime.now())
        assert result is None
        assert tracker.trades == []

    def test_check_sl_tp_stop_loss(self, tracker):
        ts = datetime(2025, 1, 6, 9, 15, 0)
        tracker.record_open(fill_price=1300.0, qty=1, side="LONG", timestamp=ts,
                            stop_loss=1280.0, take_profit=1340.0)
        bar = make_bar(low=1275.0, high=1305.0)  # low hits SL
        assert tracker.check_sl_tp(bar) == "STOP_LOSS"

    def test_check_sl_tp_take_profit(self, tracker):
        ts = datetime(2025, 1, 6, 9, 15, 0)
        tracker.record_open(fill_price=1300.0, qty=1, side="LONG", timestamp=ts,
                            stop_loss=1280.0, take_profit=1340.0)
        bar = make_bar(low=1305.0, high=1345.0)  # high hits TP
        assert tracker.check_sl_tp(bar) == "TAKE_PROFIT"

    def test_check_sl_tp_no_trigger(self, tracker):
        ts = datetime(2025, 1, 6, 9, 15, 0)
        tracker.record_open(fill_price=1300.0, qty=1, side="LONG", timestamp=ts,
                            stop_loss=1280.0, take_profit=1340.0)
        bar = make_bar(low=1295.0, high=1320.0)  # neither hits
        assert tracker.check_sl_tp(bar) is None

    def test_check_sl_while_flat_returns_none(self, tracker):
        bar = make_bar(low=1000.0, high=2000.0)  # would trigger anything
        assert tracker.check_sl_tp(bar) is None

    def test_short_sl_triggered_by_high(self, tracker):
        ts = datetime(2025, 1, 6, 9, 15, 0)
        tracker.record_open(fill_price=1300.0, qty=1, side="SHORT", timestamp=ts,
                            stop_loss=1330.0, take_profit=1260.0)
        bar = make_bar(low=1295.0, high=1335.0)  # high > SL
        assert tracker.check_sl_tp(bar) == "STOP_LOSS"

    def test_equity_snapshot(self, tracker):
        ts = datetime(2025, 1, 6, 9, 15, 0)
        tracker.equity_snapshot(ts)
        assert len(tracker.equity_snapshots) == 1
        assert tracker.equity_snapshots[0] == (ts, tracker.equity)


# ═══════════════════════════════════════════════════════════════════════════
# PositionTracker — cash accounting correctness
# ═══════════════════════════════════════════════════════════════════════════

class TestPositionTrackerCashAccounting:
    """
    Numerical tests for PositionTracker cash / equity accounting.

    These tests use real commission rates and the contract multiplier.
    They are designed to catch double-commission or incorrect gross-PnL
    accounting bugs — a bug of exactly this type was fixed in record_close.

    Key invariant after any number of completed round-trips:
        final_cash == initial_capital + sum(trade.pnl for trade in closed_trades)
    """

    INITIAL = 10_000_000
    RATE = 0.0003          # 0.03% per leg — realistic VN30 commission rate
    MULT = 100_000.0       # VN30 contract multiplier

    @pytest.fixture()
    def tracker(self):
        from src.paper.position_tracker import PositionTracker
        return PositionTracker(
            initial_capital=self.INITIAL,
            commission_rate=self.RATE,
            contract_multiplier=self.MULT,
        )

    def _calc_comm(self, price: float, qty: int = 1) -> float:
        return price * qty * self.MULT * self.RATE

    # ── single-trade invariants ──────────────────────────────────────────

    def test_long_winning_trade_cash_equals_initial_plus_pnl(self, tracker):
        """Cash after a profitable LONG round-trip must equal initial + trade.pnl."""
        ts = datetime(2025, 1, 6, 9, 15)
        tracker.record_open(1300.0, qty=1, side="LONG", timestamp=ts)
        tracker.record_close(1350.0, timestamp=ts, exit_reason="TP")

        trade = tracker.trades[0]
        expected_cash = self.INITIAL + trade.pnl
        assert tracker.cash == pytest.approx(expected_cash, rel=1e-9), (
            f"Cash mismatch: got {tracker.cash:,.0f}, expected {expected_cash:,.0f}"
        )

    def test_long_losing_trade_cash_equals_initial_plus_pnl(self, tracker):
        """Cash after a losing LONG round-trip must equal initial + trade.pnl (negative)."""
        ts = datetime(2025, 1, 6, 9, 15)
        tracker.record_open(1300.0, qty=1, side="LONG", timestamp=ts)
        tracker.record_close(1270.0, timestamp=ts, exit_reason="SL")

        trade = tracker.trades[0]
        expected_cash = self.INITIAL + trade.pnl
        assert tracker.cash == pytest.approx(expected_cash, rel=1e-9)

    def test_short_winning_trade_cash_invariant(self, tracker):
        """Cash after a profitable SHORT round-trip is correct."""
        ts = datetime(2025, 1, 6, 9, 15)
        tracker.record_open(1300.0, qty=1, side="SHORT", timestamp=ts)
        tracker.record_close(1250.0, timestamp=ts, exit_reason="TP")

        trade = tracker.trades[0]
        expected_cash = self.INITIAL + trade.pnl
        assert tracker.cash == pytest.approx(expected_cash, rel=1e-9)

    def test_trade_pnl_equals_gross_minus_total_commission(self, tracker):
        """trade.pnl must equal (exit - entry) * qty * multiplier - total_commission."""
        ts = datetime(2025, 1, 6, 9, 15)
        entry, exit_ = 1300.0, 1350.0
        tracker.record_open(entry, qty=1, side="LONG", timestamp=ts)
        tracker.record_close(exit_, timestamp=ts, exit_reason="TP")

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
        entry, exit_ = 1300.0, 1300.0   # break-even trade (gross = 0)
        tracker.record_open(entry, qty=1, side="LONG", timestamp=ts)
        tracker.record_close(exit_, timestamp=ts, exit_reason="EOD")

        total_comm = self._calc_comm(entry) + self._calc_comm(exit_)
        expected_cash = self.INITIAL - total_comm
        assert tracker.cash == pytest.approx(expected_cash, rel=1e-9), (
            "Break-even trade should only reduce cash by total commission (not 2x or 3x)"
        )

    def test_flat_equity_equals_cash_after_close(self, tracker):
        """When position is flat, equity should equal cash exactly."""
        ts = datetime(2025, 1, 6, 9, 15)
        tracker.record_open(1300.0, qty=1, side="LONG", timestamp=ts)
        tracker.record_close(1320.0, timestamp=ts, exit_reason="TP")
        tracker.update_unrealized(1320.0)  # re-mark to market (position flat)

        assert tracker.equity == pytest.approx(tracker.cash, rel=1e-9)

    # ── multi-trade invariants ───────────────────────────────────────────

    def test_multi_trade_cash_invariant(self, tracker):
        """
        After N completed round-trips:
          final_cash == initial_capital + sum(trade.pnl)

        This is the fundamental accounting identity. If it fails, there is a
        bug in how cash is updated on open or close.
        """
        from src.paper.position_tracker import PositionTracker

        ts = datetime(2025, 1, 6, 9, 0)
        trades_data = [
            ("LONG",  1300.0, 1350.0),  # +50 pts
            ("SHORT", 1350.0, 1300.0),  # +50 pts
            ("LONG",  1300.0, 1280.0),  # -20 pts (loss)
            ("SHORT", 1280.0, 1290.0),  # -10 pts (loss)
            ("LONG",  1290.0, 1310.0),  # +20 pts
        ]

        for i, (side, entry, exit_) in enumerate(trades_data):
            t_open  = ts.replace(minute=i * 10)
            t_close = ts.replace(minute=i * 10 + 5)
            tracker.record_open(entry, qty=1, side=side, timestamp=t_open)
            tracker.record_close(exit_, timestamp=t_close, exit_reason="test")

        total_pnl = sum(t.pnl for t in tracker.trades)
        expected_cash = self.INITIAL + total_pnl
        assert tracker.cash == pytest.approx(expected_cash, rel=1e-9), (
            f"After {len(tracker.trades)} trades: cash={tracker.cash:,.0f}, "
            f"expected={expected_cash:,.0f} (diff={tracker.cash - expected_cash:,.0f})"
        )

    def test_multi_trade_final_equity_consistent_with_net_pnl(self, tracker):
        """Final Equity = Initial Capital + Net P&L (when all positions closed)."""
        ts = datetime(2025, 1, 6, 9, 0)
        for i, (side, entry, exit_) in enumerate([
            ("LONG",  1300.0, 1340.0),
            ("LONG",  1340.0, 1330.0),
        ]):
            tracker.record_open(entry, qty=1, side=side, timestamp=ts.replace(minute=i*10))
            tracker.record_close(exit_, timestamp=ts.replace(minute=i*10+5), exit_reason="test")

        tracker.update_unrealized(0.0)   # position is flat, unrealized = 0
        net_pnl = sum(t.pnl for t in tracker.trades)
        assert tracker.equity == pytest.approx(self.INITIAL + net_pnl, rel=1e-9)


# ═══════════════════════════════════════════════════════════════════════════
# SessionStats
# ═══════════════════════════════════════════════════════════════════════════

class TestSessionStats:
    """Tests for SessionStats — metrics computation."""

    def _make_tracker_with_trades(self, n_win: int = 3, n_lose: int = 2):
        """Helper: create a PositionTracker with pre-populated closed trades."""
        from src.paper.position_tracker import PositionTracker

        tracker = PositionTracker(
            initial_capital=10_000_000,
            commission_rate=0.0,
            contract_multiplier=1.0,
        )

        # Simulate winning trades
        base_ts = datetime(2025, 1, 6, 9, 15, 0)
        for i in range(n_win):
            ts = base_ts.replace(minute=15 + i * 5)
            tracker.record_open(fill_price=1300.0, qty=1, side="LONG", timestamp=ts)
            close_ts = ts.replace(minute=ts.minute + 2)
            tracker.record_close(fill_price=1350.0, timestamp=close_ts, exit_reason="Take Profit")
            tracker.equity_snapshot(close_ts)

        # Simulate losing trades
        for i in range(n_lose):
            ts = base_ts.replace(hour=13, minute=0 + i * 5)
            tracker.record_open(fill_price=1300.0, qty=1, side="LONG", timestamp=ts)
            close_ts = ts.replace(minute=ts.minute + 2)
            tracker.record_close(fill_price=1270.0, timestamp=close_ts, exit_reason="Stop Loss")
            tracker.equity_snapshot(close_ts)

        return tracker

    def test_compute_returns_nonempty_dict_with_trades(self):
        from src.paper.stats import SessionStats

        tracker = self._make_tracker_with_trades()
        stats = SessionStats(tracker)
        metrics = stats.compute()

        assert isinstance(metrics, dict)
        assert len(metrics) > 0

    def test_compute_total_trades_correct(self):
        from src.paper.stats import SessionStats

        tracker = self._make_tracker_with_trades(n_win=4, n_lose=2)
        stats = SessionStats(tracker)
        metrics = stats.compute()

        assert metrics.get("total_trades") == 6

    def test_compute_win_rate(self):
        from src.paper.stats import SessionStats

        tracker = self._make_tracker_with_trades(n_win=3, n_lose=1)
        stats = SessionStats(tracker)
        metrics = stats.compute()

        assert metrics.get("win_rate_pct") == pytest.approx(75.0, abs=0.01)

    def test_compute_empty_tracker_returns_empty(self):
        from src.paper.position_tracker import PositionTracker
        from src.paper.stats import SessionStats

        tracker = PositionTracker(initial_capital=10_000_000)
        stats = SessionStats(tracker)
        metrics = stats.compute()

        assert metrics == {}

    def test_print_summary_does_not_raise(self, capsys):
        from src.paper.stats import SessionStats

        tracker = self._make_tracker_with_trades()
        stats = SessionStats(tracker)
        stats.print_summary()  # Should not raise

        captured = capsys.readouterr()
        assert "PAPER TRADING SESSION SUMMARY" in captured.out
        assert "Total Trades" in captured.out

    def test_save_creates_csvs(self, tmp_path):
        from src.paper.stats import SessionStats

        tracker = self._make_tracker_with_trades()
        stats = SessionStats(tracker)
        out_dir = stats.save(str(tmp_path / "paper_test"))

        trades_csv = out_dir / "trades.csv"
        equity_csv = out_dir / "equity_curve.csv"
        assert trades_csv.exists(), "trades.csv should be written"
        assert equity_csv.exists(), "equity_curve.csv should be written"

        trades_df = pd.read_csv(trades_csv)
        assert len(trades_df) == 5  # 3 win + 2 lose
        assert "pnl" in trades_df.columns
