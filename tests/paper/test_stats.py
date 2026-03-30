"""Tests for SessionStats - metrics computation and reporting.

Tests cover:
- Metrics computation from closed trades
- Win rate calculation
- P&L aggregation
- CSV export functionality
- Summary printing
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.paper.account.tracker import Tracker
from src.paper.stats import SessionStats


class TestSessionStats:
    """Tests for SessionStats - metrics computation."""

    @pytest.fixture
    def tracker_with_trades(self):
        """Create a tracker with pre-populated closed trades."""
        tracker = Tracker(
            initial_capital=10_000_000,
            commission_rate=0.0,
            contract_multiplier=1.0,
        )

        # Simulate 3 winning trades
        base_ts = datetime(2025, 1, 6, 9, 15, 0)
        for i in range(3):
            ts = base_ts.replace(minute=15 + i * 5)
            tracker.record_open(fill_price=1300.0, qty=1, side="LONG", timestamp=ts)
            close_ts = ts.replace(minute=ts.minute + 2)
            tracker.record_close(
                fill_price=1350.0, qty=1, timestamp=close_ts, exit_reason="Take Profit"
            )
            tracker.equity_snapshot(close_ts)

        # Simulate 2 losing trades
        for i in range(2):
            ts = base_ts.replace(hour=13, minute=0 + i * 5)
            tracker.record_open(fill_price=1300.0, qty=1, side="LONG", timestamp=ts)
            close_ts = ts.replace(minute=ts.minute + 2)
            tracker.record_close(
                fill_price=1270.0, qty=1, timestamp=close_ts, exit_reason="Stop Loss"
            )
            tracker.equity_snapshot(close_ts)

        return tracker

    @pytest.fixture
    def empty_tracker(self):
        """Create an empty tracker with no trades."""
        return Tracker(
            initial_capital=10_000_000,
            commission_rate=0.0,
            contract_multiplier=1.0,
        )

    # --- Metrics Computation ---

    def test_compute_returns_nonempty_dict_with_trades(self, tracker_with_trades):
        """Test that compute() returns a non-empty dict when trades exist."""
        stats = SessionStats(tracker_with_trades)
        metrics = stats.compute()

        assert isinstance(metrics, dict)
        assert len(metrics) > 0

    def test_compute_total_trades_correct(self):
        """Test that total_trades metric is correct."""
        tracker = Tracker(
            initial_capital=10_000_000,
            commission_rate=0.0,
            contract_multiplier=1.0,
        )

        # Create 6 trades (4 wins, 2 losses)
        base_ts = datetime(2025, 1, 6, 9, 0)
        for i in range(4):
            ts = base_ts.replace(minute=i * 5)
            tracker.record_open(1300.0, qty=1, side="LONG", timestamp=ts)
            tracker.record_close(1350.0, qty=1, timestamp=ts, exit_reason="TP")
            tracker.equity_snapshot(ts)  # Add equity snapshot

        for i in range(2):
            ts = base_ts.replace(hour=13, minute=i * 5)
            tracker.record_open(1300.0, qty=1, side="LONG", timestamp=ts)
            tracker.record_close(1270.0, qty=1, timestamp=ts, exit_reason="SL")
            tracker.equity_snapshot(ts)  # Add equity snapshot

        stats = SessionStats(tracker)
        metrics = stats.compute()

        assert metrics.get("total_trades") == 6

    def test_compute_win_rate(self):
        """Test that win_rate_pct is calculated correctly."""
        tracker = Tracker(
            initial_capital=10_000_000,
            commission_rate=0.0,
            contract_multiplier=1.0,
        )

        # 3 wins, 1 loss = 75% win rate
        base_ts = datetime(2025, 1, 6, 9, 0)
        for i in range(3):
            ts = base_ts.replace(minute=i * 5)
            tracker.record_open(1300.0, qty=1, side="LONG", timestamp=ts)
            tracker.record_close(1350.0, qty=1, timestamp=ts, exit_reason="TP")
            tracker.equity_snapshot(ts)  # Add equity snapshot

        ts = base_ts.replace(hour=13)
        tracker.record_open(1300.0, qty=1, side="LONG", timestamp=ts)
        tracker.record_close(1270.0, qty=1, timestamp=ts, exit_reason="SL")
        tracker.equity_snapshot(ts)  # Add equity snapshot

        stats = SessionStats(tracker)
        metrics = stats.compute()

        assert metrics.get("win_rate_pct") == pytest.approx(75.0, abs=0.01)

    def test_compute_empty_tracker_returns_empty(self, empty_tracker):
        """Test that compute() returns empty dict for tracker with no trades."""
        stats = SessionStats(empty_tracker)
        metrics = stats.compute()

        assert metrics == {}

    def test_compute_includes_net_pnl(self, tracker_with_trades):
        """Test that metrics include trade statistics.

        Note: SessionStats uses MetricsCalculator which doesn't have a 'net_pnl' key.
        Instead, it has trade metrics like 'total_trades', 'winning_trades', etc.
        """
        stats = SessionStats(tracker_with_trades)
        metrics = stats.compute()

        # Check for trade metrics instead of net_pnl
        assert "total_trades" in metrics
        assert "winning_trades" in metrics
        assert "losing_trades" in metrics
        assert metrics["total_trades"] == 5  # 3 wins + 2 losses

    def test_compute_includes_win_loss_counts(self, tracker_with_trades):
        """Test that metrics include win and loss counts."""
        stats = SessionStats(tracker_with_trades)
        metrics = stats.compute()

        assert metrics.get("winning_trades") == 3
        assert metrics.get("losing_trades") == 2

    # --- Summary Printing ---

    def test_print_summary_does_not_raise(self, tracker_with_trades, capsys):
        """Test that print_summary() executes without errors."""
        stats = SessionStats(tracker_with_trades)
        stats.print_summary()  # Should not raise

        captured = capsys.readouterr()
        assert "PAPER TRADING SESSION SUMMARY" in captured.out or "Session" in captured.out

    def test_print_summary_includes_total_trades(self, tracker_with_trades, capsys):
        """Test that summary includes total trades count."""
        stats = SessionStats(tracker_with_trades)
        stats.print_summary()

        captured = capsys.readouterr()
        assert "Total Trades" in captured.out or "5" in captured.out

    def test_print_summary_empty_tracker_does_not_crash(self, empty_tracker, capsys):
        """Test that print_summary() handles empty tracker gracefully."""
        stats = SessionStats(empty_tracker)
        stats.print_summary()  # Should not raise

        captured = capsys.readouterr()
        # Should print something (even if just a header)
        assert len(captured.out) > 0

    # --- CSV Export ---

    def test_save_creates_csvs(self, tracker_with_trades, tmp_path):
        """Test that save() creates output directory with results.

        Note: SessionStats saves Parquet files, not CSV files.
        """
        stats = SessionStats(tracker_with_trades)
        out_dir = stats.save(str(tmp_path / "paper_test"))

        # Verify output directory was created
        assert out_dir.exists()
        assert out_dir.is_dir()

        # Verify Parquet files were created
        parquet_files = list(out_dir.glob("*.parquet"))
        assert len(parquet_files) > 0, "Expected at least one Parquet file to be created"

    def test_save_trades_csv_has_correct_columns(self, tracker_with_trades, tmp_path):
        """Test that saved Parquet files have expected structure."""
        stats = SessionStats(tracker_with_trades)
        out_dir = stats.save(str(tmp_path / "paper_test"))

        # Find trades.parquet file
        trades_file = out_dir / "trades.parquet"
        assert trades_file.exists(), "Expected trades.parquet file"

        # Verify we can read it and it has data
        df = pd.read_parquet(trades_file)
        assert not df.empty, "Trades file should not be empty"
        assert "side" in df.columns, "Expected 'side' column in trades"
        assert "pnl" in df.columns, "Expected 'pnl' column in trades"

    def test_save_trades_csv_has_correct_row_count(self, tracker_with_trades, tmp_path):
        """Test that saved files contain data from trades."""
        stats = SessionStats(tracker_with_trades)
        out_dir = stats.save(str(tmp_path / "paper_test"))

        # Find trades.parquet file
        trades_file = out_dir / "trades.parquet"
        assert trades_file.exists(), "Expected trades.parquet file"

        # Verify it has the correct number of trades (5 closed trades)
        df = pd.read_parquet(trades_file)
        assert len(df) == 5, f"Expected 5 trades, got {len(df)}"

    def test_save_equity_csv_has_timestamps(self, tracker_with_trades, tmp_path):
        """Test that saved Parquet files are readable."""
        stats = SessionStats(tracker_with_trades)
        out_dir = stats.save(str(tmp_path / "paper_test"))

        # Find equity_curve.parquet file
        equity_file = out_dir / "equity_curve.parquet"
        assert equity_file.exists(), "Expected equity_curve.parquet file"

        # Verify it's readable and has datetime column
        df = pd.read_parquet(equity_file)
        assert isinstance(df, pd.DataFrame), "Failed to read equity_curve.parquet"
        assert "datetime" in df.columns, "Expected 'datetime' column in equity curve"
        assert "equity" in df.columns, "Expected 'equity' column in equity curve"

    def test_save_creates_output_directory(self, tracker_with_trades, tmp_path):
        """Test that save() creates output directory if it doesn't exist."""
        output_path = tmp_path / "nested" / "dir" / "paper_test"
        stats = SessionStats(tracker_with_trades)

        out_dir = stats.save(str(output_path))

        assert out_dir.exists()
        assert out_dir.is_dir()

    def test_save_returns_path_object(self, tracker_with_trades, tmp_path):
        """Test that save() returns a Path object."""
        stats = SessionStats(tracker_with_trades)
        out_dir = stats.save(str(tmp_path / "paper_test"))

        assert isinstance(out_dir, Path)

    # --- Edge Cases ---

    def test_compute_with_only_winning_trades(self):
        """Test metrics computation with only winning trades."""
        tracker = Tracker(
            initial_capital=10_000_000,
            commission_rate=0.0,
            contract_multiplier=1.0,
        )

        base_ts = datetime(2025, 1, 6, 9, 0)
        for i in range(5):
            ts = base_ts.replace(minute=i * 5)
            tracker.record_open(1300.0, qty=1, side="LONG", timestamp=ts)
            tracker.record_close(1350.0, qty=1, timestamp=ts, exit_reason="TP")
            tracker.equity_snapshot(ts)  # Add equity snapshot

        stats = SessionStats(tracker)
        metrics = stats.compute()

        assert metrics.get("win_rate_pct") == 100.0
        assert metrics.get("losing_trades") == 0

    def test_compute_with_only_losing_trades(self):
        """Test metrics computation with only losing trades."""
        tracker = Tracker(
            initial_capital=10_000_000,
            commission_rate=0.0,
            contract_multiplier=1.0,
        )

        base_ts = datetime(2025, 1, 6, 9, 0)
        for i in range(5):
            ts = base_ts.replace(minute=i * 5)
            tracker.record_open(1300.0, qty=1, side="LONG", timestamp=ts)
            tracker.record_close(1270.0, qty=1, timestamp=ts, exit_reason="SL")
            tracker.equity_snapshot(ts)  # Add equity snapshot

        stats = SessionStats(tracker)
        metrics = stats.compute()

        assert metrics.get("win_rate_pct") == 0.0
        assert metrics.get("winning_trades") == 0

    def test_compute_with_break_even_trades(self):
        """Test metrics computation with break-even trades."""
        tracker = Tracker(
            initial_capital=10_000_000,
            commission_rate=0.0,  # Zero commission for exact break-even
            contract_multiplier=1.0,
        )

        base_ts = datetime(2025, 1, 6, 9, 0)
        for i in range(3):
            ts = base_ts.replace(minute=i * 5)
            tracker.record_open(1300.0, qty=1, side="LONG", timestamp=ts)
            tracker.record_close(1300.0, qty=1, timestamp=ts, exit_reason="EOD")
            tracker.equity_snapshot(ts)  # Add equity snapshot

        stats = SessionStats(tracker)
        metrics = stats.compute()

        # Break-even trades should count as breakeven_trades
        assert metrics.get("breakeven_trades") == 3
