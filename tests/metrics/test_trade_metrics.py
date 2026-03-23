"""Tests for trade metrics."""

from datetime import datetime, timedelta

import pytest

from src.metrics.trade_metrics import Trade, TradeSide, calculate_trade_metrics


@pytest.fixture
def sample_trades():
    """Sample trades for testing."""
    base_time = datetime(2024, 1, 1, 9, 0)

    return [
        Trade(
            trade_id="T1",
            side=TradeSide.LONG,
            entry_time=base_time,
            entry_price=1000,
            quantity=1,
            exit_time=base_time + timedelta(minutes=30),
            exit_price=1020,
            gross_pnl=200,
            commission=10,
            pnl=190,
            mae=50,
            mfe=250,
        ),
        Trade(
            trade_id="T2",
            side=TradeSide.SHORT,
            entry_time=base_time + timedelta(hours=1),
            entry_price=1020,
            quantity=1,
            exit_time=base_time + timedelta(hours=1, minutes=45),
            exit_price=1030,
            gross_pnl=-100,
            commission=10,
            pnl=-110,
            mae=150,
            mfe=80,
        ),
        Trade(
            trade_id="T3",
            side=TradeSide.LONG,
            entry_time=base_time + timedelta(hours=2),
            entry_price=1030,
            quantity=1,
            exit_time=base_time + timedelta(hours=2, minutes=20),
            exit_price=1050,
            gross_pnl=200,
            commission=10,
            pnl=190,
            mae=30,
            mfe=220,
        ),
    ]


class TestTrade:
    """Test Trade dataclass."""

    def test_trade_creation(self):
        """Test basic trade creation."""
        trade = Trade(
            trade_id="T1",
            side=TradeSide.LONG,
            entry_price=1000,
            exit_price=1020,
            pnl=190,
        )

        assert trade.trade_id == "T1"
        assert trade.side == TradeSide.LONG
        assert trade.entry_price == 1000
        assert trade.exit_price == 1020
        assert trade.pnl == 190

    def test_is_closed(self):
        """Test is_closed property."""
        trade_open = Trade(trade_id="T1", entry_time=datetime.now())
        trade_closed = Trade(trade_id="T2", entry_time=datetime.now(), exit_time=datetime.now())

        assert not trade_open.is_closed
        assert trade_closed.is_closed

    def test_is_winner(self):
        """Test is_winner property."""
        winner = Trade(trade_id="T1", pnl=100)
        loser = Trade(trade_id="T2", pnl=-100)
        breakeven = Trade(trade_id="T3", pnl=0)

        assert winner.is_winner
        assert not loser.is_winner
        assert not breakeven.is_winner

    def test_is_loser(self):
        """Test is_loser property."""
        winner = Trade(trade_id="T1", pnl=100)
        loser = Trade(trade_id="T2", pnl=-100)
        breakeven = Trade(trade_id="T3", pnl=0)

        assert not winner.is_loser
        assert loser.is_loser
        assert not breakeven.is_loser

    def test_duration_seconds(self):
        """Test duration_seconds property."""
        entry = datetime(2024, 1, 1, 9, 0)
        exit = datetime(2024, 1, 1, 9, 30)

        trade = Trade(trade_id="T1", entry_time=entry, exit_time=exit)

        assert trade.duration_seconds == 1800  # 30 minutes

    def test_duration_minutes(self):
        """Test duration_minutes property."""
        entry = datetime(2024, 1, 1, 9, 0)
        exit = datetime(2024, 1, 1, 9, 45)

        trade = Trade(trade_id="T1", entry_time=entry, exit_time=exit)

        assert trade.duration_minutes == 45

    def test_duration_not_closed(self):
        """Test duration when trade not closed."""
        trade = Trade(trade_id="T1", entry_time=datetime.now())

        assert trade.duration_seconds == 0.0
        assert trade.duration_minutes == 0.0

    def test_r_multiple(self):
        """Test R-multiple calculation."""
        trade = Trade(
            trade_id="T1",
            entry_price=1000,
            stop_loss=980,
            quantity=1,
            pnl=40,  # 2R
        )

        # Risk = (1000 - 980) * 1 = 20
        # R-multiple = 40 / 20 = 2.0
        assert trade.r_multiple == 2.0

    def test_r_multiple_no_stop(self):
        """Test R-multiple when no stop loss."""
        trade = Trade(trade_id="T1", entry_price=1000, pnl=100)

        assert trade.r_multiple is None

    def test_edge_ratio(self):
        """Test edge ratio calculation."""
        trade = Trade(trade_id="T1", mae=100, mfe=300)

        # Edge ratio = 300 / 100 = 3.0
        assert trade.edge_ratio == 3.0

    def test_edge_ratio_no_mae_mfe(self):
        """Test edge ratio when MAE/MFE not available."""
        trade = Trade(trade_id="T1", pnl=100)

        assert trade.edge_ratio is None

    def test_edge_ratio_zero_mae(self):
        """Test edge ratio when MAE is zero."""
        trade = Trade(trade_id="T1", mae=0, mfe=100)

        assert trade.edge_ratio is None

    def test_trade_repr(self):
        """Test trade string representation."""
        trade = Trade(
            trade_id="T1",
            side=TradeSide.LONG,
            entry_price=1000,
            exit_price=1020,
            pnl=190,
            exit_time=datetime.now(),
        )

        repr_str = repr(trade)

        assert "T1" in repr_str
        assert "long" in repr_str
        assert "1000" in repr_str


class TestCalculateTradeMetrics:
    """Test trade metrics calculation."""

    def test_basic_metrics(self, sample_trades):
        """Test basic trade metrics calculation."""
        metrics = calculate_trade_metrics(sample_trades)

        assert metrics["total_trades"] == 3
        assert metrics["winning_trades"] == 2
        assert metrics["losing_trades"] == 1
        assert metrics["breakeven_trades"] == 0

    def test_win_rate(self, sample_trades):
        """Test win rate calculation."""
        metrics = calculate_trade_metrics(sample_trades)

        # 2 wins out of 3 = 66.67%
        assert metrics["win_rate"] == pytest.approx(66.67, abs=0.1)

    def test_profit_factor(self, sample_trades):
        """Test profit factor calculation."""
        metrics = calculate_trade_metrics(sample_trades)

        # Net profit = 190 + 190 = 380
        # Net loss = 110
        # Net PF = 380 / 110 = 3.45
        assert metrics["net_profit_factor"] == pytest.approx(3.45, abs=0.1)

    def test_gross_profit_factor(self, sample_trades):
        """Test gross profit factor (before commission)."""
        metrics = calculate_trade_metrics(sample_trades)

        # Gross wins = 200 + 200 = 400
        # Gross losses = 100
        # Gross PF = 400 / 100 = 4.0
        assert metrics["gross_profit_factor"] == pytest.approx(4.0, abs=0.1)

    def test_avg_win_loss(self, sample_trades):
        """Test average win and loss."""
        metrics = calculate_trade_metrics(sample_trades)

        # Avg win = (190 + 190) / 2 = 190
        # Avg loss = 110 / 1 = 110
        assert metrics["avg_win"] == 190
        assert metrics["avg_loss"] == 110

    def test_payoff_ratio(self, sample_trades):
        """Test payoff ratio."""
        metrics = calculate_trade_metrics(sample_trades)

        # Payoff = 190 / 110 = 1.73
        assert metrics["payoff_ratio"] == pytest.approx(1.73, abs=0.1)

    def test_expectancy(self, sample_trades):
        """Test expectancy calculation."""
        metrics = calculate_trade_metrics(sample_trades)

        # Win rate = 2/3, Loss rate = 1/3
        # Expectancy = (2/3 * 190) - (1/3 * 110) = 126.67 - 36.67 = 90
        assert metrics["expectancy"] == pytest.approx(90, abs=1)

    def test_consecutive_wins(self):
        """Test max consecutive wins."""
        trades = [Trade(trade_id=f"T{i}", pnl=100, exit_time=datetime.now()) for i in range(5)]
        trades.append(Trade(trade_id="T6", pnl=-50, exit_time=datetime.now()))

        metrics = calculate_trade_metrics(trades)

        assert metrics["max_consecutive_wins"] == 5

    def test_consecutive_losses(self):
        """Test max consecutive losses."""
        trades = [Trade(trade_id=f"T{i}", pnl=-100, exit_time=datetime.now()) for i in range(4)]
        trades.append(Trade(trade_id="T5", pnl=50, exit_time=datetime.now()))

        metrics = calculate_trade_metrics(trades)

        assert metrics["max_consecutive_losses"] == 4

    def test_avg_duration(self, sample_trades):
        """Test average duration calculation."""
        metrics = calculate_trade_metrics(sample_trades)

        # T1: 30 min, T2: 45 min, T3: 20 min
        # Avg = (30 + 45 + 20) / 3 = 31.67
        assert metrics["avg_duration_minutes"] == pytest.approx(31.67, abs=0.1)

    def test_mae_mfe_metrics(self, sample_trades):
        """Test MAE/MFE metrics."""
        metrics = calculate_trade_metrics(sample_trades)

        # Avg MAE = (50 + 150 + 30) / 3 = 76.67
        # Avg MFE = (250 + 80 + 220) / 3 = 183.33
        assert metrics["avg_mae"] == pytest.approx(76.67, abs=0.1)
        assert metrics["avg_mfe"] == pytest.approx(183.33, abs=0.1)

    def test_edge_ratio_avg(self, sample_trades):
        """Test average edge ratio."""
        metrics = calculate_trade_metrics(sample_trades)

        # T1: 250/50 = 5.0
        # T2: 80/150 = 0.53
        # T3: 220/30 = 7.33
        # Avg = (5.0 + 0.53 + 7.33) / 3 = 4.29
        assert metrics["avg_edge_ratio"] == pytest.approx(4.29, abs=0.1)

    def test_total_commission(self, sample_trades):
        """Test total commission calculation."""
        metrics = calculate_trade_metrics(sample_trades)

        # 10 + 10 + 10 = 30
        assert metrics["total_commission"] == 30

    def test_empty_trades(self):
        """Test with empty trade list."""
        metrics = calculate_trade_metrics([])

        assert metrics["total_trades"] == 0
        assert metrics["win_rate"] == 0.0
        assert metrics["net_profit_factor"] == 0.0

    def test_only_open_trades(self):
        """Test with only open trades (not closed)."""
        trades = [
            Trade(trade_id="T1", entry_time=datetime.now()),
            Trade(trade_id="T2", entry_time=datetime.now()),
        ]

        metrics = calculate_trade_metrics(trades)

        assert metrics["total_trades"] == 0

    def test_all_winners(self):
        """Test with all winning trades."""
        trades = [
            Trade(trade_id=f"T{i}", pnl=100, gross_pnl=110, exit_time=datetime.now())
            for i in range(5)
        ]

        metrics = calculate_trade_metrics(trades)

        assert metrics["winning_trades"] == 5
        assert metrics["losing_trades"] == 0
        assert metrics["win_rate"] == 100.0
        assert metrics["net_profit_factor"] == 0.0  # No losses

    def test_all_losers(self):
        """Test with all losing trades."""
        trades = [
            Trade(trade_id=f"T{i}", pnl=-100, gross_pnl=-90, exit_time=datetime.now())
            for i in range(5)
        ]

        metrics = calculate_trade_metrics(trades)

        assert metrics["winning_trades"] == 0
        assert metrics["losing_trades"] == 5
        assert metrics["win_rate"] == 0.0
        assert metrics["net_profit_factor"] == 0.0  # No wins

    def test_breakeven_trades(self):
        """Test with breakeven trades."""
        trades = [
            Trade(trade_id="T1", pnl=100, exit_time=datetime.now()),
            Trade(trade_id="T2", pnl=0, exit_time=datetime.now()),
            Trade(trade_id="T3", pnl=-100, exit_time=datetime.now()),
        ]

        metrics = calculate_trade_metrics(trades)

        assert metrics["breakeven_trades"] == 1

    def test_trades_without_mae_mfe(self):
        """Test trades without MAE/MFE data."""
        trades = [
            Trade(trade_id="T1", pnl=100, exit_time=datetime.now()),
            Trade(trade_id="T2", pnl=-50, exit_time=datetime.now()),
        ]

        metrics = calculate_trade_metrics(trades)

        assert metrics["avg_mae"] is None
        assert metrics["avg_mfe"] is None
        assert metrics["avg_edge_ratio"] is None

    def test_mixed_mae_mfe_availability(self):
        """Test when some trades have MAE/MFE and some don't."""
        trades = [
            Trade(trade_id="T1", pnl=100, mae=50, mfe=150, exit_time=datetime.now()),
            Trade(trade_id="T2", pnl=-50, exit_time=datetime.now()),
            Trade(trade_id="T3", pnl=80, mae=30, mfe=120, exit_time=datetime.now()),
        ]

        metrics = calculate_trade_metrics(trades)

        # Should average only trades with MAE/MFE
        assert metrics["avg_mae"] == pytest.approx(40, abs=0.1)  # (50 + 30) / 2
        assert metrics["avg_mfe"] == pytest.approx(135, abs=0.1)  # (150 + 120) / 2

    def test_trades_without_duration(self):
        """Test trades without entry/exit times."""
        trades = [
            Trade(trade_id="T1", pnl=100, exit_time=datetime.now()),
            Trade(trade_id="T2", pnl=-50, exit_time=datetime.now()),
        ]

        metrics = calculate_trade_metrics(trades)

        # Duration should be 0 when times not set
        assert metrics["avg_duration_minutes"] == 0.0


class TestTradeMetricsEdgeCases:
    """Test edge cases for trade metrics."""

    def test_very_small_pnl(self):
        """Test with very small PnL values."""
        trades = [
            Trade(trade_id="T1", pnl=0.01, gross_pnl=0.02, exit_time=datetime.now()),
            Trade(trade_id="T2", pnl=-0.01, gross_pnl=-0.005, exit_time=datetime.now()),
        ]

        metrics = calculate_trade_metrics(trades)

        assert metrics["total_trades"] == 2
        assert metrics["winning_trades"] == 1
        assert metrics["losing_trades"] == 1

    def test_large_number_of_trades(self):
        """Test with large number of trades."""
        trades = [
            Trade(
                trade_id=f"T{i}",
                pnl=100 if i % 2 == 0 else -50,
                gross_pnl=110 if i % 2 == 0 else -40,
                exit_time=datetime.now(),
            )
            for i in range(1000)
        ]

        metrics = calculate_trade_metrics(trades)

        assert metrics["total_trades"] == 1000
        assert metrics["winning_trades"] == 500
        assert metrics["losing_trades"] == 500
        assert metrics["win_rate"] == 50.0

    def test_alternating_wins_losses(self):
        """Test with alternating wins and losses."""
        trades = [
            Trade(
                trade_id=f"T{i}",
                pnl=100 if i % 2 == 0 else -50,
                exit_time=datetime.now(),
            )
            for i in range(10)
        ]

        metrics = calculate_trade_metrics(trades)

        # Max consecutive should be 1 for both
        assert metrics["max_consecutive_wins"] == 1
        assert metrics["max_consecutive_losses"] == 1
