"""
Tests for src/optimization/scoring.py

Covers:
- ScorerConfig validation (__post_init__ guards)
- Hard gates (min_trades, min_return, min_profit_factor, min_win_rate)
- Base score selection (sharpe → sortino → return fallback)
- Drawdown penalty normalization
- Turnover penalty
- Trade count bonus + cap
- Module-level constants (INVALID_SCORE, ERROR_SCORE)
"""

from __future__ import annotations

import pytest

from src.optimization.scoring import (
    ERROR_SCORE,
    INVALID_SCORE,
    ScorerConfig,
    calculate_score,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _metrics(
    total_trades: int = 50,
    sharpe_ratio: float = 1.0,
    sortino_ratio: float = 1.5,
    max_drawdown_pct: float = 10.0,
    total_return_pct: float = 20.0,
    net_profit_factor: float = 1.5,
    win_rate_pct: float = 55.0,
) -> dict:
    return {
        "total_trades": total_trades,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown_pct": max_drawdown_pct,
        "total_return_pct": total_return_pct,
        "net_profit_factor": net_profit_factor,
        "win_rate_pct": win_rate_pct,
    }


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


def test_constants_values():
    assert INVALID_SCORE == -20.0
    assert ERROR_SCORE == -100.0


def test_constants_accessible_without_instance():
    # Should not require ScorerConfig() to access
    assert INVALID_SCORE < 0
    assert ERROR_SCORE < INVALID_SCORE


# ---------------------------------------------------------------------------
# ScorerConfig validation
# ---------------------------------------------------------------------------


def test_scorer_config_defaults():
    cfg = ScorerConfig()
    assert cfg.min_trades == 30
    assert cfg.drawdown_penalty == 0.5
    assert cfg.turnover_penalty == 0.0
    assert cfg.trade_count_bonus == 0.0


def test_scorer_config_negative_min_trades_raises():
    with pytest.raises(ValueError, match="min_trades"):
        ScorerConfig(min_trades=-1)


def test_scorer_config_win_rate_out_of_range_raises():
    with pytest.raises(ValueError, match="min_win_rate_pct"):
        ScorerConfig(min_win_rate_pct=101.0)
    with pytest.raises(ValueError, match="min_win_rate_pct"):
        ScorerConfig(min_win_rate_pct=-1.0)


def test_scorer_config_negative_drawdown_penalty_raises():
    with pytest.raises(ValueError, match="drawdown_penalty"):
        ScorerConfig(drawdown_penalty=-0.1)


def test_scorer_config_negative_turnover_penalty_raises():
    with pytest.raises(ValueError, match="turnover_penalty"):
        ScorerConfig(turnover_penalty=-1.0)


def test_scorer_config_negative_trade_bonus_raises():
    with pytest.raises(ValueError, match="trade_count_bonus"):
        ScorerConfig(trade_count_bonus=-0.5)


def test_scorer_config_zero_trade_bonus_cap_raises():
    with pytest.raises(ValueError, match="trade_bonus_cap"):
        ScorerConfig(trade_bonus_cap=0.0)


# ---------------------------------------------------------------------------
# Hard gates
# ---------------------------------------------------------------------------


def test_gate_min_trades():
    cfg = ScorerConfig(min_trades=30)
    # Exactly at threshold passes (strict <)
    assert calculate_score(_metrics(total_trades=30), cfg) != INVALID_SCORE
    # One below fails
    assert calculate_score(_metrics(total_trades=29), cfg) == INVALID_SCORE


def test_gate_min_return():
    cfg = ScorerConfig(min_return_pct=5.0)
    assert calculate_score(_metrics(total_return_pct=5.0), cfg) != INVALID_SCORE
    assert calculate_score(_metrics(total_return_pct=4.9), cfg) == INVALID_SCORE


def test_gate_min_profit_factor():
    cfg = ScorerConfig(min_profit_factor=1.2)
    assert calculate_score(_metrics(net_profit_factor=1.2), cfg) != INVALID_SCORE
    assert calculate_score(_metrics(net_profit_factor=1.19), cfg) == INVALID_SCORE


def test_gate_min_win_rate():
    cfg = ScorerConfig(min_win_rate_pct=40.0)
    assert calculate_score(_metrics(win_rate_pct=40.0), cfg) != INVALID_SCORE
    assert calculate_score(_metrics(win_rate_pct=39.9), cfg) == INVALID_SCORE


def test_gate_returns_invalid_not_error():
    """Hard gate failures return INVALID_SCORE, not ERROR_SCORE."""
    cfg = ScorerConfig(min_trades=100)
    score = calculate_score(_metrics(total_trades=5), cfg)
    assert score == INVALID_SCORE
    assert score != ERROR_SCORE


# ---------------------------------------------------------------------------
# Base score selection
# ---------------------------------------------------------------------------


def test_base_score_uses_sharpe_when_positive():
    cfg = ScorerConfig(drawdown_penalty=0.0)
    m = _metrics(sharpe_ratio=1.5, sortino_ratio=2.0, max_drawdown_pct=0.0)
    assert calculate_score(m, cfg) == pytest.approx(1.5)


def test_base_score_falls_back_to_sortino_when_sharpe_zero():
    cfg = ScorerConfig(drawdown_penalty=0.0)
    m = _metrics(sharpe_ratio=0.0, sortino_ratio=2.0, max_drawdown_pct=0.0)
    assert calculate_score(m, cfg) == pytest.approx(2.0 * 0.9)


def test_base_score_falls_back_to_return_when_both_nonpositive():
    cfg = ScorerConfig(drawdown_penalty=0.0)
    m = _metrics(sharpe_ratio=0.0, sortino_ratio=0.0, total_return_pct=30.0, max_drawdown_pct=0.0)
    assert calculate_score(m, cfg) == pytest.approx(3.0)  # 30 / 10


def test_base_score_negative_return_fallback():
    cfg = ScorerConfig(drawdown_penalty=0.0, min_return_pct=-999.0)
    m = _metrics(
        sharpe_ratio=-0.5, sortino_ratio=-0.5, total_return_pct=-20.0, max_drawdown_pct=0.0
    )
    assert calculate_score(m, cfg) == pytest.approx(-2.0)  # -20 / 10


# ---------------------------------------------------------------------------
# Drawdown penalty
# ---------------------------------------------------------------------------


def test_drawdown_penalty_normalization():
    """10% DD with penalty=0.5 should cost 0.5 Sharpe points."""
    cfg = ScorerConfig(drawdown_penalty=0.5)
    m = _metrics(sharpe_ratio=2.0, max_drawdown_pct=10.0)
    # base=2.0, dd_penalty=0.5*(10/10)=0.5 → score=1.5
    assert calculate_score(m, cfg) == pytest.approx(1.5)


def test_drawdown_penalty_20pct():
    """20% DD with penalty=0.5 should cost 1.0 Sharpe point."""
    cfg = ScorerConfig(drawdown_penalty=0.5)
    m = _metrics(sharpe_ratio=2.0, max_drawdown_pct=20.0)
    assert calculate_score(m, cfg) == pytest.approx(1.0)


def test_drawdown_penalty_zero():
    cfg = ScorerConfig(drawdown_penalty=0.0)
    m = _metrics(sharpe_ratio=1.5, max_drawdown_pct=50.0)
    assert calculate_score(m, cfg) == pytest.approx(1.5)


def test_drawdown_penalty_uses_abs_value():
    """max_drawdown_pct stored as negative in some backends - should still work."""
    cfg = ScorerConfig(drawdown_penalty=0.5)
    pos = calculate_score(_metrics(sharpe_ratio=2.0, max_drawdown_pct=10.0), cfg)
    neg = calculate_score(_metrics(sharpe_ratio=2.0, max_drawdown_pct=-10.0), cfg)
    assert pos == pytest.approx(neg)


# ---------------------------------------------------------------------------
# Turnover penalty
# ---------------------------------------------------------------------------


def test_turnover_penalty():
    cfg = ScorerConfig(drawdown_penalty=0.0, turnover_penalty=1.0)
    # 500 trades → penalty = 1.0 * (500/1000) = 0.5
    m = _metrics(sharpe_ratio=2.0, max_drawdown_pct=0.0, total_trades=500)
    assert calculate_score(m, cfg) == pytest.approx(1.5)


def test_turnover_penalty_zero_by_default():
    cfg = ScorerConfig(drawdown_penalty=0.0)
    m = _metrics(sharpe_ratio=1.0, max_drawdown_pct=0.0, total_trades=10_000)
    assert calculate_score(m, cfg) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Trade count bonus
# ---------------------------------------------------------------------------


def test_trade_count_bonus():
    cfg = ScorerConfig(drawdown_penalty=0.0, trade_count_bonus=1.0, trade_bonus_cap=2.0)
    # 500 trades → bonus = 1.0 * min(0.5, 2.0) = 0.5
    m = _metrics(sharpe_ratio=1.0, max_drawdown_pct=0.0, total_trades=500)
    assert calculate_score(m, cfg) == pytest.approx(1.5)


def test_trade_count_bonus_capped():
    cfg = ScorerConfig(drawdown_penalty=0.0, trade_count_bonus=1.0, trade_bonus_cap=0.5)
    # 5000 trades → min(5.0, 0.5) = 0.5 → bonus = 0.5
    m = _metrics(sharpe_ratio=1.0, max_drawdown_pct=0.0, total_trades=5000)
    assert calculate_score(m, cfg) == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Missing / empty metrics
# ---------------------------------------------------------------------------


def test_empty_metrics_returns_invalid():
    """Empty dict → 0 trades → fails min_trades gate."""
    cfg = ScorerConfig(min_trades=1)
    assert calculate_score({}, cfg) == INVALID_SCORE


def test_partial_metrics_uses_defaults():
    """Only total_trades provided - other fields default to 0."""
    cfg = ScorerConfig(min_trades=1, drawdown_penalty=0.0)
    # sharpe=0, sortino=0, return=0 → base=0.0
    score = calculate_score({"total_trades": 10}, cfg)
    assert score == pytest.approx(0.0)
