"""
src/optimization/scoring.py

Pluggable scoring functions for optimization.

Separating scoring from OptunaSearch / GridSearch means:
- Easy to swap scoring logic without changing optimizer code
- Scoring can be unit-tested independently
- Walk-forward and Optuna can share the same scorer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScorerConfig:
    """
    Configuration for composite scoring.

    Score formula:
        Hard gates (return INVALID_SCORE if any fail):
            total_trades < min_trades
            total_return < min_return_pct
            net_profit_factor < min_profit_factor
            win_rate < min_win_rate_pct

        base  = sharpe_ratio  (if > 0)
              | sortino_ratio * 0.9  (if sharpe <= 0 but sortino > 0)
              | total_return / 10.0  (last resort, same scale as Sharpe)

        score = base
              - drawdown_penalty * (max_drawdown_pct / 10.0)   ← normalized to ~Sharpe scale
              - turnover_penalty * (trades / 1000.0)
              + trade_count_bonus * min(trades / 1000.0, trade_bonus_cap)

    Drawdown penalty normalization:
        max_drawdown_pct is in % (e.g. 20.0 for 20% drawdown).
        Dividing by 10 maps 10% DD → 1.0 penalty unit, same scale as Sharpe.
        So drawdown_penalty=0.5 means: 20% DD costs 1.0 Sharpe point.

    Notes:
        - INVALID_SCORE and ERROR_SCORE are class-level constants, not configurable.
        - min_trades uses strict < (V1 bug: used <= so exactly min_trades was invalid).
        - All penalty weights must be >= 0.
    """

    # Hard gates
    min_trades: int = 30
    min_return_pct: float = -999.0
    min_profit_factor: float = -999.0
    min_win_rate_pct: float = 0.0  # e.g. 30.0 to require at least 30% win rate

    # Penalty / bonus weights
    drawdown_penalty: float = 0.5  # per 10% drawdown → 0.5 Sharpe points
    turnover_penalty: float = 0.0
    trade_count_bonus: float = 0.0
    trade_bonus_cap: float = 1.0  # cap bonus at this many "thousands of trades"

    # Sentinel values - class-level constants, not configurable via __init__
    INVALID_SCORE: float = field(default=-10.0, init=False, repr=False, compare=False)
    ERROR_SCORE: float = field(default=-100.0, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.min_trades < 0:
            raise ValueError(f"min_trades must be >= 0, got {self.min_trades}")
        if self.min_win_rate_pct < 0 or self.min_win_rate_pct > 100:
            raise ValueError(f"min_win_rate_pct must be in [0, 100], got {self.min_win_rate_pct}")
        if self.drawdown_penalty < 0:
            raise ValueError(f"drawdown_penalty must be >= 0, got {self.drawdown_penalty}")
        if self.turnover_penalty < 0:
            raise ValueError(f"turnover_penalty must be >= 0, got {self.turnover_penalty}")
        if self.trade_count_bonus < 0:
            raise ValueError(f"trade_count_bonus must be >= 0, got {self.trade_count_bonus}")
        if self.trade_bonus_cap <= 0:
            raise ValueError(f"trade_bonus_cap must be > 0, got {self.trade_bonus_cap}")


def calculate_score(
    metrics: dict[str, Any],
    cfg: ScorerConfig,
) -> float:
    """
    Calculate composite optimization score from backtest metrics.

    Args:
        metrics: dict from PerformanceMetrics.to_dict() or BacktestResult.metrics.
        cfg:     ScorerConfig with penalty/bonus weights.

    Returns:
        Float score. Higher is better.
        cfg.INVALID_SCORE (-10) = failed hard gate.
        cfg.ERROR_SCORE (-100)  = exception during trial.
    """
    total_trades = int(metrics.get("total_trades", 0))
    sharpe = float(metrics.get("sharpe_ratio", 0.0))
    sortino = float(metrics.get("sortino_ratio", 0.0))
    max_dd = abs(float(metrics.get("max_drawdown_pct", 0.0)))
    total_return = float(metrics.get("total_return_pct", 0.0))
    profit_factor = float(metrics.get("net_profit_factor", 0.0))
    win_rate = float(metrics.get("win_rate_pct", 0.0))

    # Hard gates - return INVALID immediately
    if total_trades < cfg.min_trades:
        return cfg.INVALID_SCORE
    if total_return < cfg.min_return_pct:
        return cfg.INVALID_SCORE
    if profit_factor < cfg.min_profit_factor:
        return cfg.INVALID_SCORE
    if win_rate < cfg.min_win_rate_pct:
        return cfg.INVALID_SCORE

    # Base score - consistent scale across all three paths
    # Sharpe and Sortino are already dimensionless risk-adjusted ratios.
    # total_return / 10 maps: +30% return → +3.0, -20% → -2.0 (Sharpe-like range).
    if sharpe > 0:
        base_score = sharpe
    elif sortino > 0:
        base_score = sortino * 0.9  # slight discount: Sortino ignores upside vol
    else:
        base_score = total_return / 10.0

    # Drawdown penalty - normalized so 10% DD = 1.0 penalty unit (Sharpe scale)
    # drawdown_penalty=0.5 → 20% DD costs 1.0 Sharpe point
    dd_penalty = cfg.drawdown_penalty * (max_dd / 10.0)

    # Turnover penalty - discourages over-trading
    turnover = cfg.turnover_penalty * (total_trades / 1000.0)

    # Trade count bonus - capped to prevent HFT strategies dominating
    bonus = cfg.trade_count_bonus * min(total_trades / 1000.0, cfg.trade_bonus_cap)

    return base_score - dd_penalty - turnover + bonus
