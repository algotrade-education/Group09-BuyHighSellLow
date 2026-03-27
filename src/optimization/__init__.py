"""
Optimization tools for parameter search, scoring, and robustness checks.
"""

from src.optimization.grid_search import GridResult, GridSearch
from src.optimization.monte_carlo import MonteCarlo, MonteCarloResult
from src.optimization.optuna_search import OptunaResult, OptunaSearch
from src.optimization.scoring import ScorerConfig, calculate_score
from src.optimization.walk_forward import WalkForwardOptimizer, WalkForwardResult, WalkForwardWindow

__all__ = [
    # Search engines
    "GridSearch",
    "OptunaSearch",
    "WalkForwardOptimizer",
    # Result classes
    "GridResult",
    "OptunaResult",
    "WalkForwardWindow",
    "WalkForwardResult",
    "MonteCarlo",
    "MonteCarloResult",
    # Scoring
    "ScorerConfig",
    "calculate_score",
]
