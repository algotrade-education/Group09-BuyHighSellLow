# Optimization module
# Optimization module
from .grid_search import GridSearch, OptimizationResult
from .walk_forward import WalkForwardOptimizer, WalkForwardResult, WalkForwardWindow

try:
    from .optuna_search import OptunaSearch, OptunaResult

    _OPTUNA_EXPORTS = ["OptunaSearch", "OptunaResult"]
except ImportError:
    _OPTUNA_EXPORTS = []

__all__ = [
    "GridSearch",
    "OptimizationResult",
    "WalkForwardOptimizer",
    "WalkForwardResult",
    "WalkForwardWindow",
    *_OPTUNA_EXPORTS,
]
