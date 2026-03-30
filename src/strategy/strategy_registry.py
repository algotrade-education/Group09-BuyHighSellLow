"""
src/strategy/strategy_registry.py

Central registry for all strategies - backtest, optimization, and walk-forward.

Each strategy registers a StrategyPlugin that bundles everything the run scripts
need: how to load the strategy, config class, param spaces, and trial fn builders.

Adding a new strategy requires only:
    1. Create src/strategy/<name>_plugin.py
    2. Define a StrategyPlugin and call register_strategy_plugin()
    3. Add one import line to _ensure_plugins_loaded() below

No changes to run_backtest.py, run_optimize.py, or run_walk_forward.py are needed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

# --- Type aliases ---

# load_fn(config_path) -> (strategy, registry, config)
LoadFn = Callable[[str], tuple[Any, Any, Any]]

# trial_fn for standalone optimization: (params) -> BacktestResult
StandaloneTrial = Callable[[dict[str, Any]], Any]

# trial_fn for walk-forward: (params, data_slice, capital) -> BacktestResult
WFOTrial = Callable[[dict[str, Any], pd.DataFrame, float], Any]


@dataclass
class StrategyPlugin:
    """
    Everything the run scripts need to work with a strategy.

    Fields:
        name:               Short identifier, e.g. "orb". Must be unique.
        display_name:       Human-readable name for CLI output.
        default_config:     Path to the default JSON config file.
        load_fn:            Callable(config_path) -> (strategy, registry, config).
                            Used by run_backtest to instantiate the strategy.
        param_spaces:       Named Optuna param spaces. Must include at least "full".
        build_trial_fn:     Factory -> standalone trial_fn for run_optimize.
        build_wfo_trial_fn: Factory -> WFO trial_fn for run_walk_forward.
        wfo_grid_space:     Param grid (list values) for WFO grid optimizer.
        wfo_optuna_space:   Param space (range specs) for WFO Optuna optimizer.
        risk_keys:          Param names routed to config["risk"] during optimization.
        session_name:       Session identifier for display, e.g. "vn30".
    """

    name: str
    display_name: str
    default_config: str
    load_fn: LoadFn
    param_spaces: dict[str, dict[str, Any]]
    build_trial_fn: Callable[..., StandaloneTrial]
    build_wfo_trial_fn: Callable[..., WFOTrial]
    wfo_grid_space: dict[str, list[Any]]
    wfo_optuna_space: dict[str, dict[str, Any]]
    risk_keys: set[str] = field(default_factory=set)
    session_name: str = "vn30"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_registry: dict[str, StrategyPlugin] = {}


def register_strategy_plugin(plugin: StrategyPlugin) -> None:
    """Register a strategy plugin. Raises if name already taken."""
    if plugin.name in _registry:
        raise ValueError(
            f"Strategy plugin {plugin.name!r} is already registered. "
            "Use a unique name or call unregister_strategy_plugin() first."
        )
    _registry[plugin.name] = plugin


def get_strategy_plugin(name: str) -> StrategyPlugin:
    """Get a registered plugin by name. Raises KeyError if not found."""
    _ensure_plugins_loaded()
    if name not in _registry:
        available = list_strategy_names()
        raise KeyError(f"Strategy {name!r} not registered. Available: {available}")
    return _registry[name]


def list_strategy_names() -> list[str]:
    """Return all registered strategy names."""
    _ensure_plugins_loaded()
    return sorted(_registry.keys())


def list_param_space_keys(strategy_name: str) -> list[str]:
    """Return available param space names for a strategy."""
    plugin = get_strategy_plugin(strategy_name)
    return list(plugin.param_spaces.keys())


def unregister_strategy_plugin(name: str) -> None:
    """Remove a plugin from the registry (mainly for testing)."""
    _registry.pop(name, None)


# ---------------------------------------------------------------------------
# Lazy plugin loading
# ---------------------------------------------------------------------------

_plugins_loaded = False


def _ensure_plugins_loaded() -> None:
    """Import all plugin modules so they self-register."""
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True

    # Import each strategy plugin module so they self-register.
    # Add new strategies here.
    import src.strategy.orb_plugin  # noqa: F401
