"""
src/strategy/orb_plugin.py

ORB strategy plugin - registers ORB with the global strategy registry.

Single source of truth for:
  - ORB param spaces (full, core, wfo_grid, wfo_optuna)
  - Trial function builders (standalone + WFO)
  - Risk key routing
  - Strategy loader (for run_backtest)

To add a new strategy, create a similar plugin and register it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.strategy.strategy_registry import StrategyPlugin, register_strategy_plugin
from src.utils.frequency import parse_frequency_to_minutes

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RISK_KEYS: set[str] = {
    "use_trailing_stop",
    "trailing_atr_multiplier",
    "risk_per_trade_pct",
    "entry_ord_type",
}

# ---------------------------------------------------------------------------
# Param spaces
# ---------------------------------------------------------------------------

_FULL: dict[str, dict[str, Any]] = {
    # Frequency
    "resample_freq": {"type": "categorical", "choices": ["15min"]},
    # Core strategy
    "orb_minutes": {"type": "int", "low": 0, "high": 60, "step": 15},
    "atr_period": {"type": "int", "low": 14, "high": 30},
    "atr_tp_multiplier": {"type": "float", "low": 1.0, "high": 4.0, "step": 0.05},
    "atr_sl_multiplier": {"type": "float", "low": 0.5, "high": 2.0, "step": 0.05},
    "breakout_buffer": {"type": "float", "low": 0.0, "high": 1.0, "step": 0.05},
    "require_close_confirmation": {"type": "categorical", "choices": [True, False]},
    "use_range_sl": {"type": "categorical", "choices": [True, False]},
    "min_range_atr": {"type": "float", "low": 0.3, "high": 2.0, "step": 0.1},
    "max_range_atr": {"type": "float", "low": 2.0, "high": 6.0, "step": 0.1},
    # Direction / trade limits
    "long_only": {"type": "categorical", "choices": [False]},
    "max_trades_per_session": {"type": "int", "low": 1, "high": 3},
    # Optional filters
    "use_volume_filter": {"type": "categorical", "choices": [True, False]},
    "use_adx_filter": {"type": "categorical", "choices": [True, False]},
    "adx_min": {"type": "float", "low": 15.0, "high": 35.0, "step": 1.0},
    # Adaptive volatility
    "use_adaptive_volatility": {"type": "categorical", "choices": [True, False]},
    "atr_lookback_period": {"type": "int", "low": 10, "high": 30, "step": 5},
    "volatility_low_threshold": {"type": "float", "low": 0.6, "high": 0.85, "step": 0.05},
    "volatility_high_threshold": {"type": "float", "low": 1.2, "high": 1.5, "step": 0.05},
    "low_vol_range_multiplier": {"type": "float", "low": 0.5, "high": 0.9, "step": 0.1},
    "high_vol_range_multiplier": {"type": "float", "low": 1.1, "high": 1.5, "step": 0.1},
    "low_vol_buffer_multiplier": {"type": "float", "low": 0.5, "high": 0.9, "step": 0.1},
    "high_vol_buffer_multiplier": {"type": "float", "low": 1.1, "high": 1.5, "step": 0.1},
    # Risk
    "use_trailing_stop": {"type": "categorical", "choices": [True, False]},
    "trailing_atr_multiplier": {"type": "float", "low": 1.0, "high": 4.0, "step": 0.25},
    "risk_per_trade_pct": {"type": "float", "low": 0.5, "high": 3.0, "step": 0.25},
    "entry_ord_type": {"type": "categorical", "choices": ["LIMIT"]},
}

_CORE: dict[str, dict[str, Any]] = {
    k: v
    for k, v in _FULL.items()
    if k
    in {
        "orb_minutes",
        "atr_period",
        "atr_tp_multiplier",
        "atr_sl_multiplier",
        "breakout_buffer",
        "require_close_confirmation",
        "use_range_sl",
        "min_range_atr",
        "max_range_atr",
    }
}

_WFO_GRID: dict[str, list[Any]] = {
    "orb_minutes": [10, 15, 20, 30],
    "atr_period": [10, 14, 20],
    "atr_tp_multiplier": [1.5, 2.0, 3.0],
    "atr_sl_multiplier": [0.75, 1.0, 1.5],
    "breakout_buffer": [0.0, 0.1, 0.2],
    "require_close_confirmation": [True, False],
    "use_range_sl": [True, False],
    "min_range_atr": [0.3, 0.5, 0.8],
    "max_range_atr": [2.5, 3.0, 4.0],
    "long_only": [False],
    "use_volume_filter": [False],
    "use_adx_filter": [False],
    # Adaptive volatility
    "use_adaptive_volatility": [False, True],
    "atr_lookback_period": [15, 20],
    "volatility_low_threshold": [0.7, 0.75],
    "volatility_high_threshold": [1.25, 1.3],
    "low_vol_range_multiplier": [0.7, 0.8],
    "high_vol_range_multiplier": [1.2, 1.3],
    "low_vol_buffer_multiplier": [0.7, 0.8],
    "high_vol_buffer_multiplier": [1.2, 1.3],
}

# WFO Optuna: no resample_freq - data is pre-sliced per window
_WFO_OPTUNA: dict[str, dict[str, Any]] = {k: v for k, v in _FULL.items() if k != "resample_freq"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _invalid_result() -> Any:
    return type("R", (), {"metrics": {"total_trades": 0}})()


def _split_params(
    params: dict[str, Any],
    base_freq: str,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Return (trial_freq, strategy_params, risk_params) from a flat params dict."""
    trial_freq = params.get("resample_freq", base_freq)
    strategy_params = {
        k: v for k, v in params.items() if k not in _RISK_KEYS and k != "resample_freq"
    }
    risk_params = {k: v for k, v in params.items() if k in _RISK_KEYS}
    return trial_freq, strategy_params, risk_params


def _make_trial_raw(
    base_raw: dict[str, Any],
    strategy_params: dict[str, Any],
    risk_params: dict[str, Any],
    trial_freq: str,
) -> dict[str, Any]:
    """Merge trial params into base config dict."""
    return {
        **base_raw,
        "strategy": {**base_raw["strategy"], "resample_freq": trial_freq, **strategy_params},
        "risk": {**base_raw["risk"], **risk_params},
    }


def _run_backtest(
    config: Any,
    data: Any,
    capital: float,
    commission_rate: float,
    slippage_points: float,
    contract_multiplier: float,
    margin_rate: float,
    cache_dir: str,
    freq_minutes: int,
    use_cache: bool = True,
    processed_data: Any = None,
    bars_list: list[dict[str, Any]] | None = None,
) -> Any:
    """
    Build pipeline + backtester from a validated config and run it.

    Args:
        processed_data: Pre-computed DataFrame with indicators (optional)
        bars_list: Pre-converted list of dicts from processed_data (optional)
                  If provided along with processed_data, skips expensive to_dict("records")
    """
    from src.data.pipeline import DataPipeline
    from src.engine.account.sizer import PercentRiskSizer
    from src.engine.backtester import Backtester
    from src.engine.execution.slippage import FixedSlippage
    from src.strategy.orb import ORBStrategy

    strategy = ORBStrategy(config)
    registry = ORBStrategy.build_registry(
        atr_period=config.strategy.atr_period,
        adx_period=config.strategy.adx_period,
        volume_ma_period=config.strategy.volume_ma_period,
        atr_lookback_period=config.strategy.atr_lookback_period,
        use_adaptive_volatility=config.strategy.use_adaptive_volatility,
    )

    if processed_data is not None:
        processed = processed_data
        warmup = registry.get_required_lookback()
    else:
        pipeline = DataPipeline(registry, cache_dir=cache_dir, use_cache=use_cache)
        processed = pipeline.run(data)
        warmup = pipeline.get_required_lookback()

    sizer = PercentRiskSizer(
        risk_per_trade_pct=config.risk.risk_per_trade_pct,
        min_size=config.risk.min_position_size,
        max_size=config.risk.max_position_size,
    )
    bt = Backtester(
        strategy=strategy,
        initial_capital=capital,
        commission_rate=commission_rate,
        contract_multiplier=contract_multiplier,
        margin_rate=margin_rate,
        position_sizer=sizer,
        slippage_model=FixedSlippage(slippage_points),
        use_trailing_stop=config.risk.use_trailing_stop,
        trailing_atr_multiplier=config.risk.trailing_atr_multiplier,
        max_daily_loss_pct=config.risk.max_daily_loss,
        entry_cutoff_seconds=float(config.risk.entry_cutoff_seconds),
        allow_late_entry=config.risk.allow_late_entry,
        freq_minutes=freq_minutes,
    )
    return bt.run(processed, warmup_bars=warmup, bars_list=bars_list)


# ---------------------------------------------------------------------------
# Strategy loader  (run_backtest)
# ---------------------------------------------------------------------------


def load_fn(config_path: str) -> tuple[Any, Any, Any]:
    """Return (ORBStrategy, IndicatorRegistry, ORBConfig) from a config file."""
    from config.schemas.orb import ORBConfig
    from src.strategy.orb import ORBStrategy

    config = ORBConfig.from_json(config_path)
    strategy = ORBStrategy(config)
    registry = ORBStrategy.build_registry(
        atr_period=config.strategy.atr_period,
        adx_period=config.strategy.adx_period,
        volume_ma_period=config.strategy.volume_ma_period,
        atr_lookback_period=config.strategy.atr_lookback_period,
        use_adaptive_volatility=config.strategy.use_adaptive_volatility,
    )
    return strategy, registry, config


# ---------------------------------------------------------------------------
# Trial function builders
# ---------------------------------------------------------------------------


def build_trial_fn(
    preprocessed_data: Any,
    base_config_path: str,
    capital: float,
    commission_rate: float,
    slippage_points: float,
    contract_multiplier: float,
    margin_rate: float,
    cache_dir: str,
    freq: str,
) -> Any:
    """Return a standalone optimization trial fn: (params) -> BacktestResult."""
    from config.schemas.orb import ORBConfig
    from src.data.pipeline import DataPipeline
    from src.data.preprocessor import DataPreprocessor
    from src.strategy.orb import ORBStrategy

    base_raw = json.loads(Path(base_config_path).read_text(encoding="utf-8"))

    # In-memory indicator cache keyed by (atr_period, adx_period, volume_ma_period, data_id).
    # Avoids recomputing indicators when Optuna revisits the same combo.
    # DataPipeline disk cache is still used as a secondary layer for cross-run persistence.
    # Cache both DataFrame AND bars list to avoid repeated to_dict("records") calls.
    _indicator_cache: dict[tuple, tuple[Any, list[dict]]] = {}

    def _get_processed_and_bars(config: Any, data: Any, data_id: int) -> tuple[Any, list[dict]]:
        """
        Get processed DataFrame with indicators AND pre-converted bars list.

        Args:
            config: Strategy config with indicator periods
            data: Input DataFrame
            data_id: Unique ID for this data (use id(data) for identity-based caching)

        Returns:
            (processed_df, bars_list) tuple - both cached together
        """
        key = (
            config.strategy.atr_period,
            config.strategy.adx_period,
            config.strategy.volume_ma_period,
            config.strategy.atr_lookback_period,
            config.strategy.use_adaptive_volatility,
            data_id,  # Use data identity instead of shape for reliable caching
        )
        if key not in _indicator_cache:
            registry = ORBStrategy.build_registry(
                atr_period=config.strategy.atr_period,
                adx_period=config.strategy.adx_period,
                volume_ma_period=config.strategy.volume_ma_period,
                atr_lookback_period=config.strategy.atr_lookback_period,
                use_adaptive_volatility=config.strategy.use_adaptive_volatility,
            )
            pipeline = DataPipeline(registry, cache_dir=cache_dir, use_cache=True)
            processed_df = pipeline.run(data)
            # Pre-convert to bars list once and cache it
            bars_list = processed_df.to_dict("records")
            _indicator_cache[key] = (processed_df, bars_list)

        return _indicator_cache[key]

    def trial_fn(params: dict[str, Any]) -> Any:
        trial_freq, strategy_params, risk_params = _split_params(params, freq)
        trial_raw = _make_trial_raw(base_raw, strategy_params, risk_params, trial_freq)

        try:
            config = ORBConfig.from_dict(trial_raw)
        except Exception:
            return _invalid_result()

        data = (
            DataPreprocessor().prepare(preprocessed_data, freq=trial_freq)  # type: ignore
            if trial_freq != freq
            else preprocessed_data
        )

        # Get both processed DataFrame and pre-converted bars list from cache
        # This avoids expensive to_dict("records") call on every trial
        data_id = id(data)  # Use object identity for reliable cache key
        processed_df, bars = _get_processed_and_bars(config, data, data_id)

        return _run_backtest(
            config=config,
            data=data,
            capital=capital,
            commission_rate=commission_rate,
            slippage_points=slippage_points,
            contract_multiplier=contract_multiplier,
            margin_rate=margin_rate,
            cache_dir=cache_dir,
            freq_minutes=parse_frequency_to_minutes(trial_freq),
            use_cache=True,
            processed_data=processed_df,
            bars_list=bars,
        )

    return trial_fn


def build_wfo_trial_fn(
    base_config_path: str,
    capital: float,
    commission_rate: float,
    slippage_points: float,
    contract_multiplier: float,
    margin_rate: float,
    cache_dir: str,
    freq: str,
) -> Any:
    """Return a WFO trial fn: (params, data_slice, window_capital) -> BacktestResult."""
    from config.schemas.orb import ORBConfig

    base_raw = json.loads(Path(base_config_path).read_text(encoding="utf-8"))
    freq_minutes = parse_frequency_to_minutes(freq)

    def trial_fn(params: dict[str, Any], data: pd.DataFrame, window_capital: float) -> Any:
        trial_freq, strategy_params, risk_params = _split_params(params, freq)
        trial_raw = _make_trial_raw(base_raw, strategy_params, risk_params, trial_freq)
        try:
            config = ORBConfig.from_dict(trial_raw)
        except Exception:
            return _invalid_result()

        return _run_backtest(
            config=config,
            data=data,
            capital=window_capital,
            commission_rate=commission_rate,
            slippage_points=slippage_points,
            contract_multiplier=contract_multiplier,
            margin_rate=margin_rate,
            cache_dir=cache_dir,
            freq_minutes=freq_minutes,
            use_cache=False,  # each window slice is unique
        )

    return trial_fn


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

register_strategy_plugin(
    StrategyPlugin(
        name="orb",
        display_name="Opening Range Breakout",
        default_config="config/strategy_params/orb_default.json",
        load_fn=load_fn,
        param_spaces={"full": _FULL, "core": _CORE},
        build_trial_fn=build_trial_fn,
        build_wfo_trial_fn=build_wfo_trial_fn,
        wfo_grid_space=_WFO_GRID,
        wfo_optuna_space=_WFO_OPTUNA,
        risk_keys=_RISK_KEYS,
        session_name="vn30",
    )
)
