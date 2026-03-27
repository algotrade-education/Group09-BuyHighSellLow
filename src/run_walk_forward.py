"""
src/run_walk_forward.py

Walk-forward optimization for a strategy over a date range.

Walk-forward splits data into N train/test windows, optimizes params on each
train window, then validates on the out-of-sample test window. This detects
whether a strategy's edge is persistent or just curve-fitting.

Usage:
    # Anchored WFO with Optuna inner optimizer (default)
    python -m src.run_walk_forward --strategy orb --start 2022-01-01 --end 2024-12-31

    # Rolling windows with grid search
    python -m src.run_walk_forward --strategy orb --mode rolling --optimizer grid

    # More windows, more Optuna trials per window
    python -m src.run_walk_forward --strategy orb --n-windows 6 --n-trials 150

    # Save results to custom dir
    python -m src.run_walk_forward --strategy orb --output-dir results/wfo/orb_2024
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from config.constants import ExecutionConfig
from config.schemas.orb import ORBConfig
from src.data.loader import DataLoader
from src.data.pipeline import DataPipeline
from src.data.preprocessor import DataPreprocessor
from src.database.data_service import get_data_service
from src.engine.account.sizer import PercentRiskSizer
from src.engine.backtester import Backtester
from src.engine.execution.slippage import FixedSlippage
from src.optimization.scoring import ScorerConfig
from src.optimization.walk_forward import WalkForwardOptimizer
from src.strategy.orb import ORBStrategy
from src.utils.cli_helpers import print_exception, print_kv_rows, print_section, print_status
from src.utils.logger import setup_logging

logger = setup_logging(
    name="run_walk_forward",
    log_file="logs/walk_forward.log",
    capture_all_loggers=False,
)

# --- Param spaces (same as run_optimize, but as grid lists for grid optimizer) ---

_ORB_GRID_SPACE: dict[str, list[Any]] = {
    "orb_minutes": [10, 15, 20, 30],
    "atr_period": [10, 14, 20],
    "atr_tp_multiplier": [1.5, 2.0, 3.0],
    "atr_sl_multiplier": [0.75, 1.0, 1.5],
    "breakout_buffer": [0.0, 0.1, 0.2],
    "use_range_sl": [True, False],
    "min_range_atr": [0.3, 0.5, 0.8],
    "max_range_atr": [2.5, 3.0, 4.0],
    "long_only": [False],
    "use_volume_filter": [False],
    "use_adx_filter": [False],
}

# Optuna space for WFO (continuous ranges, no freq - data is pre-sliced per window)
_ORB_OPTUNA_SPACE: dict[str, dict[str, Any]] = {
    "orb_minutes": {"type": "int", "low": 10, "high": 60, "step": 5},
    "atr_period": {"type": "int", "low": 5, "high": 30},
    "atr_tp_multiplier": {"type": "float", "low": 1.0, "high": 4.0, "step": 0.25},
    "atr_sl_multiplier": {"type": "float", "low": 0.5, "high": 2.0, "step": 0.25},
    "breakout_buffer": {"type": "float", "low": 0.0, "high": 1.0, "step": 0.1},
    "use_range_sl": {"type": "categorical", "choices": [True, False]},
    "min_range_atr": {"type": "float", "low": 0.3, "high": 2.0, "step": 0.1},
    "max_range_atr": {"type": "float", "low": 2.0, "high": 6.0, "step": 0.5},
    "long_only": {"type": "categorical", "choices": [True, False]},
    "use_trailing_stop": {"type": "categorical", "choices": [True, False]},
    "trailing_atr_multiplier": {"type": "float", "low": 1.0, "high": 4.0, "step": 0.25},
    "risk_per_trade_pct": {"type": "float", "low": 0.5, "high": 3.0, "step": 0.25},
}

# Keys that belong to RiskConfig, not ORBStrategyConfig
_RISK_KEYS = {"use_trailing_stop", "trailing_atr_multiplier", "risk_per_trade_pct"}


def _build_orb_wfo_trial_fn(
    base_config_path: str,
    capital: float,
    commission_rate: float,
    slippage_points: float,
    contract_multiplier: float,
    margin_rate: float,
    cache_dir: str,
    freq: str,
) -> Any:
    """
    Build a WFO trial_fn for ORB.

    Signature: (params, data_slice, capital) → BacktestResult
    Each window passes its own data slice and chained capital.
    """
    base_raw = json.loads(Path(base_config_path).read_text(encoding="utf-8"))
    freq_minutes = int(freq.replace("min", ""))

    def trial_fn(params: dict[str, Any], data: pd.DataFrame, window_capital: float) -> Any:
        strategy_params = {k: v for k, v in params.items() if k not in _RISK_KEYS}
        risk_params = {k: v for k, v in params.items() if k in _RISK_KEYS}

        trial_raw = {
            **base_raw,
            "strategy": {**base_raw["strategy"], **strategy_params},
            "risk": {**base_raw["risk"], **risk_params},
        }

        try:
            config = ORBConfig.from_dict(trial_raw)
        except Exception:
            from src.run_optimize import _invalid_result

            return _invalid_result()

        strategy = ORBStrategy(config)
        registry = ORBStrategy.build_registry(
            atr_period=config.strategy.atr_period,
            adx_period=config.strategy.adx_period,
            volume_ma_period=config.strategy.volume_ma_period,
        )

        pipeline = DataPipeline(registry, cache_dir=cache_dir, use_cache=False)
        processed = pipeline.run(data)
        warmup = pipeline.get_required_lookback()

        sizer = PercentRiskSizer(
            risk_per_trade_pct=config.risk.risk_per_trade_pct,
            min_size=config.risk.min_position_size,
            max_size=config.risk.max_position_size,
        )
        bt = Backtester(
            strategy=strategy,
            initial_capital=window_capital,
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
        return bt.run(processed, warmup_bars=warmup)

    return trial_fn


def run(args: argparse.Namespace) -> int:
    # --- 1. Resolve config ---
    config_path = args.config or f"config/strategy_params/{args.strategy}_default.json"
    if not Path(config_path).exists():
        print_status(f"Config not found: {config_path}", "error")
        return 1

    base_raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    freq = args.freq or base_raw.get("strategy", {}).get("resample_freq", "5min")

    print_section(f"WALK-FORWARD: {args.strategy.upper()}", width=60)
    print_kv_rows(
        {
            "Period": f"{args.start} → {args.end}",
            "Windows": args.n_windows,
            "Mode": args.mode,
            "Optimizer": args.optimizer,
            "Trials/window": args.n_trials if args.optimizer == "optuna" else "N/A (grid)",
            "Freq": freq,
            "Capital": f"{args.capital:,.0f}",
        },
        label_width=14,
    )
    print()

    # --- 2. Load + preprocess data ---
    print_status("Loading data...", "info")
    try:
        loader = DataLoader(
            data_service=get_data_service(),
            cache_dir=args.cache_dir,
        )
        raw = loader.load(
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            use_cache=not args.force_refresh,
        )
        preprocessor = DataPreprocessor()
        prep = preprocessor.prepare(raw, freq=freq)
        print_status(f"{len(prep):,} bars loaded ({freq})", "success")
    except Exception as e:
        logger.error("Data load failed: %s", e, exc_info=True)
        print_exception("Data load", e)
        return 1

    # --- 3. Build trial function ---
    if args.strategy == "orb":
        trial_fn = _build_orb_wfo_trial_fn(
            base_config_path=config_path,
            capital=args.capital,
            commission_rate=args.commission_rate,
            slippage_points=args.slippage_points,
            contract_multiplier=args.contract_multiplier,
            margin_rate=args.margin_rate,
            cache_dir=args.cache_dir,
            freq=freq,
        )
        param_space = _ORB_OPTUNA_SPACE if args.optimizer == "optuna" else _ORB_GRID_SPACE
    else:
        print_status(f"Strategy {args.strategy!r} not yet supported for WFO.", "error")
        return 1

    # --- 4. Run WFO ---
    scorer = ScorerConfig(
        min_trades=args.min_trades,
        min_return_pct=args.min_return,
        drawdown_penalty=0.5,
    )

    wfo = WalkForwardOptimizer(
        trial_fn=trial_fn,
        param_space=param_space,
        optimizer=args.optimizer,
        n_windows=args.n_windows,
        train_pct=args.train_pct,
        anchored=args.mode == "anchored",
        scorer=scorer,
        embargo_bars=args.embargo_bars,
        chain_capital=not args.no_chain_capital,
        n_trials=args.n_trials,
        optuna_storage=args.storage,
    )

    print_status(f"Running {args.n_windows} windows ({args.mode})...", "info")
    t0 = time.perf_counter()

    try:
        result = wfo.optimize(
            data=prep,
            initial_capital=args.capital,
            show_progress=True,
        )
    except Exception as e:
        logger.error("WFO failed: %s", e, exc_info=True)
        print_exception("Walk-forward", e)
        return 1

    elapsed = time.perf_counter() - t0
    print_status(f"Done in {elapsed:.1f}s", "success")

    # --- 5. Print summary ---
    result.print_summary()

    # --- 6. Save results ---
    out_dir = Path(args.output_dir)
    try:
        paths = result.save(out_dir)
        for name, path in paths.items():
            print_status(f"{name} → {path}", "success")
    except Exception as e:
        logger.warning("Save failed: %s", e)

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_walk_forward",
        description="Walk-forward optimization for strategy robustness validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--strategy", "-s", default="orb", choices=["orb"])
    parser.add_argument("--config", "-c", default=None, help="Base config JSON")

    # Data
    parser.add_argument("--symbol", default="VN30F1M")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--freq", default=None, choices=["1min", "5min", "15min", "30min"])
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--force-refresh", action="store_true")

    # Capital / costs
    ExecutionConfig.add_args(parser)

    # WFO settings
    parser.add_argument("--n-windows", type=int, default=5, help="Number of WFO windows")
    parser.add_argument(
        "--mode",
        default="anchored",
        choices=["anchored", "rolling"],
        help="anchored=train grows each window, rolling=fixed-size sliding window",
    )
    parser.add_argument(
        "--optimizer",
        default="optuna",
        choices=["optuna", "grid"],
        help="Inner optimizer per window. optuna=Bayesian TPE, grid=exhaustive",
    )
    parser.add_argument(
        "--n-trials", type=int, default=100, help="Optuna trials per window (optuna only)"
    )
    parser.add_argument(
        "--train-pct", type=float, default=0.7, help="Train fraction (rolling mode only)"
    )
    parser.add_argument(
        "--embargo-bars", type=int, default=0, help="Bars to skip at test window start"
    )
    parser.add_argument(
        "--no-chain-capital", action="store_true", help="Disable capital chaining between windows"
    )
    parser.add_argument(
        "--storage",
        default=None,
        help="SQLite path for Optuna persistence per window, e.g. results/wfo_study.db",
    )

    # Scorer gates
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--min-return", type=float, default=-999.0)

    # Output
    parser.add_argument("--output-dir", default="results/walk_forward")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
