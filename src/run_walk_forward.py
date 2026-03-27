"""
src/run_walk_forward.py

Walk-forward optimization for any registered strategy.

Usage:
    # Anchored WFO with Optuna inner optimizer (default)
    python -m src.run_walk_forward --strategy orb --start 2022-01-01 --end 2024-12-31

    # Rolling windows with grid search
    python -m src.run_walk_forward --strategy orb --mode rolling --optimizer grid

    # More windows, more Optuna trials per window
    python -m src.run_walk_forward --strategy orb --n-windows 6 --n-trials 150

Adding a new strategy:
    Create src/strategy/<name>_plugin.py and call register_strategy_plugin().
    It will appear automatically in --strategy choices.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from config.constants import ExecutionConfig
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.database.data_service import get_data_service
from src.optimization.scoring import ScorerConfig
from src.optimization.walk_forward import WalkForwardOptimizer
from src.strategy.strategy_registry import get_strategy_plugin, list_strategy_names
from src.utils.cli_helpers import print_exception, print_kv_rows, print_section, print_status
from src.utils.logger import setup_logging

logger = setup_logging(
    name="run_walk_forward",
    log_file="logs/walk_forward.log",
    capture_all_loggers=False,
)


def run(args: argparse.Namespace) -> int:
    # --- 1. Resolve plugin + config ---
    try:
        plugin = get_strategy_plugin(args.strategy)
    except KeyError as e:
        print_status(str(e), "error")
        return 1

    config_path = args.config or plugin.default_config
    if not Path(config_path).exists():
        print_status(f"Config not found: {config_path}", "error")
        return 1

    base_raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    freq = args.freq or base_raw.get("strategy", {}).get("resample_freq", "5min")

    print_section(f"WALK-FORWARD: {plugin.display_name}", width=60)
    print_kv_rows(
        {
            "Period": f"{args.start} → {args.end}",
            "Windows": args.n_windows,
            "Mode": args.mode,
            "Optimizer": args.optimizer,
            "Trials/window": args.n_trials if args.optimizer == "optuna" else "N/A (grid)",
            "Freq": freq,
            "Session": plugin.session_name,
            "Capital": f"{args.capital:,.0f}",
        },
        label_width=14,
    )
    print()

    # --- 2. Load + preprocess data ---
    print_status("Loading data...", "info")
    try:
        loader = DataLoader(data_service=get_data_service(), cache_dir=args.cache_dir)
        raw = loader.load(
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            use_cache=not args.force_refresh,
        )
        prep = DataPreprocessor().prepare(raw, freq=freq)
        print_status(f"{len(prep):,} bars loaded ({freq})", "success")
    except Exception as e:
        logger.error("Data load failed: %s", e, exc_info=True)
        print_exception("Data load", e)
        return 1

    # --- 3. Build trial function + param space via plugin ---
    trial_fn = plugin.build_wfo_trial_fn(
        base_config_path=config_path,
        capital=args.capital,
        commission_rate=args.commission_rate,
        slippage_points=args.slippage_points,
        contract_multiplier=args.contract_multiplier,
        margin_rate=args.margin_rate,
        cache_dir=args.cache_dir,
        freq=freq,
    )
    param_space = plugin.wfo_optuna_space if args.optimizer == "optuna" else plugin.wfo_grid_space

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
        result = wfo.optimize(data=prep, initial_capital=args.capital, show_progress=True)
    except Exception as e:
        logger.error("WFO failed: %s", e, exc_info=True)
        print_exception("Walk-forward", e)
        return 1

    elapsed = time.perf_counter() - t0
    print_status(f"Done in {elapsed:.1f}s", "success")

    # --- 5. Print + save ---
    result.print_summary()
    try:
        paths = result.save(Path(args.output_dir))
        for name, path in paths.items():
            print_status(f"{name} → {path}", "success")
    except Exception as e:
        logger.warning("Save failed: %s", e)

    return 0


def build_parser() -> argparse.ArgumentParser:
    strategies = list_strategy_names()

    parser = argparse.ArgumentParser(
        prog="run_walk_forward",
        description="Walk-forward optimization for strategy robustness validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--strategy", "-s", default="orb", choices=strategies)
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
    parser.add_argument("--n-windows", type=int, default=5)
    parser.add_argument(
        "--mode",
        default="anchored",
        choices=["anchored", "rolling"],
        help="anchored=train grows, rolling=fixed-size sliding window",
    )
    parser.add_argument("--optimizer", default="optuna", choices=["optuna", "grid"])
    parser.add_argument("--n-trials", type=int, default=100, help="Optuna trials per window")
    parser.add_argument(
        "--train-pct", type=float, default=0.7, help="Train fraction (rolling only)"
    )
    parser.add_argument("--embargo-bars", type=int, default=0)
    parser.add_argument("--no-chain-capital", action="store_true")
    parser.add_argument(
        "--storage", default=None, help="SQLite path for Optuna persistence per window"
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
