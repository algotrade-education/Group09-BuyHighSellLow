"""
src/run_optimize.py

Run Optuna optimization for any registered strategy.

Full pipeline:
    Load data once → for each trial: build strategy → run backtest → score

Usage:
    # Basic: 200 trials, ORB full param space
    python -m src.run_optimize --strategy orb --start 2023-01-01 --end 2024-12-31

    # Core params only (faster, no freq/risk search)
    python -m src.run_optimize --strategy orb --param-space core --n-trials 100

    # Crash-safe + resumable via SQLite
    python -m src.run_optimize --strategy orb --n-trials 500 --storage results/orb_study.db

    # Multivariate TPE (better when params interact)
    python -m src.run_optimize --strategy orb --sampler tpe_multivariate

    # Save best config to file
    python -m src.run_optimize --strategy orb --save-best config/strategy_params/orb_optimized.json

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
from src.optimization.optuna_search import OptunaSearch
from src.optimization.scoring import ScorerConfig
from src.strategy.strategy_registry import (
    get_strategy_plugin,
    list_param_space_keys,
    list_strategy_names,
)
from src.utils.cli_helpers import print_exception, print_kv_rows, print_section, print_status
from src.utils.logger import setup_logging

logger = setup_logging(
    name="run_optimize",
    log_file="logs/optimize.log",
    capture_all_loggers=False,
)


def run(args: argparse.Namespace) -> int:
    # --- 1. Resolve plugin + config ---
    try:
        plugin = get_strategy_plugin(args.strategy)
    except KeyError as e:
        print_status(str(e), "error")
        return 1

    if args.param_space not in plugin.param_spaces:
        available = list_param_space_keys(args.strategy)
        print_status(
            f"Unknown param space {args.param_space!r} for {args.strategy!r}. "
            f"Available: {available}",
            "error",
        )
        return 1

    config_path = args.config or plugin.default_config
    if not Path(config_path).exists():
        print_status(f"Config not found: {config_path}", "error")
        return 1

    base_raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    freq = args.freq or base_raw.get("strategy", {}).get("resample_freq", "5min")

    print_section(f"OPTIMIZE: {plugin.display_name}", width=60)
    print_kv_rows(
        {
            "Period": f"{args.start} → {args.end}",
            "Trials": args.n_trials,
            "Sampler": args.sampler,
            "Param space": args.param_space,
            "Session": plugin.session_name,
            "Capital": f"{args.capital:,.0f}",
        },
        label_width=12,
    )
    print()

    # --- 2. Load + preprocess data once ---
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

    # --- 3. Build trial function via plugin ---
    param_space = plugin.param_spaces[args.param_space]
    trial_fn = plugin.build_trial_fn(
        preprocessed_data=prep,
        base_config_path=config_path,
        capital=args.capital,
        commission_rate=args.commission_rate,
        slippage_points=args.slippage_points,
        contract_multiplier=args.contract_multiplier,
        margin_rate=args.margin_rate,
        cache_dir=args.cache_dir,
        freq=freq,
    )

    # --- 4. Run optimization ---
    scorer = ScorerConfig(
        min_trades=args.min_trades,
        min_return_pct=args.min_return,
        drawdown_penalty=0.3,
        trade_count_bonus=0.1,
    )
    search = OptunaSearch(
        trial_fn=trial_fn,
        param_space=param_space,
        scorer=scorer,
        n_trials=args.n_trials,
        sampler=args.sampler,
        study_name=f"{args.strategy}_opt",
        storage_path=args.storage,
        seed=args.seed or 42,
    )

    # --- 4b. Handle storage conflict ---
    if args.storage:
        conflict = search.check_storage_conflict()
        if conflict is not None:
            print_status("Param space conflict detected!", "error")
            print(f"  {conflict.describe()}")
            print()

            on_conflict = args.on_conflict
            if on_conflict == "prompt":
                print("  Options:")
                print("    [r] Resume anyway  (mix old + new param space trials)")
                print("    [o] Overwrite      (delete existing study, start fresh)")
                print("    [a] Abort          (exit, rename --storage manually)")
                print()
                choice = input("  Choice [r/o/a]: ").strip().lower()
                if choice == "o":
                    on_conflict = "overwrite"
                elif choice == "r":
                    on_conflict = "resume"
                else:
                    print_status("Aborted.", "info")
                    return 1

            if on_conflict == "overwrite":
                search.delete_study()
                print_status("Existing study deleted. Starting fresh.", "info")
            elif on_conflict == "resume":
                print_status("Resuming with mismatched param space.", "info")
            else:  # abort
                print_status("Aborted. Use --on-conflict=overwrite or rename --storage.", "info")
                return 1

    print_status(f"Starting {args.n_trials} trials...", "info")
    t0 = time.perf_counter()
    try:
        results = search.optimize(show_progress=True)
    except Exception as e:
        logger.error("Optimization failed: %s", e, exc_info=True)
        print_exception("Optimization", e)
        return 1

    elapsed = time.perf_counter() - t0
    print_status(f"Done in {elapsed:.1f}s ({elapsed / args.n_trials:.2f}s/trial)", "success")

    # --- 5. Print results ---
    search.print_study_summary()
    search.print_top(n=args.top_n)

    # --- 6. Save CSV ---
    try:
        csv_path = search.save(Path(args.output_dir))
        print_status(f"Results saved → {csv_path}", "success")
    except Exception as e:
        logger.warning("Save failed: %s", e)

    # --- 7. Save best config ---
    if results:
        best = results[0]
        # Split optimized params: use plugin's declared risk_keys as the authority
        risk_keys = plugin.risk_keys
        best_strategy_params = {k: v for k, v in best.params.items() if k not in risk_keys}
        best_risk_params = {k: v for k, v in best.params.items() if k in risk_keys}
        best_raw = {
            **base_raw,
            "strategy": {**base_raw["strategy"], **best_strategy_params},
            "risk": {**base_raw.get("risk", {}), **best_risk_params},
        }

        current_time = time.strftime("%Y%m%d-%H%M%S")
        save_path = Path(args.output_dir) / f"{args.strategy}_best_{current_time}.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(best_raw, indent=2), encoding="utf-8")

        print_status(f"Best config saved → {save_path}", "success")
        print_kv_rows(
            {
                "Score": f"{best.score:.4f}",
                "Sharpe": f"{best.metrics.get('sharpe_ratio', 0):.3f}",
                "Return": f"{best.metrics.get('total_return_pct', 0):.2f}%",
                "Max DD": f"{best.metrics.get('max_drawdown_pct', 0):.2f}%",
                "Trades": best.metrics.get("total_trades", 0),
            },
            label_width=10,
        )

    return 0


def build_parser() -> argparse.ArgumentParser:
    strategies = list_strategy_names()

    parser = argparse.ArgumentParser(
        prog="run_optimize",
        description="Optimize strategy parameters using Optuna",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--strategy", "-s", default="orb", choices=strategies)
    parser.add_argument("--config", "-c", default=None, help="Base config JSON (overrides default)")
    parser.add_argument(
        "--param-space",
        default="full",
        help="Param space name. Use 'full' (all params) or 'core' (strategy only, faster). "
        "Available spaces depend on the strategy.",
    )

    # Data
    parser.add_argument("--symbol", default="VN30F1M")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2025-03-31")
    parser.add_argument("--freq", default=None, choices=["1min", "5min", "15min", "30min"])
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--force-refresh", action="store_true")

    # Capital / costs
    ExecutionConfig.add_args(parser)

    # Optimization
    parser.add_argument("--n-trials", type=int, default=300)
    parser.add_argument(
        "--sampler",
        default="tpe",
        choices=["tpe", "tpe_multivariate", "cmaes", "qmc"],
        help="tpe=default, tpe_multivariate=correlated params, cmaes=continuous only",
    )
    parser.add_argument(
        "--storage",
        default=None,
        help="SQLite path for crash-safe persistence. Run same command again to resume.",
    )
    parser.add_argument(
        "--on-conflict",
        default="prompt",
        choices=["prompt", "resume", "overwrite", "abort"],
        help=(
            "What to do when --storage has an existing study with a different param space. "
            "'prompt' asks interactively (default), 'resume' continues anyway, "
            "'overwrite' deletes the old study and starts fresh, 'abort' exits."
        ),
    )
    parser.add_argument("--min-trades", type=int, default=50)
    parser.add_argument("--min-return", type=float, default=-999.0)
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--output-dir", default="results/optimization")
    parser.add_argument("--top-n", type=int, default=5)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
