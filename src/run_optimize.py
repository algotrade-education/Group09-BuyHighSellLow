"""
src/run_optimize.py

Run Optuna optimization for a strategy over a date range.

Full pipeline:
    Load data once → for each trial: build strategy → run backtest → score

Usage:
    # Basic: 200 trials with default ORB param space
    python -m src.run_optimize --strategy orb --start 2023-01-01 --end 2024-12-31

    # More trials, save study to SQLite (crash-safe, resumable)
    python -m src.run_optimize --strategy orb --n-trials 500 --storage results/orb_study.db

    # Resume interrupted study (just run the same command again)
    python -m src.run_optimize --strategy orb --n-trials 500 --storage results/orb_study.db

    # Use multivariate TPE (better when params interact)
    python -m src.run_optimize --strategy orb --sampler tpe_multivariate

    # Save best config to file
    python -m src.run_optimize --strategy orb --save-best config/strategy_params/orb_optimized.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from config.constants import ExecutionConfig
from config.schemas.orb import ORBConfig
from src.data.loader import DataLoader
from src.data.pipeline import DataPipeline
from src.data.preprocessor import DataPreprocessor
from src.database.data_service import get_data_service
from src.engine.account.sizer import PercentRiskSizer
from src.engine.backtester import Backtester
from src.engine.execution.slippage import FixedSlippage
from src.optimization.optuna_search import OptunaSearch
from src.optimization.scoring import ScorerConfig
from src.strategy.orb import ORBStrategy
from src.utils.cli_helpers import print_exception, print_kv_rows, print_section, print_status
from src.utils.logger import setup_logging

logger = setup_logging(
    name="run_optimize",
    log_file="logs/optimize.log",
    capture_all_loggers=False,
)

# --- Default param spaces per strategy ---
# These are sensible ranges based on the default config values.
# Adjust low/high to narrow or widen the search.

_ORB_PARAM_SPACE: dict[str, dict[str, Any]] = {
    "orb_minutes": {"type": "int", "low": 10, "high": 45, "step": 5},
    "atr_period": {"type": "int", "low": 10, "high": 20},
    "atr_tp_multiplier": {"type": "float", "low": 1.0, "high": 4.0, "step": 0.5},
    "breakout_buffer": {"type": "float", "low": 0.0, "high": 1.0, "step": 0.1},
    "min_range_atr": {"type": "float", "low": 0.3, "high": 1.5, "step": 0.1},
    "max_range_atr": {"type": "float", "low": 2.0, "high": 5.0, "step": 0.5},
}

_PARAM_SPACES: dict[str, dict[str, dict[str, Any]]] = {
    "orb": _ORB_PARAM_SPACE,
}


# --- Trial function builder ---


def _build_orb_trial_fn(
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
    """
    Build a trial_fn for ORB optimization.

    The returned function accepts a params dict and returns a BacktestResult.
    Data is loaded once outside and reused across all trials.
    """
    # Load base config once - trials only override strategy params
    base_raw = json.loads(Path(base_config_path).read_text(encoding="utf-8"))
    freq_minutes = int(freq.replace("min", ""))

    def trial_fn(params: dict[str, Any]) -> Any:
        # Merge trial params into base config
        trial_raw = {
            **base_raw,
            "strategy": {**base_raw["strategy"], **params},
        }

        try:
            config = ORBConfig.from_dict(trial_raw)
        except Exception:
            # Pydantic validation failed (e.g. min_range_atr >= max_range_atr)
            # Return a dummy result with 0 trades so scorer returns INVALID_SCORE
            return _invalid_result()

        strategy = ORBStrategy(config)
        registry = ORBStrategy.build_registry(
            atr_period=config.strategy.atr_period,
            adx_period=config.strategy.adx_period,
            volume_ma_period=config.strategy.volume_ma_period,
        )

        pipeline = DataPipeline(registry, cache_dir=cache_dir, use_cache=True)
        data = pipeline.run(preprocessed_data)
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
        return bt.run(data, warmup_bars=warmup)

    return trial_fn


def _invalid_result() -> Any:
    """Dummy result with 0 trades - scorer will return INVALID_SCORE."""
    return type("R", (), {"metrics": {"total_trades": 0}})()


# --- Main run ---


def run(args: argparse.Namespace) -> int:
    # --- 1. Resolve config ---
    config_path = args.config or f"config/strategy_params/{args.strategy}_default.json"
    if not Path(config_path).exists():
        print_status(f"Config not found: {config_path}", "error")
        return 1

    # Load config to get freq
    base_raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    freq = args.freq or base_raw.get("strategy", {}).get("resample_freq", "5min")

    print_section(f"OPTIMIZE: {args.strategy.upper()}", width=60)
    print_kv_rows(
        {
            "Period": f"{args.start} → {args.end}",
            "Trials": args.n_trials,
            "Sampler": args.sampler,
            "Freq": freq,
            "Capital": f"{args.capital:,.0f}",
        },
        label_width=10,
    )
    print()

    # --- 2. Load + preprocess data once ---
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
    if args.strategy not in _PARAM_SPACES:
        print_status(f"No param space defined for strategy: {args.strategy!r}", "error")
        return 1

    param_space = _PARAM_SPACES[args.strategy]

    if args.strategy == "orb":
        trial_fn = _build_orb_trial_fn(
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
    else:
        print_status(f"Strategy {args.strategy!r} not yet supported for optimization.", "error")
        return 1

    # --- 4. Run optimization ---
    scorer = ScorerConfig(
        min_trades=args.min_trades,
        min_return_pct=args.min_return,
        drawdown_penalty=0.5,
    )

    search = OptunaSearch(
        trial_fn=trial_fn,
        param_space=param_space,
        scorer=scorer,
        n_trials=args.n_trials,
        sampler=args.sampler,
        study_name=f"{args.strategy}_opt",
        storage_path=args.storage,
    )

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
    out_dir = Path(args.output_dir)
    try:
        csv_path = search.save(out_dir)
        print_status(f"Results saved → {csv_path}", "success")
    except Exception as e:
        logger.warning("Save failed: %s", e)

    # --- 7. Save best config ---
    if args.save_best and results:
        best = results[0]
        best_raw = {
            **base_raw,
            "strategy": {**base_raw["strategy"], **best.params},
        }
        save_path = Path(args.save_best)
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


# --- CLI ---


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_optimize",
        description="Optimize strategy parameters using Optuna",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--strategy", "-s", default="orb", choices=list(_PARAM_SPACES))
    parser.add_argument("--config", "-c", default=None, help="Base config JSON (overrides default)")

    # Data
    parser.add_argument("--symbol", default="VN30F1M")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--freq", default=None, choices=["1min", "5min", "15min", "30min"])
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--force-refresh", action="store_true")

    # Capital / costs - grouped via ExecutionConfig
    ExecutionConfig.add_args(parser)

    # Optimization
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument(
        "--sampler",
        default="tpe",
        choices=["tpe", "tpe_multivariate", "cmaes", "qmc"],
        help="tpe=default, tpe_multivariate=correlated params, cmaes=continuous only",
    )
    parser.add_argument(
        "--storage",
        default=None,
        help="SQLite path for crash-safe persistence, e.g. results/orb_study.db. "
        "Run same command again to resume.",
    )
    parser.add_argument("--min-trades", type=int, default=30, help="Min trades gate for scorer")
    parser.add_argument("--min-return", type=float, default=-999.0, help="Min return%% gate")

    # Output
    parser.add_argument("--output-dir", default="results/optimization")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument(
        "--save-best",
        default=None,
        metavar="PATH",
        help="Save best params as JSON config, e.g. config/strategy_params/orb_optimized.json",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
