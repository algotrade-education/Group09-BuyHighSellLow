"""
Script to run Optuna-based Bayesian optimization for ORB strategy parameters.

This runner specifically targets the Opening Range Breakout (ORB) strategy. 
It optimizes both the strategy parameters and the data resampling timeframe 
simultaneously by using tick data as the base input.

Run as:
    python -m src.run_optimization --sample is --trials 300

Optimization Objectives:
    - Minimum Activity: total_trades > 500
    - Profitability: total_return_pct > 0 and profit_factor > 1.0
    - Composite Score (Maximized):
        score = sharpe - 0.1 * |max_drawdown_pct| + 0.1 * (total_trades / 1000)
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data.preprocessor import Preprocessor
from src.optimization.optuna_search import OptunaSearch
from src.strategy.ORB import OpeningRangeBreakout
from src.utils.cli_helpers import (
    load_orb_config_context,
    load_sample_data,
    prepare_optuna_dataset,
)
from src.utils.logger import setup_logging

logger = setup_logging(__name__, log_file="logs/optuna_orb.log")


def preprocess_data(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Trial-specific data preparation pipeline.

    Invoked by the Optuna objective function for every trial. This allows 
    the optimizer to explore different bar timeframes (e.g., 5m vs 15m) 
    rather than being locked to a single preprocessed file.

    Steps:
    1. Resample raw tick data to the trial's timeframe.
    2. Filter for VN30 trading session hours.
    3. Calculate all strategy indicators with the trial's specific periods.

    Args:
        df: Raw or cleaned tick data.
        params: Trial parameters sampled by Optuna (resample_freq, etc.).

    Returns:
        A ready-to-test DataFrame with OHLCV bars and indicators.
    """
    resample_freq = params.get("resample_freq", "1min")

    preprocessor = Preprocessor(
        atr_period=params.get("atr_period", 14),
    )

    # Resample tick data to OHLC bars
    df = preprocessor.resample_to_ohlc(df, freq=resample_freq)

    # Filter trading hours
    df = preprocessor.filter_trading_hours(df, include_atc=True)

    # Add all indicators
    df = preprocessor.add_all_indicators(df, copy=False)

    # Drop NaN from indicator warmup period
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def run_optuna(data: pd.DataFrame, config: dict, n_trials: int = 300) -> None:
    """
    Configure and execute the Optuna study.

    Defines the search space, initializes composite scoring parameters, 
    and triggers the TPE (Bayesian) search. After completion, it persists 
    the full results to CSV and the best configuration to JSON.
    """
    logger.info("Starting ORB Optuna Optimization with %d trials...", n_trials)

    # Search space for ORB parameters and optional risk controls.
    param_space = {
        "resample_freq": {
            "type": "categorical",
            "choices": ["5min", "15min", "1h"],
        },
        "orb_minutes": {"type": "int", "low": 15, "high": 60, "step": 5},
        "atr_period": {"type": "int", "low": 5, "high": 30, "step": 1},
        "atr_tp_multiplier": {"type": "float", "low": 1.5, "high": 6.0, "step": 0.1},
        "atr_sl_multiplier": {"type": "float", "low": 0.5, "high": 3.0, "step": 0.1},
        "breakout_buffer": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.05},
        "use_range_sl": {"type": "categorical", "choices": [True, False]},
        "min_range_atr": {"type": "float", "low": 0.3, "high": 2.0, "step": 0.1},
        "max_range_atr": {"type": "float", "low": 2.0, "high": 5.0, "step": 0.2},
        "max_trades_per_session": {"type": "int", "low": 1, "high": 3, "step": 1},
        "long_only": {"type": "categorical", "choices": [True, False]},
        "use_volume_filter": {"type": "categorical", "choices": [True, False]},
        "use_adx_filter": {"type": "categorical", "choices": [True, False]},
        "adx_min": {"type": "float", "low": 15.0, "high": 35.0, "step": 1.0},
        
        # Risk management
        "use_trailing_stop": {"type": "categorical", "choices": [True, False]},
        "trailing_atr_multiplier": {"type": "float", "low": 1.0, "high": 4.0, "step": 0.25},
    }

    # Extract risk params for backtester
    risk_params = config.get("risk", {})
    backtester_kwargs = {
        "max_daily_loss_pct": risk_params.get("max_daily_loss", 0.0),
    }

    # Initialize Optuna optimizer
    # Note: min_trades is enforced inside OptunaSearch via the composite score.
    optimizer = OptunaSearch(
        strategy_class=OpeningRangeBreakout,
        param_space=param_space,
        indicator_fn=preprocess_data,
        min_trades=500,          # ensure at least ~medium activity
        drawdown_penalty=0.1,    # penalize larger drawdowns
        turnover_penalty=0.0,    # do not penalize higher trade count
        trade_count_bonus=0.1,   # reward more trades among profitable configs
        min_return_pct=0.0,      # require non-negative total return
        min_profit_factor=1.0,   # require PF > 1.0
        n_trials=n_trials,
        backtester_kwargs=backtester_kwargs,
    )

    # Run optimization with raw_data=True since we pass tick data
    optimizer.optimize(data, raw_data=True)

    # Print results
    optimizer.print_study_summary()
    optimizer.print_top_results(10)

    # Save results
    output_path = optimizer.save_results()
    logger.info("Results saved to: %s", output_path)

    # Save best params as config
    if optimizer.best_params:
        params_dir = Path("config/strategy_params")
        params_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d")
        params_path = params_dir / f"orb_optuna_{timestamp}.json"

        best_config = {
            "name": f"ORB Optuna Optimized {timestamp}",
            "description": "Auto-optimized ORB parameters from Optuna TPE",
            "version": config.get("version", "1.0.0"),
            "strategy": config.get("strategy", {}).copy(),
            "risk": config.get("risk", {}).copy(),
        }

        # Update with optimized parameters (including resample_freq)
        # Separate risk params from strategy params
        risk_keys = {"use_trailing_stop", "trailing_atr_multiplier"}
        strategy_update = {k: v for k, v in optimizer.best_params.items() if k not in risk_keys}
        risk_update = {k: v for k, v in optimizer.best_params.items() if k in risk_keys}

        best_config["strategy"].update(strategy_update)
        best_config["risk"].update(risk_update)

        with open(params_path, "w") as f:
            json.dump(best_config, f, indent=2)

        logger.info("Best params saved to: %s", params_path)
        print(f"\nBest config saved to: {params_path}")
        print("Run backtest with:")
        print(
            f"  python -m src.run_backtest --sample is --config {params_path}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ORB Optuna optimization.")
    parser.add_argument(
        "--sample",
        choices=["is", "os"],
        default="is",
        help="Sample type: is (in-sample) or os (out-of-sample).",
    )
    parser.add_argument(
        "--contract", default="VN30F1M", help="Contract symbol (default: VN30F1M)."
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=300,
        help="Number of Optuna trials (default: 300).",
    )
    args = parser.parse_args()

    # Load configuration
    config, _, _ = load_orb_config_context("config/strategy_params/orb_default.json")
    logger.info("Loaded ORB configuration")

    # Load raw data (tick data)
    data = load_sample_data(sample=args.sample, contract=args.contract)
    data = prepare_optuna_dataset(data)
    logger.info("Cleaned tick data shape: %s", data.shape)

    run_optuna(data, config, n_trials=args.trials)
