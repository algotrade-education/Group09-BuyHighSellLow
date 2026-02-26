"""
Script to run Optuna-based Bayesian optimization for ORB strategy parameters.
Run as:
    python -m src.run_optimization_orb --sample is --trials 300

Composite objective:
    score = sharpe - |0.1 * max_drawdown| - |0.1 * trades/1000|
    If trades <= 50: score = -1.0 (invalid)
    If sharpe <= 0: use total_return / 100 as fallback

Also optimizes the resampling timeframe (1min, 5min) alongside strategy parameters.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data.preprocessor import Preprocessor
from src.optimization.optuna_search import OptunaSearch
from src.run_data_loader import load_data
from src.strategy.ORB import OpeningRangeBreakout
from src.utils.config_loader import load_config
from src.utils.logger import setup_logging

logger = setup_logging(__name__, log_file="logs/optuna_orb.log")


def preprocess_data(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Full preprocessing pipeline per Optuna trial:
    1. Resample tick data to OHLC bars at the trial's timeframe
    2. Filter trading hours
    3. Add all indicators with the trial's parameters

    Args:
        df: Cleaned tick data with datetime, price, volume columns.
        params: Trial parameters including resample_freq and indicator params.

    Returns:
        OHLC DataFrame with all indicators calculated.
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
    """Run Optuna optimization for ORB strategy."""
    logger.info("Starting ORB Optuna Optimization with %d trials...", n_trials)

    # Focused search space — narrowed around known-good ORB params
    # Fixed: 5min, long_only=True (proven best from previous runs)
    param_space = {
        "resample_freq": {
            "type": "categorical",
            "choices": ["5min", "15min", "1h"],
        },
        "orb_minutes": {"type": "int", "low": 15, "high": 60, "step": 5},
        "atr_period": {"type": "int", "low": 5, "high": 30, "step": 1},
        "atr_tp_multiplier": {"type": "float", "low": 1.5, "high": 5.0, "step": 0.1},
        "atr_sl_multiplier": {"type": "float", "low": 1.0, "high": 3.0, "step": 0.1},
        "breakout_buffer": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.05},
        "use_range_sl": {"type": "categorical", "choices": [True, False]},
        "min_range_atr": {"type": "float", "low": 0.3, "high": 1.5, "step": 0.1},
        "max_range_atr": {"type": "float", "low": 2.0, "high": 5.0, "step": 0.5},
        "long_only": {"type": "categorical", "choices": [True, False]},
        "use_volume_filter": {"type": "categorical", "choices": [True, False]},
        "use_adx_filter": {"type": "categorical", "choices": [True, False]},
        "adx_min": {"type": "float", "low": 15.0, "high": 35.0, "step": 1.0},
        
        # Risk management
        "use_trailing_stop": {"type": "categorical", "choices": [True, False]},
        "trailing_atr_multiplier": {"type": "float", "low": 1.0, "high": 4.0, "step": 0.5},
    }

    # Extract risk params for backtester
    risk_params = config.get("risk", {})
    backtester_kwargs = {
        "max_daily_loss_pct": risk_params.get("max_daily_loss", 0.0),
    }

    # Initialize Optuna optimizer
    optimizer = OptunaSearch(
        strategy_class=OpeningRangeBreakout,
        param_space=param_space,
        indicator_fn=preprocess_data,
        min_trades=100,
        drawdown_penalty=0.1,
        turnover_penalty=0.1,
        n_trials=n_trials,
        backtester_kwargs=backtester_kwargs,
    )

    # Run optimization with raw_data=True since we pass tick data
    results = optimizer.optimize(data, raw_data=True)

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
            f"  python -m src.run_backtest_orb --sample is --config {params_path}"
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
    config = load_config("config/strategy_params/orb_default.json")
    logger.info("Loaded ORB configuration")

    # Load raw data (tick data)
    data = load_data(sample=args.sample, contract=args.contract)

    # Only clean and derive volume — do NOT resample here
    # Each Optuna trial will resample to its own timeframe
    preprocessor = Preprocessor()
    data = preprocessor.clean_data(data)
    data = preprocessor._derive_volume(data, copy=False)
    logger.info("Cleaned tick data shape: %s", data.shape)

    run_optuna(data, config, n_trials=args.trials)
