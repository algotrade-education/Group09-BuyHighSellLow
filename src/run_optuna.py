"""
Script to run Optuna-based Bayesian optimization on strategy parameters.
Run as:
    python src/run_optuna.py --sample is --trials 200

This uses Optuna's TPE sampler to intelligently explore the parameter space
instead of exhaustively testing every combination (grid search).

Composite objective:
    score = sharpe - |0.1 * max_drawdown| - |0.1 * trades/1000|
    If trades <= 50: score = -1.0 (invalid)
    If sharpe <= 0: use total_return / 100 as fallback

Also optimizes the resampling timeframe (1min, 5min, 15min, 1h) alongside
strategy parameters.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data.preprocessor import Preprocessor
from src.optimization.optuna_search import OptunaSearch
from src.run_data_loader import load_data
from src.strategy.BB import BollingerMeanReversion
from src.utils.config_loader import load_config
from src.utils.logger import setup_logging

logger = setup_logging(__name__, log_file="logs/optuna.log")


def preprocess_data(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Full preprocessing pipeline per Optuna trial:
    1. Resample tick data to OHLC bars at the trial's timeframe
    2. Filter trading hours
    3. Add all indicators with the trial's parameters

    This function receives raw (cleaned) tick data and returns
    a fully preprocessed DataFrame ready for backtesting.

    Args:
        df: Cleaned tick data with datetime, price, volume columns.
        params: Trial parameters including resample_freq and indicator params.

    Returns:
        OHLC DataFrame with all indicators calculated.
    """
    resample_freq = params.get("resample_freq", "5min")

    preprocessor = Preprocessor(
        sma_period=params.get("bb_period", 20),
        bb_std=params.get("bb_std", 2.0),
        rsi_period=params.get("rsi_period", 14),
        adx_period=params.get("adx_period", 14),
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


def run_optuna(data: pd.DataFrame, config: dict, n_trials: int = 200) -> None:
    """Run Optuna optimization."""
    logger.info("Starting Optuna Optimization with %d trials...", n_trials)

    # Define parameter search space
    # Includes resample_freq for timeframe optimization
    param_space = {
        "resample_freq": {
            "type": "categorical",
            "choices": ["1min", "5min", "15min", "1h"],
        },
        "bb_period": {"type": "int", "low": 10, "high": 30, "step": 1},
        "bb_std": {"type": "float", "low": 1.5, "high": 5.0, "step": 0.1},
        "adx_threshold": {"type": "int", "low": 20, "high": 40, "step": 1},
        "rsi_oversold": {"type": "int", "low": 10, "high": 40, "step": 1},
        "rsi_overbought": {"type": "int", "low": 55, "high": 75, "step": 1},
        "atr_sl_multiplier": {"type": "float", "low": 0.5, "high": 3.0, "step": 0.1},
    }

    # Initialize Optuna optimizer
    optimizer = OptunaSearch(
        strategy_class=BollingerMeanReversion,
        param_space=param_space,
        indicator_fn=preprocess_data,  # Handles resampling + indicators
        min_trades=100,
        drawdown_penalty=0.1,
        turnover_penalty=0.1,
        n_trials=n_trials,
    )

    # Run optimization with raw_data=True since we pass tick data
    # and preprocess_data handles resampling per trial
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
        params_path = params_dir / f"optuna_{timestamp}.json"

        best_config = {
            "name": f"Optuna Optimized {timestamp}",
            "description": "Auto-optimized parameters from Optuna TPE",
            "version": config.get("version", "1.0.0"),
            "strategy": config.get("strategy", {}).copy(),
            "risk": config.get("risk", {}).copy(),
            "trading_hours": config.get("trading_hours", {}).copy(),
        }

        # Update with optimized parameters (including resample_freq)
        best_config["strategy"].update(optimizer.best_params)

        with open(params_path, "w") as f:
            json.dump(best_config, f, indent=2)

        logger.info("Best params saved to: %s", params_path)
        print(f"\nBest config saved to: {params_path}")
        print("Run backtest with:")
        print(f"  python src/run_backtest.py --sample is --config {params_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Optuna optimization.")
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
        default=200,
        help="Number of Optuna trials (default: 200).",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config()
    logger.info("Loaded configuration from default path")

    # Load raw data (tick data)
    data = load_data(sample=args.sample, contract=args.contract)

    # Only clean and derive volume — do NOT resample here
    # Each Optuna trial will resample to its own timeframe
    preprocessor = Preprocessor()
    data = preprocessor.clean_data(data)
    data = preprocessor._derive_volume(data, copy=False)
    logger.info("Cleaned tick data shape: %s", data.shape)

    run_optuna(data, config, n_trials=args.trials)
