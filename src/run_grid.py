"""
Script to run optimization (grid search) on strategy parameters.
Run as:
    python src/run_grid.py --sample is
to optimize on in-sample data, or
    python src/run_grid.py --sample os
to optimize on out-of-sample data.

This script will:
1. Load the specified data (in-sample or out-of-sample).
2. Preprocess the data (resample, add indicators).
3. Run grid search optimization over specified parameter ranges.
4. Print top results to console and save detailed results to the results directory.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data.preprocessor import Preprocessor
from src.optimization.grid_search import GridSearch
from src.run_data_loader import load_data
from src.strategy.BB import BollingerMeanReversion
from src.utils.config_loader import load_config
from src.utils.logger import setup_logging

logger = setup_logging(__name__, log_file="logs/optimization.log")


def recalculate_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Recalculate indicators using parameters from each grid combination.
    This function will be called by the optimization process for each parameter set.
    """
    preprocessor = Preprocessor(
        sma_period=params.get("bb_period", 20),
        bb_std=params.get("bb_std", 2.0),
        rsi_period=params.get("rsi_period", 14),
        adx_period=params.get("adx_period", 14),
        atr_period=params.get("atr_period", 14),
    )
    df = preprocessor.add_all_indicators(df, copy=False)
    df.dropna(inplace=True)
    return df


def run_optimization(data: pd.DataFrame, config: dict) -> None:
    """Run grid search optimization."""
    logger.info("Starting Grid Search Optimization...")

    # Parameter grid: test a range around defaults for the most impactful params
    # Keeping bb_period, rsi_period, adx_period, atr_period fixed to reduce
    # search space (changing periods requires indicator recalculation anyway).
    param_grid = {
        "bb_period": [20],
        "bb_std": [2.0, 2.5, 3.0, 5.0],
        "adx_threshold": [25, 30, 35],
        "rsi_oversold": [30, 35, 40],
        "rsi_overbought": [60, 65, 70],
        "atr_sl_multiplier": [1.0, 1.5, 2.0, 3.0],
        "atr_tp_multiplier": [2.0, 3.0, 4.0, 5.0],
    }

    total = 1
    for v in param_grid.values():
        total *= len(v)
    logger.info("Total combinations: %d", total)

    # Initialize GridSearch
    # Using profit_factor as objective: it balances win rate AND magnitude
    # (gross_profit / gross_loss). Unlike Sharpe alone, it penalizes strategies
    # that win often but lose big, or win big but lose often.
    grid_search = GridSearch(
        strategy_class=BollingerMeanReversion,
        param_grid=param_grid,
        objective="profit_factor",
        indicator_fn=recalculate_indicators,
    )

    results = grid_search.optimize(data)

    print("\n" + "=" * 60)
    print("OPTIMIZATION TOP RESULTS")
    print("=" * 60)
    grid_search.print_top_results(10)
    print("=" * 60)

    # Save results
    output_path = grid_search.save_results()
    logger.info("Results saved to: %s", output_path)

    # Save best params
    if grid_search.best_params:
        params_dir = Path("config/strategy_params")
        params_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d")
        params_path = params_dir / f"optimized_{timestamp}.json"

        # Start with the full default config structure
        best_config = {
            "name": f"Optimized {timestamp}",
            "description": "Auto-optimized parameters from grid search",
            "version": config.get("version", "1.0.0"),
            "strategy": config.get("strategy", {}).copy(),  # Copy all strategy params
            "risk": config.get("risk", {}).copy(),  # Copy all risk params
            "trading_hours": config.get(
                "trading_hours", {}
            ).copy(),  # Copy trading hours
        }

        # Update only the optimized parameters
        best_config["strategy"].update(grid_search.best_params)

        with open(params_path, "w") as f:
            json.dump(best_config, f, indent=2)

        logger.info("Best params saved to: %s", params_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run optimization.")
    parser.add_argument(
        "--sample",
        choices=["is", "os"],
        default="is",
        help="Sample type: is (in-sample) or os (out-of-sample).",
    )
    parser.add_argument(
        "--contract", default="VN30F1M", help="Contract symbol (default: VN30F1M)."
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config()
    logger.info("Loaded configuration from default path")

    # Load data based on arguments
    data = load_data(sample=args.sample, contract=args.contract)

    resample_freq = config.get("strategy", {}).get("resample_freq", "5min")
    logger.info(
        "Preprocessing %s data for %s (resample: %s)...",
        args.sample,
        args.contract,
        resample_freq,
    )
    data = Preprocessor().prepare_for_optimization(data, resample_freq=resample_freq)
    logger.info("Data shape for optimization: %s", data.shape)

    run_optimization(data, config)
