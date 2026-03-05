"""
Script to run optimization (grid search) on strategy parameters.
Run as:
    python -m src.run_grid --sample is
to optimize on in-sample data, or
    python -m src.run_grid --sample os
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
from src.strategy.ORB import OpeningRangeBreakout
from src.utils.cli_helpers import (
    load_orb_config_context,
    load_sample_data,
    prepare_optimization_dataset,
)
from src.utils.logger import setup_logging

logger = setup_logging(__name__, log_file="logs/optimization.log")


def recalculate_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Recalculate indicators using parameters from each grid combination.
    This function will be called by the optimization process for each parameter set.
    """
    preprocessor = Preprocessor(
        adx_period=params.get("adx_period", 14),
        atr_period=params.get("atr_period", 14),
    )
    df = preprocessor.add_all_indicators(df, copy=False)
    df.dropna(inplace=True)
    return df


def run_optimization(data: pd.DataFrame, config: dict) -> None:
    """Run ORB grid search and persist ranked parameter results."""
    logger.info("Starting Grid Search Optimization...")

    # Parameter grid for ORB strategy.
    param_grid = {
        "orb_minutes": [10, 15, 20, 30],
        "atr_period": [10, 14, 20],
        "breakout_buffer": [0.0, 0.1, 0.2, 0.3],
        "min_range_atr": [0.3, 0.5, 0.8, 1.0],
        "max_range_atr": [2.5, 3.0, 4.0],
        "atr_sl_multiplier": [1.0, 1.5, 2.0, 3.0],
        "atr_tp_multiplier": [2.0, 3.0, 4.0],
        "use_range_sl": [True, False],
        "long_only": [True, False],
        "use_volume_filter": [False, True],
        "use_adx_filter": [False, True],
        "adx_min": [15.0, 20.0, 25.0, 30.0],
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
        strategy_class=OpeningRangeBreakout,
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
    config, _, resample_freq = load_orb_config_context(
        "config/strategy_params/orb_default.json",
        default_resample_freq="5min",
    )
    logger.info("Loaded ORB configuration")

    # Load data based on arguments
    data = load_sample_data(sample=args.sample, contract=args.contract)

    logger.info(
        "Preprocessing %s data for %s (resample: %s)...",
        args.sample,
        args.contract,
        resample_freq,
    )
    data = prepare_optimization_dataset(data, resample_freq=resample_freq)
    logger.info("Data shape for optimization: %s", data.shape)

    run_optimization(data, config)
