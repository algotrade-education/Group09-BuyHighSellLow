import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data.preprocessor import Preprocessor
from src.optimization.grid_search import GridSearch
from src.run_data_loader import load_data
from src.utils.config_loader import load_config
from src.utils.logger import setup_logging

logger = setup_logging(__name__, log_file="logs/optimization.log")


def run_optimization(data: pd.DataFrame, config: dict) -> None:
    """Run grid search optimization."""
    logger.info("Starting Grid Search Optimization...")

    # Function to recalculate indicators based on params
    def recalculate_indicators(data: pd.DataFrame, params: dict) -> pd.DataFrame:
        """Recalculate indicators using params from each grid search combination."""
        preprocessor = Preprocessor(
            sma_period=params.get("sma_period", 20),
            bb_std=params.get("bb_std", 2.0),
            slope_lookback=params.get("slope_lookback", 1),
        )
        return preprocessor.add_all_indicators(data, copy=False)

    # Define parameter grid for optimization
    param_grid = {
        # For example, optimizing SMA period and Bollinger Bands std deviation
        # "sma_period": [10, 20, 50],
    }

    # Initialize GridSearch
    grid_search = GridSearch(
        strategy_class=None,
        param_grid=param_grid,
        objective="sharpe_ratio",
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

    resample_freq = config.get("strategy", {}).get("resample_freq", "1min")
    logger.info(
        "Preprocessing %s data for %s (resample: %s)...",
        args.sample,
        args.contract,
        resample_freq,
    )
    data = Preprocessor().prepare_for_optimization(data, resample_freq=resample_freq)
    logger.info("Data shape for optimization: %s", data.shape)

    run_optimization(data, config)
