"""
Script to run walk-forward optimization.
Run as:
    python src/run_walk_forward.py --sample is
to run on in-sample data.

Walk-forward optimization guards against overfitting by:
1. Splitting data into overlapping train/test windows.
2. Optimizing parameters on each train window.
3. Testing best parameters on the unseen test window.
4. Checking if in-sample performance holds out-of-sample (robustness ratio).
"""

import pandas as pd

from src.data.preprocessor import Preprocessor
from src.optimization.walk_forward import WalkForwardOptimizer
from src.run_data_loader import load_data
from src.strategy.BB import BollingerMeanReversion
from src.utils.config_loader import load_config
from src.utils.logger import setup_logging

logger = setup_logging(__name__, log_file="logs/walk_forward.log")


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


def run_walk_forward(data: pd.DataFrame, config: dict) -> None:
    """Run walk-forward optimization."""
    logger.info("Starting Walk-Forward Optimization...")

    # Same param grid as grid search optimization
    param_grid = {
        "bb_std": [1.5, 2.0, 2.5],
        "adx_threshold": [25, 30, 35],
        "rsi_oversold": [30, 35, 40],
        "rsi_overbought": [60, 65, 70],
        "atr_sl_multiplier": [1.0, 1.5, 2.0],
    }

    wf_optimizer = WalkForwardOptimizer(
        strategy_class=BollingerMeanReversion,
        param_grid=param_grid,
        n_windows=5,
        train_pct=0.7,
        anchored=True,
        objective="profit_factor",
        indicator_fn=recalculate_indicators,
    )

    result = wf_optimizer.optimize(data)

    print("\n" + "=" * 60)
    print("WALK-FORWARD SUMMARY")
    print("=" * 60)
    wf_optimizer.print_summary()
    print("=" * 60)

    # Save results
    output_path = wf_optimizer.save_results()
    logger.info("Results saved to: %s", output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run walk-forward optimization.")
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

    run_walk_forward(data, config)
