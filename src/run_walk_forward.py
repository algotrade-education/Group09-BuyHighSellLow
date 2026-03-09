"""
Script to run walk-forward optimization for the ORB strategy.

Walk-forward optimization is a robust validation technique that splits
historical data into moving windows. Parameters are optimized on a 'Train'
period and then tested on an immediately following 'Test' (Out-Of-Sample)
period. This process detects if a strategy's edge is persistent or just a
result of curve-fitting.

Run as:
    python -m src.run_walk_forward --sample is

Safeguards:
1. Anchored splitting to ensure training data grows over time.
2. Direct comparison of In-Sample vs Out-of-Sample Sharpe ratios.
3. Calculation of a Robustness Ratio (Test / Train performance).
"""

import pandas as pd

from src.data.preprocessor import Preprocessor
from src.optimization.walk_forward import WalkForwardOptimizer
from src.run_data_loader import load_data
from src.strategy.ORB import OpeningRangeBreakout
from src.utils.config_loader import load_config
from src.utils.logger import setup_logging

logger = setup_logging(__name__, log_file="logs/walk_forward.log")


def recalculate_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Indicator calculation hook for the walk-forward cycle.

    Invoked by the GridSearch optimizer inside the walk-forward loop.
    Resets indicators (ATR, ADX) using the specific periods chosen
    for the current trial.
    """
    preprocessor = Preprocessor(
        adx_period=params.get("adx_period", 14),
        atr_period=params.get("atr_period", 14),
    )
    df = preprocessor.add_all_indicators(df, copy=False)
    df.dropna(inplace=True)
    return df


def run_walk_forward(data: pd.DataFrame, config: dict) -> None:
    """
    Initialize and run the Walk-Forward Optimizer.

    Defines the parameter grid to search, partitions the data into
    N windows, and performs the train-test cycles. Outputs a detailed
    summary of performance degradation and final robustness.
    """
    logger.info("Starting Walk-Forward Optimization...")

    # ORB parameter grid for walk-forward optimization
    param_grid = {
        "orb_minutes": [10, 15, 20, 30],
        "atr_period": [10, 14, 20],
        "breakout_buffer": [0.0, 0.1, 0.2],
        "min_range_atr": [0.3, 0.5, 0.8],
        "max_range_atr": [2.5, 3.0, 4.0],
        "atr_sl_multiplier": [1.0, 1.5, 2.0],
        "atr_tp_multiplier": [2.0, 3.0, 4.0],
        "use_range_sl": [True, False],
        "long_only": [True, False],
        "use_volume_filter": [False, True],
        "use_adx_filter": [False, True],
        "adx_min": [15.0, 20.0, 25.0],
    }

    wf_optimizer = WalkForwardOptimizer(
        strategy_class=OpeningRangeBreakout,
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
    config = load_config("config/strategy_params/orb_default.json")
    logger.info("Loaded ORB configuration")

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
