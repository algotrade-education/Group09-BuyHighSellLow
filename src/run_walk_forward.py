import pandas as pd

from src.data.preprocessor import Preprocessor
from src.optimization.walk_forward import WalkForwardOptimizer
from src.run_data_loader import load_data
from src.utils.config_loader import load_config
from src.utils.logger import setup_logging

logger = setup_logging(__name__, log_file="logs/walk_forward.log")


def recalculate_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Recalculate indicators using parameters from each grid combination.
    This function will be called by the optimization process for each parameter set.

    Optional: You can implement specific indicator recalculations based on the parameters being optimized.
    For example, if optimizing SMA periods, you would recalculate the SMA indicators here.
    """
    preprocessor = Preprocessor()
    df = preprocessor.add_all_indicators(df, copy=False)
    df.dropna(inplace=True)
    return df


def run_walk_forward(data: pd.DataFrame, config: dict) -> None:
    """Run walk-forward optimization."""
    logger.info("Starting Walk-Forward Optimization...")

    # Use ranges around default/config values if possible
    # For now keep the hardcoded grid
    param_grid = {}

    wf_optimizer = WalkForwardOptimizer(
        strategy_class=None,
        param_grid=param_grid,
        n_windows=5,
        train_pct=0.7,
        anchored=True,
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

    resample_freq = config.get("strategy", {}).get("resample_freq", "1min")
    logger.info(
        "Preprocessing %s data for %s (resample: %s)...",
        args.sample,
        args.contract,
        resample_freq,
    )
    data = Preprocessor().prepare_for_optimization(data, resample_freq=resample_freq)
    logger.info("Data shape for optimization: %s", data.shape)

    run_walk_forward(data, config)
