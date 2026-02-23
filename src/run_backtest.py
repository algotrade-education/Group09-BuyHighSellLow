from datetime import datetime
from pathlib import Path

import pandas as pd

from config.config import (
    COMMISSION_RATE,
    CONTRACT_MULTIPLIER,
    DEFAULT_INITIAL_CAPITAL,
    MARGIN_RATE,
    RESULTS_DIR,
)
from src.data.preprocessor import Preprocessor
from src.engine.backtester import Backtester
from src.engine.result import BacktestResult
from src.run_data_loader import load_data
from src.utils.config_loader import load_config
from src.utils.logger import setup_logging

logger = setup_logging(__name__, log_file="logs/backtest.log")


def run_backtest(
    data: pd.DataFrame,
    params: dict,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    commission_rate: float = COMMISSION_RATE,
    contract_multiplier: float = CONTRACT_MULTIPLIER,
    margin_rate: float = MARGIN_RATE,
) -> None:
    """Run backtest with given parameters."""

    # Fetch the parameters
    # YOUR CODE HERE

    # Initialize strategy
    # YOUR CODE HERE

    # Create backtester
    # backtester = Backtester(
    #     strategy=None,
    #     initial_capital=initial_capital,
    #     commission_rate=commission_rate,
    #     contract_multiplier=contract_multiplier,
    #     margin_rate=margin_rate,
    # )

    # Run backtest
    logger.info("Starting backtest execution...")

    # Print summary to console
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"\nStrategy: {strategy.name}")
    print(f"Parameters: {strategy.params}")

    print("\nPerformance Metrics:")
    for key, value in result.metrics.items():
        if isinstance(value, float):
            print(f"  {key:25}: {value:.4f}")
        else:
            print(f"  {key:25}: {value}")

    print(f"\nTotal Trades: {result.total_trades}")
    print(f"Win Rate:     {result.win_rate:.2f}%")
    print(f"Total P&L:    {result.total_pnl:.2f}")
    print("=" * 60)

    # Save results
    output_dir = Path(RESULTS_DIR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Subdirectory for this run
    run_dir = output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save equity curve
    equity_path = run_dir / "equity_curve.csv"
    result.equity_curve.to_csv(equity_path, index=False)
    logger.info("Equity curve saved to: %s", equity_path)

    # Save trades
    if result.trades:
        trades_path = run_dir / "trades.csv"
        result.to_dataframe().to_csv(trades_path, index=False)
        logger.info("Trades saved to: %s", trades_path)

    # --- PLOTTING ---
    logger.info("Generating plots...")
    plotter = BacktestPlotter(output_dir=run_dir)

    # 1. Equity Curve
    plotter.plot_equity_curve(
        result.equity_curve,
        initial_capital=initial_capital,
        filename="equity_curve.png",
    )

    # 2. Backtest Results (Price, Indicators, Signals)
    # Reconstruct data with indicators directly from data passed to run_backtest
    # The 'data' passed to this function already has the indicators from Preprocessor
    plotter.plot_backtest_results(
        data=data, trades=result.trades, filename="backtest_results.png"
    )

    # 3. Trade Analysis
    plotter.plot_trade_analysis(trades=result.trades, filename="trade_analysis.png")

    # 4. Exit Reasons
    plotter.plot_exit_reasons(trades=result.trades, filename="exit_reasons.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run strategy backtest.")
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
        "--config",
        default="config/strategy_params/default.json",
        help="Path to strategy configuration file (default: config/strategy_params/default.json).",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    logger.info("Loaded configuration from %s", args.config)
    # Load data based on arguments
    data = load_data(sample=args.sample, contract=args.contract)

    # Get resample frequency from config, default to 1min if not specified
    resample_freq = config.get("strategy", {}).get("resample_freq", "1min")
    logger.info(
        "Resampling %s data for %s to %s...", args.sample, args.contract, resample_freq
    )

    # Get strategy parameters to initialize preprocessor with matching parameters
    strategy_params = config.get("strategy", {})
    sma_period = strategy_params.get("sma_period", 20)
    bb_std = strategy_params.get("bb_std", 2.0)
    slope_lookback = strategy_params.get("slope_lookback", 1)

    # Create preprocessor with matching parameters
    preprocessor = Preprocessor(
        sma_period=sma_period,
        bb_std=bb_std,
        slope_lookback=slope_lookback,
    )

    # Resample to OHLC bars and add indicators
    data = preprocessor.prepare_for_backtest(data, resample_freq=resample_freq)

    logger.info(
        "Data prepared: %s bars with indicators for SMA(%s)", len(data), sma_period
    )

    run_backtest(data, config)
