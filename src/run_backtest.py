"""
Script to run backtest for ORB (Opening Range Breakout) strategy.
Run as:
    python -m src.run_backtest_orb --sample is --config config/strategy_params/orb_default.json

This script will:
1. Load the specified data (in-sample or out-of-sample).
2. Preprocess the data (resample, add ATR indicator).
3. Initialize the ORB strategy with parameters from the config file.
4. Run the backtest and collect results.
5. Save detailed results (equity curve, trades) to the results directory.
"""

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
from src.engine.position_sizer import PercentRiskSizer
from src.engine.result import BacktestResult
from src.metrics.plotter import BacktestPlotter
from src.run_data_loader import load_data
from src.strategy.ORB import OpeningRangeBreakout
from src.utils.config_loader import load_config
from src.utils.logger import setup_logging

logger = setup_logging(__name__, log_file="logs/backtest_orb.log")


def run_backtest(
    data: pd.DataFrame,
    params: dict,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    commission_rate: float = COMMISSION_RATE,
    contract_multiplier: float = CONTRACT_MULTIPLIER,
    margin_rate: float = MARGIN_RATE,
) -> None:
    """Run backtest with given parameters."""

    # Extract strategy parameters from config
    strategy_params = params.get("strategy", {})

    # Remove non-strategy keys that the constructor doesn't accept
    strategy_kwargs = {
        k: v for k, v in strategy_params.items() if k not in ("resample_freq",)
    }

    # Initialize strategy
    strategy = OpeningRangeBreakout(**strategy_kwargs)

    # Extract risk management params
    risk_params = params.get("risk", {})

    # Build position sizer if configured
    position_sizer = None
    if risk_params.get("risk_per_trade_pct"):
        position_sizer = PercentRiskSizer(
            risk_per_trade_pct=risk_params["risk_per_trade_pct"],
            min_size=risk_params.get("min_position_size", 1),
            max_size=risk_params.get("max_position_size", 10),
        )
        logger.info(
            "Using PercentRiskSizer: %.1f%% risk per trade",
            risk_params["risk_per_trade_pct"],
        )

    # Create backtester
    backtester = Backtester(
        strategy=strategy,
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        contract_multiplier=contract_multiplier,
        margin_rate=margin_rate,
        position_sizer=position_sizer,
        use_trailing_stop=risk_params.get("use_trailing_stop", False),
        trailing_atr_multiplier=risk_params.get("trailing_atr_multiplier", 2.0),
        max_daily_loss_pct=risk_params.get("max_daily_loss", 0.0),
    )

    # Run backtest
    logger.info("Starting backtest execution...")
    result: BacktestResult = backtester.run(data)

    # Print summary to console
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS — ORB Strategy")
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
    run_dir = output_dir / f"orb_{timestamp}"
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
    plotter.plot_backtest_results(
        data=data, trades=result.trades, filename="backtest_results.png"
    )

    # 3. Trade Analysis
    plotter.plot_trade_analysis(trades=result.trades, filename="trade_analysis.png")

    # 4. Exit Reasons
    plotter.plot_exit_reasons(trades=result.trades, filename="exit_reasons.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ORB strategy backtest.")
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
        default="config/strategy_params/orb_default.json",
        help="Path to strategy configuration file.",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    logger.info("Loaded configuration from %s", args.config)
    # Load data based on arguments
    data = load_data(sample=args.sample, contract=args.contract)

    # Get resample frequency from config — ORB typically uses 1min
    resample_freq = config.get("strategy", {}).get("resample_freq", "1min")
    logger.info(
        "Resampling %s data for %s to %s...", args.sample, args.contract, resample_freq
    )

    # Get strategy parameters to initialize preprocessor with matching parameters
    strategy_params = config.get("strategy", {})

    # Create preprocessor — ORB only needs ATR, but we include all indicators
    preprocessor = Preprocessor(
        atr_period=strategy_params.get("atr_period", 14),
    )

    # Resample to OHLC bars and add indicators
    data = preprocessor.prepare_for_backtest(data, resample_freq=resample_freq)

    run_backtest(data, config)
