"""
src/run_backtest.py

Run a single backtest for a given strategy and date range.

Full pipeline:
    DataLoader -> DataPreprocessor -> DataPipeline -> Backtester -> Metrics -> Plotter

Usage:
    # Basic run with default config
    python -m src.run_backtest --strategy orb --start 2023-01-01 --end 2024-12-31

    # Custom config file
    python -m src.run_backtest --strategy orb --config config/strategy_params/orb_aggressive.json

    # Interactive HTML plots instead of PNG
    python -m src.run_backtest --strategy orb --plot-html

    # Skip plotting (metrics only)
    python -m src.run_backtest --strategy orb --no-plot

    # Print trade details to console
    python -m src.run_backtest --strategy orb --show-trades

    # Compare Approach 2 (loop) vs Approach 1 (event-driven) - validate both engines
    python -m src.run_backtest --strategy orb --compare-engines

    # Force fresh data from DB (bypass cache)
    python -m src.run_backtest --strategy orb --force-refresh
"""

from __future__ import annotations

import argparse
import datetime
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.data.indicators.registry import IndicatorRegistry
from src.strategy.base import StrategyBase
from src.utils.cli_helpers import (
    print_exception,
    print_kv,
    print_kv_rows,
    print_section,
    print_section_end,
    print_status,
)
from src.utils.logger import setup_logging

if TYPE_CHECKING:
    from src.engine.backtester import Backtester
    from src.engine.result import BacktestResult

logger = setup_logging(
    name="run_backtest",
    log_file="logs/backtest.log",
    capture_all_loggers=False,
)

# --- Strategy registry ---
# Add new strategies here as they are implemented.

_STRATEGY_CONFIGS: dict[str, str] = {
    "orb": "config/strategy_params/orb_default.json",
}


# --- Strategy loader ---


def _load_strategy(name: str, config_path: str) -> tuple[StrategyBase, IndicatorRegistry, Any]:
    """
    Load strategy, config, and indicator registry by name.

    Returns (strategy, registry, config).
    """
    if name == "orb":
        from config.schemas.orb import ORBConfig
        from src.strategy.orb import ORBStrategy

        config = ORBConfig.from_json(config_path)
        strategy = ORBStrategy(config)
        registry = ORBStrategy.build_registry(
            atr_period=config.strategy.atr_period,
            adx_period=config.strategy.adx_period,
            volume_ma_period=config.strategy.volume_ma_period,
        )
        return strategy, registry, config

    raise ValueError(f"Unknown strategy: {name!r}. Available: {list(_STRATEGY_CONFIGS)}")


# --- Pipeline helpers ---


def _load_data(args: argparse.Namespace, config: Any) -> tuple[pd.DataFrame, str]:
    """Load and preprocess raw data. Returns preprocessed DataFrame."""
    from src.data.loader import DataLoader
    from src.data.preprocessor import DataPreprocessor
    from src.database.data_service import get_data_service

    # Use freq from config if not overridden by CLI
    freq = args.freq if args.freq else config.strategy.resample_freq

    print_status("Loading data...", "info")
    loader = DataLoader(
        data_service=get_data_service(),
        cache_dir=args.cache_dir,
    )
    raw = loader.load(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        use_cache=not args.force_refresh,
    )
    print_status(f"{len(raw):,} raw 1min bars loaded", "success")

    print_status("Preprocessing...", "info")
    preprocessor = DataPreprocessor()
    prep = preprocessor.prepare(raw, freq=freq)
    print_status(f"{len(prep):,} bars after resample + filter ({freq})", "success")

    return prep, freq


def _run_pipeline(
    prep: pd.DataFrame,
    registry: IndicatorRegistry,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, int]:
    """Run indicator pipeline on preprocessed data."""
    from src.data.pipeline import DataPipeline

    print_status("Computing indicators...", "info")
    pipeline = DataPipeline(
        registry=registry,
        cache_dir=args.cache_dir,
        use_cache=not args.force_refresh,
    )
    data = pipeline.run(prep)
    warmup = pipeline.get_required_lookback()
    print_status(f"Indicators computed (warmup = {warmup} bars)", "success")
    return data, warmup


def _build_backtester(
    strategy: StrategyBase,
    config: Any,
    args: argparse.Namespace,
    freq: str,
) -> Backtester:
    """Instantiate Backtester from config and CLI args."""
    from src.engine.account.sizer import PercentRiskSizer
    from src.engine.backtester import Backtester
    from src.engine.execution.slippage import FixedSlippage

    sizer = PercentRiskSizer(
        risk_per_trade_pct=config.risk.risk_per_trade_pct,
        min_size=config.risk.min_position_size,
        max_size=config.risk.max_position_size,
    )

    return Backtester(
        strategy=strategy,
        initial_capital=args.capital,
        commission_rate=args.commission_rate,
        contract_multiplier=args.contract_multiplier,
        margin_rate=args.margin_rate,
        position_sizer=sizer,
        slippage_model=FixedSlippage(args.slippage_points),
        use_trailing_stop=config.risk.use_trailing_stop,
        trailing_atr_multiplier=config.risk.trailing_atr_multiplier,
        max_daily_loss_pct=config.risk.max_daily_loss,
        entry_cutoff_seconds=float(config.risk.entry_cutoff_seconds),
        allow_late_entry=config.risk.allow_late_entry,
        freq_minutes=int(freq.replace("min", "")),
    )


# --- Output helpers ---


def _print_metrics(result: BacktestResult, elapsed: float) -> None:
    metrics = result.metrics
    print_section("BACKTEST RESULTS", width=60)

    print("\n  Returns")
    print_kv_rows(
        {
            "Total Return": f"{metrics.get('total_return_pct', 0):.2f}%",
            "Annualized Return": f"{metrics.get('annualized_return_pct', 0):.2f}%",
            "CAGR": f"{metrics.get('cagr_pct', 0):.2f}%",
            "Volatility": f"{metrics.get('volatility_pct', 0):.2f}%",
        },
        label_width=20,
    )

    print("\n  Risk-Adjusted")
    print_kv_rows(
        {
            "Sharpe Ratio": f"{metrics.get('sharpe_ratio', 0):.2f}",
            "Sortino Ratio": f"{metrics.get('sortino_ratio', 0):.2f}",
        },
        label_width=20,
    )

    print("\n  Drawdown")
    print_kv_rows(
        {
            "Max Drawdown": f"{metrics.get('max_drawdown_pct', 0):.2f}%",
            "Longest Drawdown": f"{metrics.get('longest_drawdown_bars', 0)} bars",
        },
        label_width=20,
    )

    print("\n  Trades")
    winning_trades, losing_trades, breakeven_trades = (
        result.winning_trades,
        result.losing_trades,
        result.breakeven_trades,
    )
    print_kv_rows(
        {
            "Total Trades": result.total_trades,
            "Win / Loss / Even": f"{winning_trades} / {losing_trades} / {breakeven_trades}",
            "Win Rate": f"{result.win_rate:.1f}%",
            "Net Profit Factor": f"{metrics.get('net_profit_factor', 0):.2f}",
            "Gross Profit Factor": f"{metrics.get('gross_profit_factor', 0):.2f}",
            "Avg Win": f"{metrics.get('avg_win', 0):.0f}",
            "Avg Loss": f"{metrics.get('avg_loss', 0):.0f}",
            "Payoff Ratio": f"{metrics.get('payoff_ratio', 0):.2f}",
            "Expectancy": f"{metrics.get('expectancy', 0):.0f}",
        },
        label_width=20,
    )

    print("\n  Streaks")
    print_kv_rows(
        {
            "Max Consec. Wins": metrics.get("max_consecutive_wins", 0),
            "Max Consec. Losses": metrics.get("max_consecutive_losses", 0),
        },
        label_width=20,
    )

    print("\n  Cost")
    print_kv_rows(
        {
            "Total PnL": f"{result.total_pnl:.0f}",
            "Total Commission": f"{metrics.get('total_commission', 0):.0f}",
            "Avg Duration": f"{metrics.get('avg_duration_minutes', 0):.1f} min",
        },
        label_width=20,
    )

    print("\n  Runtime")
    print_kv("Elapsed", f"{elapsed:.2f}s", label_width=20)

    print_section_end(width=60)


def _print_trades(result: BacktestResult, n: int = 20) -> None:
    trades = result.trades[:n]
    if not trades:
        print("\n  No trades.\n")
        return
    print(f"\n{'-' * 80}")
    print(f"  TRADES (first {min(n, len(result.trades))} of {len(result.trades)})")
    print(f"{'-' * 80}")
    print(
        f"  {'#':>3}  {'Side':5}  {'Entry':>8}  {'Exit':>8}  {'PnL':>10}  {'Duration':>8}  Reason"
    )
    print(f"  {'-' * 3}  {'-' * 5}  {'-' * 8}  {'-' * 8}  {'-' * 10}  {'-' * 8}  {'-' * 20}")
    for i, t in enumerate(trades, 1):
        dur = f"{t.duration_minutes:.0f}m" if t.duration_minutes else "-"
        print(
            f"  {i:>3}  {t.side.value:5}  {t.entry_price:>8.1f}  "
            f"{t.exit_price:>8.1f}  {t.pnl:>+10.0f}  {dur:>8}  "
            f"{t.exit_reason[:25]}"
        )
    print()


def _compare_engines(
    strategy: StrategyBase,
    config: Any,
    data: pd.DataFrame,
    warmup: int,
    args: argparse.Namespace,
    freq: str,
) -> None:
    """Run both H2 (loop) and H1 (event-driven), compare results."""
    from src.engine.account.account import AccountState
    from src.engine.core.engine import EventDrivenBacktester
    from src.engine.execution.slippage import FixedSlippage

    print_status("Comparing engines (H2 loop vs H1 event-driven)...", "info")

    # H2
    bt2 = _build_backtester(strategy, config, args, freq)
    t0 = time.perf_counter()
    r2 = bt2.run(data, warmup_bars=warmup)
    t2 = time.perf_counter() - t0

    # H1 - same AccountState settings + T+1 execution
    strategy.reset()
    from src.engine.account.sizer import PercentRiskSizer

    sizer = PercentRiskSizer(
        risk_per_trade_pct=config.risk.risk_per_trade_pct,
        min_size=config.risk.min_position_size,
        max_size=config.risk.max_position_size,
    )
    account = AccountState(
        initial_capital=args.capital,
        commission_rate=args.commission_rate,
        contract_multiplier=args.contract_multiplier,
        margin_rate=args.margin_rate,
        position_sizer=sizer,
        slippage_model=FixedSlippage(args.slippage_points),
        use_trailing_stop=config.risk.use_trailing_stop,
        trailing_atr_multiplier=config.risk.trailing_atr_multiplier,
        max_daily_loss_pct=config.risk.max_daily_loss,
    )
    bt1 = EventDrivenBacktester(
        strategy=strategy,
        account=account,
        freq_minutes=int(freq.replace("min", "")),
        use_t1_execution=True,  # Match H2's pending order behavior
        entry_cutoff_seconds=float(config.risk.entry_cutoff_seconds),
        allow_late_entry=config.risk.allow_late_entry,
    )
    t0 = time.perf_counter()
    r1 = bt1.run(data, warmup_bars=warmup)
    t1 = time.perf_counter() - t0

    pnl_diff = abs(r2.total_pnl - r1.total_pnl)
    trades_match = r2.total_trades == r1.total_trades

    print(f"\n{'-' * 55}")
    print(f"  {'Metric':28}  {'H2 (Loop)':>10}  {'H1 (Event)':>10}")
    print(f"  {'-' * 28}  {'-' * 10}  {'-' * 10}")
    print(f"  {'Total Trades':28}  {r2.total_trades:>10}  {r1.total_trades:>10}")
    print(f"  {'Total PnL':28}  {r2.total_pnl:>10.0f}  {r1.total_pnl:>10.0f}")
    print(f"  {'Win Rate':28}  {r2.win_rate:>10.1f}  {r1.win_rate:>10.1f}")
    print(f"  {'Runtime (s)':28}  {t2:>10.2f}  {t1:>10.2f}")
    print(f"\n  PnL diff:     {pnl_diff:.0f}")
    print(f"  Trades match: {'✅ YES' if trades_match else '❌ NO'}")
    if pnl_diff < 1 and trades_match:
        print("\n  ✅ Both engines produce identical results!")
    else:
        print("\n  ⚠️  Results differ - check implementation.")
    print(f"{'-' * 55}\n")


def _save_and_plot(result: BacktestResult, args: argparse.Namespace) -> None:
    """Save result files and generate plots."""
    output_dir = Path(args.output)

    # Save JSON + parquet
    print_status(f"Saving to {output_dir}...", "info")
    try:
        saved = result.save(output_dir)
        for name, path in saved.items():
            print_kv(name, path.name, label_width=18)
    except Exception as e:
        logger.warning("Save failed: %s", e)
        print_exception("Save", e)

    if args.no_plot:
        return

    print_status("Generating plots...", "info")
    try:
        from src.metrics.metrics import PerformanceMetrics
        from src.metrics.plotter import BacktestPlotter, PlotData

        # Reconstruct PerformanceMetrics from dict
        valid_fields = PerformanceMetrics.__dataclass_fields__
        perf = PerformanceMetrics(**{k: v for k, v in result.metrics.items() if k in valid_fields})

        plot_data = PlotData(
            equity=result.equity_curve,
            trades=result.trades,
            metrics=perf,
            initial_capital=args.capital,
        )
        plotter = BacktestPlotter(
            data=plot_data,
            output_dir=output_dir / "plots",
        )
        fmt = "html" if args.plot_html else "png"
        paths = plotter.plot_all(fmt=fmt)
        for p in paths:
            print_status(f"Created {p.name}", "success")
    except Exception as e:
        logger.warning("Plotting failed: %s", e, exc_info=True)
        print_exception("Plotting", e)


# --- Main ---


def run(args: argparse.Namespace) -> int:
    # --- 1. Load strategy ---
    config_path = args.config or _STRATEGY_CONFIGS.get(args.strategy)
    if config_path is None:
        print_status(f"Unknown strategy: {args.strategy!r}", "error")
        return 1

    logger.info("Strategy: %s | Config: %s", args.strategy, config_path)
    try:
        strategy, registry, config = _load_strategy(args.strategy, config_path)
    except Exception as e:
        logger.error("Failed to load strategy: %s", e, exc_info=True)
        print_exception("Strategy load", e)
        return 1

    # Use freq from config if not overridden by CLI
    freq = args.freq if args.freq else config.strategy.resample_freq

    print_section(f"BACKTEST: {args.strategy.upper()}", width=60)
    print_kv_rows(
        {
            "Period": f"{args.start} -> {args.end}",
            "Capital": f"{args.capital:,.0f}",
            "Freq": freq,
        },
        label_width=8,
    )
    print()

    # --- 2. Load + preprocess data ---
    try:
        prep, freq = _load_data(args, config)
    except Exception as e:
        logger.error("Data load failed: %s", e, exc_info=True)
        print_exception("Data load", e)
        return 1

    # --- 3. Run indicator pipeline ---
    try:
        data, warmup = _run_pipeline(prep, registry, args)
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        print_exception("Pipeline", e)
        return 1

    # --- 4. Run backtest ---
    bt = _build_backtester(strategy, config, args, freq)

    print_status("Running backtest...", "info")
    t_start = time.perf_counter()

    # Use tqdm for better progress tracking
    try:
        from tqdm import tqdm

        pbar = None

        def progress(current: int, total: int) -> None:
            nonlocal pbar
            if pbar is None:
                pbar = tqdm(total=total, desc="Backtest", unit="bar", ncols=80)
            pbar.update(1)
            if current == total:
                pbar.close()

        result = bt.run(data, warmup_bars=warmup, progress_callback=progress)
    except ImportError:
        # Fallback to simple progress bar if tqdm not available
        last_pct = -1

        def progress_simple(current: int, total: int) -> None:
            nonlocal last_pct
            pct = int(current / total * 100)
            if pct % 5 == 0 and pct != last_pct:
                last_pct = pct
                bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                print(f"\r  [{bar}] {pct:3d}%  ({current:,}/{total:,})", end="", flush=True)

        result = bt.run(data, warmup_bars=warmup, progress_callback=progress_simple)
        print()
    except Exception as e:
        logger.error("Backtest failed: %s", e, exc_info=True)
        print()
        print_exception("Backtest", e)
        return 1

    elapsed = time.perf_counter() - t_start
    print_status(f"Done in {elapsed:.2f}s", "success")

    # --- 5. Compare engines (optional) ---
    if args.compare_engines:
        try:
            _compare_engines(strategy, config, data, warmup, args, freq)
        except Exception as e:
            logger.warning("Engine comparison failed: %s", e)
            print_exception("Engine comparison", e)

    # --- 6. Print metrics ---
    _print_metrics(result, elapsed)

    if args.show_trades:
        _print_trades(result, n=args.show_trades_n)

    # --- 7. Save + plot ---
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output = f"results/{args.strategy}_{current_time}"
    _save_and_plot(result, args)

    # --- 8. Monte Carlo (optional) ---
    if args.monte_carlo and result.trades:
        try:
            from src.optimization.monte_carlo import MonteCarlo

            print_status(f"Running Monte Carlo ({args.mc_simulations:,} simulations)...", "info")
            mc = MonteCarlo(
                trades=result.trades,
                initial_capital=args.capital,
                ruin_threshold_pct=args.mc_ruin_pct,
            )
            mc_result = mc.run(n_simulations=args.mc_simulations)
            mc_result.print_summary()
            mc_result.save(Path(args.output) / "monte_carlo")
        except Exception as e:
            logger.warning("Monte Carlo failed: %s", e)
            print_exception("Monte Carlo", e)

    print_status(f"Backtest complete -> {args.output}", "success")
    return 0


# --- CLI ---


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_backtest",
        description="Run a VN30 strategy backtest end-to-end",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Strategy
    parser.add_argument("--strategy", "-s", choices=list(_STRATEGY_CONFIGS), default="orb")
    parser.add_argument(
        "--config", "-c", default=None, help="Path to strategy JSON config (overrides default)"
    )

    # Data
    parser.add_argument("--symbol", default="VN30F1M")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2025-03-31")
    parser.add_argument(
        "--freq",
        default=None,
        choices=["1min", "5min", "15min", "30min"],
        help="Override resample_freq from config (optional)",
    )
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Bypass cache, fetch fresh data and recompute indicators",
    )

    # Capital / costs — grouped via ExecutionConfig
    from config.constants import ExecutionConfig

    ExecutionConfig.add_args(parser)

    # Output
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--plot-html", action="store_true", help="Generate interactive HTML plots instead of PNG"
    )

    # Analysis
    parser.add_argument("--show-trades", action="store_true")
    parser.add_argument("--show-trades-n", type=int, default=30)
    parser.add_argument(
        "--compare-engines", action="store_true", help="Run H2 and H1 engines, compare results"
    )
    parser.add_argument(
        "--monte-carlo", action="store_true", help="Run Monte Carlo simulation on trade list"
    )
    parser.add_argument("--mc-simulations", type=int, default=1000)
    parser.add_argument(
        "--mc-ruin-pct", type=float, default=20.0, help="Drawdown %% considered ruin"
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
