"""
src/run_data_loader.py

Data management CLI - fetch, validate, inspect, cache data.

DataLoader now uses monthly parquet cache with incremental fetching.
Always returns 1-minute bars. Use DataPreprocessor for resampling.

Usage:
    # Fetch và cache 1min data từ DB (monthly chunks)
    python -m src.run_data_loader fetch --symbol VN30F1M --start 2023-01-01 --end 2024-12-31

    # Force refresh specific months
    python -m src.run_data_loader fetch --symbol VN30F1M --start 2023-01-01 --end 2023-03-31 --force-months 2023_01,2023_02

    # Import tick CSV và aggregate to 1min
    python -m src.run_data_loader import-csv --symbol VN30F1M --path "data/ticks_*.csv"

    # Kiểm tra cached months
    python -m src.run_data_loader list-cache --symbol VN30F1M

    # Inspect data
    python -m src.run_data_loader inspect --symbol VN30F1M --start 2023-01-01 --end 2024-12-31

    # Validate data quality
    python -m src.run_data_loader validate --symbol VN30F1M --start 2023-01-01 --end 2024-12-31

    # Xóa cache (specific symbol or month)
    python -m src.run_data_loader clear-cache --symbol VN30F1M --month 2023_01

    # Xem thống kê data sau khi preprocess + resample
    python -m src.run_data_loader stats --symbol VN30F1M --start 2023-01-01 --end 2024-12-31 --freq 5min
"""

from __future__ import annotations

import argparse
import sys

from src.utils.cli_helpers import (
    print_exception,
    print_kv,
    print_kv_rows,
    print_rule,
    print_section,
    print_section_end,
    print_status,
)
from src.utils.logger import setup_logging

logger = setup_logging(
    name="run_data_loader",
    log_file="logs/data_loader.log",
    capture_all_loggers=False,
)


# --- Commands ---


def cmd_fetch(args: argparse.Namespace) -> int:
    """Fetch 1min data từ DB và save monthly cache."""
    from src.data.loader import DataLoader
    from src.data.validators import DataValidator
    from src.database.data_service import get_data_service

    logger.info(
        "Fetching %s from %s to %s...",
        args.symbol,
        args.start,
        args.end,
    )

    try:
        svc = get_data_service()
        loader = DataLoader(svc, cache_dir=args.cache_dir)

        # Parse force_months if provided
        force_months = None
        if args.force_months:
            force_months = [m.strip() for m in args.force_months.split(",")]
            logger.info("Force refresh months: %s", force_months)

        # Load returns 1min bars with monthly cache
        df = loader.load(
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            use_cache=not args.force_refresh,
            force_months=force_months,
        )

        if df.empty:
            logger.error("No data returned for %s [%s -> %s]", args.symbol, args.start, args.end)
            return 1

        print_section("FETCH COMPLETE (1min bars)", width=55)
        print_kv_rows(
            {
                "Symbol": args.symbol,
                "Period": f"{args.start} -> {args.end}",
                "Rows": f"{len(df):,}",
                "Columns": list(df.columns),
                "Date range": f"{df['datetime'].min()} -> {df['datetime'].max()}",
                "Cache dir": args.cache_dir,
            }
        )

        # Show cached months
        cached = loader.list_cached_months(args.symbol)
        print_kv("Cached months", len(cached))
        print_section_end(width=55)

        # Quick validation
        validator = DataValidator()
        result = validator.validate_ohlcv(df)
        if result.is_valid:
            print_status("Data validation PASSED", status="success")
        else:
            print_status(f"Data validation WARNINGS:\n{result.summary()}", status="warning")
        if result.warnings:
            print(f"\nWarnings:\n{result.summary()}")

        return 0

    except Exception as e:
        logger.error("Fetch failed: %s", e, exc_info=True)
        print_exception("Fetch", e)
        return 1


def cmd_import_csv(args: argparse.Namespace) -> int:
    """Import tick CSV files, aggregate to 1min, and cache."""
    from src.data.loader import DataLoader
    from src.database.data_service import get_data_service

    logger.info("Importing tick CSV from %s for %s...", args.path, args.symbol)

    try:
        svc = get_data_service()
        loader = DataLoader(svc, cache_dir=args.cache_dir)

        # Use load_tick_csv method
        df = loader.load_tick_csv(
            path_pattern=args.path,
            symbol=args.symbol,
            cache_result=True,
            force_refresh=args.force_refresh,
        )

        if df.empty:
            logger.error("No data produced from CSV files: %s", args.path)
            return 1

        print_status(f"Imported and aggregated to {len(df):,} 1min bars", status="success")
        print(f"   Date range: {df['datetime'].min()} -> {df['datetime'].max()}")
        print("   Cached as monthly parquet files")
        print("   Now you can run 'inspect', 'stats', or Backtest normally!")
        return 0

    except Exception as e:
        logger.error("Import failed: %s", e, exc_info=True)
        print_exception("Import CSV", e)
        return 1


def cmd_inspect(args: argparse.Namespace) -> int:
    """Inspect cached 1min data - xem sample, statistics, missing data."""
    from src.data.loader import DataLoader
    from src.database.data_service import get_data_service

    try:
        svc = get_data_service()
        loader = DataLoader(svc, cache_dir=args.cache_dir)

        # Load 1min bars
        df = loader.load(
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            use_cache=True,
        )

        if df.empty:
            print_status(
                f"No data found for {args.symbol} [{args.start} -> {args.end}]", status="error"
            )
            print("   Run: python -m src.run_data_loader fetch ... to fetch data first")
            return 1

        import pandas as pd

        df["datetime"] = pd.to_datetime(df["datetime"])

        # --- Basic info ---
        print_section(f"DATA INSPECTION: {args.symbol} (1min bars)", width=60)
        print_kv_rows(
            {
                "Period": f"{df['datetime'].min().date()} -> {df['datetime'].max().date()}",
                "Total rows": f"{len(df):,}",
                "Columns": list(df.columns),
            },
            label_width=14,
        )

        # --- Price stats ---
        print_rule("PRICE STATISTICS")
        print(f"  Close min:  {df['close'].min():,.2f}")
        print(f"  Close max:  {df['close'].max():,.2f}")
        print(f"  Close mean: {df['close'].mean():,.2f}")
        print(f"  Close std:  {df['close'].std():,.2f}")

        # --- Volume stats ---
        if "volume" in df.columns:
            print_rule("VOLUME STATISTICS")
            print(f"  Vol mean:   {df['volume'].mean():,.0f}")
            print(f"  Vol max:    {df['volume'].max():,.0f}")
            zero_vol = (df["volume"] == 0).sum()
            print(f"  Zero vol:   {zero_vol:,} bars ({zero_vol / len(df) * 100:.1f}%)")

        # --- Missing data ---
        print_rule("MISSING DATA")
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                n_null = df[col].isna().sum()
                print(f"  {col:8s}: {n_null:,} NaN")

        # --- Bars per day ---
        bars_per_day = df.groupby(df["datetime"].dt.date).size()
        print_rule("BARS PER DAY")
        print(f"  Trading days: {len(bars_per_day):,}")
        print(f"  Median bars/day: {bars_per_day.median():.0f}")
        print(f"  Min bars/day:    {bars_per_day.min()}")
        print(f"  Max bars/day:    {bars_per_day.max()}")

        low_bar_threshold = int(bars_per_day.median() * 0.5)
        low_bars_days = bars_per_day[bars_per_day < low_bar_threshold]
        if not low_bars_days.empty:
            print(f"\n  Days with unusually few bars ({len(low_bars_days)}):")
            for d, n in low_bars_days.head(5).items():
                print(f"     {d}: {n} bars")

        # --- Sample data ---
        print_rule("FIRST 5 ROWS")
        print(df.head(5).to_string(index=False))

        print_rule("LAST 5 ROWS")
        print(df.tail(5).to_string(index=False))
        print_section_end(width=60)

        return 0

    except Exception as e:
        logger.error("Inspect failed: %s", e, exc_info=True)
        print_exception("Inspect", e)
        return 1


def cmd_validate(args: argparse.Namespace) -> int:
    """Full data validation - OHLC relationships, gaps, anomalies."""
    from src.data.loader import DataLoader
    from src.data.validators import DataValidator
    from src.database.data_service import get_data_service

    try:
        svc = get_data_service()
        loader = DataLoader(svc, cache_dir=args.cache_dir)

        df = loader.load(
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            use_cache=True,
        )

        if df.empty:
            print_status("No data found. Run fetch first.", status="error")
            return 1

        validator = DataValidator()
        result = validator.validate_ohlcv(df)

        print_section(f"VALIDATION: {args.symbol}", width=55)
        print_kv("Rows checked", f"{len(df):,}")
        print()

        if result.is_valid and not result.warnings:
            print_status("All checks PASSED - data is clean", status="success")
        elif result.is_valid:
            print_status("Validation PASSED with warnings", status="success")
        else:
            print_status("Validation FAILED", status="error")

        if result.errors:
            print(f"\n  ERRORS ({len(result.errors)}):")
            for err in result.errors:
                print(f"    ✗ {err}")

        if result.warnings:
            print(f"\n  WARNINGS ({len(result.warnings)}):")
            for w in result.warnings:
                print(f"    ⚠ {w}")

        print_section_end(width=55)
        return 0 if result.is_valid else 1

    except Exception as e:
        logger.error("Validate failed: %s", e, exc_info=True)
        print_exception("Validate", e)
        return 1


def cmd_stats(args: argparse.Namespace) -> int:
    """Thống kê data sau khi preprocess + resample to target frequency."""
    from src.data.loader import DataLoader
    from src.data.preprocessor import DataPreprocessor
    from src.database.data_service import get_data_service

    try:
        svc = get_data_service()
        loader = DataLoader(svc, cache_dir=args.cache_dir)
        preprocessor = DataPreprocessor()

        # Load 1min bars
        raw = loader.load(args.symbol, args.start, args.end, use_cache=True)

        # Preprocess and resample to target frequency
        prep = preprocessor.prepare(raw, freq=args.freq)

        print_section(f"PREPROCESSED DATA STATS: {args.symbol} ({args.freq})", width=60)
        print_kv_rows(
            {
                "Raw rows (1min)": f"{len(raw):,}",
                "Processed rows": f"{len(prep):,}",
                "Removed rows": f"{len(raw) - len(prep):,}",
            },
            label_width=16,
        )

        import pandas as pd

        prep["datetime"] = pd.to_datetime(prep["datetime"])
        bars_per_day = prep.groupby(prep["datetime"].dt.date).size()

        print(f"\n  Trading days:  {len(bars_per_day):,}")
        print(f"  Bars/day:      {bars_per_day.median():.0f} median")

        from config.schemas.session import VN30SessionConfig

        expected = VN30SessionConfig().bars_per_year(int(args.freq.replace("min", "")))
        actual = len(prep)
        coverage = actual / expected * 100 if expected > 0 else 0

        print(f"\n  Expected bars/year: {expected:,}")
        print(f"  Actual bars:        {actual:,}")
        print(f"  Coverage:           {coverage:.1f}%")
        print_section_end(width=60)

        return 0

    except Exception as e:
        logger.error("Stats failed: %s", e, exc_info=True)
        print_exception("Stats", e)
        return 1


def cmd_clear_cache(args: argparse.Namespace) -> int:
    """Xóa cache files (symbol or specific month)."""
    from src.data.loader import DataLoader
    from src.database.data_service import get_data_service

    try:
        svc = get_data_service()
        loader = DataLoader(svc, cache_dir=args.cache_dir)

        if not args.yes:
            if args.month:
                msg = f"Delete cache for {args.symbol}/{args.month}?"
            elif args.symbol:
                msg = f"Delete all cached months for {args.symbol}?"
            else:
                msg = f"Delete ALL cache in {args.cache_dir}?"

            confirm = input(f"\n{msg} [y/N] ").strip().lower()
            if confirm != "y":
                print("Cancelled.")
                return 0

        count = loader.invalidate_cache(
            symbol=args.symbol if args.symbol else None,
            month_key=args.month if args.month else None,
        )

        if count > 0:
            print_status(f"Cleared {count} cache file(s)", status="success")
        else:
            print_status("No cache files found to delete", status="warning")

        return 0

    except Exception as e:
        logger.error("Clear cache failed: %s", e, exc_info=True)
        print_exception("Clear cache", e)
        return 1


# --- CLI ---


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_data_loader",
        description="Data management CLI for VN30 trading system (monthly cache)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Shared args
    def add_data_args(p: argparse.ArgumentParser, require_dates: bool = True) -> None:
        p.add_argument("--symbol", default="VN30F1M")
        if require_dates:
            p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
            p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
        p.add_argument("--cache-dir", default="data/cache")

    # fetch
    p_fetch = subparsers.add_parser("fetch", help="Fetch 1min data from DB and cache monthly")
    add_data_args(p_fetch)
    p_fetch.add_argument(
        "--force-refresh", action="store_true", help="Ignore cache, always fetch from DB"
    )
    p_fetch.add_argument(
        "--force-months", help="Comma-separated month keys to force refresh (e.g., 2023_01,2023_02)"
    )

    # import-csv
    p_import = subparsers.add_parser(
        "import-csv", help="Import tick CSV, aggregate to 1min, cache monthly"
    )
    p_import.add_argument(
        "--path", required=True, help="Path to CSV or glob pattern (e.g. data/ticks_*.csv)"
    )
    p_import.add_argument("--symbol", required=True, help="Symbol for cache directory")
    p_import.add_argument("--cache-dir", default="data/cache")
    p_import.add_argument(
        "--force-refresh", action="store_true", help="Re-aggregate even if cached"
    )

    # inspect
    p_inspect = subparsers.add_parser("inspect", help="Inspect cached 1min data")
    add_data_args(p_inspect)

    # validate
    p_validate = subparsers.add_parser("validate", help="Validate data quality")
    add_data_args(p_validate)

    # stats
    p_stats = subparsers.add_parser("stats", help="Show preprocessed data statistics")
    add_data_args(p_stats)
    p_stats.add_argument("--freq", default="5min", choices=["1min", "5min", "15min", "30min"])

    # clear-cache
    p_clear = subparsers.add_parser("clear-cache", help="Clear cache files")
    p_clear.add_argument("--symbol", help="Symbol to clear (omit to clear all)")
    p_clear.add_argument("--month", help="Specific month to clear (e.g., 2023_01)")
    p_clear.add_argument("--cache-dir", default="data/cache")
    p_clear.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "fetch": cmd_fetch,
        "import-csv": cmd_import_csv,
        "inspect": cmd_inspect,
        "validate": cmd_validate,
        "stats": cmd_stats,
        "clear-cache": cmd_clear_cache,
    }

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
