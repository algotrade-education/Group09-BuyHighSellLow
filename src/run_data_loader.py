"""
Market Data Pipeline Runner.

Orchestrates the lifecycle of market data, from raw database extraction
to local Parquet disk caching. Supports 'In-Sample' (IS) and
'Out-of-Sample' (OS) partitioning used across the project.

Usage:
    python -m src.run_data_loader --mode fetch --contract VN30F1M
    python -m src.run_data_loader --mode load --sample is

Modes:
- fetch: Connects to the main database, pulls raw ticks, and saves monthly
         CSV chunks to `data/`.
- load:  Reads CSV chunks, performs initial cleaning, and serializes to
         highly-optimized Parquet files for fast backtesting.

Arguments:
    --mode:     'fetch' or 'load'.
    --contract: Symbol (default: VN30F1M).
    --sample:   'is' or 'os' (only for 'load' mode).
    --chunk-by: Granularity of saved files ('month' or 'year').
"""

import argparse
import sys

import pandas as pd

from config.config import (
    IS_SAMPLE_START,
    OUT_SAMPLE_END,
)
from src.data.loader import DataLoader
from src.utils.logger import setup_logging

# Configure logging
logger = setup_logging(__name__, log_file="logs/data_loader.log")


def fetch_data(contract: str = "VN30F1M", chunk_by: str = "month"):
    """
    Execute raw data extraction from SQL Database.

    Pulls the full range defined in global config (`IS_SAMPLE_START` to
    `OUT_SAMPLE_END`) and shards it into archival CSVs. Note that
    database credentials must be present in `.env`.

    Args:
        contract: The market ticker.
        chunk_by: Partition size for saved files.
    """
    loader = DataLoader()

    # Fetch the full range covering both IS and OS periods in one go.
    # The IS/OS split happens at load time via load_in_sample / load_out_of_sample.
    full_start = IS_SAMPLE_START
    full_end = OUT_SAMPLE_END

    try:
        logger.info(
            "Fetching %s data (%s to %s) with chunk_by='%s'",
            contract,
            full_start,
            full_end,
            chunk_by,
        )
        loader.fetch_from_database(
            contract_name=contract,
            start_date=full_start,
            end_date=full_end,
            chunk_by=chunk_by,
        )
        logger.info("Data fetch complete. Chunks saved to data/.")
    except Exception as e:
        logger.error("Error fetching data: %s", e)
        logger.error("Ensure database credentials are set in .env")
        sys.exit(1)


def load_data(sample: str = "is", contract: str = "VN30F1M") -> pd.DataFrame:
    """
    Load data from local storage into memory.

    This function attempts to use the `.parquet` cache if it exists.
    If not, it rebuilds the cache from the `.csv` chunks found in `data/`.

    Args:
        sample: Dataset partition ('is' or 'os').
        contract: Ticker symbol.

    Returns:
        Cleaned and timestamp-indexed DataFrame.
    """
    loader = DataLoader()
    try:
        if sample == "is":
            return loader.load_in_sample(contract_name=contract)

        return loader.load_out_of_sample(contract_name=contract)
    except FileNotFoundError as e:
        logger.error("Error: %s", e)
        logger.info("\nPlease ensure data files exist in the data/ directory.")
        logger.info(
            "Run 'python src/run_data_loader.py --mode fetch' to fetch from database."
        )
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch or load market data.")
    parser.add_argument(
        "--mode",
        choices=["fetch", "load"],
        default="fetch",
        help="Operation mode: 'fetch' to get data from database, 'load' to load existing data.",
    )
    parser.add_argument(
        "--contract", default="VN30F1M", help="Contract symbol (default: VN30F1M)."
    )
    parser.add_argument(
        "--sample",
        choices=["is", "os"],
        default="is",
        help="Sample type for --mode load: 'is' or 'os' (default: is).",
    )
    parser.add_argument(
        "--chunk-by",
        choices=["month", "year"],
        default="month",
        dest="chunk_by",
        help="Chunk granularity when saving CSVs (default: month).",
    )
    args = parser.parse_args()

    if args.mode == "fetch":
        fetch_data(args.contract, args.chunk_by)
    else:
        df = load_data(args.sample, args.contract)
        logger.info("Loaded %s rows.", len(df))
