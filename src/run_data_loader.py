"""
Script to fetch and load market data for specified contracts.
Defaults to VN30F1M contract.

Run as:
    py -m src.run_data_loader --mode fetch
to fetch from the database and save monthly CSV chunks to data/.

    py -m src.run_data_loader --mode load --sample is
to load IS data (reads parquet cache or rebuilds from chunks).
"""

import argparse
import sys

import pandas as pd

from config.config import (
    IS_SAMPLE_END,
    IS_SAMPLE_START,
    OUT_SAMPLE_END,
    OUT_SAMPLE_START,
)
from src.data.loader import DataLoader
from src.utils.logger import setup_logging

# Configure logging
logger = setup_logging(__name__, log_file="logs/data_loader.log")


def fetch_data(contract: str = "VN30F1M", chunk_by: str = "month"):
    """
    Fetch market data from the database and save as CSV chunks in data/.

    Fetches the full IS+OS date range in one request and splits it into
    monthly (or yearly) CSV files: data/<contract>_YYYYMM.csv.
    IS and OS parquets are generated automatically on the first load.

    Args:
        contract (str): Contract symbol to fetch data for
        chunk_by (str): 'month' or 'year' - controls chunk file granularity
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
    Load data (uses cache if available, creates if missing).

    Args:
        sample (str): "is" for in-sample, "os" for out-of-sample
        contract (str): Contract symbol to load data for

    Returns:
        pd.DataFrame: Loaded data
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
