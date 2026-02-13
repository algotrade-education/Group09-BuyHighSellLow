"""
Script to fetch and load market data for specified contracts.
Defaults to VN30F1M contract.

Run as:
python src/run_data_loader.py --mode fetch
to fetch data from database, or
python src/run_data_loader.py --mode load
to load existing data.
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


def fetch_data(contract: str = "VN30F1M"):
    """
    Fetch in-sample and out-of-sample data from database.

    Args:
        contract (str): Contract symbol to fetch data for
    """
    loader = DataLoader()

    try:
        # Fetch in-sample
        logger.info(
            "Fetching in-sample data for %s (%s to %s)",
            contract,
            IS_SAMPLE_START,
            IS_SAMPLE_END,
        )
        loader.fetch_from_database(
            contract_name=contract,
            start_date=IS_SAMPLE_START,
            end_date=IS_SAMPLE_END,
            save_path=f"data/is/{contract}_data.csv",
        )

        # Fetch out-of-sample
        logger.info(
            "Fetching out-of-sample data for %s (%s to %s)",
            contract,
            OUT_SAMPLE_START,
            OUT_SAMPLE_END,
        )
        loader.fetch_from_database(
            contract_name=contract,
            start_date=OUT_SAMPLE_START,
            end_date=OUT_SAMPLE_END,
            save_path=f"data/os/{contract}_data.csv",
        )
        logger.info("Data fetch complete.")
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
        help="Sample type: is (in-sample) or os (out-of-sample).",
    )
    args = parser.parse_args()

    if args.mode == "fetch":
        fetch_data(args.contract)
    else:
        df = load_data(args.sample, args.contract)
        logger.info("Loaded %s rows.", len(df))
