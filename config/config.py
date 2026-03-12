"""
Configuration for trading algorithm backtest.
"""

import os

from dotenv import load_dotenv

# Load environment variables from .env file
_ = load_dotenv()

# === DATABASE ===
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
database = os.getenv("DB_NAME")

DB_CONFIG = {
    "host": host,
    "port": port,
    "user": user,
    "password": password,
    "database": database,
}

# === PATHS ===
DATA_DIR = "data"
CACHE_DIR = "data/cache"
RESULTS_DIR = "results"
REPORTS_DIR = "reports"

# === DATA SAMPLE PERIODS ===
IS_SAMPLE_START = "2024-01-01 00:00:00"
IS_SAMPLE_END = "2025-03-01 23:59:59"
OUT_SAMPLE_START = "2025-04-01 00:00:00"
OUT_SAMPLE_END = "2025-12-31 23:59:59"

# === TRADING SESSION TIMES (HOSE) ===
TRADING_START = "09:00:00"
TRADING_END = "14:30:00"
ATC_START = "14:30:00"
ATC_END = "14:45:00"

# === SESSION DEFAULTS ===
DEFAULT_INITIAL_CAPITAL = 500_000_000  # 500 million VND
COMMISSION_RATE = 0.00015  # 0.015% per trade
CONTRACT_MULTIPLIER = 100000  # 100k VND per point
MARGIN_RATE = 0.18  # 18% margin requirement
