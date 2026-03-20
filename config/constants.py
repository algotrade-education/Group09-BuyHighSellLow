"""
Pure constants - values that are unlikely to change across different environments or runs, and do not contain sensitive information.
Examples include fixed time intervals, mathematical constants, or any other hardcoded values that are essential
for the functioning of the code but do not vary based on user input or environment settings.

Credentials: secrets.py
Session times: schemas/session.py
Strategy params: schemas/*.py
"""

# --- Capital ---
DEFAULT_INITIAL_CAPITAL: float = 500_000_000.00  # 500 million VND

# --- Contract Spec VN30 ---
VN30F_CONTRACT_MULTIPLIER: float = 100_000.00  # 1 point = 100,000 VND for VN30F futures
VN30F_COMMISSION_PER_CONTRACT: float = 4_750.00  # 4,750 VND per contract per side (futures)
VN30F_MARGIN_PER_CONTRACT: float = 9_000_000.00  # 9 million VND margin per contract (futures)

# --- Data ---
DEFAULT_SYMBOL: str = "VN30F1M"
DATETIME_COLUMN: str = "datetime"
CACHE_DIR: str = "data/cache/"

# --- Optimization ---
DEFAULT_N_OPTUNA_TRIALS: int = 300
DEFAULT_N_WALK_FORWARD_WINDOWS: int = 5
DEFAULT_MIN_TRADES_REQUIRED: int = 30

# --- Paths ---
RESULTS_DIR: str = "results"
LOG_DIR: str = "logs"
STRATEGY_PARAMS_DIR: str = "config/strategy_params"
