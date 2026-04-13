"""
Pure constants - values that are unlikely to change across different environments or runs, and do not contain sensitive information.
Examples include fixed time intervals, mathematical constants, or any other hardcoded values that are essential
for the functioning of the code but do not vary based on user input or environment settings.

Credentials: secrets.py
Session times: schemas/session.py
Strategy params: schemas/*.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

# --- Capital ---
DEFAULT_INITIAL_CAPITAL: float = 500_000_000.00  # 500 million VND

# --- Contract Spec VN30 ---
VN30F_CONTRACT_MULTIPLIER: float = 100_000.00  # 1 point = 100,000 VND for VN30F futures
VN30F_COMMISSION_PER_CONTRACT: float = 4_750.00  # 4,750 VND per contract per side (futures)
VN30F_MARGIN_PER_CONTRACT: float = 9_000_000.00  # 9 million VND margin per contract (futures)
VN30F_COMMISSION_RATE: float = 0.00015  # 0.015% of notional
VN30F_MARGIN_RATE: float = 0.18  # 18% margin requirement
VN30F_SLIPPAGE_POINTS: float = 0.5  # 0.5 index points slippage per side

# --- Order Execution ---
MARKET_ORDER_PRICE_BUFFER: float = 20.0  # Price buffer for MARKET orders (points above/below entry)

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


@dataclass
class ExecutionConfig:
    """
    Shared execution parameters used by run_backtest, run_optimize, run_walk_forward.

    Groups the 5 args that are identical across all run scripts and rarely change.
    Loaded from constants by default - override only when needed.

    Usage in argparse:
        from config.constants import ExecutionConfig
        ExecutionConfig.add_args(parser)          # adds --capital, --commission-rate, etc.
        exec_cfg = ExecutionConfig.from_args(args) # parse back from Namespace
    """

    capital: float = DEFAULT_INITIAL_CAPITAL
    commission_rate: float = VN30F_COMMISSION_RATE
    slippage_points: float = VN30F_SLIPPAGE_POINTS
    contract_multiplier: float = VN30F_CONTRACT_MULTIPLIER
    margin_rate: float = VN30F_MARGIN_RATE

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        """Add execution args to an argparse.ArgumentParser."""
        parser.add_argument("--capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
        parser.add_argument("--commission-rate", type=float, default=VN30F_COMMISSION_RATE)
        parser.add_argument("--slippage-points", type=float, default=VN30F_SLIPPAGE_POINTS)
        parser.add_argument("--contract-multiplier", type=float, default=VN30F_CONTRACT_MULTIPLIER)
        parser.add_argument("--margin-rate", type=float, default=VN30F_MARGIN_RATE)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> ExecutionConfig:
        """Build ExecutionConfig from parsed argparse.Namespace."""
        return cls(
            capital=args.capital,
            commission_rate=args.commission_rate,
            slippage_points=args.slippage_points,
            contract_multiplier=args.contract_multiplier,
            margin_rate=args.margin_rate,
        )
