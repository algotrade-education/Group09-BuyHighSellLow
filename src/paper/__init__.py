"""Paper trading engine package.

Provides the live/sim paper trading stack, including:
- Engine orchestration
- Feed/bar aggregation
- Risk and account state management
- Session statistics and data-quality utilities

Subpackages:
- account: tracker and broker reconciliation
- execution: order lifecycle management
- feeds: live/sim market data feeds
- handlers: bar/risk/signal pipeline handlers
"""

from src.paper.bar_aggregator import BarAggregator
from src.paper.data_quality import (
    BarState,
    DataQualityConfig,
    get_quality_reasons,
    maybe_merge_db_bar,
    merge_db_bar,
)
from src.paper.engine import PaperEngine
from src.paper.risk_manager import RiskManager
from src.paper.stats import SessionStats

__all__ = [
    "PaperEngine",
    "BarAggregator",
    "RiskManager",
    "SessionStats",
    "BarState",
    "DataQualityConfig",
    "get_quality_reasons",
    "merge_db_bar",
    "maybe_merge_db_bar",
]
