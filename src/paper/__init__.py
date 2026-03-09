"""
Paper trading module - live & simulated ORB trading via PaperBrokerClient.

Quick start:
    from src.paper.engine import PaperTrader
    from src.paper.stats import SessionStats
"""

from src.paper.engine import PaperTrader
from src.paper.stats import SessionStats

__all__ = ["PaperTrader", "SessionStats"]