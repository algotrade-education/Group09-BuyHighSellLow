"""
Base session manager interface for trading hours and session rules.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime

logger = logging.getLogger(__name__)


class SessionManager(ABC):
    """
    Abstract session manager interface.

    Manages trading session rules including:
    - Trading hours validation
    - End-of-day close logic
    - Signal generation restrictions
    - Entry/exit timing controls
    """

    @abstractmethod
    def is_trading_hours(self, dt: datetime) -> bool:
        """
        Check if given datetime is within trading hours.

        Args:
            dt: Datetime to check

        Returns:
            True if within trading hours, False otherwise
        """
        ...

    @abstractmethod
    def should_close_eod(self, dt: datetime) -> bool:
        """
        Check if positions should be closed at end of day.

        Args:
            dt: Current datetime

        Returns:
            True if should close positions, False otherwise
        """
        ...

    @abstractmethod
    def should_skip_signal(self, dt: datetime) -> bool:
        """
        Check if signal generation should be skipped.

        Args:
            dt: Current datetime

        Returns:
            True if should skip signal, False otherwise
        """
        ...

    def is_entry_blocked(
        self,
        dt: datetime,
        cutoff_seconds: float = 0.0,
        allow_late: bool = False,
    ) -> bool:
        """
        Check if new entries should be blocked (e.g., too close to session end).

        Args:
            dt: Current datetime
            cutoff_seconds: Minimum seconds before session end to allow entry
            allow_late: If True, ignore cutoff and allow late entries

        Returns:
            True if entry should be blocked, False otherwise
        """
        return False

    def get_force_close_reason(self, dt: datetime, preclose_seconds: float = 0.0) -> str | None:
        """
        Get reason for forced position close (e.g., approaching session end).

        Args:
            dt: Current datetime
            preclose_seconds: Seconds before session end to force close

        Returns:
            Reason string if should force close, None otherwise
        """
        return None

    def is_atc(self, dt: datetime) -> bool:
        """
        Check if currently in ATC (At-The-Close) period.

        Default implementation returns False (no ATC period).
        Override in session-specific subclasses (e.g. VN30Session).

        Args:
            dt: Datetime to check

        Returns:
            True if in ATC period, False otherwise
        """
        return False

    def should_cancel_pending_entry(self, created_at: datetime, now: datetime) -> bool:
        """
        Check whether a queued entry order should be cancelled due to session transition.

        Default behavior keeps pending entries alive. Session-specific managers can
        override this to enforce stricter order lifecycle rules.

        Args:
            created_at: Datetime when pending order was created.
            now: Current bar datetime.

        Returns:
            True if pending entry should be cancelled.
        """
        return False


class AlwaysOpenSession(SessionManager):
    """
    No session restrictions - always open.

    Useful for:
    - Testing and development
    - 24/7 markets (crypto)
    - Backtesting without session constraints
    """

    def is_trading_hours(self, dt: datetime) -> bool:
        """Always returns True."""
        return True

    def should_close_eod(self, dt: datetime) -> bool:
        """Always returns False - never force close."""
        return False

    def should_skip_signal(self, dt: datetime) -> bool:
        """Always returns False - never skip signals."""
        return False
