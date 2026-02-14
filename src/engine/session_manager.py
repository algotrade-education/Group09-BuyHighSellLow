"""
Trading session management for the backtesting engine.
This module defines the SessionManager class, which is responsible for managing trading sessions,
including starting and ending sessions, tracking session information,
and providing utilities for session management.
"""

from abc import ABC, abstractmethod
from datetime import datetime, time

from pytest import Session


class SessionManager(ABC):
    """
    Abstract base class for trading session management.
    This class defines the interface for managing trading sessions
    """

    @abstractmethod
    def is_trading_hours(self, dt: datetime) -> bool:
        """
        Check if the given datetime is within trading hours.

        Args:
            dt: The datetime to check

        Returns:
            True if the datetime is within trading hours, False otherwise.
        """
        pass

    @abstractmethod
    def should_close_eod(self, dt: datetime) -> bool:
        """
        Check if the given datetime is close to the end of the trading day.

        Args:
            dt: The datetime to check

        Returns:
            True if the datetime is close to the end of the trading day, False otherwise.
        """
        pass

    @abstractmethod
    def should_skip_signal_generation(self, dt: datetime) -> bool:
        """
        Check if signal generation should be skipped at the given datetime.

        Args:
            dt: The datetime to check

        Returns:
            True if signal generation should be skipped, False otherwise.
        """
        pass

    def get_session_name(self) -> str:
        """
        Get the name of the current trading session.

        Returns:
            The name of the current trading session.
        """
        return self.__class__.__name__


class AlwaysOpenSession(SessionManager):
    """
    A simple session manager that treats all times as trading hours.
    This is useful for testing and backtesting without time restrictions.
    """

    def is_trading_hours(self, dt: datetime) -> bool:
        """Always return True, indicating that all times are trading hours."""
        return True

    def should_close_eod(self, dt: datetime) -> bool:
        """Never closes EOD - since no end of day in this session."""
        return False

    def should_skip_signal_generation(self, dt: datetime) -> bool:
        """Never skip signal generation - since all times are valid for trading."""
        return False


class StandardSession(SessionManager):
    """
    A standard session manager that defines typical trading hours (e.g., 9:30 AM to 4:00 PM).
    This is useful for simulating real-world trading conditions.
    """

    def __init__(
        self,
        trading_start_time: str = "09:00:00",
        trading_end_time: str = "16:00:00",
        close_at_eod: bool = True,
    ):
        """
        Initialize the standard session manager.

        Args:
            trading_start_time: The start time of the trading session (e.g., "09:00:00")
            trading_end_time: The end time of the trading session (e.g., "16:00:00")
            close_at_eod: Whether to close positions at the end of the day
        """
        self.trading_start = time.fromisoformat(trading_start_time)
        self.trading_end = time.fromisoformat(trading_end_time)
        self.close_at_eod = close_at_eod

    def is_trading_hours(self, dt: datetime) -> bool:
        """Check if the given datetime is within the defined trading hours."""
        current_time = dt.time()
        return self.trading_start <= current_time < self.trading_end

    def should_close_eod(self, dt: datetime) -> bool:
        """Check if it's time to close positions at the end of the day."""
        if not self.close_at_eod:
            return False

        current_time = dt.time()
        return current_time >= self.trading_end

    def should_skip_signal_generation(self, dt: datetime) -> bool:
        """Skip signal generation outside of trading hours."""
        return not self.is_trading_hours(dt)


class VN30Session(SessionManager):
    """
    Vietnamese VN30 index session manager with specific trading hours and rules.

    VN30 trading hours:
    - Morning session: 9:00 AM to 11:30 AM
    - Afternoon session: 1:00 PM to 3:00 PM
    - No overnight trading, so close at EOD is always True.
    """

    def __init__(
        self,
        close_at_eod: bool = True,
    ):
        """
        Initialize the VN30 session manager.

        Args:
            close_at_eod: Whether to close positions at the end of the day
        """
        self.morning = (time.fromisoformat("09:00:00"), time.fromisoformat("11:30:00"))
        self.afternoon = (
            time.fromisoformat("13:00:00"),
            time.fromisoformat("14:30:00"),
        )
        self.atc_start = time.fromisoformat("14:30:00")
        self.atc_end = time.fromisoformat("15:45:00")

        self.close_at_eod = close_at_eod

    def is_trading_hours(self, dt: datetime) -> bool:
        """Check if within trading hours (including ATC)"""
        dt_time = dt.time()

        return (
            (self.morning[0] <= dt_time < self.morning[1])
            or (self.afternoon[0] <= dt_time < self.afternoon[1])
            or (self.atc_start <= dt_time < self.atc_end)
        )

    def is_atc_session(self, dt: datetime) -> bool:
        """Check if within ATC session"""
        dt_time = dt.time()
        return self.atc_start <= dt_time < self.atc_end

    def should_close_eod(self, dt: datetime) -> bool:
        """Check if it's time to close positions at the end of the day."""
        if not self.close_at_eod:
            return False

        return self.is_atc_session(dt)

    def should_skip_signal_generation(self, dt: datetime) -> bool:
        """Skip signal generation during ATC or outside of trading hours."""
        return self.is_atc_session(dt) or not self.is_trading_hours(dt)
