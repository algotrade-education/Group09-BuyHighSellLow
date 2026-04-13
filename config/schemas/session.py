"""
Single source of truth for session configuration.
"""

from abc import ABC, abstractmethod
from datetime import time
from enum import StrEnum

from src.utils.frequency import parse_frequency_to_minutes


class Session(StrEnum):
    MORNING = "morning"
    AFTERNOON = "afternoon"
    ATC = "atc"
    CLOSED = "closed"


class SessionConfig(ABC):
    """
    Abstract interface for market session configuration.
    Any market (VN30, S&P500, etc.) should implement this interface.
    """

    # Default: no ATC session - subclass can override if needed (e.g., VN30)
    ATC_START: time | None = None
    ATC_END: time | None = None

    def has_atc(self) -> bool:
        """Check if this market has an ATC (At-The-Close) session."""
        return self.ATC_START is not None and self.ATC_END is not None

    # Session boundaries - subclasses must define these as class attributes
    MORNING_START: time
    MORNING_END: time
    AFTERNOON_START: time
    AFTERNOON_END: time
    TRADING_DAYS_PER_YEAR: int

    @abstractmethod
    def get_session(self, current_time: time) -> Session:
        """
        Determine the current session based on the given time.

        Args:
            current_time: The time to check.

        Returns:
            The current session (MORNING, AFTERNOON, ATC, or CLOSED).
        """
        pass

    @abstractmethod
    def is_trading_time(self, current_time: time) -> bool:
        """
        Check if the given time is within any trading session (excluding ATC).

        Args:
            current_time: The time to check.

        Returns:
            True if it's trading time, False otherwise.
        """
        pass

    @abstractmethod
    def is_signal_allowed(self, current_time: time) -> bool:
        """
        Check if signal generation is allowed at the given time.

        Args:
            current_time: The time to check.

        Returns:
            True if signal generation is allowed, False otherwise.
        """
        pass

    @abstractmethod
    def bars_per_year(self, freq_minutes: int | str) -> float:
        """
        Calculate the number of bars per year based on the given frequency.

        Args:
            freq_minutes: The bar frequency as minutes (e.g., 1, 5, 15)
                or timeframe text (e.g., "1H", "1D").

        Returns:
            The estimated number of bars per year.
        """
        pass

    @staticmethod
    def _bars_per_day_from_frequency(
        trading_minutes_per_day: int, freq_minutes: int | str
    ) -> float:
        """Calculate bars per day from frequency.

        Args:
            trading_minutes_per_day: Total trading minutes in a day
            freq_minutes: Frequency as minutes (int) or string ('5min', '1H', etc.)

        Returns:
            Number of bars per day
        """
        minutes = parse_frequency_to_minutes(freq_minutes)
        return trading_minutes_per_day / minutes


class VN30SessionConfig(SessionConfig):
    """
    Defines all trading schedule of VN30 Futures

    Morning:    09:00 - 11:30
    Afternoon:  13:00 - 14:30
    ATC:        14:30 - 14:45
    Closed:     Not mentioned above

    Total trading time: 255 minutes/day

    Notes:
        - ATC (At-the-close): no generate signals - only execute existing orders
        - Bars per day (5min): 30 + 18 + 3 = 51 bars/day (exclude ATC)
    """

    # --- Session Boundaries ---
    MORNING_START: time = time(9, 0)
    MORNING_END: time = time(11, 30)  # Exclusive

    AFTERNOON_START: time = time(13, 0)
    AFTERNOON_END: time = time(14, 30)  # Exclusive

    ATC_START = time(14, 30)
    ATC_END = time(14, 45)  # Exclusive

    # --- Derived constants ---
    # Used for annualize metrics
    TRADING_DAYS_PER_YEAR: int = 252
    BARS_PER_DAY_1MIN: int = 255  # Exclude ATC bars
    BARS_PER_DAY_5MIN: int = 51  # Exclude ATC bars
    BARS_PER_DAY_15MIN: int = 17  # Exclude ATC bars

    def get_session(self, current_time: time) -> Session:
        """
        Determine the current session based on the given time.

        Args:
            current_time (time): The time to check.

        Returns:
            Session: The current session (MORNING, AFTERNOON, ATC, or CLOSED).
        """
        if self.MORNING_START <= current_time < self.MORNING_END:
            return Session.MORNING
        elif self.AFTERNOON_START <= current_time < self.AFTERNOON_END:
            return Session.AFTERNOON
        elif self.ATC_START and self.ATC_END and self.ATC_START <= current_time < self.ATC_END:
            return Session.ATC
        else:
            return Session.CLOSED

    def is_trading_time(self, current_time: time) -> bool:
        """
        Check if the given time is within any trading session (excluding ATC).

        For VN30: ATC (14:30-14:45) is trading time but NOT signal generation time.
        This method returns False for ATC to prevent new signals during close.

        Args:
            current_time (time): The time to check.

        Returns:
            bool: True if it's trading time, False otherwise.
        """
        return self.get_session(current_time) in {Session.MORNING, Session.AFTERNOON}

    def is_signal_allowed(self, current_time: time) -> bool:
        """
        Check if signal generation is allowed at the given time.

        For VN30: Same as is_trading_time() - no new signals during ATC.
        ATC (14:30-14:45) is for executing existing orders only.

        Args:
            current_time (time): The time to check.

        Returns:
            bool: True if signal generation is allowed, False otherwise.
        """
        return self.get_session(current_time) in {Session.MORNING, Session.AFTERNOON}

    def bars_per_year(self, freq_minutes: int | str) -> float:
        """
        Calculate the number of bars per year based on the given frequency.

        Args:
            freq_minutes (int | str): The frequency in minutes (e.g., 1, 5, 15)
                or timeframe text (e.g., "1H", "1D").

        Returns:
            float: The estimated number of bars per year.
        """
        trading_minutes_per_day = 255  # Exclude ATC
        bars_per_day = self._bars_per_day_from_frequency(trading_minutes_per_day, freq_minutes)
        return bars_per_day * self.TRADING_DAYS_PER_YEAR


class SPXSessionConfig(SessionConfig):
    """
    S&P 500 Futures session configuration.

    Trading hours: 9:30 AM - 4:00 PM ET (no lunch break)
    Total trading time: 390 minutes/day
    """

    MORNING_START: time = time(9, 30)
    MORNING_END: time = time(16, 0)  # Exclusive

    # S&P 500 has no afternoon session - use same as morning for interface compliance
    AFTERNOON_START: time = time(16, 0)
    AFTERNOON_END: time = time(16, 0)  # No afternoon session

    TRADING_DAYS_PER_YEAR: int = 252

    def get_session(self, current_time: time) -> Session:
        """
        Determine the current session for S&P 500.

        Args:
            current_time: The time to check.

        Returns:
            Session: MORNING (9:30-16:00) or CLOSED
        """
        if self.MORNING_START <= current_time < self.MORNING_END:
            return Session.MORNING
        else:
            return Session.CLOSED

    def is_trading_time(self, current_time: time) -> bool:
        """Check if within trading hours (9:30-16:00 ET)."""
        return self.get_session(current_time) == Session.MORNING

    def is_signal_allowed(self, current_time: time) -> bool:
        """Signal generation allowed during trading hours."""
        return self.is_trading_time(current_time)

    def bars_per_year(self, freq_minutes: int | str) -> float:
        """
        Calculate bars per year for S&P 500.

        Args:
            freq_minutes: Frequency in minutes (e.g., 1, 5, 15)
                or timeframe text (e.g., "1H", "1D").

        Returns:
            Estimated number of bars per year
        """
        trading_minutes_per_day = 390  # 9:30 AM - 4:00 PM
        bars_per_day = self._bars_per_day_from_frequency(trading_minutes_per_day, freq_minutes)
        return bars_per_day * self.TRADING_DAYS_PER_YEAR
