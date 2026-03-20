"""
Single source of truth for session configuration.
"""

from datetime import time
from enum import StrEnum


class Session(StrEnum):
    MORNING = "morning"
    AFTERNOON = "afternoon"
    ATC = "atc"
    CLOSED = "closed"


class VN30SessionConfig:
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

    @classmethod
    def get_session(cls, current_time: time) -> Session:
        """
        Determine the current session based on the given time.

        Args:
            current_time (time): The time to check.

        Returns:
            Session: The current session (MORNING, AFTERNOON, ATC, or CLOSED).
        """
        if cls.MORNING_START <= current_time < cls.MORNING_END:
            return Session.MORNING
        elif cls.AFTERNOON_START <= current_time < cls.AFTERNOON_END:
            return Session.AFTERNOON
        elif cls.ATC_START <= current_time < cls.ATC_END:
            return Session.ATC
        else:
            return Session.CLOSED

    @classmethod
    def is_trading_time(cls, current_time: time) -> bool:
        """
        Check if the given time is within any trading session (excluding ATC).

        Args:
            current_time (time): The time to check.

        Returns:
            bool: True if it's trading time, False otherwise.
        """
        return cls.get_session(current_time) in {Session.MORNING, Session.AFTERNOON}

    @classmethod
    def is_signal_allowed(cls, current_time: time) -> bool:
        """
        Check if signal generation is allowed at the given time.

        Args:
            current_time (time): The time to check.

        Returns:
            bool: True if signal generation is allowed, False otherwise.
        """
        return cls.get_session(current_time) in {Session.MORNING, Session.AFTERNOON}

    @classmethod
    def bars_per_year(cls, freq_minutes: int) -> int:
        """
        Calculate the number of bars per year based on the given frequency.

        Args:
            freq_minutes (int): The frequency in minutes (e.g., 1, 5, 15).

        Returns:
            int: The estimated number of bars per year.

        Raises:
            ValueError: If freq_minutes is not a valid divisor of trading minutes per day.
        """
        trading_minutes_per_day = 255  # Exclude ATC
        if trading_minutes_per_day % freq_minutes != 0:
            raise ValueError(
                f"freq_minutes={freq_minutes} không chia đều vào trading_minutes_per_day={trading_minutes_per_day}"
            )
        bars_per_day = trading_minutes_per_day // freq_minutes
        return bars_per_day * cls.TRADING_DAYS_PER_YEAR
