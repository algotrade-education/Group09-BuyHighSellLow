"""
VN30 futures session manager with ATC (At-The-Close) handling.
"""

from __future__ import annotations

import logging
from datetime import datetime

from config.schemas.session import Session, VN30SessionConfig
from src.engine.session.base import SessionManager

logger = logging.getLogger(__name__)


class VN30Session(SessionManager):
    """
    VN30 futures session manager.

    Session schedule:
    - Morning: 09:00 - 11:30
    - Afternoon: 13:00 - 14:30
    - ATC (At-The-Close): 14:30 - 14:45

    Key behaviors:
    - is_trading_hours() excludes ATC (signal generation not allowed)
    - is_atc() checks if currently in ATC period
    - should_skip_signal() skips ATC and closed periods
    - Supports entry cutoff and preclose logic
    """

    def __init__(
        self,
        close_at_eod: bool = True,
        session_cfg: type[VN30SessionConfig] = VN30SessionConfig,
    ) -> None:
        """
        Args:
            close_at_eod: If True, force close positions after ATC ends
            session_cfg: Session configuration class (default: VN30SessionConfig)
        """

        self.close_at_eod = close_at_eod
        self._cfg = session_cfg()
        logger.debug("VN30Session initialized: close_at_eod=%s", close_at_eod)

    def is_trading_hours(self, dt: datetime) -> bool:
        """
        Check if within trading hours (morning or afternoon session).

        Note: ATC period is NOT considered trading hours for signal generation.

        Args:
            dt: Datetime to check

        Returns:
            True if in morning or afternoon session, False otherwise
        """
        if dt.weekday() >= 5:  # Weekend
            return False

        session = self._cfg.get_session(dt.time())
        return session in (Session.MORNING, Session.AFTERNOON)

    def is_atc(self, dt: datetime) -> bool:
        """
        Check if currently in ATC (At-The-Close) period.

        Args:
            dt: Datetime to check

        Returns:
            True if in ATC period (14:30-14:45), False otherwise
        """
        return self._cfg.get_session(dt.time()) == Session.ATC

    def should_close_eod(self, dt: datetime) -> bool:
        """
        Check if positions should be closed at end of day.

        Args:
            dt: Current datetime

        Returns:
            True if after ATC end and close_at_eod is enabled
        """
        if not self.close_at_eod:
            return False
        if self._cfg.ATC_END is None:
            return False

        t = dt.time()
        should_close = t >= self._cfg.ATC_END

        if should_close:
            logger.debug("EOD close triggered at %s", dt)

        return should_close

    def should_skip_signal(self, dt: datetime) -> bool:
        """
        Check if signal generation should be skipped.

        Signals are skipped during:
        - ATC period (14:30-14:45)
        - Outside trading hours
        - Weekends

        Args:
            dt: Current datetime

        Returns:
            True if should skip signal generation
        """
        return not self.is_trading_hours(dt)

    def is_entry_blocked(
        self,
        dt: datetime,
        cutoff_seconds: float = 0.0,
        allow_late: bool = False,
    ) -> bool:
        """
        Check if new entries should be blocked.

        Entries are blocked when too close to session end (within cutoff_seconds).

        Args:
            dt: Current datetime
            cutoff_seconds: Minimum seconds before session end to allow entry
            allow_late: If True, ignore cutoff and allow late entries

        Returns:
            True if entry should be blocked
        """
        if allow_late or cutoff_seconds <= 0:
            return False

        seconds = self._seconds_to_session_end(dt)

        if seconds is None:
            # Outside session - block entry
            return True

        is_blocked = seconds <= cutoff_seconds

        if is_blocked:
            logger.debug(
                "Entry blocked: %s seconds to session end (cutoff: %s)",
                seconds,
                cutoff_seconds,
            )

        return is_blocked

    def get_force_close_reason(self, dt: datetime, preclose_seconds: float = 0.0) -> str | None:
        """
        Get reason for forced position close.

        Positions are force closed when approaching session end (within preclose_seconds).

        Args:
            dt: Current datetime
            preclose_seconds: Seconds before session end to force close

        Returns:
            Reason string if should force close, None otherwise
        """
        if preclose_seconds <= 0:
            return None

        seconds = self._seconds_to_session_end(dt)

        if seconds is not None and seconds <= preclose_seconds:
            reason = f"Session preclose ({seconds:.0f}s remaining)"
            logger.info("Force close triggered: %s at %s", reason, dt)
            return reason

        return None

    def _seconds_to_session_end(self, dt: datetime) -> float | None:
        """
        Calculate seconds remaining until current session ends.

        Args:
            dt: Current datetime

        Returns:
            Seconds to session end, or None if outside any session
        """
        cfg = self._cfg
        t = dt.time()

        # Morning session
        if cfg.MORNING_START <= t < cfg.MORNING_END:
            end = dt.replace(
                hour=cfg.MORNING_END.hour,
                minute=cfg.MORNING_END.minute,
                second=0,
                microsecond=0,
            )
            return max(0.0, (end - dt).total_seconds())

        # Afternoon session
        if cfg.AFTERNOON_START <= t < cfg.AFTERNOON_END:
            end = dt.replace(
                hour=cfg.AFTERNOON_END.hour,
                minute=cfg.AFTERNOON_END.minute,
                second=0,
                microsecond=0,
            )
            return max(0.0, (end - dt).total_seconds())

        # ATC session
        if (
            cfg.ATC_START is not None
            and cfg.ATC_END is not None
            and cfg.ATC_START <= t < cfg.ATC_END
        ):
            end = dt.replace(
                hour=cfg.ATC_END.hour,
                minute=cfg.ATC_END.minute,
                second=0,
                microsecond=0,
            )
            return max(0.0, (end - dt).total_seconds())

        return None

    def get_session_name(self, dt: datetime) -> str:
        """
        Get human-readable session name.

        Args:
            dt: Datetime to check

        Returns:
            Session name (e.g., "MORNING", "AFTERNOON", "ATC", "CLOSED")
        """
        return self._cfg.get_session(dt.time()).value.upper()

    def __repr__(self) -> str:
        return f"VN30Session(close_at_eod={self.close_at_eod})"
