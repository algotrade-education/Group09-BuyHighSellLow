"""
Base class for intraday trading strategies.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Any

from config.schemas.session import Session, SessionConfig, VN30SessionConfig
from src.strategy.base import StrategyBase

logger = logging.getLogger(__name__)


class IntradayStrategy(StrategyBase, ABC):
    """
    Base class for intraday trading strategies.
    Inherits from StrategyBase.

    Session state reset when:
    - New trading day starts (e.g. at 9:30 AM).
    - New session starts (e.g. morning to afternoon).
    - Detected data gap (e.g. missing bars > 60 minutes).
    """

    # Subclass override if needed more fields from bar
    REQUIRED_FIELDS: list[str] = ["datetime", "open", "high", "low", "close"]

    def __init__(
        self,
        name: str,
        session: SessionConfig | None = None,
        gap_threshold_minutes: int = 60,
    ):
        super().__init__(name)
        self._session_cfg = session if session is not None else VN30SessionConfig()
        self._gap_threshold = gap_threshold_minutes

        # Session Tracking
        self._current_date: date | None = None
        self._current_session: Session | None = None
        self._last_bar_dt: datetime | None = None

    # --- Session Helpers ---

    def _get_session(self, dt: datetime) -> Session:
        """Determine session based on datetime."""
        return self._session_cfg.get_session(dt.time())

    def _is_signal_allowed(self, dt: datetime) -> bool:
        """Check if signal generation is allowed at this datetime."""
        return self._session_cfg.is_signal_allowed(dt.time())

    def _update_session_state(self, dt: datetime, session: Session) -> bool:
        """
        Check if session state needs reset. Returns True if reset occurred.

        Returns:
            bool: True if session state was reset, False otherwise.
        """
        current_date = dt.date()
        is_new_day = self._current_date != current_date
        is_new_session = self._current_session != session

        # Detect data gap
        has_gap = False
        gap_minutes = 0.0
        if self._last_bar_dt is not None:
            gap_minutes = (dt - self._last_bar_dt).total_seconds() / 60
            has_gap = gap_minutes > self._gap_threshold

        self._last_bar_dt = dt

        if is_new_day or is_new_session or has_gap:
            if has_gap and not is_new_day and not is_new_session:
                logger.warning(
                    "%s: Data gap detected (%.0f minutes) at %s. Resetting session state.",
                    self.name,
                    gap_minutes,
                    dt,
                )

            self._current_date = current_date
            self._current_session = session
            self._on_session_reset(session)
            return True

        return False

    @abstractmethod
    def _on_session_reset(self, session: Session) -> None:
        """
        Hook for subclasses to reset any session-specific state.
        Called when a new day/session starts or a data gap is detected.
        """
        return None

    # --- State serialization ---

    def save_state(self) -> dict[str, Any]:
        return {
            "current_date": self._current_date.isoformat() if self._current_date else None,
            "current_session": self._current_session.name if self._current_session else None,
            "last_bar_dt": self._last_bar_dt.isoformat() if self._last_bar_dt else None,
            **self._get_strategy_state(),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        d = state.get("current_date")
        self._current_date = date.fromisoformat(d) if d else None

        s = state.get("current_session")
        self._current_session = Session(s) if s else None

        lbd = state.get("last_bar_dt")
        self._last_bar_dt = datetime.fromisoformat(lbd) if lbd else None

        self._set_strategy_state(state)

    def reset(self) -> None:
        """
        Reset all session tracking state back to initial conditions.
        Called by engine before each backtest or paper trading session.

        This clears all session tracking and calls _on_session_reset()
        to allow subclasses to reset their own state.
        """
        self._current_date = None
        self._current_session = None
        self._last_bar_dt = None
        # Call subclass hook to reset strategy-specific state
        # Pass CLOSED to indicate we're in a reset state (no active session)
        self._on_session_reset(Session.CLOSED)

    def _get_strategy_state(self) -> dict[str, Any]:
        """Subclasses can override to add more state fields for serialization."""
        return {}

    def _set_strategy_state(self, state: dict[str, Any]) -> None:
        """Subclasses can override to load additional state fields."""
        pass
