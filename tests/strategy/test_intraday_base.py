"""Tests for IntradayStrategy base class."""

from datetime import datetime
from typing import Any

from config.schemas.session import Session, SessionConfig
from src.data.indicators.registry import IndicatorRegistry
from src.strategy.base import PositionSnapshot
from src.strategy.intraday_base import IntradayStrategy
from src.strategy.signal import Signal, TradeSignal


class DummyIntradayStrategy(IntradayStrategy):
    """Dummy intraday strategy for testing."""

    def __init__(self, name: str = "DummyIntraday", session: SessionConfig | None = None):
        super().__init__(name, session, gap_threshold_minutes=60)
        self.reset_count = 0
        self.last_reset_session: Session | None = None

    def _on_session_reset(self, session: Session) -> None:
        self.reset_count += 1
        self.last_reset_session = session

    def generate_signal(
        self,
        bar: dict[str, Any],
        position: PositionSnapshot | None = None,
        is_warmup: bool = False,
    ) -> TradeSignal:
        return TradeSignal(Signal.HOLD)

    @classmethod
    def build_registry(cls, **params: Any) -> IndicatorRegistry:
        return IndicatorRegistry()


class TestIntradayStrategy:
    """Test IntradayStrategy base class."""

    def test_init_default_session(self):
        """Test initialization with default VN30 session."""
        strategy = DummyIntradayStrategy()
        assert strategy.name == "DummyIntraday"
        assert strategy._session_cfg is not None
        assert strategy._gap_threshold == 60

    def test_init_custom_session(self):
        """Test initialization with custom session config."""
        from config.schemas.session import VN30SessionConfig

        custom = VN30SessionConfig()
        strategy = DummyIntradayStrategy(session=custom)
        assert strategy._session_cfg == custom

    def test_get_session(self):
        """Test _get_session method."""
        strategy = DummyIntradayStrategy()

        # Morning session
        dt_morning = datetime(2024, 1, 1, 9, 30)
        assert strategy._get_session(dt_morning) == Session.MORNING

        # Afternoon session
        dt_afternoon = datetime(2024, 1, 1, 13, 30)
        assert strategy._get_session(dt_afternoon) == Session.AFTERNOON

        # Closed
        dt_closed = datetime(2024, 1, 1, 8, 0)
        assert strategy._get_session(dt_closed) == Session.CLOSED

    def test_is_signal_allowed(self):
        """Test _is_signal_allowed method."""
        strategy = DummyIntradayStrategy()

        # Allowed during morning session
        dt_morning = datetime(2024, 1, 1, 9, 30)
        assert strategy._is_signal_allowed(dt_morning)

        # Allowed during afternoon session
        dt_afternoon = datetime(2024, 1, 1, 13, 30)
        assert strategy._is_signal_allowed(dt_afternoon)

        # Not allowed when closed
        dt_closed = datetime(2024, 1, 1, 8, 0)
        assert not strategy._is_signal_allowed(dt_closed)

    def test_update_session_state_new_day(self):
        """Test session state update on new day."""
        strategy = DummyIntradayStrategy()

        # First bar
        dt1 = datetime(2024, 1, 1, 9, 30)
        session1 = strategy._get_session(dt1)
        reset1 = strategy._update_session_state(dt1, session1)
        assert reset1  # First bar triggers reset
        assert strategy.reset_count == 1
        assert strategy.last_reset_session == Session.MORNING

        # Same day, same session
        dt2 = datetime(2024, 1, 1, 9, 35)
        session2 = strategy._get_session(dt2)
        reset2 = strategy._update_session_state(dt2, session2)
        assert not reset2  # No reset
        assert strategy.reset_count == 1

        # New day
        dt3 = datetime(2024, 1, 2, 9, 30)
        session3 = strategy._get_session(dt3)
        reset3 = strategy._update_session_state(dt3, session3)
        assert reset3  # New day triggers reset
        assert strategy.reset_count == 2

    def test_update_session_state_new_session(self):
        """Test session state update on new session."""
        strategy = DummyIntradayStrategy()

        # Morning session
        dt1 = datetime(2024, 1, 1, 9, 30)
        session1 = strategy._get_session(dt1)
        strategy._update_session_state(dt1, session1)
        assert strategy.reset_count == 1

        # Same morning session
        dt2 = datetime(2024, 1, 1, 10, 0)
        session2 = strategy._get_session(dt2)
        strategy._update_session_state(dt2, session2)
        assert strategy.reset_count == 1  # No reset

        # Afternoon session (same day)
        dt3 = datetime(2024, 1, 1, 13, 0)
        session3 = strategy._get_session(dt3)
        strategy._update_session_state(dt3, session3)
        assert strategy.reset_count == 2  # New session triggers reset
        assert strategy.last_reset_session == Session.AFTERNOON

    def test_update_session_state_data_gap(self):
        """Test session state update on data gap."""
        strategy = DummyIntradayStrategy()

        # First bar
        dt1 = datetime(2024, 1, 1, 9, 30)
        session1 = strategy._get_session(dt1)
        strategy._update_session_state(dt1, session1)
        assert strategy.reset_count == 1

        # Normal bar (5 minutes later)
        dt2 = datetime(2024, 1, 1, 9, 35)
        session2 = strategy._get_session(dt2)
        strategy._update_session_state(dt2, session2)
        assert strategy.reset_count == 1  # No reset

        # Data gap (70 minutes later, exceeds 60 minute threshold)
        dt3 = datetime(2024, 1, 1, 10, 45)
        session3 = strategy._get_session(dt3)
        strategy._update_session_state(dt3, session3)
        assert strategy.reset_count == 2  # Gap triggers reset

    def test_reset(self):
        """Test reset method."""
        strategy = DummyIntradayStrategy()

        # Set some state
        dt = datetime(2024, 1, 1, 9, 30)
        session = strategy._get_session(dt)
        strategy._update_session_state(dt, session)
        assert strategy._current_date is not None
        assert strategy._current_session is not None
        assert strategy._last_bar_dt is not None

        # Reset
        strategy.reset()
        assert strategy._current_date is None
        assert strategy._current_session is None
        assert strategy._last_bar_dt is None
        assert strategy.last_reset_session == Session.CLOSED

    def test_save_and_load_state(self):
        """Test state serialization."""
        strategy = DummyIntradayStrategy()

        # Set some state
        dt = datetime(2024, 1, 1, 9, 30)
        session = strategy._get_session(dt)
        strategy._update_session_state(dt, session)

        # Save state
        state = strategy.save_state()
        assert state["current_date"] == "2024-01-01"
        assert state["current_session"] == "morning"  # Session.MORNING.value = "morning"
        assert state["last_bar_dt"] == dt.isoformat()

        # Create new strategy and load state
        strategy2 = DummyIntradayStrategy()
        strategy2.load_state(state)
        assert strategy2._current_date == strategy._current_date
        assert strategy2._current_session == strategy._current_session
        assert strategy2._last_bar_dt == strategy._last_bar_dt

    def test_save_state_empty(self):
        """Test save_state with no state set."""
        strategy = DummyIntradayStrategy()
        state = strategy.save_state()
        assert state["current_date"] is None
        assert state["current_session"] is None
        assert state["last_bar_dt"] is None

    def test_load_state_empty(self):
        """Test load_state with empty state."""
        strategy = DummyIntradayStrategy()
        strategy.load_state({})
        assert strategy._current_date is None
        assert strategy._current_session is None
        assert strategy._last_bar_dt is None

    def test_get_strategy_state_default(self):
        """Test _get_strategy_state returns empty dict by default."""
        strategy = DummyIntradayStrategy()
        state = strategy._get_strategy_state()
        assert state == {}

    def test_set_strategy_state_default(self):
        """Test _set_strategy_state does nothing by default."""
        strategy = DummyIntradayStrategy()
        strategy._set_strategy_state({"key": "value"})  # Should not raise
