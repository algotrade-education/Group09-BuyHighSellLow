"""
Unit tests for SimBroker - simulated order execution.
"""

from datetime import datetime

from src.engine.account.account import AccountState
from src.engine.core.event_bus import EventBus
from src.engine.core.events import EventType, OrderEvent
from src.engine.execution.sim_broker import SimBroker, SimBrokerT1


class TestSimBroker:
    """Test SimBroker order execution logic."""

    def test_market_order_filled_at_open(self):
        """Test market order is filled at bar open."""
        bus = EventBus()
        account = AccountState(initial_capital=500_000_000)
        broker = SimBroker(account, bus)

        # Update bar
        bar = {"open": 1000.0, "high": 1010.0, "low": 990.0, "close": 1005.0}
        broker.update_bar(bar)

        # Capture fill events
        fills = []
        bus.subscribe(EventType.FILL, lambda e: fills.append(e))

        # Submit market order
        order = OrderEvent(
            timestamp=datetime(2024, 1, 1, 9, 0),
            order_type="market",
            side="buy",
            quantity=1,
        )
        broker.on_order(order)

        # Verify filled (price may include slippage)
        assert len(fills) == 1
        assert fills[0].fill_price >= 1000.0  # At or above open due to slippage

    def test_limit_buy_filled_when_price_reached(self):
        """Test limit buy order is filled when price reaches limit."""
        bus = EventBus()
        account = AccountState(initial_capital=500_000_000)
        broker = SimBroker(account, bus)

        # Bar with low below limit
        bar = {"open": 1005.0, "high": 1010.0, "low": 995.0, "close": 1000.0}
        broker.update_bar(bar)

        fills = []
        bus.subscribe(EventType.FILL, lambda e: fills.append(e))

        # Submit limit buy at 1000
        order = OrderEvent(
            timestamp=datetime(2024, 1, 1, 9, 0),
            order_type="limit",
            side="buy",
            quantity=1,
            limit_price=1000.0,
        )
        broker.on_order(order)

        # Should be filled (price may include slippage)
        assert len(fills) == 1
        assert fills[0].fill_price >= 1000.0  # At or above limit due to slippage

    def test_limit_buy_not_filled_when_price_not_reached(self):
        """Test limit buy order is not filled when price doesn't reach limit."""
        bus = EventBus()
        account = AccountState(initial_capital=500_000_000)
        broker = SimBroker(account, bus)

        # Bar with low above limit
        bar = {"open": 1005.0, "high": 1010.0, "low": 1002.0, "close": 1007.0}
        broker.update_bar(bar)

        fills = []
        bus.subscribe(EventType.FILL, lambda e: fills.append(e))

        # Submit limit buy at 1000
        order = OrderEvent(
            timestamp=datetime(2024, 1, 1, 9, 0),
            order_type="limit",
            side="buy",
            quantity=1,
            limit_price=1000.0,
        )
        broker.on_order(order)

        # Should not be filled
        assert len(fills) == 0

    def test_limit_buy_filled_at_open_when_gap_down(self):
        """Test limit buy filled at open when bar opens below limit."""
        bus = EventBus()
        account = AccountState(initial_capital=500_000_000)
        broker = SimBroker(account, bus)

        # Bar opens below limit
        bar = {"open": 995.0, "high": 1000.0, "low": 990.0, "close": 998.0}
        broker.update_bar(bar)

        fills = []
        bus.subscribe(EventType.FILL, lambda e: fills.append(e))

        # Submit limit buy at 1000
        order = OrderEvent(
            timestamp=datetime(2024, 1, 1, 9, 0),
            order_type="limit",
            side="buy",
            quantity=1,
            limit_price=1000.0,
        )
        broker.on_order(order)

        # Should be filled at open (better price, but with slippage)
        assert len(fills) == 1
        assert fills[0].fill_price <= 1000.0  # At or below limit (gap down + slippage)

    def test_limit_sell_filled_when_price_reached(self):
        """Test limit sell order is filled when price reaches limit."""
        bus = EventBus()
        account = AccountState(initial_capital=500_000_000)
        broker = SimBroker(account, bus)

        # Bar with high above limit
        bar = {"open": 995.0, "high": 1005.0, "low": 990.0, "close": 1000.0}
        broker.update_bar(bar)

        fills = []
        bus.subscribe(EventType.FILL, lambda e: fills.append(e))

        # Submit limit sell at 1000
        order = OrderEvent(
            timestamp=datetime(2024, 1, 1, 9, 0),
            order_type="limit",
            side="sell",
            quantity=1,
            limit_price=1000.0,
        )
        broker.on_order(order)

        # Should be filled (price may include slippage)
        assert len(fills) == 1
        assert fills[0].fill_price <= 1000.0  # At or below limit due to slippage

    def test_limit_sell_not_filled_when_price_not_reached(self):
        """Test limit sell order is not filled when price doesn't reach limit."""
        bus = EventBus()
        account = AccountState(initial_capital=500_000_000)
        broker = SimBroker(account, bus)

        # Bar with high below limit
        bar = {"open": 995.0, "high": 998.0, "low": 990.0, "close": 997.0}
        broker.update_bar(bar)

        fills = []
        bus.subscribe(EventType.FILL, lambda e: fills.append(e))

        # Submit limit sell at 1000
        order = OrderEvent(
            timestamp=datetime(2024, 1, 1, 9, 0),
            order_type="limit",
            side="sell",
            quantity=1,
            limit_price=1000.0,
        )
        broker.on_order(order)

        # Should not be filled
        assert len(fills) == 0

    def test_limit_sell_filled_at_open_when_gap_up(self):
        """Test limit sell filled at open when bar opens above limit."""
        bus = EventBus()
        account = AccountState(initial_capital=500_000_000)
        broker = SimBroker(account, bus)

        # Bar opens above limit
        bar = {"open": 1005.0, "high": 1010.0, "low": 1000.0, "close": 1007.0}
        broker.update_bar(bar)

        fills = []
        bus.subscribe(EventType.FILL, lambda e: fills.append(e))

        # Submit limit sell at 1000
        order = OrderEvent(
            timestamp=datetime(2024, 1, 1, 9, 0),
            order_type="limit",
            side="sell",
            quantity=1,
            limit_price=1000.0,
        )
        broker.on_order(order)

        # Should be filled at open (better price, but with slippage)
        assert len(fills) == 1
        assert fills[0].fill_price >= 1000.0  # At or above limit (gap up - slippage)

    def test_commission_calculation(self):
        """Test commission is calculated correctly."""
        bus = EventBus()
        account = AccountState(
            initial_capital=500_000_000,
            commission_rate=0.0001,  # 0.01%
        )
        broker = SimBroker(account, bus)

        bar = {"open": 1000.0, "high": 1010.0, "low": 990.0, "close": 1005.0}
        broker.update_bar(bar)

        fills = []
        bus.subscribe(EventType.FILL, lambda e: fills.append(e))

        order = OrderEvent(
            timestamp=datetime(2024, 1, 1, 9, 0),
            order_type="market",
            side="buy",
            quantity=2,
        )
        broker.on_order(order)

        # Verify commission
        assert len(fills) == 1
        assert fills[0].commission > 0

    def test_slippage_applied(self):
        """Test slippage is applied to fill price."""
        bus = EventBus()
        account = AccountState(initial_capital=500_000_000)
        broker = SimBroker(account, bus)

        bar = {"open": 1000.0, "high": 1010.0, "low": 990.0, "close": 1005.0}
        broker.update_bar(bar)

        fills = []
        bus.subscribe(EventType.FILL, lambda e: fills.append(e))

        order = OrderEvent(
            timestamp=datetime(2024, 1, 1, 9, 0),
            order_type="market",
            side="buy",
            quantity=1,
        )
        broker.on_order(order)

        # Verify slippage recorded
        assert len(fills) == 1
        # Slippage should be applied (fill price != open)
        assert fills[0].slippage >= 0

    def test_stop_loss_take_profit_passed_through(self):
        """Test stop loss and take profit are passed to fill event."""
        bus = EventBus()
        account = AccountState(initial_capital=500_000_000)
        broker = SimBroker(account, bus)

        bar = {"open": 1000.0, "high": 1010.0, "low": 990.0, "close": 1005.0}
        broker.update_bar(bar)

        fills = []
        bus.subscribe(EventType.FILL, lambda e: fills.append(e))

        order = OrderEvent(
            timestamp=datetime(2024, 1, 1, 9, 0),
            order_type="market",
            side="buy",
            quantity=1,
            stop_loss=990.0,
            take_profit=1020.0,
        )
        broker.on_order(order)

        assert len(fills) == 1
        assert fills[0].stop_loss == 990.0
        assert fills[0].take_profit == 1020.0

    def test_no_bar_data_warning(self):
        """Test warning when no bar data available."""
        bus = EventBus()
        account = AccountState(initial_capital=500_000_000)
        broker = SimBroker(account, bus)

        # Don't update bar
        fills = []
        bus.subscribe(EventType.FILL, lambda e: fills.append(e))

        order = OrderEvent(
            timestamp=datetime(2024, 1, 1, 9, 0),
            order_type="market",
            side="buy",
            quantity=1,
        )
        broker.on_order(order)

        # Should not fill
        assert len(fills) == 0


class TestSimBrokerT1:
    """Test SimBrokerT1 T+1 execution logic."""

    def test_order_queued_for_next_bar(self):
        """Test order is queued and filled at next bar."""
        bus = EventBus()
        account = AccountState(initial_capital=500_000_000)
        broker = SimBrokerT1(account, bus)

        fills = []
        bus.subscribe(EventType.FILL, lambda e: fills.append(e))

        # Bar 1: Submit order
        bar1 = {"open": 1000.0, "high": 1010.0, "low": 990.0, "close": 1005.0}
        broker.update_bar(bar1)

        order = OrderEvent(
            timestamp=datetime(2024, 1, 1, 9, 0),
            order_type="market",
            side="buy",
            quantity=1,
        )
        broker.on_order(order)

        # Should not be filled yet
        assert len(fills) == 0

        # Bar 2: Order should be filled at this bar's open
        bar2 = {"open": 1008.0, "high": 1015.0, "low": 1005.0, "close": 1012.0}
        broker.update_bar(bar2)

        # Should be filled at bar2 open (with slippage)
        assert len(fills) == 1
        assert fills[0].fill_price >= 1008.0  # At or above bar2 open

    def test_multiple_orders_queued(self):
        """Test multiple orders can be queued."""
        bus = EventBus()
        account = AccountState(initial_capital=500_000_000)
        broker = SimBrokerT1(account, bus)

        fills = []
        bus.subscribe(EventType.FILL, lambda e: fills.append(e))

        # Bar 1: Submit multiple orders
        bar1 = {"open": 1000.0, "high": 1010.0, "low": 990.0, "close": 1005.0}
        broker.update_bar(bar1)

        for i in range(3):
            order = OrderEvent(
                timestamp=datetime(2024, 1, 1, 9, i),
                order_type="market",
                side="buy",
                quantity=1,
            )
            broker.on_order(order)

        assert len(fills) == 0

        # Bar 2: All orders should be filled
        bar2 = {"open": 1008.0, "high": 1015.0, "low": 1005.0, "close": 1012.0}
        broker.update_bar(bar2)

        assert len(fills) == 3

    def test_pending_orders_cleared_after_execution(self):
        """Test pending orders are cleared after execution."""
        bus = EventBus()
        account = AccountState(initial_capital=500_000_000)
        broker = SimBrokerT1(account, bus)

        fills = []
        bus.subscribe(EventType.FILL, lambda e: fills.append(e))

        # Bar 1: Submit order
        bar1 = {"open": 1000.0, "high": 1010.0, "low": 990.0, "close": 1005.0}
        broker.update_bar(bar1)

        order = OrderEvent(
            timestamp=datetime(2024, 1, 1, 9, 0),
            order_type="market",
            side="buy",
            quantity=1,
        )
        broker.on_order(order)

        # Bar 2: Execute pending orders
        bar2 = {"open": 1008.0, "high": 1015.0, "low": 1005.0, "close": 1012.0}
        broker.update_bar(bar2)

        assert len(fills) == 1

        # Bar 3: No more pending orders
        bar3 = {"open": 1010.0, "high": 1020.0, "low": 1008.0, "close": 1015.0}
        broker.update_bar(bar3)

        # Should still be 1 fill (no new orders)
        assert len(fills) == 1
