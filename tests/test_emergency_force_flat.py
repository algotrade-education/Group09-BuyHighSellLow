
import pytest
from datetime import datetime, time
from unittest.mock import MagicMock, patch
from src.paper.engine import PaperTrader
from src.strategy.base import Strategy, TradeSignal, Signal
from src.engine.session_manager import VN30Session

class MockStrategy(Strategy):
    def generate_signal(self, bar, current_position=None, is_warmup=False):
        # Mô phỏng tình huống của USER: Strategy báo HOLD vì hết lượt
        return TradeSignal(signal=Signal.HOLD, reason="Session trade limit reached (1/1)")

def test_force_flat_bypasses_strategy_limit():
    """
    Test kiểm chứng: Lệnh Force Flat PHẢI chạy khi đến giờ, 
    kể cả khi Strategy báo HOLD hoặc hết lượt.
    """
    # 1. Setup cấu hình giống của USER
    config = {
        "strategy": {"atr_period": 14},
        "risk": {
            "force_flat_on_session_close": True,
            "force_flat_preclose_seconds": 60, # Đóng trước 60s (14:29)
        }
    }
    
    symbol = "HNXDS:VN30F2603"
    mock_client = MagicMock()
    mock_redis = MagicMock()
    strategy = MockStrategy("TestORB")
    
    # 2. Khởi tạo Engine
    engine = PaperTrader(
        strategy=strategy,
        symbol=symbol,
        config=config,
        client=mock_client,
        redis_client=mock_redis,
        dry_run=False
    )
    # Giả lập đã nhận được giá thị trường
    engine._last_close = 1845.0
    
    # 3. GIẢ LẬP TÌNH HUỐNG: Đang có vị thế LONG (để không bị dính lỗi FLAT)
    engine._tracker.record_open(
        fill_price=1500.0,
        qty=1,
        side="LONG",
        timestamp=datetime.now()
    )
    assert not engine._tracker.is_flat
    
    # 4. GIẢ LẬP THỜI GIAN: 14:29:00 (Đúng thời điểm Force Flat 60s)
    test_time = datetime(2026, 3, 13, 14, 29, 0)
    
    # Mock hàm place_order của client để xem nó có bị gọi không
    mock_client.place_order.return_value = "order_123"
    
    print(f"\n[TEST] Current time: {test_time.strftime('%H:%M:%S')}")
    print(f"[TEST] Position before: {engine._tracker.position.side}")
    
    # 5. CHẠY LOGIC KIỂM TRA CỦA ENGINE
    engine._maybe_force_flat_by_clock(test_time)
    
    # 6. KIỂM CHỨNG (ASSERT)
    # Lệnh place_order PHẢI được gọi với side="SELL" để đóng vị thế LONG
    call_args = mock_client.place_order.call_args
    if call_args:
        print(f"[TEST] Success! place_order was called with: {call_args}")
    else:
        print("[TEST] Failed! place_order was NOT called.")

    assert mock_client.place_order.called, "Engine phải gửi lệnh Bán dù Strategy báo HOLD!"
    assert mock_client.place_order.call_args[1]['side'] == "SELL"
    assert "Preclose" in engine._order_mgr._pending_exits[mock_client.place_order.return_value]

if __name__ == "__main__":
    # Chạy test thủ công nếu không dùng pytest
    try:
        test_force_flat_bypasses_strategy_limit()
        print("\n===> KẾT QUẢ: TEST PASS! Engine hoạt động đúng lý thuyết.")
    except Exception as e:
        print(f"\n===> KẾT QUẢ: TEST FAIL! {e}")
