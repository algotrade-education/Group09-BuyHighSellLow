#!/usr/bin/env python3
"""
Trading Toolkit - Một script tổng hợp cho các tình huống khẩn cấp.
Dùng để:
1. Huỷ toàn bộ lệnh đang chờ (Cancel All)
2. Bán tháo toàn bộ danh mục (Panic Sell / Liquidate)
3. Đặt lệnh nhanh (Quick Order)
4. Kiểm tra trạng thái tài khoản (Status)

"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.paper.bootstrap import build_broker_client

# Thêm project root vào path
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TradingToolkit")

# --- CẤU HÌNH NHANH (Cho việc sửa tay trong code) ---
DEFAULT_SYMBOL = "HNXDS:VN30F2605"
DEFAULT_SIDE = "SHORT"  # "BUY" hoặc "SELL"
DEFAULT_QTY = 5
DEFAULT_PRICE = 2016  # Giá giới hạn (LIMIT)
DEFAULT_TYPE = "LIMIT"  # "LIMIT" hoặc "MARKET"
# ---------------------------------------------------


def get_active_orders(client: Any) -> list[dict]:
    """Lấy danh sách các lệnh đang hoạt động."""
    today = datetime.now().strftime("%Y-%m-%d")
    response = client.get_orders(today, today)
    if not response.get("success"):
        logger.error(f"Lỗi khi lấy danh sách lệnh: {response.get('error')}")
        return []

    orders = response.get("items", [])
    # 0: New, 1: Partial, A: Pending New, E: Pending Replace
    return [o for o in orders if str(o.get("ordStatus")) in ("0", "1", "A", "E")]


def cancel_all(client: Any) -> None:
    """Huỷ toàn bộ lệnh đang chờ."""
    active_orders = get_active_orders(client)
    if not active_orders:
        logger.info("Không có lệnh nào đang chờ để huỷ.")
        return

    logger.info(f"Tìm thấy {len(active_orders)} lệnh đang chờ. Đang huỷ...")
    for order in active_orders:
        cl_ord_id = order.get("clOrdId")
        symbol = order.get("symbol")
        side = "BUY" if str(order.get("side")) == "1" else "SELL"
        qty = order.get("orderQty")

        if cl_ord_id is None:
            logger.warning(f"Bỏ qua lệnh không có clOrdId: {order}")
            continue

        logger.info(f"Huỷ lệnh: {cl_ord_id[:8]}... | {side} {qty} {symbol}")
        try:
            client.cancel_order(cl_ord_id)
        except Exception as e:
            logger.error(f"Lỗi khi huỷ {cl_ord_id}: {e}")

    logger.info("Đã gửi yêu cầu huỷ toàn bộ lệnh.")


def liquidate(client: Any, slippage: float = 20.0) -> None:
    """Bán tháo toàn bộ danh mục (Đóng mọi vị thế)."""
    logger.info("Đang kiểm tra danh mục để bán tháo...")
    portfolio = client.get_portfolio_by_sub()
    if not portfolio.get("success"):
        logger.error(f"Không thể lấy thông tin danh mục: {portfolio.get('error')}")
        return

    items = portfolio.get("items", [])
    positions = [item for item in items if float(item.get("quantity", 0)) != 0]

    if not positions:
        logger.info("✨ Tài khoản hiện đang trống. Không có gì để bán tháo.")
        return

    # Huỷ lệnh trước khi bán tháo để tránh conflict
    logger.info("Bước 1: Huỷ các lệnh đang chờ để giải phóng ký quỹ/vị thế...")
    cancel_all(client)
    time.sleep(1)

    logger.info(f"Bước 2: Đang đóng {len(positions)} vị thế (Slippage: {slippage})...")
    for pos in positions:
        symbol = pos.get("instrument")
        qty = float(pos.get("quantity"))
        side = "SELL" if qty > 0 else "BUY"
        abs_qty = int(abs(qty))

        # Tính toán giá quyết liệt: Nếu bán thì bán thấp hơn, nếu mua (đóng short) thì mua cao hơn
        price_base = pos.get("currentPrice") or pos.get("avgPrice")
        if side == "SELL":
            exec_price = price_base - slippage
        else:
            exec_price = price_base + slippage

        logger.info(
            f"ĐÓNG VỊ THẾ: {symbol} | Qty: {abs_qty} | Side: {side} | Price: {exec_price:,.1f}"
        )
        try:
            client.place_order(
                full_symbol=symbol,
                side=side,
                qty=abs_qty,
                price=exec_price,
                ord_type="LIMIT",
                tif="GTC",
            )
        except Exception as e:
            logger.error(f"Lỗi khi đóng {symbol}: {e}")

    logger.info("✅ Đã gửi yêu cầu đóng toàn bộ vị thế.")


def quick_order(client: Any, symbol: str, side: str, qty: int, price: float, ord_type: str) -> None:
    """Đặt một lệnh nhanh."""
    logger.info(f"🚀 Đang đặt lệnh: {side} {qty} {symbol} @ {price} ({ord_type})")
    try:
        cl_ord_id = client.place_order(
            full_symbol=symbol, side=side, qty=qty, price=price, ord_type=ord_type, tif="GTC"
        )
        logger.info(f"✅ ĐẶT LỆNH THÀNH CÔNG! ID: {cl_ord_id}")
    except Exception as e:
        logger.error(f"❌ Lỗi khi đặt lệnh: {e}")


def show_status(client: Any, loop: bool = False) -> None:
    """Hiển thị trạng thái tài khoản hiện tại."""
    try:
        while True:
            if loop:
                # Clear màn hình để nhìn cho sạch (Windows 'cls' | Unix 'clear')
                os.system("cls" if os.name == "nt" else "clear")

            logger.info("=" * 60)
            logger.info(f"📊 TRẠNG THÁI TÀI KHOẢN - {datetime.now().strftime('%H:%M:%S')}")
            logger.info("=" * 60)

            # 1. Tiền mặt & Tổng tài sản
            cash = client.get_cash_balance()
            remain = float(cash.get("remainCash", 0))
            logger.info(f"💰 Tiền mặt khả dụng: {remain:,.0f} VND")

            # 2. Vị thế
            portfolio = client.get_portfolio_by_sub()
            items = portfolio.get("items", [])
            logger.info("\n📈 Vị thế hiện tại:")
            found_pos = False
            total_pnl = 0.0
            for item in items:
                qty = float(item.get("quantity", 0))
                if qty != 0:
                    found_pos = True
                    sym = item.get("instrument")
                    avg = float(item.get("avgPrice", 0))
                    pnl = float(item.get("pnl", 0))
                    total_pnl += pnl
                    pnl_icon = "🟢" if pnl >= 0 else "🔴"
                    logger.info(
                        f"  - {sym}: {qty:g} HĐ | Giá TB: {avg:,.1f} | PnL: {pnl_icon} {pnl:,.0f}"
                    )

            if not found_pos:
                logger.info("  (Trống)")
            else:
                pnl_all_icon = "🟢" if total_pnl >= 0 else "🔴"
                logger.info(f"  >> TỔNG PNL: {pnl_all_icon} {total_pnl:,.0f} VND")

            # 3. Lệnh đang chờ
            active = get_active_orders(client)
            logger.info(f"\n⏳ Lệnh đang chờ ({len(active)}):")
            for o in active:
                clOrdId = o.get("clOrdId")
                side = "BUY" if str(o.get("side")) == "1" else "SELL"
                orderQty = o.get("orderQty")
                symbol = o.get("symbol")
                price = o.get("price")

                if clOrdId is None:
                    logger.warning(f"Bỏ qua lệnh không có clOrdId: {o}")
                    continue

                logger.info(f"  - {clOrdId[:8]} | {side} {orderQty} {symbol} @ {price}")

            logger.info("=" * 60)

            if not loop:
                break

            logger.info("Đang theo dõi... Bấm Ctrl+C để thoát.")
            time.sleep(2)
    except KeyboardInterrupt:
        logger.info("\nDừng theo dõi.")


def main() -> None:
    parser = argparse.ArgumentParser(description="PaperBroker Emergency Toolkit")
    parser.add_argument(
        "action",
        choices=["status", "cancel", "liquidate", "order", "auto"],
        help="Hành động muốn thực hiện",
    )
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--side", choices=["BUY", "SELL"], default=DEFAULT_SIDE)
    parser.add_argument("--qty", type=int, default=DEFAULT_QTY)
    parser.add_argument("--price", type=float, default=DEFAULT_PRICE)
    parser.add_argument("--type", choices=["LIMIT", "MARKET"], default=DEFAULT_TYPE)
    parser.add_argument(
        "--slippage",
        type=float,
        default=20.0,
        help="Độ lệch giá khi bán tháo để đảm bảo khớp nhanh",
    )
    args = parser.parse_args()

    load_dotenv()

    client = build_broker_client()

    if client is None:
        logger.error("Không thể khởi tạo PaperBrokerClient. Vui lòng kiểm tra cấu hình.")
        sys.exit(1)

    try:
        if args.action == "status":
            show_status(client)
        elif args.action == "cancel":
            cancel_all(client)
        elif args.action == "liquidate":
            liquidate(client, slippage=args.slippage)
        elif args.action == "order":
            quick_order(client, args.symbol, args.side, args.qty, args.price, args.type)
        elif args.action == "auto":
            # Chạy theo cấu hình biến tay ở trên đầu file
            logger.info("Đang chạy theo cấu hình mặc định trong code...")
            quick_order(
                client, DEFAULT_SYMBOL, DEFAULT_SIDE, DEFAULT_QTY, DEFAULT_PRICE, DEFAULT_TYPE
            )

        # Đợi một chút để các yêu cầu được gửi đi
        time.sleep(2)

    except Exception as e:
        logger.error(f"Lỗi hệ thống: {e}")
    finally:
        client.disconnect()
        # Dùng os._exit để đóng hẳn các thread của QuickFIX
        os._exit(0)


if __name__ == "__main__":
    # Nếu không truyền tham số, mặc định hiện status
    if len(sys.argv) == 1:
        sys.argv.append("status")

    main()
