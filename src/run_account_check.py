"""
Standalone account inspector — prints balance, portfolio, orders,
and transactions from the PaperBroker REST API.

No FIX session or market data connection required.

Usage:
    python -m src.run_account_check            # last 7 days
    python -m src.run_account_check --days 30  # last 30 days
    python -m src.run_account_check --no-disconnect
"""

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta

from paperbroker.client import PaperBrokerClient
from src.paper.connect import resolve_fix_sender_comp_id

from dotenv import load_dotenv

from src.utils.logger import setup_logging

logger = setup_logging(__name__, log_file="logs/account_check.log")


# ── Formatting helpers ─────────────────────────────────────────────────────


def sep(char: str = "=", w: int = 70) -> None:
    print(char * w)


def section(title: str) -> None:
    print()
    sep()
    print(f"  {title}")
    sep()


def fmt(value, decimals: int = 2) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):,.{decimals}f}"
    except (ValueError, TypeError):
        return str(value)


def pnl_color(pnl) -> str:
    try:
        v = float(pnl)
        if v > 0:
            return f"🟢 +{fmt(v)}"
        elif v < 0:
            return f"🔴 {fmt(v)}"
        return f"   {fmt(v)}"
    except (ValueError, TypeError):
        return fmt(pnl)


# ── Sections ───────────────────────────────────────────────────────────────


def print_balances(client) -> None:
    section("💰 BALANCES")
    try:
        cash = client.get_cash_balance()
        total = client.get_account_balance()
        print(f"\n  Available Cash : {fmt(cash.get('remainCash', 0)):>20} VND")
        print(f"  Total Balance  : {fmt(total.get('totalBalance', 0)):>20} VND")
    except Exception as exc:
        print(f"  ❌ Could not fetch balance: {exc}")


def print_portfolio(client) -> None:
    section("📊 PORTFOLIO (Positions)")
    try:
        result = client.get_portfolio_by_sub()
        if not result.get("success"):
            print(f"  ❌ {result.get('error', 'Unknown error')}")
            return
        items = result.get("items", [])
        if not items:
            print("\n  No open positions.")
            return

        header = f"  {'Instrument':<16} {'Qty':>8} {'Cost':>14} {'Market Val':>14} {'PnL':>16}"
        print(f"\n{header}")
        print("  " + "-" * 68)

        total_pnl = 0.0
        for item in items:
            pnl = item.get("pnl")
            try:
                total_pnl += float(pnl or 0)
            except (ValueError, TypeError):
                pass
            print(
                f"  {item.get('instrument', ''):<16} "
                f"{fmt(item.get('quantity'), 0):>8} "
                f"{fmt(item.get('totalCost')):>14} "
                f"{fmt(item.get('marketValue')):>14} "
                f"{pnl_color(pnl):>16}"
            )

        print("  " + "-" * 68)
        direction = "+" if total_pnl >= 0 else ""
        print(f"\n  Total Unrealised P&L: {direction}{fmt(total_pnl)} VND")
    except Exception as exc:
        print(f"  ❌ {exc}")


def print_orders(client, start_date: str, end_date: str) -> None:
    section(f"📜 ORDERS  ({start_date} → {end_date})")
    try:
        result = client.get_orders(start_date, end_date)
        if not result.get("success"):
            print(f"  ❌ {result.get('error', 'Unknown error')}")
            return
        orders = result.get("items", [])
        print(f"\n  Found {len(orders)} order(s)")
        if not orders:
            return

        for i, o in enumerate(orders, 1):
            side_code = o.get("side", "")
            side = (
                "BUY" if side_code == "1" else "SELL" if side_code == "2" else side_code
            )
            status = o.get("ordStatus", "N/A")
            status_text = o.get("statusText", "")
            print(
                f"\n  [{i:02d}] {o.get('symbol', 'N/A'):<18} {side:<5} "
                f"Qty:{fmt(o.get('orderQty'), 0)}  "
                f"Filled:{fmt(o.get('cumQty'), 0)}  "
                f"AvgPx:{fmt(o.get('avgPx'))}  "
                f"Status:{status} ({status_text})  "
                f"Date:{o.get('orderDate', 'N/A')}"
            )
    except Exception as exc:
        print(f"  ❌ {exc}")


def print_transactions(client, start_date: str, end_date: str) -> None:
    section(f"💳 TRANSACTIONS  ({start_date} → {end_date})")
    try:
        result = client.get_transactions_by_date(start_date, end_date)
        if not result.get("success"):
            print(f"  ❌ {result.get('error', 'Unknown error')}")
            return
        txns = result.get("items", [])
        print(f"\n  Found {len(txns)} transaction(s)")
        if not txns:
            return

        # Per-symbol breakdown
        by_symbol = defaultdict(
            lambda: {
                "buy_qty": 0,
                "sell_qty": 0,
                "buy_cost": 0.0,
                "sell_cost": 0.0,
                "fees": 0.0,
                "pnl": 0.0,
                "count": 0,
            }
        )
        total_fees = 0.0
        total_pnl = 0.0

        for txn in txns:
            sym = txn.get("symbol", "UNKNOWN")
            t = txn.get("type", "")
            qty = float(txn.get("quantity", 0) or 0)
            cost = float(txn.get("totalCost", 0) or 0)
            fee = float(txn.get("totalFee", 0) or 0)
            pnl = float(txn.get("pnl", 0) or 0)

            by_symbol[sym]["count"] += 1
            by_symbol[sym]["fees"] += fee
            by_symbol[sym]["pnl"] += pnl
            total_fees += fee
            total_pnl += pnl

            if t == "BUY":
                by_symbol[sym]["buy_qty"] += qty
                by_symbol[sym]["buy_cost"] += cost
            elif t == "SELL":
                by_symbol[sym]["sell_qty"] += qty
                by_symbol[sym]["sell_cost"] += cost

        # Per-symbol table
        print(
            f"\n  {'Symbol':<20} {'Txns':>5} {'BuyQty':>8} {'SellQty':>8} {'Fees':>14} {'P&L':>14}"
        )
        print("  " + "-" * 72)
        for sym, s in sorted(by_symbol.items()):
            net_pnl = pnl_color(s["pnl"])
            print(
                f"  {sym:<20} {s['count']:>5} "
                f"{fmt(s['buy_qty'], 0):>8} "
                f"{fmt(s['sell_qty'], 0):>8} "
                f"{fmt(s['fees']):>14} "
                f"{net_pnl:>14}"
            )

        print("  " + "-" * 72)
        print(f"\n  Total Fees: {fmt(total_fees)} VND")
        print(f"  Total P&L : {pnl_color(total_pnl)} VND")
    except Exception as exc:
        print(f"  ❌ {exc}")


# ── Main ───────────────────────────────────────────────────────────────────


def main(args: argparse.Namespace) -> None:
    load_dotenv()

    username = os.getenv("PAPER_USERNAME", "BL01")
    password = os.getenv("PAPER_PASSWORD", "123")
    rest_url = os.getenv("PAPER_REST_BASE_URL", "http://localhost:9090")
    host = os.getenv("SOCKET_CONNECT_HOST", "localhost")
    port = int(os.getenv("SOCKET_CONNECT_PORT", "5001"))
    sender = os.getenv("SENDER_COMP_ID", "cross-FIX")
    target = os.getenv("TARGET_COMP_ID", "SERVER")
    sub_account = os.getenv("PAPER_ACCOUNT_ID_D1", "D1")

    if not username or not password:
        print("❌ PAPER_USERNAME and PAPER_PASSWORD must be set in .env")
        sys.exit(1)

    # Must resolve the correct FIX SenderCompID from REST API BEFORE building the client.
    # The server matches logon by fixAccountID (a UUID) — using the .env SENDER_COMP_ID
    # directly causes an immediate logout because the IDs don't match.
    resolved_sender = resolve_fix_sender_comp_id(rest_url, username, password)
    sender = resolved_sender or sender

    sep()
    print("  PAPER ACCOUNT CHECK")
    sep()
    print(f"  REST API    : {rest_url}")
    print(f"  Username    : {username}")
    print(f"  Sub-Account : {sub_account}")
    print(f"  SenderCompID: {sender}")
    print(f"  Date range  : last {args.days} day(s)")
    sep()

    print("\n🔌 Connecting to PaperBroker…")
    client = PaperBrokerClient(
        default_sub_account=sub_account,
        username=username,
        password=password,
        rest_base_url=rest_url,
        socket_connect_host=host,
        socket_connect_port=port,
        sender_comp_id=sender,
        target_comp_id=target,
        console=False,
    )
    client.connect()
    print("  (waiting up to 60s — server may need time to drop previous session…)")
    if not client.wait_until_logged_on(timeout=60):
        err = client.last_logon_error()
        print(f"❌ FIX logon failed: {err or 'no reason returned'}")
        sys.exit(1)
    print("✅ Connected!\n")

    # Date range
    today = datetime.now().date()
    start_str = (today - timedelta(days=args.days)).strftime("%Y-%m-%d")
    end_str = today.strftime("%Y-%m-%d")

    print_balances(client)
    print_portfolio(client)
    print_orders(client, start_str, end_str)
    print_transactions(client, start_str, end_str)

    print()
    sep()
    print("  ✅  Account check complete")
    sep()
    print()

    # Avoid QuickFIX cleanup segfault
    import os as _os

    _os._exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect your PaperBroker account: balances, portfolio, orders, transactions."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of past days to fetch orders/transactions for (default: 7).",
    )
    main(parser.parse_args())
