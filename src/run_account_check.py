"""
Standalone Account & Portfolio Inspector.

Connects to the PaperBroker REST API to fetch and display the current
account state without requiring a persistent FIX session or Redis
market data feed. Useful for verifying balances and trade history.

Usage:
    python -m src.run_account_check            # Fetch last 7 days
    python -m src.run_account_check --days 30  # Fetch last 30 days

Dependencies:
    - .env: Requires PAPER_REST_BASE_URL, PAPER_USERNAME, and PAPER_PASSWORD
    - paperbroker: PaperBrokerClient for REST API access

Workflow:
    1. Load environment variables from .env
    2. Resolve FIX SenderCompID (UUID) via REST logon
    3. Initialize PaperBrokerClient (REST-only mode)
    4. Query and display:
       - Cash balances (available and total)
       - Open positions (portfolio with unrealized P&L)
       - Order history (for specified date range)
       - Transaction history (closed trades with realized P&L)
    5. Clean exit (avoid QuickFIX cleanup issues)

Output format:
    - Balances: Available cash and total balance
    - Portfolio: Per-position breakdown with unrealized P&L
    - Orders: Order history with status and fill details
    - Transactions: Per-symbol summary with fees and realized P&L
"""

import argparse
import contextlib
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta

from dotenv import load_dotenv
from paperbroker.client import PaperBrokerClient

from config.secrets import _resolve_sender_comp_id_from_api
from src.utils.logger import setup_logging

logger = setup_logging(__name__, log_file="logs/account_check.log")


# ──────────────────────────────────────────────────────────────────────────
# Formatting Helpers
# ──────────────────────────────────────────────────────────────────────────


def sep(char: str = "=", width: int = 70) -> None:
    """Print a separator line."""
    print(char * width)


def section(title: str) -> None:
    """Print a section header with separators."""
    print()
    sep()
    print(f"  {title}")
    sep()


def fmt(value: float | str, decimals: int = 2) -> str:
    """Format numeric value with thousands separator.

    Args:
        value: Numeric value to format
        decimals: Number of decimal places (default: 2)

    Returns:
        Formatted string with thousands separator
    """
    if value is None:
        return "N/A"
    try:
        return f"{float(value):,.{decimals}f}"
    except (ValueError, TypeError):
        return str(value)


def pnl_color(pnl: float | str) -> str:
    """Format P&L with color indicator (🟢 for profit, 🔴 for loss).

    Args:
        pnl: P&L value to format

    Returns:
        Formatted string with color emoji
    """
    try:
        v = float(pnl)
        if v > 0:
            return f"🟢 +{fmt(v)}"
        elif v < 0:
            return f"🔴 {fmt(v)}"
        return f"   {fmt(v)}"
    except (ValueError, TypeError):
        return fmt(pnl)


# ──────────────────────────────────────────────────────────────────────────
# Display Sections
# ──────────────────────────────────────────────────────────────────────────


def print_balances(client: PaperBrokerClient) -> None:
    """Display cash balances and total account balance.

    Args:
        client: Connected PaperBrokerClient instance
    """
    section("💰 BALANCES")
    try:
        cash = client.get_cash_balance()
        total = client.get_account_balance()
        print(f"\n  Available Cash : {fmt(cash.get('remainCash', 0)):>20} VND")
        print(f"  Total Balance  : {fmt(total.get('totalBalance', 0)):>20} VND")
    except Exception as exc:
        logger.error("Failed to fetch balances: %s", exc, exc_info=True)
        print(f"  ❌ Could not fetch balance: {exc}")


def print_portfolio(client: PaperBrokerClient) -> None:
    """Display open positions with unrealized P&L.

    Shows per-position breakdown including:
    - Instrument symbol
    - Quantity held
    - Total cost basis
    - Current market value
    - Unrealized P&L

    Args:
        client: Connected PaperBrokerClient instance
    """
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

        # Table header
        header = f"  {'Instrument':<16} {'Qty':>8} {'Cost':>14} {'Market Val':>14} {'PnL':>16}"
        print(f"\n{header}")
        print("  " + "-" * 68)

        # Per-position rows
        total_pnl = 0.0
        for item in items:
            pnl = item.get("pnl")
            with contextlib.suppress(ValueError, TypeError):
                total_pnl += float(pnl or 0)

            print(
                f"  {item.get('instrument', ''):<16} "
                f"{fmt(item.get('quantity'), 0):>8} "
                f"{fmt(item.get('totalCost')):>14} "
                f"{fmt(item.get('marketValue')):>14} "
                f"{pnl_color(pnl):>16}"
            )

        # Total row
        print("  " + "-" * 68)
        direction = "+" if total_pnl >= 0 else ""
        print(f"\n  Total Unrealised P&L: {direction}{fmt(total_pnl)} VND")

    except Exception as exc:
        logger.error("Failed to fetch portfolio: %s", exc, exc_info=True)
        print(f"  ❌ {exc}")


def print_orders(client: PaperBrokerClient, start_date: str, end_date: str) -> None:
    """Display order history for specified date range.

    Shows order details including:
    - Symbol and side (BUY/SELL)
    - Order quantity and filled quantity
    - Average fill price
    - Order status
    - Order date

    Args:
        client: Connected PaperBrokerClient instance
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    """
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

        # Per-order details
        for i, o in enumerate(orders, 1):
            # Map side code to readable text
            side_code = o.get("side", "")
            side = "BUY" if side_code == "1" else "SELL" if side_code == "2" else side_code
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
        logger.error("Failed to fetch orders: %s", exc, exc_info=True)
        print(f"  ❌ {exc}")


def print_transactions(client: PaperBrokerClient, start_date: str, end_date: str) -> None:
    """Display transaction history with per-symbol P&L breakdown.

    Shows aggregated statistics per symbol:
    - Number of transactions
    - Buy and sell quantities
    - Total fees paid
    - Realized P&L

    Args:
        client: Connected PaperBrokerClient instance
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    """
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

        # Aggregate per-symbol statistics
        by_symbol: defaultdict[str, dict] = defaultdict(
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

        # Process each transaction
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

        # Total row
        print("  " + "-" * 72)
        print(f"\n  Total Fees: {fmt(total_fees)} VND")
        print(f"  Total P&L : {pnl_color(total_pnl)} VND")

    except Exception as exc:
        logger.error("Failed to fetch transactions: %s", exc, exc_info=True)
        print(f"  ❌ {exc}")


# ──────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────────────────


def main(args: argparse.Namespace) -> None:
    """
    Main entry point for account inspection.

    Workflow:
    1. Load environment variables from .env
    2. Resolve FIX SenderCompID via REST API
    3. Connect to PaperBroker
    4. Display account information sections
    5. Clean exit

    Args:
        args: Command-line arguments (days for date range)
    """
    load_dotenv()

    # Load configuration from environment
    username = os.getenv("PAPER_USERNAME", "BL01")
    password = os.getenv("PAPER_PASSWORD", "123")
    rest_url = os.getenv("PAPER_REST_BASE_URL", "http://localhost:9090")
    host = os.getenv("SOCKET_CONNECT_HOST", "localhost")
    port = int(os.getenv("SOCKET_CONNECT_PORT", "5001"))
    sender = os.getenv("SENDER_COMP_ID", "cross-FIX")
    target = os.getenv("TARGET_COMP_ID", "SERVER")
    sub_account = os.getenv("PAPER_ACCOUNT_ID_D1", "D1")

    # Validate required credentials
    if not username or not password:
        print("❌ PAPER_USERNAME and PAPER_PASSWORD must be set in .env")
        sys.exit(1)

    # Resolve FIX SenderCompID from REST API
    # The server matches logon by fixAccountID (a UUID), not the .env SENDER_COMP_ID
    # Using the wrong ID causes immediate logout
    resolved_sender = _resolve_sender_comp_id_from_api(rest_url, username, password)
    sender = resolved_sender or sender

    # Print configuration
    sep()
    print("  PAPER ACCOUNT CHECK")
    sep()
    print(f"  REST API    : {rest_url}")
    print(f"  Username    : {username}")
    print(f"  Sub-Account : {sub_account}")
    print(f"  SenderCompID: {sender}")
    print(f"  Date range  : last {args.days} day(s)")
    sep()

    # Connect to PaperBroker
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
    print("  (waiting up to 60s - server may need time to drop previous session…)")

    if not client.wait_until_logged_on(timeout=60):
        err = client.last_logon_error()
        print(f"❌ FIX logon failed: {err or 'no reason returned'}")
        sys.exit(1)

    print("✅ Connected!\n")

    # Calculate date range
    today = datetime.now().date()
    start_str = (today - timedelta(days=args.days)).strftime("%Y-%m-%d")
    end_str = today.strftime("%Y-%m-%d")

    # Display all sections
    print_balances(client)
    print_portfolio(client)
    print_orders(client, start_str, end_str)
    print_transactions(client, start_str, end_str)

    # Success message
    print()
    sep()
    print("  ✅  Account check complete")
    sep()
    print()

    # Clean exit to avoid QuickFIX cleanup segfault
    # QuickFIX C++ cleanup can cause segfaults on normal exit
    # Using os._exit(0) bypasses Python cleanup and exits immediately
    import os as _os

    _os._exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect your PaperBroker account: balances, portfolio, orders, transactions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.run_account_check              # Last 7 days
  python -m src.run_account_check --days 30    # Last 30 days
  python -m src.run_account_check --days 1     # Today only

Environment variables required in .env:
  PAPER_REST_BASE_URL    - PaperBroker REST API URL
  PAPER_USERNAME         - Account username
  PAPER_PASSWORD         - Account password
  PAPER_ACCOUNT_ID_D1    - Sub-account ID (default: D1)
        """,
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of past days to fetch orders/transactions for (default: 7).",
    )

    main(parser.parse_args())
