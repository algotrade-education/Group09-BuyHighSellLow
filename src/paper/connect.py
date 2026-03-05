"""
Simple FIX login example using event callbacks.

Demonstrates:
- Event-driven login lifecycle
- SenderCompID resolution from REST metadata
- Basic account checks after FIX logon
"""

import os
import sys
import logging
from pathlib import Path
import requests
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from paperbroker.client import PaperBrokerClient

# Load environment variables
load_dotenv()

# Setup logger for this example (separate from library logger)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def on_logon(session_id, **kw):
    """Handle successful FIX logon events."""
    logger.info(f"✅ FIX session established: {session_id}")


def on_logout(session_id, reason=None, **kw):
    """Handle FIX logout events."""
    logger.info(f"👋 FIX session closed: {session_id}, reason: {reason}")


def on_reject(reason, msg_type, **kw):
    """Handle broker/FIX message rejects."""
    logger.error(f"❌ Message rejected - Type: {msg_type}, Reason: {reason}")


def on_logon_error(session_id=None, reason=None, **kw):
    """Handle FIX logon failures."""
    logger.error(f"❌ FIX logon error: session={session_id}, reason={reason}")


def resolve_fix_sender_comp_id(
    rest_base_url: str, username: str, password: str
) -> str | None:
    """Resolve SenderCompID via REST `fixAccountID` metadata."""
    url = f"{rest_base_url.rstrip('/')}/api/fix-account-info/get-fix-id"
    try:
        response = requests.post(
            url,
            json={"username": username, "password": password},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json() if response.content else {}
        if isinstance(payload, dict):
            return payload.get("fixAccountID") or payload.get("accountID")
    except Exception as exc:
        logger.warning(f"⚠️ Could not resolve fixAccountID from REST: {exc}")
    return None


def main():
    """Run login demo and print a minimal connectivity sanity check."""

    username = os.getenv("PAPER_USERNAME", "BL01")
    password = os.getenv("PAPER_PASSWORD", "123")
    rest_base_url = os.getenv("PAPER_REST_BASE_URL", "http://localhost:9090")

    env_sender_comp_id = os.getenv("SENDER_COMP_ID")
    resolved_sender_comp_id = resolve_fix_sender_comp_id(
        rest_base_url, username, password
    )
    sender_comp_id = resolved_sender_comp_id or env_sender_comp_id or "cross-FIX"

    if (
        resolved_sender_comp_id
        and env_sender_comp_id
        and resolved_sender_comp_id != env_sender_comp_id
    ):
        logger.warning(
            "⚠️ SENDER_COMP_ID from .env differs from server fixAccountID; using server fixAccountID for FIX logon"
        )

    logger.info(f"🧾 Using SenderCompID: {sender_comp_id}")

    # Create client
    # Config file auto-generated from connection parameters
    client = PaperBrokerClient(
        default_sub_account=os.getenv("PAPER_ACCOUNT_ID_D1", "D1"),
        username=username,
        password=password,
        rest_base_url=rest_base_url,
        socket_connect_host=os.getenv("SOCKET_CONNECT_HOST", "localhost"),
        socket_connect_port=int(os.getenv("SOCKET_CONNECT_PORT", "5001")),
        sender_comp_id=sender_comp_id,
        target_comp_id=os.getenv("TARGET_COMP_ID", "SERVER"),
        console=True,  # Only show WARNING/ERROR in console, DEBUG/INFO go to file only
    )

    # Subscribe to events (clean event-based design)
    client.on("fix:logon", on_logon)
    client.on("fix:logout", on_logout)
    client.on("fix:logon_error", on_logon_error)
    client.on("fix:reject", on_reject)

    # Connect (non-blocking)
    logger.info("🔌 Connecting to PaperBroker...")
    client.connect()

    try:
        # Wait for logon (with timeout)
        if client.wait_until_logged_on(timeout=20):
            logger.info("✅ Successfully logged on!")

            # Get account info via REST API
            cash = client.get_cash_balance()
            logger.info(f"💰 Available cash: {cash.get('remainCash', 0):,.0f} VND")

            total = client.get_account_balance()
            logger.info(f"💰 Total balance: {total.get('totalBalance', 0):,.0f} VND")

        else:
            error = client.last_logon_error()
            logger.error(
                f"❌ Logon failed: {error or 'No reason returned by package/server'}"
            )
            return

        # Keep connection alive for a moment
        logger.info("⏳ Staying connected for 5 seconds...")
        import time

        time.sleep(5)

    finally:
        # Note: Using os._exit() to avoid QuickFIX cleanup segfault
        # This is a known issue with QuickFIX Python bindings
        logger.info("✅ Example completed!")
        os._exit(0)


if __name__ == "__main__":
    main()
