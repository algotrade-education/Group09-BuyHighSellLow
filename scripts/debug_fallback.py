"""
Diagnostic script for paper-trade bar fallback.

Tests every layer independently so you can pinpoint exactly where 0 rows comes from:
  1. DB connection
  2. futurecontractcode mapping (futurecode -> tickersymbol) for today
  3. Raw quote.matched rows for today's contract in the bucket window
  4. get_matched_data_in_range (MATCHED_RANGE_QUERY)
  5. get_last_matched_before (MATCHED_LAST_BEFORE_QUERY)
  6. fetch_bucket_bar (volume derivation included)
  7. load_fallback_bar_for_bucket (full paper-trade path)

Usage
-----
    # Check the default bucket (today 09:15 -> 09:30) for VN30F1M at 15-min freq:
    python debug_fallback.py

    # Custom bucket & freq:
    python debug_fallback.py --symbol HNXDS:VN30F2601 --bucket "2026-03-11 09:15" --freq 15

    # Probe a different contract root:
    python debug_fallback.py --db-symbol VN30F2603 --bucket "2026-03-11 09:15" --freq 15
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEP = "-" * 70


def section(title: str) -> None:
    print(f"\n{_SEP}")
    print(f"  {title}")
    print(_SEP)


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def info(msg: str) -> None:
    print(f"         {msg}")


# ---------------------------------------------------------------------------
# Layer checks
# ---------------------------------------------------------------------------


def check_connection(ds) -> bool:
    section("LAYER 1 — DB connection")
    try:
        ds._ensure_connection()
        rows = ds._execute_query("SELECT 1", ())
        ok(f"Connected. SELECT 1 → {rows}")
        return True
    except Exception as exc:
        fail(str(exc))
        return False


def check_futurecontractcode(ds, db_symbol: str, bucket_dt: datetime) -> list[str]:
    """Return the list of tickersymbols mapped to db_symbol on bucket date."""
    section(
        f"LAYER 2 — futurecontractcode join  (futurecode='{db_symbol}', date={bucket_dt.date()})"
    )
    query = """
        SELECT tickersymbol
        FROM quote.futurecontractcode
        WHERE futurecode = %s
          AND datetime = %s
        ORDER BY tickersymbol;
    """
    try:
        rows = ds._execute_query(query, (db_symbol, bucket_dt.date()))
        if rows:
            tickers = [r[0] for r in rows]
            ok(f"Mapped tickersymbol(s): {tickers}")
            return tickers
        else:
            fail(
                f"No rows in futurecontractcode for futurecode='{db_symbol}' on {bucket_dt.date()}.\n"
                "         → The contract roll may not be registered for today."
            )
            return []
    except Exception as exc:
        fail(str(exc))
        return []


def check_raw_matched(
    ds, db_symbol: str, bucket_dt: datetime, bucket_end: datetime
) -> int:
    """Count raw matched rows using the futurecontractcode join (same as MATCHED_RANGE_QUERY)."""
    section(
        f"LAYER 3 — Raw quote.matched rows  [{bucket_dt.strftime('%H:%M')} – {bucket_end.strftime('%H:%M')})"
    )
    query = """
        SELECT COUNT(*), MIN(m.datetime), MAX(m.datetime)
        FROM quote.matched m
        JOIN quote.futurecontractcode fc ON
            date(m.datetime) = fc.datetime AND
            m.tickersymbol = fc.tickersymbol
        WHERE
            fc.futurecode = %s AND
            m.datetime >= %s AND
            m.datetime < %s;
    """
    try:
        rows = ds._execute_query(query, (db_symbol, bucket_dt, bucket_end))
        count, first_ts, last_ts = rows[0]
        count = count or 0
        if count > 0:
            ok(
                f"{count} ticks in [{bucket_dt.strftime('%H:%M')}, {bucket_end.strftime('%H:%M')})"
            )
            info(f"first={first_ts}  last={last_ts}")
        else:
            fail(
                f"0 matched rows for futurecode='{db_symbol}' "
                f"in [{bucket_dt.strftime('%H:%M')}, {bucket_end.strftime('%H:%M')}).\n"
                "         Possible causes:\n"
                "           a) Data not yet written to DB (live latency)\n"
                "           b) futurecontractcode join returns no row (→ see layer 2)\n"
                "           c) The market was halted / zero liquidity in this bucket"
            )
        return count
    except Exception as exc:
        fail(str(exc))
        return 0


def check_matched_range_query(
    ds, db_symbol: str, bucket_dt: datetime, bucket_end: datetime
) -> None:
    section("LAYER 4 — get_matched_data_in_range  (MATCHED_RANGE_QUERY)")
    try:
        df = ds.get_matched_data_in_range(
            from_datetime=bucket_dt,
            to_datetime=bucket_end,
            contract_name=db_symbol,
        )
        if df.empty:
            fail("Returned empty DataFrame.")
        else:
            ok(f"{len(df)} rows.  Columns: {list(df.columns)}")
            info(str(df.head(3).to_string(index=False)))
    except Exception as exc:
        fail(str(exc))


def check_last_matched_before(ds, db_symbol: str, bucket_dt: datetime) -> None:
    section(
        f"LAYER 5 — get_last_matched_before  (before {bucket_dt.strftime('%H:%M')})"
    )
    try:
        df = ds.get_last_matched_before(
            before_datetime=bucket_dt,
            contract_name=db_symbol,
        )
        if df.empty:
            fail(
                "No tick found before bucket start — volume delta calc will be impaired."
            )
        else:
            ok(f"Last tick before bucket: {df.iloc[0].to_dict()}")
    except Exception as exc:
        fail(str(exc))


def check_fetch_bucket_bar(
    db_symbol: str, bucket_dt: datetime, bucket_end: datetime
) -> None:
    section("LAYER 6 — fetch_bucket_bar  (full OHLCV assembly)")
    try:
        from src.database.data_service import fetch_bucket_bar

        df = fetch_bucket_bar(
            contract_name=db_symbol,
            bucket_start=bucket_dt,
            bucket_end=bucket_end,
        )
        if df.empty:
            fail("fetch_bucket_bar returned empty DataFrame.")
        else:
            ok(f"Bar assembled: {df.iloc[0].to_dict()}")
    except Exception as exc:
        fail(str(exc))


def check_load_fallback(symbol: str, bucket_dt: datetime, freq_minutes: int) -> None:
    section("LAYER 7 — load_fallback_bar_for_bucket  (paper-trade path)")
    import logging

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("debug_fallback")
    try:
        from src.paper.bar_fallback import load_fallback_bar_for_bucket

        result = load_fallback_bar_for_bucket(
            symbol=symbol,
            bucket_dt=bucket_dt,
            freq_minutes=freq_minutes,
            enabled=True,
            logger=logger,
        )
        if result is None:
            fail("load_fallback_bar_for_bucket returned None.")
        else:
            ok(f"Fallback bar: {result}")
    except Exception as exc:
        fail(str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose bar fallback failures in paper trading.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__ or ""),
    )
    parser.add_argument(
        "--symbol",
        default="HNXDS:VN30F2601",
        help="Full market symbol (default: HNXDS:VN30F2601)",
    )
    parser.add_argument(
        "--db-symbol",
        default=None,
        help="DB futurecode override (default: derived from --symbol)",
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="Bucket start datetime, e.g. '2026-03-11 09:15' (default: most recent completed bucket)",
    )
    parser.add_argument(
        "--freq",
        type=int,
        default=15,
        help="Bar frequency in minutes (default: 15)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve db_symbol
    contract = args.symbol.split(":")[-1]
    db_symbol = args.db_symbol or (
        "VN30F1M" if contract.startswith("VN30F") else contract
    )

    # Resolve bucket
    if args.bucket:
        bucket_dt = datetime.fromisoformat(args.bucket)
    else:
        now = datetime.now()
        total_minutes = now.hour * 60 + now.minute
        bucket_start_minutes = (total_minutes // args.freq) * args.freq
        bucket_dt = now.replace(
            hour=bucket_start_minutes // 60,
            minute=bucket_start_minutes % 60,
            second=0,
            microsecond=0,
        ) - timedelta(minutes=args.freq)  # last *completed* bucket

    bucket_end = bucket_dt + timedelta(minutes=args.freq)

    print(f"\nFallback Diagnostic")
    print(f"  symbol     : {args.symbol}")
    print(f"  db_symbol  : {db_symbol}")
    print(f"  bucket     : [{bucket_dt}  –  {bucket_end})")
    print(f"  freq       : {args.freq} min")

    # Bootstrap env/config
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    # Import DataService singleton
    try:
        from src.database.data_service import data_service as ds
    except Exception as exc:
        fail(f"Could not import data_service: {exc}")
        sys.exit(1)

    # Run layers
    if not check_connection(ds):
        print("\nAborting: cannot continue without a DB connection.")
        sys.exit(1)

    tickers = check_futurecontractcode(ds, db_symbol, bucket_dt)
    raw_count = check_raw_matched(ds, db_symbol, bucket_dt, bucket_end)

    if raw_count == 0 and tickers:
        # Extra: check if ticks exist for those tickers without the JOIN to narrow down
        section(
            "LAYER 3b — Raw matched rows by tickersymbol directly (no futurecode join)"
        )
        for ticker in tickers:
            query = """
                SELECT COUNT(*), MIN(datetime), MAX(datetime)
                FROM quote.matched
                WHERE tickersymbol = %s
                  AND datetime >= %s
                  AND datetime < %s;
            """
            try:
                rows = ds._execute_query(query, (ticker, bucket_dt, bucket_end))
                count, first_ts, last_ts = rows[0]
                if count:
                    ok(
                        f"tickersymbol='{ticker}': {count} ticks  ({first_ts} → {last_ts})"
                    )
                    info(
                        "  The JOIN to futurecontractcode may be broken for this date."
                    )
                else:
                    fail(
                        f"tickersymbol='{ticker}': 0 ticks in window — data not in DB yet."
                    )
            except Exception as exc:
                fail(f"tickersymbol='{ticker}': {exc}")

    check_matched_range_query(ds, db_symbol, bucket_dt, bucket_end)
    check_last_matched_before(ds, db_symbol, bucket_dt)
    check_fetch_bucket_bar(db_symbol, bucket_dt, bucket_end)
    check_load_fallback(args.symbol, bucket_dt, args.freq)

    print(f"\n{_SEP}")
    print("  Diagnostic complete.")
    print(_SEP)


if __name__ == "__main__":
    main()
