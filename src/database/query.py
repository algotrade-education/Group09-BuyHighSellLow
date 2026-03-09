"""
This module contains SQL queries for fetching market data from the PostgreSQL database.
"""

# Parameters:
# 1. futurecode (str): The common contract code (e.g., 'VN30F1M')
# 2. start_date (str/date): The start of the date range (inclusive)
# 3. end_date (str/date): The end of the date range (inclusive)
MATCHED_QUERY = """
    SELECT m.datetime, m.tickersymbol, m.price, v.quantity
    FROM quote.matched m
    JOIN quote.futurecontractcode fc ON
        date(m.datetime) = fc.datetime AND
        m.tickersymbol = fc.tickersymbol
    LEFT JOIN
        quote.total v ON
        m.tickersymbol = v.tickersymbol AND
        m.datetime = v.datetime
    WHERE
        fc.futurecode = %s AND
        date(m.datetime) BETWEEN %s AND %s AND
        ((EXTRACT(HOUR FROM m.datetime) >= 9 AND EXTRACT(HOUR FROM m.datetime) < 14) OR
            (EXTRACT(HOUR FROM m.datetime) = 14 AND EXTRACT(MINUTE FROM m.datetime) <= 30))
    ORDER BY m.datetime;
"""

# Parameters:
# 1. futurecode (str): The common contract code (e.g., 'VN30F1M')
# 2. start_date (str/date): The start of the date range (inclusive)
# 3. end_date (str/date): The end of the date range (inclusive)
BID_ASK_QUERY = """
    SELECT b.datetime, b.tickersymbol, b.price, a.price, a.price - b.price
    FROM quote.bidprice b
    JOIN quote.askprice a ON
        b.datetime = a.datetime AND
        b.tickersymbol = a.tickersymbol AND
        b.depth = a.depth
    JOIN quote.futurecontractcode fc ON
        date(b.datetime) = fc.datetime AND
        b.tickersymbol = fc.tickersymbol
    WHERE
        b.depth = 1 AND
        fc.futurecode = %s AND
        date(b.datetime) BETWEEN %s AND %s AND
        ((EXTRACT(HOUR FROM b.datetime) >= 9 AND EXTRACT(HOUR FROM b.datetime) < 14) OR
            (EXTRACT(HOUR FROM b.datetime) = 14 AND EXTRACT(MINUTE FROM b.datetime) <= 30))
    ORDER BY b.datetime;
"""

# Parameters:
# 1. futurecode (str): The common contract code (e.g., 'VN30F1M')
# 2. start_date (str/date): The start of the date range (inclusive)
# 3. end_date (str/date): The end of the date range (inclusive)
CLOSE_QUERY = """
    SELECT c.datetime, c.tickersymbol, c.price
    FROM quote.close c
    JOIN quote.futurecontractcode fc ON
        date(c.datetime) = fc.datetime AND
        c.tickersymbol = fc.tickersymbol
    WHERE
        fc.futurecode = %s AND
        date(c.datetime) BETWEEN %s AND %s
    ORDER BY c.datetime;
"""
