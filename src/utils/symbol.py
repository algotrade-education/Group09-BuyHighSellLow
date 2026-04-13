"""Symbol normalization utilities.

Handles mapping between specific contract codes (e.g., VN30F2604) and
generic contract codes (e.g., VN30F1M) used for database queries and caching.
"""

import re


def normalize_symbol(symbol: str) -> str:
    """Normalize a symbol to its generic form for database queries and caching.

    Maps specific contract codes to their generic equivalents:
    - VN30F2604 -> VN30F1M
    - VN30F2605 -> VN30F1M
    - VN30F2606 -> VN30F1M
    - etc.

    Args:
        symbol: Symbol to normalize (e.g., "VN30F2604" or "VN30F1M").

    Returns:
        Normalized symbol (e.g., "VN30F1M").

    Examples:
        >>> normalize_symbol("VN30F2604")
        'VN30F1M'
        >>> normalize_symbol("VN30F1M")
        'VN30F1M'
        >>> normalize_symbol("AAPL")
        'AAPL'
    """
    # Strip exchange prefix if present (e.g., "HNXDS:VN30F2604" -> "VN30F2604")
    contract = symbol.split(":")[-1] if ":" in symbol else symbol

    # Pattern: VN30F followed by 4 digits (YYMM format)
    # Example: VN30F2604 (April 2026 contract)
    if re.match(r"^VN30F\d{4}$", contract):
        return "VN30F1M"

    # Return the contract part (without exchange prefix) if no pattern matches
    return contract
