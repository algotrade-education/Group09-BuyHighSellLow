"""
Generic hashing utilities.

Provides stable, deterministic SHA-256 fingerprints for dicts and arbitrary
strings. Used by the optimization layer (param space conflict detection) and
the data pipeline (indicator cache keys).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def hash_str(value: str, length: int = 64) -> str:
    """
    Return a hex SHA-256 digest of a string.

    Args:
        value:  The string to hash.
        length: Number of hex characters to return (max 64). Default: full digest.
    """
    digest = hashlib.sha256(value.encode()).hexdigest()
    return digest[:length]


def hash_dict(
    data: dict[str, Any],
    *,
    length: int = 16,
    exclude_keys: set[str] | None = None,
) -> str:
    """
    Return a stable hex SHA-256 fingerprint of a JSON-serialisable dict.

    Keys are sorted before serialisation so insertion order doesn't matter.

    Args:
        data:         Dict to fingerprint.
        length:       Number of hex characters to return (max 64). Default: 16.
        exclude_keys: Keys to drop before hashing (e.g. volatile metadata).
    """
    payload = {k: v for k, v in data.items() if k not in (exclude_keys or set())}
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hash_str(canonical, length=length)
