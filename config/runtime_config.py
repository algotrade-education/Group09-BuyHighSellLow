import os
from typing import Any, Dict, Mapping, Optional

from dotenv import load_dotenv

_ = load_dotenv()

_TRUTHY_VALUES = {"1", "true", "yes", "on"}


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in _TRUTHY_VALUES


def _to_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _env_value(name: str) -> Optional[str]:
    return os.getenv(name)


def _resolve_float(
    risk_value: Any,
    env_name: str,
    default: float,
) -> float:
    if risk_value is not None:
        return _to_float(risk_value, default)

    env_value = _env_value(env_name)
    if env_value is not None:
        return _to_float(env_value, default)

    return float(default)


def _resolve_int(
    risk_value: Any,
    env_name: str,
    default: int,
) -> int:
    if risk_value is not None:
        return _to_int(risk_value, default)

    env_value = _env_value(env_name)
    if env_value is not None:
        return _to_int(env_value, default)

    return int(default)


def _resolve_bool(
    risk_value: Any,
    env_name: str,
    default: bool,
) -> bool:
    if risk_value is not None:
        return _to_bool(risk_value, default)

    env_value = _env_value(env_name)
    if env_value is not None:
        return _to_bool(env_value, default)

    return default


def get_paper_runtime_config(risk: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    risk_config = risk or {}

    return {
        "enable_db_bar_fallback": _resolve_bool(
            risk_config.get("enable_db_bar_fallback"),
            "PAPER_ENABLE_DB_BAR_FALLBACK",
            True,
        ),
        "force_hard_exit": _resolve_bool(
            risk_config.get("force_hard_exit"),
            "PAPER_FORCE_HARD_EXIT",
            False,
        ),
        "entry_cutoff_seconds": _resolve_float(
            risk_config.get("entry_cutoff_seconds"),
            "PAPER_ENTRY_CUTOFF_SECONDS",
            60.0,
        ),
        "allow_late_entry": _resolve_bool(
            risk_config.get("allow_late_entry"),
            "PAPER_ALLOW_LATE_ENTRY",
            False,
        ),
        "force_flat_on_session_close": _resolve_bool(
            risk_config.get("force_flat_on_session_close"),
            "PAPER_FORCE_FLAT_ON_SESSION_CLOSE",
            False,
        ),
        "defer_exit_outside_session": _resolve_bool(
            risk_config.get("defer_exit_outside_session"),
            "PAPER_DEFER_EXIT_OUTSIDE_SESSION",
            True,
        ),
    }


def get_paper_bar_runtime_config(
    freq_minutes: int,
    risk: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    risk_config = risk or {}

    default_stale_seconds = max(5, int(freq_minutes * 60 * 0.1))
    stale_seconds = _resolve_float(
        risk_config.get("bar_stale_seconds"),
        "PAPER_BAR_STALE_SECONDS",
        float(default_stale_seconds),
    )
    default_preclose_seconds = max(2, int(stale_seconds))

    return {
        "stale_trade_seconds": stale_seconds,
        "preclose_db_fetch_seconds": _resolve_float(
            risk_config.get("bar_preclose_fetch_seconds"),
            "PAPER_BAR_PRECLOSE_FETCH_SECONDS",
            float(default_preclose_seconds),
        ),
        "min_live_updates": _resolve_int(
            risk_config.get("bar_min_live_updates"),
            "PAPER_BAR_MIN_UPDATES",
            2,
        ),
        "debug_quotes": _resolve_bool(
            risk_config.get("debug_quotes"),
            "PAPER_DEBUG_QUOTES",
            False,
        ),
    }


def get_backtest_runtime_config(
    entry_cutoff_seconds: Optional[float] = None,
    allow_late_entry: Optional[bool] = None,
    risk: Optional[Mapping[str, Any]] = None,
    *,
    default_entry_cutoff_seconds: float = 0.0,
    default_allow_late_entry: bool = False,
) -> Dict[str, Any]:
    risk_config = risk or {}

    if entry_cutoff_seconds is not None:
        resolved_entry_cutoff = float(entry_cutoff_seconds)
    else:
        resolved_entry_cutoff = _resolve_float(
            risk_config.get("entry_cutoff_seconds"),
            "BACKTEST_ENTRY_CUTOFF_SECONDS",
            default_entry_cutoff_seconds,
        )

    if allow_late_entry is not None:
        resolved_allow_late = bool(allow_late_entry)
    else:
        resolved_allow_late = _resolve_bool(
            risk_config.get("allow_late_entry"),
            "BACKTEST_ALLOW_LATE_ENTRY",
            default_allow_late_entry,
        )

    return {
        "entry_cutoff_seconds": resolved_entry_cutoff,
        "allow_late_entry": resolved_allow_late,
    }
