"""Paper trading configuration schemas.

Pydantic models for paper trading runtime configuration, loaded from environment
variables with validation and type safety.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PaperBarConfig(BaseSettings):
    """Bar construction runtime configuration.

    Controls bar quality assessment and database fallback behavior.
    Loaded from PAPER_BAR_* environment variables.
    """

    model_config = SettingsConfigDict(
        env_prefix="PAPER_BAR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        validate_assignment=True,
    )

    stale_seconds: Annotated[float, Field(ge=0.0, le=300.0)] = Field(
        default=5.0,
        description="Tick staleness threshold (seconds) for bar builder. "
        "Gaps exceeding this trigger DB fallback merge.",
    )

    preclose_fetch_seconds: Annotated[float, Field(ge=0.0, le=300.0)] = Field(
        default=30.0,
        description="Seconds before bar close to attempt DB preclose fallback fetch. "
        "Proactively fills gaps before bar emission.",
    )

    min_updates: Annotated[int, Field(ge=0, le=100)] = Field(
        default=2,
        description="Minimum live updates required before bar is considered sufficiently formed. "
        "Bars with fewer ticks trigger DB fallback.",
    )

    debug_quotes: bool = Field(
        default=False,
        description="Enable verbose quote-level logging for bar builder diagnostics.",
        alias="PAPER_DEBUG_QUOTES",
    )

    enable_db_bar_fallback: bool = Field(
        default=True,
        description="Enable fallback closed-bar fetch from DB when live ticks are missing or sparse.",
        alias="PAPER_ENABLE_DB_BAR_FALLBACK",
    )


class PaperEngineConfig(BaseSettings):
    """Paper engine lifecycle and session management configuration.

    Loaded from PAPER_* environment variables.
    """

    model_config = SettingsConfigDict(
        env_prefix="PAPER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        validate_assignment=True,
    )

    close_on_shutdown: bool = Field(
        default=False,
        description="Close open position when bot stops/shuts down. "
        "Recommended false for debugging to preserve positions across restarts.",
    )

    force_hard_exit: bool = Field(
        default=False,
        description="Force hard process exit (os._exit) after shutdown cleanup. "
        "Workaround for FIX teardown issues that can hang the process.",
    )

    entry_cutoff_seconds: Annotated[float, Field(ge=0.0, le=3600.0)] = Field(
        default=0.0,
        description="Block new entries this many seconds before session end. "
        "0 = no cutoff (entries allowed until session close).",
    )

    allow_late_entry: bool = Field(
        default=False,
        description="Allow entries near session boundary (overrides entry_cutoff_seconds). "
        "Use with caution as late entries may not have time to develop.",
    )

    force_flat_on_session_close: bool = Field(
        default=True,
        description="Force flat when crossing out of trading session in bar loop. "
        "Ensures no overnight positions in intraday strategies.",
    )

    force_flat_preclose_seconds: Annotated[float, Field(ge=0.0, le=3600.0)] = Field(
        default=240.0,
        description="Force flat this many seconds before VN30 sub-session end (0=disabled). "
        "If force_flat_on_session_close=true and this=0 and force_flat_on_last_candle=false, "
        "engine auto-fallbacks to 15s.",
    )

    force_flat_on_last_candle: bool = Field(
        default=True,
        description="Force flat on the final bar of each VN30 sub-session window. "
        "Alternative to preclose_seconds for bar-aligned exits.",
    )

    defer_exit_outside_session: bool = Field(
        default=False,
        description="If an exit trigger occurs outside session, defer until next tradable bar. "
        "Prevents rejected orders during closed hours.",
    )


def get_paper_bar_config() -> PaperBarConfig:
    """Get paper bar configuration from environment with validation."""
    return PaperBarConfig()


def get_paper_engine_config() -> PaperEngineConfig:
    """Get paper engine configuration from environment with validation."""
    return PaperEngineConfig()
