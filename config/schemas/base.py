"""
Base Pydantic schemas for all strategies.
Strategy-specific schemas should be defined in their respective modules and inherit from these base schemas.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# --- Type Aliases ---
ResampleFreq = Literal["1min", "5min", "15min", "30min", "1H", "1D", "1W", "1M"]

PositiveFloat = Annotated[float, Field(gt=0)]
PositiveInt = Annotated[int, Field(gt=0)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class BaseStrategyConfig(BaseModel):
    """
    Fields common to all strategy configurations.
    Strategy-specific fields should be added in their respective schemas
    that inherit from this base class.
    """

    model_config = ConfigDict(
        extra="forbid",  # Forbid extra fields not defined in the schema
        frozen=False,  # Allow mutation of model instances
        validate_assignment=True,  # Validate fields on assignment
    )

    resample_freq: ResampleFreq = Field(
        description="Bar timeframe. E.g., '1min', '5min', '1D', etc."
    )


class RiskConfig(BaseModel):
    """
    Risk management configuration fields.

    Notes:
        - max_daily_loss: Maximum loss allowed per day, expressed as a percentage of equity (0.02 = 2% of equity).
        - Trailing stop behavior is controlled via use_trailing_stop together with trailing_atr_multiplier.
        - entry_cutoff_seconds and allow_late_entry: Used to control late entries and only for Paper Trading.
    """

    model_config = ConfigDict(
        extra="forbid",  # Forbid extra fields not defined in the schema
        validate_assignment=True,  # Validate fields on assignment
    )

    # --- Position Sizing ---
    min_position_size: PositiveInt = Field(
        default=1,
        description="Number of contracts for the smallest position size. Should be >= 1.",
    )
    max_position_size: PositiveInt = Field(
        default=10,
        description="Number of contracts for the largest position size. Should be >= min_position_size.",
    )
    risk_per_trade_pct: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="% of equity to risk per trade. E.g., 1.0 means risk 1% of equity on each trade.",
    )

    # --- Daily Loss Limit ---
    max_daily_loss: float = Field(
        default=0.02,
        ge=0.001,
        le=0.20,
        description="Maximum loss allowed per day as a percentage of equity. E.g., 0.02 means 2% of equity.",
    )

    # --- Trailing Stop ---
    use_trailing_stop: bool = Field(
        default=False,
        description="Whether to use a trailing stop loss. If True, the trailing_stop_pct will be used to calculate the trailing stop distance.",
    )
    trailing_atr_multiplier: float = Field(
        default=2.0,
        ge=0.5,
        le=10.0,
        description="ATR multiplier for calculating the trailing stop distance. Only used if use_trailing_stop is True.",
    )

    # --- Paper Trading / Live Execution ---
    entry_ord_type: Literal["LIMIT", "MARKET"] = Field(
        default="LIMIT",
        description="Order type for entry orders. LIMIT = limit order at signal price, MARKET = market order at next bar open.",
    )
    entry_cutoff_seconds: int = Field(
        default=60,
        ge=0,
        le=3600,
        description="Number of seconds after the bar close during which entries are allowed. Only for Paper Trading.",
    )
    allow_late_entry: bool = Field(
        default=False,
        description="Whether to allow entries after the entry cutoff time. Only for Paper Trading.",
    )
    force_flat_on_session_close: bool = Field(
        default=True,
        description="Whether to force flat positions at the end of the trading session. Only for Paper Trading.",
    )
    defer_exit_outside_session: bool = Field(
        default=True,
        description="Whether to defer exit orders that occur outside of trading hours until the next session. Only for Paper Trading.",
    )

    @model_validator(mode="after")
    def validate_position_size_order(self) -> "RiskConfig":
        if self.max_position_size < self.min_position_size:
            raise ValueError(
                f"max_position_size ({self.max_position_size}) must be greater than or equal to min_position_size ({self.min_position_size})."
            )
        return self

    @model_validator(mode="after")
    def validate_trailing_stop_consistency(self) -> "RiskConfig":
        # Only warning instead of erroring out because some users might want to set a trailing stop distance even if they don't use it yet.
        # For example, they might want to set it up in advance and then toggle it on later.
        return self


class BaseConfig(BaseModel):
    """
    Top-level base configuration schema that all strategy configs should inherit from.

    Structure:
    {
        "name": "Strategy Name",
        "version": "1.0.0",
        "description": "A brief description of the strategy.",
        "strategy": { ... strategy-specific config fields ... },
        "risk": { ... risk management config fields ... }
    }
    """

    model_config = ConfigDict(
        extra="forbid",  # Forbid extra fields not defined in the schema
        validate_assignment=True,  # Validate fields on assignment
    )

    name: str = Field(description="Human-readable strategy name.")
    version: str = Field(default="2.0.0", description="Strategy version number.")
    description: str = Field(
        default="",
        description="A brief description of the strategy. Optional but recommended for clarity.",
    )
