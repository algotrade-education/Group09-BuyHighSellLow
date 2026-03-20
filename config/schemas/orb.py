"""
Pydantic schemas for ORB (Opening Range Breakout) strategy configuration.
"""

from pydantic import Field, model_validator

from config.schemas.base import BaseConfig, BaseStrategyConfig, RiskConfig


class ORBStrategyConfig(BaseStrategyConfig):
    """
    Config for ORB (Opening Range Breakout) strategy.

    Opening Range is defined as the high and low of the first N minutes of the trading day, where N is specified by orb_minutes.
    Breakout only occurs if close price over range + buffer is breached.

    Stop Loss:
        - If use_range_sl is True: SL set at the opposite end of the opening range (e.g., if long, SL is set at the opening range low).
        - If use_range_sl is False: SL set at a ATR multiplier distance from the entry price (e.g., if long, SL is set at entry price - ATR * atr_multiplier).

    Take Profit:
        - Always set at a multiple of ATR distance from the entry price (e.g., if long, TP is set at entry price + ATR * tp_atr_multiplier).

    Filiter (Optional):
        - Volume filter: Only take trades if the volume during the opening range is above a certain threshold.
        - ADX filter: Only take trades if the ADX during the opening range is above a certain threshold, indicating a strong trend.
    """

    # --- Opening Range ---
    orb_minutes: int = Field(
        default=20,
        ge=1,
        le=60,
        description=(
            "Number of minutes after the market open to define the opening range. "
            "E.g., 20 means the opening range is defined by the high and low of the first 20 minutes of trading."
        ),
    )

    # --- ATR Params ---
    atr_period: int = Field(
        default=14,
        ge=5,
        le=50,
        description="Number of bars to use for ATR calculation. E.g., 14 means ATR is calculated using the last 14 bars.",
    )
    atr_tp_multiplier: float = Field(
        default=2.0,
        ge=0.5,
        le=10.0,
        description=(
            "ATR multiplier for calculating the take profit distance from the entry price. "
            "E.g., 2.0 means TP is set at entry price + ATR * 2 for long positions."
        ),
    )
    atr_sl_multiplier: float = Field(
        default=1.0,
        ge=0.5,
        le=10.0,
        description=(
            "ATR multiplier for calculating the stop loss distance from the entry price if use_range_sl is False. "
            "E.g., 1.0 means SL is set at entry price - ATR * 1 for long positions. "
            "Note: If use_range_sl is True, the SL will be set at the opposite end of the opening range instead of using this ATR multiplier."
        ),
    )

    # --- Entry ---
    breakout_buffer: float = Field(
        default=0.0,
        ge=0.0,
        le=5.0,
        description=(
            "Buffer added to the breakout price to avoid false breakouts. "
            "Long entry when close price > opening range high + buffer. "
            "Used to prevent false breakouts by requiring the price to move a certain distance beyond the opening range before triggering an entry."
        ),
    )

    # --- Stop Loss Mode ---
    use_range_sl: bool = Field(
        default=True,
        description=(
            "Whether to set the stop loss at the opposite end of the opening range instead of using an ATR-based stop loss. "
            "If True, the SL will be set at the opening range low for long positions and at the opening range high for short positions. "
            "If False, the SL will be calculated using the atr_sl_multiplier as described above."
        ),
    )

    # --- Range quality filters ---
    min_range_atr: float = Field(
        default=0.5,
        ge=0.0,
        le=10.0,
        description=(
            "Minimum opening range size in terms of ATR to take trades. "
            "E.g., 0.5 means the opening range must be at least 0.5 ATR to consider taking trades, which helps filter out low-volatility days."
        ),
    )
    max_range_atr: float = Field(
        default=3.0,
        ge=0.0,
        le=20.0,
        description=(
            "Maximum opening range size in terms of ATR to take trades. "
            "E.g., 5.0 means the opening range must be no more than 5 ATR to consider taking trades, which helps filter out excessively volatile days."
        ),
    )

    # --- Direction ---
    long_only: bool = Field(
        default=False,
        description="Whether to only take long trades. If False, both long and short trades are allowed based on the breakout direction.",
    )

    # --- Optional Filters ---
    use_volume_filter: bool = Field(
        default=False,
        description=(
            "Whether to use a volume filter based on the opening range volume. "
            "If True, the strategy will only take trades if the total volume during the opening range is above the volume_threshold."
        ),
    )
    volume_filter_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Volume filter threshold as a percentage of the average volume. "
            "E.g., 0.5 means the opening range volume must be at least 50% of the average volume to consider taking trades."
        ),
    )
    volume_ma_period: int = Field(
        default=20,
        ge=5,
        le=100,
        description=(
            "Number of bars to use for calculating the average volume for the volume filter. "
            "E.g., 20 means the average volume is calculated using the last 20 bars."
        ),
    )

    use_adx_filter: bool = Field(
        default=False,
        description=(
            "Whether to use an ADX filter based on the opening range ADX. "
            "If True, the strategy will only take trades if the ADX during the opening range is above the adx_filter_threshold."
        ),
    )
    adx_period: int = Field(
        default=14,
        ge=5,
        le=50,
        description=(
            "Number of bars to use for calculating the ADX for the ADX filter. "
            "E.g., 14 means the ADX is calculated using the last 14 bars."
        ),
    )
    adx_min: float = Field(
        default=20.0,
        ge=0.0,
        le=60.0,
        description=(
            "Minimum ADX value for the ADX filter. "
            "E.g., 20.0 means the ADX must be above 20 to consider taking trades, which helps filter out non-trending days."
        ),
    )

    # --- Breakout confirmation ---
    require_close_confirmation: bool = Field(
        default=False,
        description=(
            "Whether to require the breakout bar to close beyond the breakout level (opening range high + buffer) before entering a trade. "
            "If True, the strategy will only enter a trade if the close price of the breakout bar is above the breakout level for long trades (or below for short trades). "
            "If False, the strategy will enter a trade as soon as the price breaches the breakout level, even if it hasn't closed beyond it yet."
        ),
    )

    # --- Trade Limits ---
    max_trades_per_session: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Maximum number of trades to take per trading session. E.g., 1 means only the first valid breakout will be traded each day.",
    )

    # --- Validators ---
    @model_validator(mode="after")
    def validate_range_atr_order(self) -> "ORBStrategyConfig":
        if (
            self.min_range_atr > 0
            and self.max_range_atr > 0
            and self.min_range_atr >= self.max_range_atr
        ):
            raise ValueError(
                f"min_range_atr ({self.min_range_atr}) must be less than max_range_atr ({self.max_range_atr})."
            )
        return self

    @model_validator(mode="after")
    def validate_tp_gt_sl(self) -> "ORBStrategyConfig":
        """Validate that the take profit distance is greater than the stop loss distance when use_range_sl is False."""
        if not self.use_range_sl and self.atr_tp_multiplier <= self.atr_sl_multiplier:
            raise ValueError(
                f"atr_tp_multiplier ({self.atr_tp_multiplier}) must be greater than atr_sl_multiplier ({self.atr_sl_multiplier}) to ensure R:R positive."
            )
        return self


class ORBConfig(BaseConfig):
    """
    Top-level config for ORB strategy, including risk management and execution parameters.
    This is object load from JSON/YAML and pass into ORB strategy constructor.

    Usage:
        config = ORBConfig.from_json("path/to/orb_config.json")
        strategy = ORBStrategy(config)
    """

    strategy: ORBStrategyConfig
    risk: RiskConfig

    @classmethod
    def from_json(cls, file_path: str) -> "ORBConfig":
        """
        Load and validate ORBConfig from a JSON file.

        Raises:
            - FileNotFoundError: If the specified file does not exist.
            - ValidationError: If the JSON structure does not match the ORBConfig schema or if any field values are invalid.
        """
        import json
        from pathlib import Path

        raw = json.loads(Path(file_path).read_text(encoding="utf-8"))
        return cls.model_validate(raw)

    @classmethod
    def from_dict(cls, data: dict) -> "ORBConfig":
        """
        Load and validate ORBConfig from a dictionary.
        Used for optimization when parameters are generated programmatically.

        Raises:
            - ValidationError: If the dictionary structure does not match the ORBConfig schema or if any field values are invalid.
        """
        return cls.model_validate(data)

    def to_json(self, file_path: str, indent: int = 2) -> None:
        """
        Save ORBConfig to a JSON file.

        Args:
            file_path: Path to the output JSON file.
            indent: Number of spaces for indentation in the output JSON file (default is 2).
        """
        from pathlib import Path

        Path(file_path).write_text(
            self.model_dump_json(indent=indent),
            encoding="utf-8",
        )
