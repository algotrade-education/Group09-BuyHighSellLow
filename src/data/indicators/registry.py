"""
Strategy registers/subscribe to indicators it needs via this registry.
DataPipeline only instantiates and updates the indicators registered here.

Usage:
    registry = IndicatorRegistry()
    registry.register(IndicatorSpec(name="atr", params={"period": 14}, output_column="atr_14"))
    registry.register(IndicatorSpec(name="adx", params={"period": 14}, output_column="adx_14"))

    pipeline = DataPipeline(indicator_registry=registry)
    df_with_indicators = pipeline.run(df)
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.data.indicators.base import IndicatorBase

# --- Indicator Factory ---
# Plain dict, populated lazily on first access to avoid circular imports and reduce initial load time.
#
# To add new indicator:
# 1. Implement the indicator class in src/data/indicators/, inheriting from IndicatorBase.
# 2. Call register_indicator("name", YourIndicatorClass) to add it to the factory.
# 3. No need for manual imports in this file - the factory will lazy load the class when first accessed.

_factory: dict[str, Callable[..., IndicatorBase]] = {}
_factory_loaded = False


def _ensure_factory_loaded() -> None:
    global _factory_loaded
    if _factory_loaded:
        return

    from src.data.indicators.adx import WilderADX
    from src.data.indicators.atr import WilderATR
    from src.data.indicators.volume_ma import VolumeMA

    _factory.update(
        {
            "atr": WilderATR,
            "adx": WilderADX,
            "volume_ma": VolumeMA,
        }
    )
    _factory_loaded = True


def register_indicator(name: str, cls: Callable[..., IndicatorBase]) -> None:
    """
    Register a new indicator class in the factory.

    Call this function at the end of the indicator module after defining the class, e.g.:
        # At the end of file of macd.py
        register_indicator("macd", MACD)

    Then use the indicator in the registry:
        registry.register(IndicatorSpec(name="macd", params={"fast_period": 12, "slow_period": 26, "signal_period": 9}, output_column="macd_12_26_9"))
    """
    _ensure_factory_loaded()
    _factory[name] = cls


def get_indicator_class(name: str) -> Callable[..., IndicatorBase] | None:
    """Get the indicator class from the factory by name. None if not found."""
    _ensure_factory_loaded()
    return _factory.get(name)


def _get_supported_indicators() -> set[str]:
    """Get the set of supported indicator names from the factory."""
    _ensure_factory_loaded()
    return set(_factory.keys())


# --- Indicator Registry and Spec ---


@dataclass
class IndicatorSpec:
    """
    Specification for an indicator.

    Attributes:
        name: Name of the indicator, e.g., "atr", "adx".
        params: Parameters for the indicator, e.g., {"period": 14}.
        output_column: Column name in the DataFrame output where the indicator values will be stored, e.g., "atr_14".

    Example:
        IndicatorSpec(name="atr", params={"period": 14}, output_column="atr_14")
    """

    name: str  # "atr", "adx", etc.
    params: dict[str, Any] = field(default_factory=dict)  # e.g., {"period": 14}
    output_column: str = ""  # Column name in DataFrame output, e.g., "atr_14"

    def __post_init__(self) -> None:
        """
        Validate the indicator name and auto-generate output_column if not provided.
        """
        supported = _get_supported_indicators()
        if self.name not in supported:
            raise ValueError(
                f"Unsupported indicator name: {self.name}.Supported indicators: {supported}"
            )

        # Auto generate output_column if not provided, e.g., "atr_14" for ATR with period 14
        if not self.output_column:
            if self.params:
                param_str = "_".join(str(v) for v in self.params.values())
                self.output_column = f"{self.name}_{param_str}"
            else:
                self.output_column = self.name


class IndicatorRegistry:
    """
    Strategy registers/subscribe to indicators it needs via this registry.
    The registry is responsible for instantiating and updating the indicators based on the provided specifications.
    """

    def __init__(self) -> None:
        self._specs: list[IndicatorSpec] = []
        self._output_columns: set[str] = set()

    def register(self, spec: IndicatorSpec) -> "IndicatorRegistry":
        """
        Add an indicator specification to the registry.

        Raises:
            ValueError: If the output column name is already registered by another indicator.

        Returns:
            self: To allow method chaining.
        """
        if spec.output_column in self._output_columns:
            raise ValueError(
                f"Output column '{spec.output_column}' is already registered by another indicator. "
                f"Please choose a unique output column name for each indicator."
            )

        self._specs.append(spec)
        self._output_columns.add(spec.output_column)
        return self

    def get_all(self) -> list[IndicatorSpec]:
        """
        Get all registered indicator specifications.

        Returns:
            List of IndicatorSpec objects.
        """
        return list(self._specs)

    def get_all_output_columns(self) -> list[str]:
        """
        Get all output column names for the registered indicators.

        Returns:
            List of output column names.
        """
        return [spec.output_column for spec in self._specs]

    def get_required_lookback(self) -> int:
        """
        Calculate the maximum lookback period required by all registered indicators.
        Used for:
            - Set embargo period in walk-forward validation to avoid look-ahead bias.
            - Validate data sufficiency before running the pipeline.
        """
        if not self._specs:
            return 0

        return max(_get_warm_up(spec) for spec in self._specs)

    def build_indicators(self) -> dict[str, "IndicatorBase"]:
        """
        Instantiates the indicator objects based on the registered specifications.

        Returns:
            A dictionary mapping output column names to their corresponding IndicatorBase instances.
        """
        results: dict[str, IndicatorBase] = {}
        for spec in self._specs:
            cls = get_indicator_class(spec.name)
            if cls is None:
                raise ValueError(f"Indicator class for '{spec.name}' not found in factory.")

            results[spec.output_column] = cls(**spec.params)

        return results

    def __len__(self) -> int:
        return len(self._specs)

    def __repr__(self) -> str:
        specs_str = ", ".join(s.output_column for s in self._specs)
        return f"IndicatorRegistry([{specs_str}])"


# ── Helpers ───────────────────────────────────────────────────────


def _get_warm_up(spec: IndicatorSpec) -> int:
    """
    Get the warm-up period required for an indicator spec.
    Returns 1 if the indicator is not found or fails to instantiate.
    """
    indicator_cls = get_indicator_class(spec.name)
    if indicator_cls is None:
        return 1

    # Temporarily instantiate the indicator to get its warm-up period.
    try:
        instance = indicator_cls(**spec.params)
        return int(instance.warm_up_required)
    except Exception:
        return 1
