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
from typing import Any, TypeVar, overload

from src.data.indicators.base import IndicatorBase

_T = TypeVar("_T")

# Set of supported indicator names. This can be expanded as new indicators are implemented.
# Populated by _INDICATOR_FACTORY at end of file
_SUPPORTED_INDICATORS = {
    "atr",
    "adx",
    "volume_ma",
}


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
        if self.name not in _SUPPORTED_INDICATORS:
            raise ValueError(
                f"Unsupported indicator name: {self.name}. Supported indicators: {_SUPPORTED_INDICATORS}"
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
        return {
            spec.output_column: _INDICATOR_FACTORY[spec.name](**spec.params) for spec in self._specs
        }

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
    indicator_cls = _INDICATOR_FACTORY.get(spec.name)
    if indicator_cls is None:
        return 1

    # Temporarily instantiate the indicator to get its warm-up period.
    try:
        instance = indicator_cls(**spec.params)
        return int(instance.warm_up_required)
    except Exception:
        return 1


def _lazy_factory() -> dict[str, Callable[..., IndicatorBase]]:
    """
    Lazy load indicator classes when first accessed.
    This avoids circular imports and reduces initial load time.
    """
    from src.data.indicators.adx import WilderADX
    from src.data.indicators.atr import WilderATR
    from src.data.indicators.volume_ma import VolumeMA

    return {
        "atr": WilderATR,
        "adx": WilderADX,
        "volume_ma": VolumeMA,
    }


class _LazyFactory(dict[str, Callable[..., IndicatorBase]]):
    """Dict lazy-load indicator classes when first accessed."""

    def __init__(self) -> None:
        super().__init__()
        self._loaded = False

    def __getitem__(self, key: str) -> Callable[..., IndicatorBase]:
        if not self._loaded:
            self.update(_lazy_factory())
            self._loaded = True
        return super().__getitem__(key)

    @overload
    def get(self, key: str, default: None = None) -> Callable[..., IndicatorBase] | None: ...

    @overload
    def get(self, key: str, default: "_T") -> Callable[..., IndicatorBase] | "_T": ...

    def get(self, key: str, default: object = None) -> object:
        if not self._loaded:
            self.update(_lazy_factory())
            self._loaded = True
        return super().get(key, default)


_INDICATOR_FACTORY = _LazyFactory()
