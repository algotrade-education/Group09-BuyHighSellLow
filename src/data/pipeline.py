"""
Calculate indicators from IndicatorRegistry and cache results.

Cache key = hash(data fingerprint + registry params)
-> 300 Optuna trials with the same atr_period=14 will only compute ATR once, then load from cache for the rest.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

from src.data.indicators.registry import IndicatorRegistry
from src.utils.hashing import hash_str

logger = logging.getLogger(__name__)

DATETIME_COL = "datetime"
_CACHE_VERSION = "v2"  # Increase when indicator logic changes to invalidate old cache
_CACHE_FILE_PREFIX = "ind"


class DataPipeline:
    """
    Build indicator columns from IndicatorRegistry.

    Usage:
        from src.strategy import ORBStrategy

        registry = ORBStrategy.build_registry(atr_period=14, adx_period=14)
        pipeline = DataPipeline(registry, cache_dir="data/cache")

        df_with_indicators = pipeline.run(df)
        # df now includes additional columns: atr_14, adx_14, volume_ma_20

    Cache:
        First run: compute indicators and save to parquet.
        Later runs: load from cache if data and params are unchanged.
        Invalidate: changing registry params or data triggers a cache miss automatically.
    """

    def __init__(
        self,
        registry: IndicatorRegistry,
        cache_dir: str = "data/cache",
        use_cache: bool = True,
    ) -> None:
        self._registry = registry
        self._cache_dir = Path(cache_dir)
        self._use_cache = use_cache

    # --- Public API ---

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all indicators in the registry and add to DataFrame.

        Args:
            df: OHLCV DataFrame after preprocessor.clean() + filter_trading_hours().

        Returns:
            DataFrame with original columns + indicator columns.
            Rows in warm_up period has NaN - use dropna() if needed.
        """
        if df.empty:
            return df

        if not self._registry.get_all():
            logger.debug("Registry is empty, no indicators to compute.")
            return df

        # Check cache
        cache_key = "placeholder"  # Set this for mypy to be happy (since it error unbound variable)
        if self._use_cache:
            cache_key = self._build_cache_key(df)
            cached = self._load_cache(cache_key)
            if cached is not None:
                logger.debug("Cache hit: %s", cache_key[:12])
                return cached

        # Compute
        logger.debug(
            "Computing %d indicators for %d bars...",
            len(self._registry),
            len(df),
        )
        result = self._compute(df)

        # Save cache
        if self._use_cache:
            self._save_cache(cache_key, result)

        return result

    def get_required_lookback(self) -> int:
        """Maximum lookback bars needed across all indicators in the registry."""
        return self._registry.get_required_lookback()

    # --- Compute ---

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all indicators and add as columns to df.
        Uses compute_vectorized() if available (fast numpy path),
        falls back to bar-by-bar update() loop otherwise.
        """
        df = df.copy()
        indicators = self._registry.build_indicators()

        for col, indicator in indicators.items():
            df[col] = indicator.compute_vectorized(df)

        logger.debug("Computed indicators: %s", list(indicators.keys()))
        return df

    def _feed_indicator(self, indicator: object, output_column: str, bar: dict) -> object:
        """Feed a single bar dict into an indicator's update() method.

        Returns the indicator value, or None if required inputs are missing
        or the indicator is still in its warm-up period.

        Args:
            indicator: An IndicatorBase instance with an update(**kwargs) method.
            output_column: Column name (unused here, kept for API symmetry).
            bar: Bar dict that may or may not contain required fields.

        Returns:
            Indicator value or None.
        """
        required: frozenset[str] = getattr(indicator, "required_inputs", frozenset())
        if required and not required.issubset(bar.keys()):
            return None
        try:
            update_fn = getattr(indicator, "update", None)
            if update_fn is None:
                return None
            return update_fn(**{k: bar[k] for k in required})
        except Exception:
            return None

    # --- Cache ---

    def _build_cache_key(self, df: pd.DataFrame) -> str:
        """
        Hash key = data fingerprint + registry params + cache version.

        Data fingerprint: shape + first/last datetime + dtypes + values hash.
        The values hash catches in-place mutations (e.g. close += 10) that
        leave shape and timestamps unchanged.
        """
        import numpy as np

        values_hash = hashlib.md5(
            np.asarray(pd.util.hash_pandas_object(df, index=False)).tobytes()
        ).hexdigest()

        data_sig = "|".join(
            [
                str(df.shape),
                str(df[DATETIME_COL].iloc[0]) if DATETIME_COL in df.columns else "",
                str(df[DATETIME_COL].iloc[-1]) if DATETIME_COL in df.columns else "",
                str(df.dtypes.to_dict()),
                values_hash,
            ]
        )

        registry_sig = "|".join(
            f"{s.name}:{s.params}:{s.output_column}" for s in self._registry.get_all()
        )

        combined = f"{_CACHE_VERSION}|{data_sig}|{registry_sig}"
        return hash_str(combined)

    def _load_cache(self, cache_key: str) -> pd.DataFrame | None:
        path = self._cache_path(cache_key)
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)

            # Verify all expected indicator columns are present - if not, invalidate cache
            expected_cols = set(self._registry.get_all_output_columns())
            if not expected_cols.issubset(df.columns):
                logger.warning(
                    "Cache '%s' thiếu columns %s. Recomputing.",
                    cache_key[:12],
                    expected_cols - set(df.columns),
                )
                path.unlink(missing_ok=True)

                return None
            return df
        except Exception as e:
            logger.warning("Cache load failed (%s): %s. Recomputing.", cache_key[:12], e)
            path.unlink(missing_ok=True)
            return None

    def _save_cache(self, cache_key: str, df: pd.DataFrame) -> None:
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            path = self._cache_path(cache_key)

            df.to_parquet(path, index=False)
            logger.debug("Cache saved: %s", cache_key[:12])
        except Exception as e:
            logger.warning("Cache save failed: %s", e)

    def _cache_path(self, cache_key: str) -> Path:
        return self._cache_dir / f"{_CACHE_FILE_PREFIX}_{cache_key}.parquet"

    def clear_cache(self) -> int:
        """Delete all cache files. Returns number of files deleted."""
        if not self._cache_dir.exists():
            return 0

        count = 0
        for f in self._cache_dir.glob(f"{_CACHE_FILE_PREFIX}_*.parquet"):
            f.unlink()
            count += 1

        logger.info("Cleared %d cache files from %s.", count, self._cache_dir)
        return count
