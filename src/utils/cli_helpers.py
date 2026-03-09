"""
Shared helpers for CLI runner orchestration.

These helpers intentionally do not manage argparse to keep each runner's
command-line surface independent and easy to extend.
"""

from typing import Any, Dict, Tuple, Optional

import pandas as pd

from src.data.preprocessor import Preprocessor
from src.run_data_loader import load_data
from src.strategy.KSB import KeltnerSqueezeBreakout
from src.strategy.ORB import OpeningRangeBreakout
from src.strategy.VWAP import VWAPBandReversion
from src.utils.config_loader import load_config


def load_orb_config_context(
    config_path: str,
    default_resample_freq: str = "5min",
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """Load ORB config and return config, strategy params, and resample frequency."""
    config = load_config(config_path)
    strategy_params = config.get("strategy", {})
    resample_freq = strategy_params.get("resample_freq", default_resample_freq)
    return config, strategy_params, resample_freq


def build_orb_strategy(strategy_params: Dict[str, Any]) -> OpeningRangeBreakout:
    """Build ORB strategy while stripping non-constructor config keys."""
    strategy_kwargs = {
        key: value for key, value in strategy_params.items() if key != "resample_freq"
    }
    return OpeningRangeBreakout(**strategy_kwargs)


def load_ksb_config_context(
    config_path: str,
    default_resample_freq: str = "5min",
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """Load KSB config and return config, strategy params, and resample frequency."""
    config = load_config(config_path)
    strategy_params = config.get("strategy", {})
    resample_freq = strategy_params.get("resample_freq", default_resample_freq)
    return config, strategy_params, resample_freq


def build_ksb_strategy(strategy_params: Dict[str, Any]) -> KeltnerSqueezeBreakout:
    """Build KSB strategy while stripping non-constructor config keys."""
    strategy_kwargs = {
        key: value for key, value in strategy_params.items() if key != "resample_freq"
    }
    return KeltnerSqueezeBreakout(**strategy_kwargs)


def load_vwap_config_context(
    config_path: str,
    default_resample_freq: str = "5min",
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """Load VWAP config and return config, strategy params, and resample frequency."""
    config = load_config(config_path)
    strategy_params = config.get("strategy", {})
    resample_freq = strategy_params.get("resample_freq", default_resample_freq)
    return config, strategy_params, resample_freq


def build_vwap_strategy(strategy_params: Dict[str, Any]) -> VWAPBandReversion:
    """Build VWAP strategy while stripping non-constructor config keys."""
    strategy_kwargs = {
        key: value for key, value in strategy_params.items() if key != "resample_freq"
    }
    return VWAPBandReversion(**strategy_kwargs)


def load_sample_data(sample: str, contract: str) -> pd.DataFrame:
    """Load raw IS/OS sample data for a contract symbol."""
    return load_data(sample=sample, contract=contract)


def prepare_backtest_dataset(
    raw_data: pd.DataFrame,
    strategy_params: Dict[str, Any],
    resample_freq: str,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """Prepare bar and indicator dataset for backtest or sim execution.

    Important: indicator columns must align with the strategy parameters
    (e.g. `mom_{mom_period}`, BB/KC computed at the configured periods),
    otherwise the strategy may never trigger entries.
    """
    bb_period = int(strategy_params.get("bb_period", 20))
    bb_std = float(strategy_params.get("bb_std", 2.0))
    kc_period = int(strategy_params.get("kc_period", 20))
    kc_mult = float(strategy_params.get("kc_mult", 1.5))
    atr_period = int(strategy_params.get("atr_period", 14))
    mom_period = int(strategy_params.get("mom_period", 12))
    vol_ma_period = int(strategy_params.get("vol_ma_period", 20))

    preprocessor = Preprocessor(
        sma_period=bb_period,
        bb_std=bb_std,
        atr_period=atr_period,
        volume_ma_period=vol_ma_period,
    )

    df = preprocessor.clean_data(raw_data)
    df = preprocessor._derive_volume(df, copy=False)
    df = preprocessor.resample_to_ohlc(df, freq=resample_freq)
    df = preprocessor.filter_trading_hours(df, include_atc=True)

    # Core indicators (align to strategy params)
    df = preprocessor.add_atr(df, period=atr_period, copy=True)
    df = preprocessor.add_bollinger_bands(
        df, period=bb_period, std_dev=bb_std, copy=False
    )
    df = preprocessor.add_volume_ma(df, period=vol_ma_period, copy=False)
    df = preprocessor.add_keltner_channels(
        df,
        ema_period=kc_period,
        atr_period=atr_period,
        multiplier=kc_mult,
        copy=False,
    )
    df = preprocessor.add_momentum(df, period=mom_period, copy=False)

    # Session VWAP + std bands (used by VWAP strategy)
    df = preprocessor.add_session_vwap(df, copy=False)

    incomplete_bar = None
    if not df.empty:
        # Check if the very last bar is "incomplete" (in the future or currently forming)
        # by looking at whether it has valid indicator values (which dropna will strip) or
        # if the system time is within its bucket. In our case, the easiest way to preserve
        # the currently forming intraday bar is to extract the last row *before* we run dropna().
        last_row = df.iloc[-1].copy()

        # Check if the last row's datetime is "today" and the time matches the current forming bucket.
        # Alternatively, we just extract it if its volume is > 0 and it would be dropped.
        # But safely, we can just extract the raw OHLCV of the last row before dropping NA
        incomplete_bar = {
            "datetime": last_row["datetime"],
            "open": float(last_row["open"]),
            "high": float(last_row["high"]),
            "low": float(last_row["low"]),
            "close": float(last_row["close"]),
            "volume": float(last_row["volume"]),
        }

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Only return the incomplete bar if we actually dropped it (meaning it was the incomplete leading edge)
    # OR if it's explicitly today's currently forming bar.
    # For paper trading, we return it so the engine can decide whether to seed it.
    return df, incomplete_bar


def prepare_optimization_dataset(
    raw_data: pd.DataFrame,
    resample_freq: str,
) -> pd.DataFrame:
    """Prepare optimization dataset with configured resampling/indicators."""
    return Preprocessor().prepare_for_optimization(
        raw_data, resample_freq=resample_freq
    )


def prepare_optuna_dataset(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Clean tick data and derive volume columns for Optuna trials."""
    preprocessor = Preprocessor()
    cleaned = preprocessor.clean_data(raw_data)
    cleaned = preprocessor._derive_volume(cleaned, copy=False)
    return cleaned
