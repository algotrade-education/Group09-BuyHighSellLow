"""
Shared helpers for CLI runner orchestration.

These helpers intentionally do not manage argparse to keep each runner's
command-line surface independent and easy to extend.
"""

from typing import Any, Dict, Tuple

import pandas as pd

from src.data.preprocessor import Preprocessor
from src.run_data_loader import load_data
from src.strategy.ORB import OpeningRangeBreakout
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


def load_sample_data(sample: str, contract: str) -> pd.DataFrame:
    """Load raw IS/OS sample data for a contract symbol."""
    return load_data(sample=sample, contract=contract)


def prepare_backtest_dataset(
    raw_data: pd.DataFrame,
    strategy_params: Dict[str, Any],
    resample_freq: str,
) -> pd.DataFrame:
    """Prepare bar+indicator dataset for backtest or sim execution."""
    preprocessor = Preprocessor(atr_period=strategy_params.get("atr_period", 14))
    return preprocessor.prepare_for_backtest(raw_data, resample_freq=resample_freq)


def prepare_optimization_dataset(
    raw_data: pd.DataFrame,
    resample_freq: str,
) -> pd.DataFrame:
    """Prepare optimization dataset with configured resampling/indicators."""
    return Preprocessor().prepare_for_optimization(raw_data, resample_freq=resample_freq)


def prepare_optuna_dataset(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Clean tick data and derive volume columns for Optuna trials."""
    preprocessor = Preprocessor()
    cleaned = preprocessor.clean_data(raw_data)
    cleaned = preprocessor._derive_volume(cleaned, copy=False)
    return cleaned