"""
Optuna optimization runner for the Keltner Squeeze Breakout (KSB) strategy.

This script uses Bayesian optimization (TPE) to find the best-performing
parameters for the KSB strategy (BB/KC periods, momentum, thresholds).
It works directly with tick data to allow timeframe optimization.

Run as:
    python -m src.run_optimization_ksb --sample is --trials 300

Search Space:
- Resampling timeframe (1min, 5min).
- Bollinger Band & Keltner Channel periods and standard deviations.
- Momentum lookbacks.
- Strategy filters (Volume, Squeeze duration).

Optimization Score (Maximized):
- Base: Sharpe Ratio.
- Penalties: Max Drawdown.
- Bonuses: Trade Count (to ensure statistical significance).
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data.preprocessor import Preprocessor
from src.optimization.optuna_search import OptunaSearch
from src.strategy.KSB import KeltnerSqueezeBreakout
from src.utils.cli_helpers import (
    load_ksb_config_context,
    load_sample_data,
    prepare_optuna_dataset,
)
from src.utils.logger import setup_logging

logger = setup_logging(__name__, log_file="logs/optuna_ksb.log")


def preprocess_data(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Trial-specific Preprocessing for KSB.

    Calculates the 'squeeze' indicators (BB vs KC) and momentum
    filters dynamically for each Optuna trial.

    Execution:
    1. Resample tick data to trial-specific bar frequency.
    2. Filter for active market hours.
    3. Calculate ATR, Bollinger Bands, Volume MA.
    4. Calculate Keltner Channels (using EMA/ATR).
    5. Calculate Momentum Oscillator.
    """
    resample_freq = params.get("resample_freq", "5min")

    bb_period = params.get("bb_period", 20)
    bb_std = params.get("bb_std", 2.0)
    kc_period = params.get("kc_period", 20)
    kc_mult = params.get("kc_mult", 1.5)
    atr_period = params.get("atr_period", 14)
    mom_period = params.get("mom_period", 12)
    vol_ma_period = params.get("vol_ma_period", 20)

    preprocessor = Preprocessor(
        sma_period=bb_period,
        bb_std=bb_std,
        atr_period=atr_period,
        volume_ma_period=vol_ma_period,
    )

    df = preprocessor.resample_to_ohlc(df, freq=resample_freq)
    df = preprocessor.filter_trading_hours(df, include_atc=True)

    # Core indicators
    df = preprocessor.add_atr(df, period=atr_period, copy=True)
    df = preprocessor.add_bollinger_bands(
        df,
        period=bb_period,
        std_dev=bb_std,
        copy=False,
    )
    df = preprocessor.add_volume_ma(df, period=vol_ma_period, copy=False)

    # Keltner Channels (depends on EMA + ATR)
    df = preprocessor.add_keltner_channels(
        df,
        ema_period=kc_period,
        atr_period=atr_period,
        multiplier=kc_mult,
        copy=False,
    )

    # Momentum oscillator
    df = preprocessor.add_momentum(df, period=mom_period, copy=False)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def run_optuna(
    data: pd.DataFrame,
    config: dict,
    n_trials: int = 300,
    min_trades: int = 120,
) -> None:
    """
    Configure and execute the KSB Optuna study.

    Args:
        data: Raw tick data (DataFrame).
        config: Base configuration for non-optimized settings.
        n_trials: Number of Bayesian iterations.
        min_trades: Hard activity gate for valid trials.
    """
    logger.info(
        "Starting KSB Optuna Optimization with %d trials (min_trades=%d)...",
        n_trials,
        min_trades,
    )

    param_space = {
        "resample_freq": {
            "type": "categorical",
            "choices": ["1min", "5min"],
        },
        # Bollinger Bands - wider range, finer step
        "bb_period": {"type": "int", "low": 10, "high": 35, "step": 1},
        "bb_std": {"type": "float", "low": 1.0, "high": 3.0, "step": 0.1},
        # Keltner Channels - kc_mult close to bb_std for meaningful squeezes
        "kc_period": {"type": "int", "low": 10, "high": 35, "step": 1},
        "kc_mult": {"type": "float", "low": 0.5, "high": 2.5, "step": 0.1},
        # Momentum
        "mom_period": {"type": "int", "low": 3, "high": 25, "step": 1},
        # ATR / Risk - finer steps for SL/TP
        "atr_period": {"type": "int", "low": 8, "high": 24, "step": 1},
        "atr_sl_mult": {"type": "float", "low": 0.5, "high": 3.0, "step": 0.1},
        "atr_tp_mult": {"type": "float", "low": 1.0, "high": 5.0, "step": 0.1},
        # Squeeze filter + signal window - wider range
        "min_squeeze_bars": {"type": "int", "low": 1, "high": 10, "step": 1},
        "signal_window": {"type": "int", "low": 1, "high": 6, "step": 1},
        "cooldown_bars": {"type": "int", "low": 0, "high": 5, "step": 1},
        # Flags
        "long_only": {"type": "categorical", "choices": [True, False]},
        "use_volume_filter": {"type": "categorical", "choices": [True, False]},
        "vol_mult": {"type": "float", "low": 0.5, "high": 2.5, "step": 0.1},
        # Risk management
        "use_trailing_stop": {"type": "categorical", "choices": [True, False]},
        "trailing_atr_multiplier": {
            "type": "float",
            "low": 0.5,
            "high": 3.5,
            "step": 0.1,
        },
    }

    risk_params = config.get("risk", {})
    backtester_kwargs = {
        "max_daily_loss_pct": risk_params.get("max_daily_loss", 0.0),
        "entry_cutoff_seconds": risk_params.get("entry_cutoff_seconds"),
        "allow_late_entry": risk_params.get("allow_late_entry"),
    }

    optimizer = OptunaSearch(
        strategy_class=KeltnerSqueezeBreakout,
        param_space=param_space,
        indicator_fn=preprocess_data,
        min_trades=min_trades,
        drawdown_penalty=0.1,
        turnover_penalty=0.0,  # do not penalize more trades
        trade_count_bonus=0.1,  # reward more trades among configs
        n_trials=n_trials,
        backtester_kwargs=backtester_kwargs,
    )

    optimizer.optimize(data, raw_data=True)

    optimizer.print_study_summary()
    optimizer.print_top_results(10)

    output_path = optimizer.save_results()
    logger.info("Results saved to: %s", output_path)

    if optimizer.best_params:
        params_dir = Path("config/strategy_params")
        params_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d")
        params_path = params_dir / f"ksb_optuna_{timestamp}.json"

        best_config = {
            "name": f"KSB Optuna Optimized {timestamp}",
            "description": "Auto-optimized KSB parameters from Optuna TPE",
            "version": config.get("version", "1.0.0"),
            "strategy": config.get("strategy", {}).copy(),
            "risk": config.get("risk", {}).copy(),
        }

        risk_keys = {"use_trailing_stop", "trailing_atr_multiplier"}
        strategy_update = {
            k: v for k, v in optimizer.best_params.items() if k not in risk_keys
        }
        risk_update = {k: v for k, v in optimizer.best_params.items() if k in risk_keys}

        best_config["strategy"].update(strategy_update)
        best_config["risk"].update(risk_update)

        with open(params_path, "w") as f:
            json.dump(best_config, f, indent=2)

        logger.info("Best params saved to: %s", params_path)
        print(f"\nBest config saved to: {params_path}")
        print("Run backtest with:")
        print(f"  python -m src.run_backtest_ksb --sample is --config {params_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run KSB Optuna optimization.")
    parser.add_argument(
        "--sample",
        choices=["is", "os"],
        default="is",
        help="Sample type: is (in-sample) or os (out-of-sample).",
    )
    parser.add_argument(
        "--contract", default="VN30F1M", help="Contract symbol (default: VN30F1M)."
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=300,
        help="Number of Optuna trials (default: 300).",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=50,
        metavar="N",
        help="Minimum trades for a valid trial (default: 50).",
    )
    args = parser.parse_args()

    config, _, _ = load_ksb_config_context("config/strategy_params/ksb_default.json")
    logger.info("Loaded KSB configuration")

    data = load_sample_data(sample=args.sample, contract=args.contract)
    data = prepare_optuna_dataset(data)
    logger.info("Cleaned tick data shape: %s", data.shape)

    run_optuna(
        data,
        config,
        n_trials=args.trials,
        min_trades=args.min_trades,
    )
