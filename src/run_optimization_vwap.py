"""
Optuna optimization runner for the VWAP Band Reversion strategy.

This script executes Bayesian optimization for VWAP entries. Since VWAP
itself has no tunable lookback (it is a volume-weighted average since
session start), the search space focuses on standard deviation band widths,
session warmup periods, and risk management filters.

Run as:
    python -m src.run_optimization_vwap --sample is --trials 300

Search Space:
- Entry band standard deviation (e.g., 1.5σ to 3.0σ).
- Session warmup duration (bars to wait before valid entries).
- ATR-based stop loss and take profit multipliers.
- VWAP slope and volume filters.

Optimization Score (Maximized):
- Base: Sharpe Ratio.
- Penalties: Max Drawdown.
- Bonuses: Trade count (weighted less than ORB/KSB due to reversion nature).
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data.preprocessor import Preprocessor
from src.optimization.optuna_search import OptunaSearch
from src.strategy.VWAP import VWAPBandReversion
from src.utils.cli_helpers import (
    load_sample_data,
    load_vwap_config_context,
    prepare_optuna_dataset,
)
from src.utils.logger import setup_logging

logger = setup_logging(__name__, log_file="logs/optuna_vwap.log")


def preprocess_data(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Trial-specific Preprocessing for VWAP.

    Execution:
    1. Resample tick data to trial-specific bar frequency.
    2. Filter for active market hours.
    3. Calculate ATR & Volume MA.
    4. Calculate Session VWAP and standard deviation bands.
    """
    resample_freq = params.get("resample_freq", "5min")
    atr_period = params.get("atr_period", 14)
    vol_ma_period = params.get("vol_ma_period", 20)

    preprocessor = Preprocessor(
        atr_period=atr_period,
        volume_ma_period=vol_ma_period,
    )

    df = preprocessor.resample_to_ohlc(df, freq=resample_freq)
    df = preprocessor.filter_trading_hours(df, include_atc=True)

    df = preprocessor.add_atr(df, period=atr_period, copy=True)
    df = preprocessor.add_volume_ma(df, period=vol_ma_period, copy=False)
    df = preprocessor.add_session_vwap(df, copy=False)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def run_optuna(
    data: pd.DataFrame,
    config: dict,
    n_trials: int = 300,
    min_trades: int = 100,
) -> None:
    """
    Configure and execute the VWAP Optuna study.

    Args:
        data: Raw tick data (DataFrame).
        config: Base configuration.
        n_trials: Iterations.
        min_trades: Validity threshold.
    """
    logger.info(
        "Starting VWAP Optuna Optimization with %d trials (min_trades=%d)...",
        n_trials,
        min_trades,
    )

    param_space = {
        "resample_freq": {
            "type": "categorical",
            "choices": ["1min", "5min"],
        },
        # Entry band width (how far from VWAP to enter)
        "entry_band": {"type": "float", "low": 1.0, "high": 3.0, "step": 0.25},
        # ATR / Risk
        "atr_period": {"type": "int", "low": 10, "high": 20, "step": 1},
        "atr_sl_mult": {"type": "float", "low": 0.5, "high": 3.0, "step": 0.1},
        "min_tp_atr": {"type": "float", "low": 0.2, "high": 1.5, "step": 0.1},
        # Timing
        "cooldown_bars": {"type": "int", "low": 0, "high": 5, "step": 1},
        "session_warmup": {"type": "int", "low": 5, "high": 20, "step": 1},
        # Flags
        "long_only": {"type": "categorical", "choices": [True, False]},
        "use_slope_filter": {"type": "categorical", "choices": [True, False]},
        "slope_period": {"type": "int", "low": 3, "high": 10, "step": 1},
        "use_volume_filter": {"type": "categorical", "choices": [True, False]},
        "vol_mult": {"type": "float", "low": 1.0, "high": 2.0, "step": 0.25},
        # Risk management
        "use_trailing_stop": {"type": "categorical", "choices": [True, False]},
        "trailing_atr_multiplier": {
            "type": "float",
            "low": 1.0,
            "high": 3.0,
            "step": 0.25,
        },
    }

    risk_params = config.get("risk", {})
    backtester_kwargs = {
        "max_daily_loss_pct": risk_params.get("max_daily_loss", 0.0),
    }

    optimizer = OptunaSearch(
        strategy_class=VWAPBandReversion,
        param_space=param_space,
        indicator_fn=preprocess_data,
        min_trades=min_trades,
        drawdown_penalty=0.1,  # penalize larger drawdowns
        turnover_penalty=0.0,  # do not penalize higher trade count
        trade_count_bonus=0.05,  # softer reward for more trades vs ORB/KSB
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
        params_path = params_dir / f"vwap_optuna_{timestamp}.json"

        best_config = {
            "name": f"VWAP Optuna Optimized {timestamp}",
            "description": "Auto-optimized VWAP parameters from Optuna TPE",
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
        print(f"  python -m src.run_backtest_vwap --sample is --config {params_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run VWAP Optuna optimization.")
    parser.add_argument(
        "--sample",
        choices=["is", "os"],
        default="is",
        help="Sample type: is (in-sample) or os (out-of-sample).",
    )
    parser.add_argument(
        "--contract",
        default="VN30F1M",
        help="Contract symbol (default: VN30F1M).",
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
        help="Minimum trades for a valid trial (default: 100).",
    )
    args = parser.parse_args()

    config, _, _ = load_vwap_config_context("config/strategy_params/vwap_default.json")
    logger.info("Loaded VWAP configuration")

    data = load_sample_data(sample=args.sample, contract=args.contract)
    data = prepare_optuna_dataset(data)
    logger.info("Cleaned tick data shape: %s", data.shape)

    run_optuna(
        data,
        config,
        n_trials=args.trials,
        min_trades=args.min_trades,
    )
