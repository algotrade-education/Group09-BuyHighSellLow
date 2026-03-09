"""
Optuna-based Bayesian optimization for strategy parameters.

Unlike grid search (exhaustive), Optuna uses Tree-structured Parzen Estimator (TPE)
to intelligently explore the parameter space - it learns which regions are promising
and focuses the search there.

Composite objective function (configurable):
    Base:
        score = sharpe - |drawdown_penalty * max_drawdown|
    Optional gates/bonuses:
        - Require a minimum number of trades (min_trades)
        - Require minimum total return / profit factor
        - Reward or penalize trade count via trade_count_bonus / turnover_penalty

Guardrails:
    - If total_trades <= min_trades: score = -10.0 (invalid)
    - If total_return/profit_factor below configured minimums: score = -10.0 (invalid)
    - If sharpe <= 0: fall back to total_return as secondary metric
    - If any error: score = -100.0
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

try:
    import optuna
    from optuna import Trial

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from config.config import CONTRACT_MULTIPLIER, DEFAULT_INITIAL_CAPITAL, RESULTS_DIR
from src.engine.backtester import Backtester

logger = logging.getLogger(__name__)


@dataclass
class OptunaResult:
    """Serializable snapshot of one Optuna trial outcome."""

    trial_number: int
    params: Dict[str, Any]
    score: float
    metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        result = {"trial": self.trial_number, "score": self.score}
        result.update(self.params)
        result.update(self.metrics)
        return result


class OptunaSearch:
    """
    Optuna-based optimizer for strategy parameters.

    Uses Bayesian optimization (TPE sampler) with a composite objective
    that balances Sharpe ratio, drawdown, and trade activity.
    """

    def __init__(
        self,
        strategy_class: type,
        param_space: Dict[str, Any],
        indicator_fn: Optional[
            Callable[[pd.DataFrame, Dict[str, Any]], pd.DataFrame]
        ] = None,
        min_trades: int = 50,
        drawdown_penalty: float = 0.1,
        turnover_penalty: float = 0.0,
        trade_count_bonus: float = 0.0,
        min_return_pct: float = -999.0,
        min_profit_factor: float = -999.0,
        n_trials: int = 200,
        seed: int = 42,
        backtester_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Optuna optimizer.

        Args:
            strategy_class: Strategy class to instantiate.
            param_space: Dictionary defining the parameter search space.
                Each key maps to a dict with:
                    - "type": "int", "float", or "categorical"
                    - "low", "high": range bounds (for int/float)
                    - "step": step size (optional, for int/float)
                    - "choices": list of values (for categorical)
                Example:
                    {
                        "bb_std": {"type": "float", "low": 1.5, "high": 3.0, "step": 0.5},
                        "adx_threshold": {"type": "int", "low": 20, "high": 40, "step": 5},
                    }
            indicator_fn: Function to recalculate indicators for each param set.
            min_trades: Minimum trades required for a valid result.
            drawdown_penalty: Weight for max drawdown in composite score.
            turnover_penalty: Weight for turnover (trade count / 1000) penalty.
            trade_count_bonus: Reward weight for trade count (trades / 1000).
            min_return_pct: Minimum total_return_pct required to be considered valid.
            min_profit_factor: Minimum profit_factor required to be considered valid.
            n_trials: Number of optimization trials.
            seed: Random seed for reproducibility.
            backtester_kwargs: Optional dictionary of keyword arguments to pass
                               to the Backtester constructor.
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for this optimizer. "
                "Install it with: pip install optuna"
            )

        self.strategy_class = strategy_class
        self.param_space = param_space
        self.indicator_fn = indicator_fn
        self.min_trades = min_trades
        self.drawdown_penalty = drawdown_penalty
        self.turnover_penalty = turnover_penalty
        self.trade_count_bonus = trade_count_bonus
        self.min_return_pct = min_return_pct
        self.min_profit_factor = min_profit_factor
        self.n_trials = n_trials
        self.seed = seed

        self.results: List[OptunaResult] = []
        self.study: Optional[optuna.Study] = None
        self.backtester_kwargs = backtester_kwargs or {}

    @property
    def _study_name(self) -> str:
        """Build study name from strategy class to avoid stale naming."""
        strategy_name = getattr(self.strategy_class, "__name__", "Strategy")
        return f"{strategy_name}_Optimization"

    def _sample_params(self, trial: "Trial") -> Dict[str, Any]:
        """
        Sample parameters from the defined search space using Optuna's trial.

        Args:
            trial: Optuna trial object.

        Returns:
            Dictionary of sampled parameters.
        """
        params = {}
        for name, spec in self.param_space.items():
            param_type = spec.get("type", "float")

            if param_type == "int":
                params[name] = trial.suggest_int(
                    name,
                    spec["low"],
                    spec["high"],
                    step=spec.get("step", 1),
                )
            elif param_type == "float":
                step = spec.get("step")
                if step:
                    params[name] = trial.suggest_float(
                        name,
                        spec["low"],
                        spec["high"],
                        step=step,
                    )
                else:
                    params[name] = trial.suggest_float(
                        name,
                        spec["low"],
                        spec["high"],
                    )
            elif param_type == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
            else:
                raise ValueError(f"Unknown param type '{param_type}' for '{name}'")

        return params

    def _calculate_score(self, metrics: Dict[str, float], total_trades: int) -> float:
        """
        Calculate composite optimization score.

        Base:
            score = Sharpe - |drawdown_penalty * MaxDrawdown|
        Optional:
            - turnover_penalty * trades/1000 (penalize excessive turnover)
            - trade_count_bonus * trades/1000 (reward trade activity)

        Guardrails:
            - trades <= min_trades -> -10.0
            - total_return_pct < min_return_pct -> -20.0
            - profit_factor < min_profit_factor -> -20.0
            - sharpe <= 0 -> use total_return / 100 as fallback (scaled)

        Args:
            metrics: Performance metrics dictionary.
            total_trades: Number of completed trades.

        Returns:
            Composite score (higher is better).
        """
        # Guard: too few trades
        if total_trades <= self.min_trades:
            return -10.0

        sharpe = metrics.get("sharpe_ratio", 0.0)
        max_dd = abs(metrics.get("max_drawdown_pct", 0.0))
        total_return = metrics.get("total_return_pct", 0.0)
        profit_factor = metrics.get("profit_factor", 0.0)

        # Guard: insufficient profitability (configurable)
        if total_return <= self.min_return_pct:
            return -20.0
        if profit_factor <= self.min_profit_factor:
            return -20.0

        # If Sharpe is non-positive, fall back to total return (scaled down)
        if sharpe <= 0:
            base_score = total_return / 100.0
        else:
            base_score = sharpe

        # Composite: penalize drawdown and optionally adjust for trade count
        dd_penalty = self.drawdown_penalty * max_dd
        turnover = self.turnover_penalty * (total_trades / 1000.0)
        trade_bonus = self.trade_count_bonus * (total_trades / 1000.0)

        score = base_score - dd_penalty - turnover + trade_bonus

        return score

    def _objective(
        self,
        trial: "Trial",
        data: pd.DataFrame,
        initial_capital: float,
        contract_multiplier: float,
    ) -> float:
        """
        Optuna objective function - runs one backtest and returns the composite score.

        Args:
            trial: Optuna trial.
            data: Clean data (OHLCV or raw tick data if raw_data=True).
            initial_capital: Starting capital.
            contract_multiplier: Contract multiplier.

        Returns:
            Composite score.
        """
        try:
            # Sample parameters
            params = self._sample_params(trial)

            # Recalculate indicators (and optionally resample) per trial
            test_data = data
            if self.indicator_fn:
                test_data = self.indicator_fn(data.copy(), params)

            # Filter out params that aren't strategy constructor args
            # (e.g., resample_freq, trailing stop are Backtester-level params)
            non_strategy_params = (
                "resample_freq",
                "use_trailing_stop",
                "trailing_atr_multiplier",
            )
            strategy_params = {
                k: v for k, v in params.items() if k not in non_strategy_params
            }

            # Build backtester kwargs with any trailing stop from trial params
            bt_kwargs = dict(self.backtester_kwargs)
            if "use_trailing_stop" in params:
                bt_kwargs["use_trailing_stop"] = params["use_trailing_stop"]
            if "trailing_atr_multiplier" in params:
                bt_kwargs["trailing_atr_multiplier"] = params["trailing_atr_multiplier"]

            # Create and run backtest
            strategy = self.strategy_class(**strategy_params)
            backtester = Backtester(
                strategy=strategy,
                initial_capital=initial_capital,
                contract_multiplier=contract_multiplier,
                **bt_kwargs,
            )
            result = backtester.run(test_data)

            # Calculate score
            total_trades = int(result.metrics.get("total_trades", 0))
            score = self._calculate_score(result.metrics, total_trades)

            # Store result
            self.results.append(
                OptunaResult(
                    trial_number=trial.number,
                    params=params,
                    score=score,
                    metrics=result.metrics,
                )
            )

            # Report intermediate values for Optuna pruning
            trial.set_user_attr("total_trades", total_trades)
            trial.set_user_attr("sharpe_ratio", result.metrics.get("sharpe_ratio", 0))
            trial.set_user_attr("profit_factor", result.metrics.get("profit_factor", 0))
            trial.set_user_attr(
                "max_drawdown", result.metrics.get("max_drawdown_pct", 0)
            )

            return score

        except Exception as e:
            logger.exception("Trial %d failed", trial.number)
            trial.set_user_attr("error", str(e))
            return -100.0

    def optimize(
        self,
        data: pd.DataFrame,
        initial_capital: float = DEFAULT_INITIAL_CAPITAL,
        contract_multiplier: float = CONTRACT_MULTIPLIER,
        show_progress: bool = True,
        raw_data: bool = False,
    ) -> List[OptunaResult]:
        """
        Run Optuna optimization.

        Args:
            data: Preprocessed OHLCV data (without indicators),
                  or raw tick data if raw_data=True.
            initial_capital: Starting capital.
            contract_multiplier: Contract multiplier.
            show_progress: Show progress output.
            raw_data: If True, pass data as-is to indicator_fn without
                      stripping to OHLCV columns. Use this when the
                      indicator_fn handles resampling from tick data
                      (e.g., for timeframe optimization).

        Returns:
            Sorted list of OptunaResult (best first).
        """
        self.results = []

        if raw_data:
            # Pass data as-is - indicator_fn will handle resampling + indicators
            clean_data = data.copy()
        else:
            # Strip indicators - keep only base OHLCV columns
            base_columns = ["datetime", "open", "high", "low", "close", "volume"]
            available_cols = [col for col in base_columns if col in data.columns]
            clean_data = data[available_cols].copy()

        # Suppress Optuna's default logging if not showing progress
        if not show_progress:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Create study (maximize the composite score)
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        self.study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=self._study_name,
        )

        if show_progress:
            print(f"Running Optuna optimization with {self.n_trials} trials...")

        self.study.optimize(
            lambda trial: self._objective(
                trial, clean_data, initial_capital, contract_multiplier
            ),
            n_trials=self.n_trials,
            show_progress_bar=show_progress,
        )

        # Sort results by score (descending)
        self.results.sort(key=lambda x: x.score, reverse=True)

        return self.results

    @property
    def best_params(self) -> Optional[Dict[str, Any]]:
        """Return params from the highest-scoring trial (if available)."""
        if not self.results:
            return None
        return self.results[0].params

    @property
    def best_result(self) -> Optional[OptunaResult]:
        """Return the highest-scoring Optuna trial result."""
        if not self.results:
            return None
        return self.results[0]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all trial results to a DataFrame."""
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in self.results])

    def save_results(
        self,
        filename: Optional[str] = None,
        directory: str = RESULTS_DIR,
    ) -> Path:
        """Persist all trial results to CSV and return output path."""
        if not self.results:
            raise ValueError("No results to save")

        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optuna_results_{timestamp}.csv"

        output_path = output_dir / filename
        df = self.to_dataframe()
        df.to_csv(output_path, index=False)
        return output_path

    def print_top_results(self, n: int = 10) -> None:
        """Print a concise leaderboard for the top N trials."""
        if not self.results:
            print("No results available")
            return

        print(f"\n{'=' * 70}")
        print(f"TOP {n} OPTUNA RESULTS (composite score)")
        print(f"{'=' * 70}\n")

        for i, result in enumerate(self.results[:n], 1):
            sharpe = result.metrics.get("sharpe_ratio", 0)
            pf = result.metrics.get("profit_factor", 0)
            dd = result.metrics.get("max_drawdown_pct", 0)
            trades = int(result.metrics.get("total_trades", 0))
            ret = result.metrics.get("total_return_pct", 0)

            print(f"  Rank {i}:")
            print(f"    Score:    {result.score:.4f}")
            print(f"    Params:   {result.params}")
            print(
                f"    Sharpe:   {sharpe:.3f}  |  PF: {pf:.2f}  |  "
                f"DD: {dd:.2f}%  |  Trades: {trades}  |  Return: {ret:.2f}%"
            )
            print()

    def print_study_summary(self) -> None:
        """Print aggregate Optuna study statistics to stdout."""
        if self.study is None:
            print("No study available")
            return

        print(f"\n{'=' * 70}")
        print("OPTUNA STUDY SUMMARY")
        print(f"{'=' * 70}")
        print(f"  Trials completed:  {len(self.study.trials)}")
        print(f"  Best trial:        #{self.study.best_trial.number}")
        print(f"  Best score:        {self.study.best_value:.4f}")
        print(f"  Best params:       {self.study.best_params}")

        # Count valid vs invalid trials
        valid = sum(
            1 for t in self.study.trials if t.value is not None and t.value > -1.0
        )
        invalid = len(self.study.trials) - valid
        print(f"  Valid trials:      {valid}")
        print(f"  Invalid trials:    {invalid} (too few trades or errors)")
        print(f"{'=' * 70}")
