"""
Bayesian optimization using Optuna (TPE sampler).

"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from src.optimization.scoring import ScorerConfig, calculate_score

INVALID_SCORE: float = -10.0
ERROR_SCORE: float = -100.0

try:
    import optuna
    from optuna import Trial
    from optuna.pruners import PatientPruner
    from optuna.samplers import CmaEsSampler, QMCSampler, TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

logger = logging.getLogger(__name__)

TrialFn = Callable[[dict[str, Any]], Any]

# Sampler presets - passed as sampler= to OptunaSearch
SamplerType = Literal["tpe", "tpe_multivariate", "cmaes", "qmc"]


@dataclass
class OptunaResult:
    """Snapshot of one Optuna trial outcome."""

    trial_number: int
    params: dict[str, Any]
    score: float
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        result = {"trial": self.trial_number, "score": self.score}
        result.update(self.params)
        result.update(self.metrics)
        return result


class OptunaSearch:
    """
    Bayesian optimization using Optuna.

    Sampler guide:
        "tpe"              - Default. TPE with QMC warmup. Good for most cases.
                             n_startup_trials random (QMC) trials, then TPE takes over.
        "tpe_multivariate" - TPE that models param correlations. Better when params
                             interact (e.g. orb_minutes + atr_period). Needs ~2x more
                             trials to converge vs standard TPE.
        "cmaes"            - CMA-ES evolution strategy. Best for continuous params
                             with no categoricals. Very efficient for 5-15 params.
        "qmc"              - Pure Quasi-Monte Carlo (Sobol). No learning, just uniform
                             coverage. Use when you want deterministic space coverage
                             without Bayesian overhead (similar to grid but continuous).

    Pruning:
        Pruning is not applicable to backtesting because there are no intermediate
        values - a backtest either completes or fails. The PatientPruner is used
        internally only to stop studies that have not improved for a long time
        (patience=n_trials // 4 by default), preventing wasted compute.

    Usage:
        def trial_fn(params: dict) -> BacktestResult:
            config   = ORBConfig.from_dict({..., "strategy": params})
            strategy = ORBStrategy(config)
            registry = ORBStrategy.build_registry(**params)
            pipeline = DataPipeline(registry)
            data     = pipeline.run(preprocessed_df)
            bt       = Backtester(strategy, ...)
            return bt.run(data)

        search = OptunaSearch(
            trial_fn=trial_fn,
            param_space={
                "orb_minutes":      {"type": "int",   "low": 10,  "high": 45, "step": 5},
                "atr_period":       {"type": "int",   "low": 10,  "high": 20},
                "breakout_buffer":  {"type": "float", "low": 0.0, "high": 0.5},
                "use_volume_filter":{"type": "categorical", "choices": [True, False]},
            },
            n_trials=200,
            sampler="tpe_multivariate",  # params interact → multivariate
        )
        results = search.optimize()
        search.print_top(5)
        search.print_study_summary()
    """

    def __init__(
        self,
        trial_fn: TrialFn,
        param_space: dict[str, dict[str, Any]],
        scorer: ScorerConfig | None = None,
        n_trials: int = 200,
        sampler: SamplerType = "tpe",
        n_startup_trials: int | None = None,
        seed: int = 42,
        study_name: str = "optimization",
        storage_path: str | None = None,
        n_jobs: int = 1,
        patience: int | None = None,
    ) -> None:
        """
        Args:
            trial_fn:          Callable (params dict) → BacktestResult.
            param_space:       Search space. Each key maps to a spec dict:
                                   int:         {"type": "int", "low": 10, "high": 50, "step": 5}
                                   float:       {"type": "float", "low": 0.1, "high": 2.0}
                                   categorical: {"type": "categorical", "choices": [True, False]}
            scorer:            ScorerConfig for composite scoring.
            n_trials:          Total number of optimization trials.
            sampler:           Sampler strategy. See class docstring for guide.
                               "tpe" is a safe default for most cases.
            n_startup_trials:  Random/QMC trials before TPE starts learning.
                               Default: max(20, n_trials // 10).
                               Higher = better TPE model but more random exploration.
            seed:              Random seed for reproducibility.
            study_name:        Optuna study name (used as storage key).
            storage_path:      SQLite path for crash-safe persistence.
                               e.g. "results/study.db". None = in-memory only.
            n_jobs:            Parallel trials. Requires storage_path for safety.
                               Automatically enables constant_liar for TPE.
            patience:          Stop early if no improvement for this many trials.
                               Default: n_trials // 4. Set 0 to disable.
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is not installed. Run: pip install optuna")

        if n_jobs > 1 and storage_path is None:
            logger.warning(
                "n_jobs=%d with in-memory storage is not thread-safe. "
                "Set storage_path to enable safe parallel trials.",
                n_jobs,
            )

        self._trial_fn = trial_fn
        self._param_space = param_space
        self._scorer = scorer or ScorerConfig()
        self._n_trials = n_trials
        self._sampler_type = sampler
        self._n_startup_trials = (
            n_startup_trials if n_startup_trials is not None else max(20, n_trials // 10)
        )
        self._seed = seed
        self._study_name = study_name
        self._storage = self._build_storage(storage_path)
        self._n_jobs = n_jobs
        self._patience = patience if patience is not None else max(n_trials // 4, 20)

        self.results: list[OptunaResult] = []
        self.study: optuna.Study | None = None

    # ── Optimize ──────────────────────────────────────────────────

    def optimize(self, show_progress: bool = True) -> list[OptunaResult]:
        """
        Run Optuna optimization.

        Resumable: if storage_path is set and the study already exists,
        calling optimize() again continues from where it left off.

        Args:
            show_progress: Show progress bar (single-job only).

        Returns:
            Trial results sorted best → worst by composite score.
        """
        self.results = []

        optuna.logging.set_verbosity(
            optuna.logging.INFO if show_progress else optuna.logging.WARNING
        )

        sampler = self._build_sampler()
        pruner = self._build_pruner()

        self.study = optuna.create_study(
            study_name=self._study_name,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            storage=self._storage,
            load_if_exists=True,
        )

        already_done = len(self.study.trials)
        remaining = max(0, self._n_trials - already_done)

        if remaining == 0:
            logger.info(
                "Study '%s' already has %d trials - loading existing results.",
                self._study_name,
                already_done,
            )
        else:
            if show_progress and already_done > 0:
                print(
                    f"Resuming '{self._study_name}': {already_done} done, {remaining} remaining..."
                )

            self.study.optimize(
                self._objective,
                n_trials=remaining,
                n_jobs=self._n_jobs,
                show_progress_bar=show_progress and self._n_jobs == 1,
            )

        self.results = self._collect_results()
        self.results.sort(key=lambda r: r.score, reverse=True)
        return self.results

    # ── Sampler / Pruner builders ─────────────────────────────────

    def _build_sampler(self) -> Any:
        """
        Build sampler based on sampler type.

        Note: independent_sampler was removed in Optuna 4.x.
        QMC warmup is achieved by using QMCSampler for the first
        n_startup_trials via a custom wrapper, or simply relying on
        TPE's built-in random startup (which is already good enough).
        """
        if self._sampler_type == "tpe":
            return TPESampler(
                n_startup_trials=self._n_startup_trials,
                seed=self._seed,
                multivariate=False,
                constant_liar=self._n_jobs > 1,
            )

        if self._sampler_type == "tpe_multivariate":
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return TPESampler(
                    n_startup_trials=self._n_startup_trials,
                    seed=self._seed,
                    multivariate=True,  # model param correlations
                    group=True,  # group correlated params together
                    constant_liar=self._n_jobs > 1,
                )

        if self._sampler_type == "cmaes":
            # CMA-ES: best for continuous params, no categoricals.
            # Falls back to TPE for categorical params automatically.
            return CmaEsSampler(
                seed=self._seed,
                n_startup_trials=self._n_startup_trials,
            )

        if self._sampler_type == "qmc":
            # QMCSampler requires scipy. Falls back to TPE if not installed.
            try:
                import scipy  # noqa: F401

                return QMCSampler(seed=self._seed)
            except ImportError:
                logger.warning(
                    "QMCSampler requires scipy (pip install scipy). Falling back to TPE."
                )
                return TPESampler(n_startup_trials=self._n_startup_trials, seed=self._seed)

        raise ValueError(
            f"Unknown sampler {self._sampler_type!r}. "
            f"Choose from: tpe, tpe_multivariate, cmaes, qmc"
        )

    def _build_pruner(self) -> Any:
        """
        Build pruner.

        Backtesting has no intermediate values, so standard pruners
        (Median, Hyperband) don't apply. PatientPruner stops the study
        if no improvement is seen for `patience` trials - prevents
        wasting compute when the search has clearly converged.
        """
        if self._patience <= 0:
            return optuna.pruners.NopPruner()

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return PatientPruner(
                wrapped_pruner=optuna.pruners.NopPruner(),
                patience=self._patience,
            )

    # ── Objective ─────────────────────────────────────────────────

    def _objective(self, trial: Trial) -> float:
        """Optuna objective function for a single trial."""
        try:
            params = self._sample_params(trial)
            bt_result = self._trial_fn(params)
            metrics = bt_result.metrics if hasattr(bt_result, "metrics") else bt_result
            score = calculate_score(metrics, self._scorer)

            # Store key metrics for dashboard / analysis
            trial.set_user_attr("total_trades", int(metrics.get("total_trades", 0)))
            trial.set_user_attr("sharpe_ratio", float(metrics.get("sharpe_ratio", 0)))
            trial.set_user_attr("net_profit_factor", float(metrics.get("net_profit_factor", 0)))
            trial.set_user_attr("max_drawdown_pct", float(metrics.get("max_drawdown_pct", 0)))
            trial.set_user_attr("win_rate_pct", float(metrics.get("win_rate_pct", 0)))

            # Serialize all scalar metrics for later recovery
            trial.set_user_attr(
                "_metrics_json",
                json.dumps(
                    {
                        k: v
                        for k, v in metrics.items()
                        if isinstance(v, int | float | str | bool)
                        and not (isinstance(v, float) and math.isnan(v))
                    }
                ),
            )

            return score

        except Exception as e:
            logger.exception("Trial %d failed", trial.number)
            trial.set_user_attr("error", str(e))
            return ERROR_SCORE

    # ── Param sampling ────────────────────────────────────────────

    def _sample_params(self, trial: Trial) -> dict[str, Any]:
        """Sample parameters from param_space using Optuna's trial API."""
        params: dict[str, Any] = {}

        for name, spec in self._param_space.items():
            ptype = spec.get("type", "float")

            if ptype == "int":
                params[name] = trial.suggest_int(
                    name,
                    spec["low"],
                    spec["high"],
                    step=spec.get("step", 1),
                )
            elif ptype == "float":
                step = spec.get("step")
                log = spec.get("log", False)  # log scale for e.g. learning rates
                params[name] = trial.suggest_float(
                    name,
                    spec["low"],
                    spec["high"],
                    **({"step": step} if step is not None else {}),
                    **({"log": log} if log else {}),
                )
            elif ptype == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
            else:
                raise ValueError(f"Unknown param type {ptype!r} for {name!r}")

        return params

    # ── Helpers ───────────────────────────────────────────────────

    def _collect_results(self) -> list[OptunaResult]:
        """Reconstruct OptunaResult list from completed study trials."""
        if self.study is None:
            return []

        results = []
        for t in self.study.trials:
            if t.value is None:
                continue
            try:
                metrics = json.loads(t.user_attrs.get("_metrics_json", "{}"))
            except (json.JSONDecodeError, ValueError):
                metrics = {}
            results.append(
                OptunaResult(
                    trial_number=t.number,
                    params=dict(t.params),
                    score=t.value,
                    metrics=metrics,
                )
            )
        return results

    @staticmethod
    def _build_storage(path: str | None) -> str | None:
        if path is None:
            return None
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{p}"

    # ── Export ────────────────────────────────────────────────────

    @property
    def best_params(self) -> dict[str, Any] | None:
        return self.results[0].params if self.results else None

    @property
    def best_result(self) -> OptunaResult | None:
        return self.results[0] if self.results else None

    def to_dataframe(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in self.results])

    def save(self, output_dir: str | Path, filename: str | None = None) -> Path:
        if not self.results:
            raise ValueError("No results to save.")
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        name = filename or f"optuna_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = out / name
        self.to_dataframe().to_csv(path, index=False)
        logger.info("Optuna results saved to %s", path)
        return path

    def print_top(self, n: int = 10) -> None:
        if not self.results:
            print("No results.")
            return
        print(f"\n{'─' * 70}")
        print(f"  TOP {n} OPTUNA RESULTS  [{self._sampler_type.upper()}]")
        print(f"{'─' * 70}")
        for i, r in enumerate(self.results[:n], 1):
            sharpe = r.metrics.get("sharpe_ratio", 0)
            pf = r.metrics.get("net_profit_factor", 0)
            dd = r.metrics.get("max_drawdown_pct", 0)
            trades = int(r.metrics.get("total_trades", 0))
            ret = r.metrics.get("total_return_pct", 0)
            wr = r.metrics.get("win_rate_pct", 0)
            print(f"\n  Rank {i} (trial #{r.trial_number}):")
            print(f"    Score:   {r.score:.4f}")
            print(f"    Params:  {r.params}")
            print(
                f"    Sharpe: {sharpe:.3f}  PF: {pf:.2f}  DD: {dd:.2f}%  "
                f"WR: {wr:.1f}%  Trades: {trades}  Return: {ret:.2f}%"
            )
        print()

    def print_study_summary(self) -> None:
        if self.study is None:
            print("No study available.")
            return
        valid = sum(1 for t in self.study.trials if t.value is not None and t.value > INVALID_SCORE)
        invalid = len(self.study.trials) - valid
        print(f"\n{'─' * 70}")
        print("  OPTUNA STUDY SUMMARY")
        print(f"{'─' * 70}")
        print(f"  Study name:        {self._study_name}")
        print(f"  Sampler:           {self._sampler_type}")
        print(f"  Startup trials:    {self._n_startup_trials}  (QMC warmup)")
        print(f"  Patience:          {self._patience}")
        print(f"  Trials completed:  {len(self.study.trials)}")
        print(f"  Valid trials:      {valid}")
        print(f"  Invalid trials:    {invalid}")
        if self.study.best_trial:
            print(f"  Best trial:        #{self.study.best_trial.number}")
            print(f"  Best score:        {self.study.best_value:.4f}")
            print(f"  Best params:       {self.study.best_params}")
        print(f"{'─' * 70}\n")
