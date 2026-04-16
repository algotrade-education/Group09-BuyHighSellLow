"""
src/run_optimize_hybrid.py

Hybrid + Conditional Optuna runner.

Workflow:
  Phase 1 (discovery): split budget into
    - core-focused branch (adaptive disabled)
    - adaptive branch (adaptive enabled)

  Phase 2 (local refine): for top seeds from phase 1,
    run local optimization in a narrow neighborhood.

  Phase 3 (final polish): refine around the best phase-2 candidate.

This runner is intended for strategies like ORB where adaptive volatility can
"rescue" mediocre core configs but also expands search dimensionality.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.constants import ExecutionConfig
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.database.data_service import get_data_service
from src.optimization.optuna_search import OptunaResult, OptunaSearch, SamplerType
from src.optimization.scoring import ScorerConfig
from src.strategy.strategy_registry import (
    get_strategy_plugin,
    list_param_space_keys,
    list_strategy_names,
)
from src.utils.cli_helpers import print_exception, print_kv_rows, print_section, print_status
from src.utils.logger import setup_logging

logger = setup_logging(
    name="run_optimize_hybrid",
    log_file="logs/optimize_hybrid.log",
    capture_all_loggers=False,
)

# ORB-oriented adaptive parameter names. The code degrades gracefully if
# some keys are absent in a strategy's param space.
_ADAPTIVE_FLAG = "use_adaptive_volatility"
_ADAPTIVE_KEYS = {
    _ADAPTIVE_FLAG,
    "atr_lookback_period",
    "volatility_low_threshold",
    "volatility_high_threshold",
    "low_vol_range_multiplier",
    "high_vol_range_multiplier",
    "low_vol_buffer_multiplier",
    "high_vol_buffer_multiplier",
}


@dataclass
class BranchRun:
    name: str
    n_trials: int
    results: list[OptunaResult]


@dataclass
class SeedCandidate:
    params: dict[str, Any]
    score: float
    source: str


def _print_trade_diagnostics(results: list[OptunaResult], label: str) -> None:
    if not results:
        print_status(f"{label}: no results", "warning")
        return

    trades = [int(r.metrics.get("total_trades", 0)) for r in results]
    score_neg21 = sum(1 for r in results if abs(float(r.score) - (-21.0)) < 1e-9)
    ratio_neg21 = score_neg21 / len(results)

    print_status(
        (
            f"{label}: trades min/median/max = {min(trades)}/"
            f"{int(statistics.median(trades))}/{max(trades)} | "
            f"-21 ratio = {ratio_neg21:.0%}"
        ),
        "info",
    )


def _copy_param_space(space: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {k: dict(v) for k, v in space.items()}


def _force_categorical_value(
    space: dict[str, dict[str, Any]],
    param_name: str,
    value: Any,
) -> None:
    if param_name not in space:
        return
    spec = dict(space[param_name])
    spec["type"] = "categorical"
    spec["choices"] = [value]
    space[param_name] = spec


def _collapse_adaptive_params_to_defaults(
    space: dict[str, dict[str, Any]],
    base_raw: dict[str, Any],
) -> None:
    strategy_cfg = base_raw.get("strategy", {})
    for key in _ADAPTIVE_KEYS:
        if key not in space or key == _ADAPTIVE_FLAG:
            continue
        if key in strategy_cfg:
            _force_categorical_value(space, key, strategy_cfg[key])


def _compute_local_numeric_bounds(
    value: float,
    low: float,
    high: float,
    rel_radius: float,
    min_span: float,
) -> tuple[float, float]:
    span = max(abs(value) * rel_radius, min_span)
    return max(low, value - span), min(high, value + span)


def _make_local_space(
    base_space: dict[str, dict[str, Any]],
    seed_params: dict[str, Any],
    rel_radius: float,
) -> dict[str, dict[str, Any]]:
    """Build a neighborhood search space around one seed config."""
    out = _copy_param_space(base_space)

    for name, spec in list(out.items()):
        if name not in seed_params:
            continue

        ptype = spec.get("type")
        center = seed_params[name]

        if ptype == "int":
            low_i = int(spec["low"])
            high_i = int(spec["high"])
            step_i = int(spec.get("step", 1))

            # Integer neighborhood with at least +/- 1 step.
            radius_abs = max(step_i, int(math.ceil(abs(int(center)) * rel_radius)))
            local_low_i = max(low_i, int(center) - radius_abs)
            local_high_i = min(high_i, int(center) + radius_abs)

            # Snap to step grid.
            def _snap(v: int, origin: int, s: int, up: bool) -> int:
                delta = v - origin
                q = int(math.ceil(delta / s)) if up else int(math.floor(delta / s))
                return origin + q * s

            local_low_i = _snap(local_low_i, low_i, step_i, up=True)
            local_high_i = _snap(local_high_i, low_i, step_i, up=False)

            if local_low_i > local_high_i:
                local_low_i = local_high_i = int(center)

            out[name] = {
                "type": "int",
                "low": local_low_i,
                "high": local_high_i,
                "step": step_i,
            }

        elif ptype == "float":
            low_f = float(spec["low"])
            high_f = float(spec["high"])
            step_f = float(spec["step"])
            log = bool(spec.get("log", False))
            center_f = float(center)

            min_span = float(step_f) if step_f is not None else max((high_f - low_f) * 0.05, 1e-9)
            local_low_f, local_high_f = _compute_local_numeric_bounds(
                center_f, low_f, high_f, rel_radius, min_span
            )

            if step_f is not None:
                step_f = float(step_f)
                # Snap local bounds to the original step lattice.
                # This avoids Optuna warnings for non-divisible [low, high] ranges.
                q_low = math.ceil((local_low_f - low_f) / step_f)
                q_high = math.floor((local_high_f - low_f) / step_f)
                local_low_f = low_f + q_low * step_f
                local_high_f = low_f + q_high * step_f

                # Normalize floating noise from binary arithmetic.
                local_low_f = round(local_low_f, 12)
                local_high_f = round(local_high_f, 12)

                if local_low_f > local_high_f:
                    # Fallback to a single-point local range around the nearest lattice point.
                    q_center = round((center_f - low_f) / step_f)
                    snapped = low_f + q_center * step_f
                    snapped = min(max(snapped, low_f), high_f)
                    snapped = round(snapped, 12)
                    local_low_f = local_high_f = snapped

            float_spec: dict[str, Any] = {
                "type": "float",
                "low": local_low_f,
                "high": local_high_f,
            }
            if step_f is not None:
                float_spec["step"] = step_f
            if log:
                float_spec["log"] = True
            out[name] = float_spec

        elif ptype == "categorical":
            # Freeze categoricals locally. Keeps neighborhood focused.
            out[name] = {
                "type": "categorical",
                "choices": [center],
            }

    return out


def _dedup_candidates(candidates: list[SeedCandidate]) -> list[SeedCandidate]:
    """Deduplicate by params fingerprint, keep best score."""
    best_by_key: dict[str, SeedCandidate] = {}
    for c in candidates:
        key = json.dumps(c.params, sort_keys=True, default=str)
        prev = best_by_key.get(key)
        if prev is None or c.score > prev.score:
            best_by_key[key] = c
    return list(best_by_key.values())


def _pick_seed_candidates(
    core_results: list[OptunaResult],
    adaptive_results: list[OptunaResult],
    n_core: int,
    n_adaptive: int,
    n_total_cap: int,
) -> list[SeedCandidate]:
    seeds: list[SeedCandidate] = []

    for r in core_results[: max(0, n_core)]:
        seeds.append(SeedCandidate(params=r.params, score=r.score, source="core"))
    for r in adaptive_results[: max(0, n_adaptive)]:
        seeds.append(SeedCandidate(params=r.params, score=r.score, source="adaptive"))

    seeds = _dedup_candidates(seeds)
    seeds.sort(key=lambda x: x.score, reverse=True)
    return seeds[: max(1, n_total_cap)]


def _make_trial_fn(
    plugin: Any,
    preprocessed_data: Any,
    config_path: str,
    args: argparse.Namespace,
    freq: str,
) -> Any:
    return plugin.build_trial_fn(
        preprocessed_data=preprocessed_data,
        base_config_path=config_path,
        capital=args.capital,
        commission_rate=args.commission_rate,
        slippage_points=args.slippage_points,
        contract_multiplier=args.contract_multiplier,
        margin_rate=args.margin_rate,
        cache_dir=args.cache_dir,
        freq=freq,
    )


def _run_one_study(
    *,
    branch_name: str,
    trial_fn: Any,
    param_space: dict[str, dict[str, Any]],
    scorer: ScorerConfig,
    n_trials: int,
    sampler: SamplerType,
    storage_path: str | None,
    study_name: str,
    seed: int,
    on_conflict: str,
    show_progress: bool,
) -> BranchRun:
    search = OptunaSearch(
        trial_fn=trial_fn,
        param_space=param_space,
        scorer=scorer,
        n_trials=n_trials,
        sampler=sampler,
        study_name=study_name,
        storage_path=storage_path,
        seed=seed,
    )

    if storage_path:
        conflict = search.check_storage_conflict()
        if conflict is not None:
            if on_conflict == "overwrite":
                search.delete_study()
                print_status(
                    f"[{branch_name}] Existing study deleted (param-space conflict).", "warning"
                )
            elif on_conflict == "abort":
                raise RuntimeError(
                    f"[{branch_name}] {conflict.describe()} | Set --on-conflict=resume/overwrite"
                )
            else:
                print_status(
                    f"[{branch_name}] Param-space conflict, resuming anyway as requested.",
                    "warning",
                )

    results = search.optimize(show_progress=show_progress)
    return BranchRun(name=branch_name, n_trials=n_trials, results=results)


def _save_best_config(
    plugin: Any,
    base_raw: dict[str, Any],
    result: OptunaResult,
    output_dir: str,
    suffix: str,
) -> Path:
    risk_keys = plugin.risk_keys
    strategy_params = {k: v for k, v in result.params.items() if k not in risk_keys}
    risk_params = {k: v for k, v in result.params.items() if k in risk_keys}

    final_raw = {
        **base_raw,
        "strategy": {**base_raw.get("strategy", {}), **strategy_params},
        "risk": {**base_raw.get("risk", {}), **risk_params},
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"{plugin.name}_hybrid_best_{suffix}_{ts}.json"
    path.write_text(json.dumps(final_raw, indent=2), encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> int:
    # --- Resolve plugin / config
    try:
        plugin = get_strategy_plugin(args.strategy)
    except KeyError as e:
        print_status(str(e), "error")
        return 1

    if args.param_space not in plugin.param_spaces:
        available = list_param_space_keys(args.strategy)
        print_status(
            f"Unknown param space {args.param_space!r} for {args.strategy!r}. Available: {available}",
            "error",
        )
        return 1

    config_path = args.config or plugin.default_config
    if not Path(config_path).exists():
        print_status(f"Config not found: {config_path}", "error")
        return 1

    base_raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    freq = args.freq or base_raw.get("strategy", {}).get("resample_freq", "5min")

    print_section(f"HYBRID OPTIMIZE: {plugin.display_name}", width=66)
    print_kv_rows(
        {
            "Period": f"{args.start} -> {args.end}",
            "Strategy": args.strategy,
            "Param space": args.param_space,
            "Sampler": args.sampler,
            "Phase1 trials": args.phase1_trials,
            "Seed": args.seed,
        },
        label_width=14,
    )
    print()

    # Progressive min-trades schedule prevents phase-1 collapse into all -21 scores
    # when the final target min_trades is strict (e.g. 300+).
    phase1_min_trades = (
        args.phase1_min_trades
        if args.phase1_min_trades is not None
        else max(30, args.min_trades // 3)
    )
    phase2_min_trades = (
        args.phase2_min_trades
        if args.phase2_min_trades is not None
        else max(50, args.min_trades // 2)
    )
    phase3_min_trades = (
        args.phase3_min_trades if args.phase3_min_trades is not None else args.min_trades
    )

    print_kv_rows(
        {
            "MinTrades P1": phase1_min_trades,
            "MinTrades P2": phase2_min_trades,
            "MinTrades P3": phase3_min_trades,
        },
        label_width=14,
    )
    if args.target_trades_min is not None or args.target_trades_max is not None:
        print_kv_rows(
            {
                "Target trades": (
                    f"{args.target_trades_min or 0}.."
                    f"{args.target_trades_max if args.target_trades_max is not None else 'inf'}"
                ),
            },
            label_width=14,
        )
    print()

    # --- Load data once
    print_status("Loading data...", "info")
    try:
        loader = DataLoader(data_service=get_data_service(), cache_dir=args.cache_dir)
        raw = loader.load(
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            use_cache=not args.force_refresh,
        )
        preprocessed = DataPreprocessor().prepare(raw, freq=freq)
        print_status(f"{len(preprocessed):,} bars loaded ({freq})", "success")
    except Exception as e:
        logger.error("Data load failed: %s", e, exc_info=True)
        print_exception("Data load", e)
        return 1

    trial_fn = _make_trial_fn(plugin, preprocessed, config_path, args, freq)

    scorer_phase1 = ScorerConfig(
        min_trades=phase1_min_trades,
        min_return_pct=args.min_return,
        drawdown_penalty=args.drawdown_penalty,
        trade_count_bonus=args.trade_count_bonus,
    )
    scorer_phase2 = ScorerConfig(
        min_trades=phase2_min_trades,
        min_return_pct=args.min_return,
        drawdown_penalty=args.drawdown_penalty,
        trade_count_bonus=args.trade_count_bonus,
    )
    scorer_phase3 = ScorerConfig(
        min_trades=phase3_min_trades,
        min_return_pct=args.min_return,
        drawdown_penalty=args.drawdown_penalty,
        trade_count_bonus=args.trade_count_bonus,
    )

    base_space = _copy_param_space(plugin.param_spaces[args.param_space])
    has_adaptive_flag = _ADAPTIVE_FLAG in base_space

    # --- Phase 1: split-budget discovery
    p1_total = max(1, args.phase1_trials)
    adaptive_trials = int(round(p1_total * args.adaptive_ratio)) if has_adaptive_flag else 0
    adaptive_trials = max(0, min(p1_total, adaptive_trials))
    core_trials = p1_total - adaptive_trials

    print_status(
        f"Phase 1: discovery split -> core={core_trials}, adaptive={adaptive_trials}",
        "info",
    )

    phase1_core_results: list[OptunaResult] = []
    phase1_adaptive_results: list[OptunaResult] = []

    if core_trials > 0:
        core_space = _copy_param_space(base_space)
        if has_adaptive_flag:
            _force_categorical_value(core_space, _ADAPTIVE_FLAG, False)
            _collapse_adaptive_params_to_defaults(core_space, base_raw)

        core_storage = args.storage
        core_study_name = f"{plugin.name}_hybrid_p1_core"
        core_branch = _run_one_study(
            branch_name="phase1-core",
            trial_fn=trial_fn,
            param_space=core_space,
            scorer=scorer_phase1,
            n_trials=core_trials,
            sampler=args.sampler,
            storage_path=core_storage,
            study_name=core_study_name,
            seed=args.seed,
            on_conflict=args.on_conflict,
            show_progress=True,
        )
        phase1_core_results = core_branch.results

    if adaptive_trials > 0:
        adaptive_space = _copy_param_space(base_space)
        _force_categorical_value(adaptive_space, _ADAPTIVE_FLAG, True)

        adaptive_storage = args.storage
        adaptive_study_name = f"{plugin.name}_hybrid_p1_adaptive"
        adaptive_branch = _run_one_study(
            branch_name="phase1-adaptive",
            trial_fn=trial_fn,
            param_space=adaptive_space,
            scorer=scorer_phase1,
            n_trials=adaptive_trials,
            sampler=args.sampler,
            storage_path=adaptive_storage,
            study_name=adaptive_study_name,
            seed=args.seed + 17,
            on_conflict=args.on_conflict,
            show_progress=True,
        )
        phase1_adaptive_results = adaptive_branch.results

    if not phase1_core_results and not phase1_adaptive_results:
        print_status("No phase-1 results produced.", "error")
        return 1

    _print_trade_diagnostics(phase1_core_results, "Phase1 core")
    _print_trade_diagnostics(phase1_adaptive_results, "Phase1 adaptive")

    # --- Phase 2: local refinement around top seeds
    seeds = _pick_seed_candidates(
        phase1_core_results,
        phase1_adaptive_results,
        n_core=args.phase2_top_core,
        n_adaptive=args.phase2_top_adaptive,
        n_total_cap=args.phase2_seed_cap,
    )

    print_status(f"Phase 2: local refinement on {len(seeds)} seeds", "info")

    phase2_results: list[OptunaResult] = []
    for idx, seed in enumerate(seeds, start=1):
        local_space = _make_local_space(
            base_space,
            seed.params,
            rel_radius=args.local_radius,
        )

        # Keep seed's adaptive regime in local search to stabilize neighborhood fit.
        if has_adaptive_flag and _ADAPTIVE_FLAG in seed.params:
            _force_categorical_value(local_space, _ADAPTIVE_FLAG, seed.params[_ADAPTIVE_FLAG])

        local_trials = max(1, args.phase2_trials_per_seed)
        storage = args.storage
        study_name = f"{plugin.name}_hybrid_p2_seed{idx:02d}"

        print_status(
            f"  Seed {idx}/{len(seeds)} from {seed.source}: score={seed.score:.4f}",
            "info",
        )

        branch = _run_one_study(
            branch_name=f"phase2-seed-{idx}",
            trial_fn=trial_fn,
            param_space=local_space,
            scorer=scorer_phase2,
            n_trials=local_trials,
            sampler=args.sampler,
            storage_path=storage,
            study_name=study_name,
            seed=args.seed + 1000 + idx,
            on_conflict=args.on_conflict,
            show_progress=False,
        )
        phase2_results.extend(branch.results[: args.phase2_keep_per_seed])

    if not phase2_results:
        # Fallback to best phase-1 result.
        fallback = phase1_core_results + phase1_adaptive_results
        fallback.sort(key=lambda r: r.score, reverse=True)
        phase2_results = fallback[:1]

    phase2_results.sort(key=lambda r: r.score, reverse=True)

    # --- Phase 3: final polish around best phase-2 candidate
    best_seed = phase2_results[0]
    phase3_space = _make_local_space(
        base_space,
        best_seed.params,
        rel_radius=args.phase3_radius,
    )

    print_status(
        f"Phase 3: final polish around best seed score={best_seed.score:.4f}",
        "info",
    )

    phase3 = _run_one_study(
        branch_name="phase3-final",
        trial_fn=trial_fn,
        param_space=phase3_space,
        scorer=scorer_phase3,
        n_trials=max(1, args.phase3_trials),
        sampler=args.sampler,
        storage_path=args.storage,
        study_name=f"{plugin.name}_hybrid_p3_final",
        seed=args.seed + 5000,
        on_conflict=args.on_conflict,
        show_progress=True,
    )
    _print_trade_diagnostics(phase3.results, "Phase3 final")

    # --- Select winner from phase 3, fallback to phase 2/1
    candidates = phase3.results + phase2_results + phase1_core_results + phase1_adaptive_results
    candidates.sort(key=lambda r: r.score, reverse=True)

    if not candidates:
        print_status("No candidate results found at the end of hybrid optimization.", "error")
        return 1

    winner = candidates[0]

    # Optional final trade-band filter (e.g. 300..600 trades in 2 years).
    if args.target_trades_min is not None or args.target_trades_max is not None:
        min_t = args.target_trades_min if args.target_trades_min is not None else 0
        max_t = args.target_trades_max if args.target_trades_max is not None else 10**9
        trade_band = [
            c for c in candidates if min_t <= int(c.metrics.get("total_trades", 0)) <= max_t
        ]
        if trade_band:
            winner = trade_band[0]
            print_status(
                f"Selected winner from trade-band [{min_t}, {max_t}] candidates={len(trade_band)}",
                "info",
            )
        else:
            print_status(
                "No candidate matched target trade band; falling back to best score.",
                "warning",
            )

    print_section("HYBRID RESULT", width=66)
    print_kv_rows(
        {
            "Winner score": f"{winner.score:.4f}",
            "Sharpe": f"{winner.metrics.get('sharpe_ratio', 0.0):.3f}",
            "Return": f"{winner.metrics.get('total_return_pct', 0.0):.2f}%",
            "Max DD": f"{winner.metrics.get('max_drawdown_pct', 0.0):.2f}%",
            "Trades": int(winner.metrics.get("total_trades", 0)),
            "Adaptive": winner.params.get(_ADAPTIVE_FLAG, "n/a"),
        },
        label_width=14,
    )

    try:
        best_path = _save_best_config(
            plugin=plugin,
            base_raw=base_raw,
            result=winner,
            output_dir=args.output_dir,
            suffix="final",
        )
        print_status(f"Best config saved -> {best_path}", "success")
    except Exception as e:
        logger.warning("Failed to save best config: %s", e)

    # Save leaderboard CSV
    try:
        rows = []
        for i, r in enumerate(candidates[: args.top_n], start=1):
            row = {
                "rank": i,
                "score": r.score,
                **r.params,
                **r.metrics,
            }
            rows.append(row)

        import pandas as pd

        df = pd.DataFrame(rows)
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{plugin.name}_hybrid_top_{time.strftime('%Y%m%d-%H%M%S')}.csv"
        df.to_csv(csv_path, index=False)
        print_status(f"Top leaderboard saved -> {csv_path}", "success")
    except Exception as e:
        logger.warning("Failed to save leaderboard CSV: %s", e)

    return 0


def build_parser() -> argparse.ArgumentParser:
    strategies = list_strategy_names()

    parser = argparse.ArgumentParser(
        prog="run_optimize_hybrid",
        description="Hybrid + Conditional Optuna optimization runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--strategy", "-s", default="orb", choices=strategies)
    parser.add_argument("--config", "-c", default=None, help="Base config JSON path")
    parser.add_argument(
        "--param-space",
        default="full",
        help="Named strategy param-space from registry (full/core/etc.)",
    )

    # Data
    parser.add_argument("--symbol", default="VN30F1M")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2025-03-31")
    parser.add_argument("--freq", default=None, choices=["1min", "5min", "15min", "30min"])
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--force-refresh", action="store_true")

    # Capital / costs
    ExecutionConfig.add_args(parser)

    # Sampler / storage
    parser.add_argument(
        "--sampler",
        default="tpe",
        choices=["tpe", "tpe_multivariate", "cmaes", "qmc"],
    )
    parser.add_argument("--storage", default=None, help="SQLite path for resumable studies")
    parser.add_argument(
        "--on-conflict",
        default="resume",
        choices=["resume", "overwrite", "abort"],
        help="How to handle existing study with different param-space fingerprint",
    )
    parser.add_argument("--seed", type=int, default=42)

    # Scoring gates
    parser.add_argument("--min-trades", type=int, default=120)
    parser.add_argument(
        "--phase1-min-trades",
        type=int,
        default=None,
        help="Override min_trades for phase1. Default=max(30, min-trades//3)",
    )
    parser.add_argument(
        "--phase2-min-trades",
        type=int,
        default=None,
        help="Override min_trades for phase2. Default=max(50, min-trades//2)",
    )
    parser.add_argument(
        "--phase3-min-trades",
        type=int,
        default=None,
        help="Override min_trades for phase3. Default=min-trades",
    )
    parser.add_argument(
        "--target-trades-min",
        type=int,
        default=None,
        help="Optional final winner filter: minimum total_trades",
    )
    parser.add_argument(
        "--target-trades-max",
        type=int,
        default=None,
        help="Optional final winner filter: maximum total_trades",
    )
    parser.add_argument("--min-return", type=float, default=-999.0)
    parser.add_argument("--drawdown-penalty", type=float, default=0.3)
    parser.add_argument("--trade-count-bonus", type=float, default=0.1)

    # Hybrid budget knobs
    parser.add_argument("--phase1-trials", type=int, default=400)
    parser.add_argument(
        "--adaptive-ratio",
        type=float,
        default=0.2,
        help="Fraction of phase1 trials allocated to adaptive-enabled branch",
    )
    parser.add_argument("--phase2-top-core", type=int, default=6)
    parser.add_argument("--phase2-top-adaptive", type=int, default=6)
    parser.add_argument("--phase2-seed-cap", type=int, default=12)
    parser.add_argument("--phase2-trials-per-seed", type=int, default=25)
    parser.add_argument("--phase2-keep-per-seed", type=int, default=2)
    parser.add_argument(
        "--local-radius",
        type=float,
        default=0.10,
        help="Neighborhood radius in phase2 (relative, e.g. 0.10 = +/-10%%)",
    )

    parser.add_argument("--phase3-trials", type=int, default=120)
    parser.add_argument(
        "--phase3-radius",
        type=float,
        default=0.05,
        help="Neighborhood radius in phase3 final polish",
    )

    # Output
    parser.add_argument("--output-dir", default="results/optimization")
    parser.add_argument("--top-n", type=int, default=20)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Arg safety checks
    if not (0.0 <= args.adaptive_ratio <= 1.0):
        print_status("--adaptive-ratio must be in [0, 1]", "error")
        return 1
    if args.phase1_trials <= 0:
        print_status("--phase1-trials must be > 0", "error")
        return 1
    if args.target_trades_min is not None and args.target_trades_min < 0:
        print_status("--target-trades-min must be >= 0", "error")
        return 1
    if args.target_trades_max is not None and args.target_trades_max < 0:
        print_status("--target-trades-max must be >= 0", "error")
        return 1
    if (
        args.target_trades_min is not None
        and args.target_trades_max is not None
        and args.target_trades_min > args.target_trades_max
    ):
        print_status("--target-trades-min cannot be greater than --target-trades-max", "error")
        return 1

    t0 = time.perf_counter()
    try:
        code = run(args)
    except Exception as e:
        logger.error("Hybrid optimization failed: %s", e, exc_info=True)
        print_exception("Hybrid optimization", e)
        return 1

    elapsed = time.perf_counter() - t0
    print_status(f"Total elapsed: {elapsed:.1f}s", "success")
    return code


if __name__ == "__main__":
    sys.exit(main())
