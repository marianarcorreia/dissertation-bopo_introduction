"""Optuna tuner entrypoint for FJSP representations (OO, OM, OJM).

Usage:
        python param.py
        python param.py --representations OO OM OJM --trials 25 --max-episodes 400
        python param.py --representations OJM --trials 30 --storage sqlite:///optuna.db
        python param.py --smoke --representations OO OM OJM

What this script does:
        1. Creates one Optuna study per selected representation.
        2. Samples critical hyperparameters from representation-specific search spaces.
        3. Runs training/validation through src.train.train for each trial.
        4. Optimises the objective: 80th-percentile gap of the LAST validation checkpoint
           (lower is better).  This rewards models that solve at least 80 % of
           validation instances well, not just the lucky best checkpoint.

Critical-parameter strategy (same search space for OO/OM/OJM):
    lr, hidden_channels, batch_size, mask_option, sel_k, num_layers, heads,
    n_cases, new_freq, K, use_greedy

Validation dataset:
    By default (--valdata fixed) the entire fixed validation set in
    val/instances + val/solutions is used, giving a stable, noise-free
    objective across all trials.

Outputs produced:
        - Per-trial training outputs under results/optuna_<rep>_<timestamp>_<trial>/
        - Aggregate tuning summary in results/optuna/tuning_summary_<timestamp>.json
        - When --storage is given (recommended), trials are persisted and resumable.

Recommended invocation:
        python param.py \\
            --representations OO OM OJM \\
            --trials 25 \\
            --max-episodes 400 \\
            --storage sqlite:///optuna.db \\
            --valdata fixed
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import optuna

from src.generate_val import generate_val
from src.train import train
from src.utils import open_dashboard


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter tuning for OO/OM/OJM representations."
    )
    parser.add_argument("--trials", type=int, default=30,
                        help="Number of Optuna trials per representation (default: 30).")
    parser.add_argument("--max-episodes", type=int, default=500,
                        help="Training episodes per trial (default: 500).")
    parser.add_argument("--validation-freq", type=int, default=10,
                        help="Validation frequency in episodes (default: 20).")
    parser.add_argument("--validation-size", type=int, default=30,
                        help="Validation sample size when valdata=our.")
    parser.add_argument("--valdata", choices=["fixed", "our", "gen"], default="fixed",
                        help="Validation dataset: 'fixed' uses all of val/instances (default), "
                             "'our' samples from val/instances, 'gen' uses val/validation_set.json.")
    parser.add_argument("--sampler-seed", type=int, default=42,
                        help="Seed for Optuna TPE sampler.")
    parser.add_argument("--storage", default=None,
                        help="Optuna storage URL, e.g. sqlite:///optuna.db, or a bare file path "
                             "(e.g. optuna.db) which is auto-converted to a SQLite URL. "
                             "Strongly recommended: allows resuming interrupted studies.")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Optional timeout (seconds) per representation study.")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick end-to-end test mode with minimal workload.")
    parser.add_argument("--representations", nargs="+", default=["oo", "om", "ojm"],
                        help="Representations to tune, any of: oo, om, ojm (default: all three).")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Do not auto-launch/open the results dashboard when tuning finishes.")
    return parser.parse_args()


def _normalize_storage_url(storage: Optional[str]) -> Optional[str]:
    """Accept a bare filesystem path (e.g. 'optuna.db', 'D:\\x\\optuna.db') as a
    convenience and turn it into a proper SQLite SQLAlchemy URL. A string that
    already looks like a URL (contains '://') is passed through unchanged."""
    if not storage:
        return storage
    if "://" in storage:
        return storage
    path = Path(storage).resolve().as_posix()
    return f"sqlite:///{path}"


# ── prerequisites ─────────────────────────────────────────────────────────────

def ensure_prerequisites(valdata: str) -> None:
    os.makedirs("results", exist_ok=True)
    os.makedirs("candidate_models", exist_ok=True)

    model_params_path = Path("candidate_models/model_params.json")
    if not model_params_path.exists():
        model_params_path.write_text("[]")

    if valdata == "fixed":
        if not Path("val/instances").is_dir() or not Path("val/solutions").is_dir():
            raise FileNotFoundError(
                "val/instances and val/solutions must exist for --valdata fixed. "
                "Ensure both directories are populated before running tuning."
            )
    elif valdata == "gen":
        validation_set_path = Path("val/validation_set.json")
        if not validation_set_path.exists():
            print("[PARAM] val/validation_set.json not found. Generating default validation set...")
            generate_val(20)


# ── search spaces ─────────────────────────────────────────────────────────────

def critical_param_names(rep: str, smoke: bool) -> List[str]:
    if smoke:
        return []
    
    return ["lr", "hidden_channels", "batch_size", "mask_option", "sel_k",
            "num_layers", "heads", "n_cases", "new_freq", "K", "use_greedy"]


def suggest_hyperparameters(rep: str, trial: optuna.Trial, smoke: bool) -> Dict:
    if smoke:
        return {
            "train_freq": 1, "new_freq": 1, "n_cases": 3,
            "mask_option": 1, "sel_k": 1, "batch_size": 8,
            "lr": 3e-4, "hidden_channels": 32, "num_layers": 1, "heads": 2,
            "K": 2, "use_greedy": True,
            "j_min": 4, "j_max": 5, "m_min": 3, "m_max": 4,
            "op_max": 5, "max_processing": 10,
        }

    return {
        "train_freq":      4,  # keep fixed; OJM updates are expensive
        "new_freq":        trial.suggest_categorical("new_freq",         [1, 10]),
        "n_cases":         trial.suggest_categorical("n_cases",          [40, 80, 120]),
        "mask_option":     trial.suggest_categorical("mask_option",      [0, 1]),
        "sel_k":           trial.suggest_categorical("sel_k",            [1, 2, 3]),
        "batch_size":      trial.suggest_categorical("batch_size",       [32, 64, 128]),
        "lr":              trial.suggest_float("lr",                     5e-5, 5e-3, log=True),
        "hidden_channels": trial.suggest_categorical("hidden_channels",  [64, 128, 256, 512]),
        "num_layers":      trial.suggest_int("num_layers",               1, 3),
        "heads":           trial.suggest_categorical("heads",            [2, 3, 4]),
        "K":               trial.suggest_categorical("K",                [2, 4, 8]),
        # use_greedy: whether one of the B rollouts is a greedy decode instead of sampled.
        "use_greedy":      trial.suggest_categorical("use_greedy",       [True, False]),
        "j_min": 5, "j_max": 15, "m_min": 4, "m_max": 13,
        "op_max": 9, "max_processing": 25,
    }


# ── objective helpers ─────────────────────────────────────────────────────────

def _run_path(run_name: str) -> Path:
    """Resolve the output directory for a run_name that may be 'study/trial_N'."""
    return Path("results") / run_name


def read_run_summary(run_name: str) -> Dict:
    direct = _run_path(run_name) / "run_summary.json"
    if direct.exists():
        return json.loads(direct.read_text())
    raise FileNotFoundError(f"No run_summary.json found at: {direct}")


def read_validation_history(run_name: str) -> List[Dict]:
    """Return the validation_history.json list for the given run, or [] if missing."""
    direct = _run_path(run_name) / "validation_history.json"
    if direct.exists():
        return json.loads(direct.read_text())
    return []


def last_val_q80(run_name: str) -> float:
    history = read_validation_history(run_name)
    if not history:
        raise RuntimeError(f"Validation history is empty for run: {run_name}")
    last_entry = history[-1]
    gaps = last_entry.get("all_gaps", [])
    if not gaps:
        raise RuntimeError(f"Last validation entry has no 'all_gaps' for run: {run_name}")
    return float(np.percentile(gaps, 80))


# ── tuning loop ───────────────────────────────────────────────────────────────

def tune_representation(rep: str, args: argparse.Namespace) -> Dict:
    rep_upper = rep.upper()
    study_name = f"fjsp_tuning_{rep}"
    study_folder = f"optuna_{rep}_{int(time.time())}"  # one folder per study, shared by all trials
    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)

    if args.storage is None:
        study = optuna.create_study(direction="minimize", sampler=sampler,
                                    study_name=study_name)
    else:
        study = optuna.create_study(
            study_name=study_name, direction="minimize",
            sampler=sampler, storage=args.storage, load_if_exists=True,
        )

    tuned_params = critical_param_names(rep, args.smoke)
    effective_trials       = 1 if args.smoke else args.trials
    effective_max_episodes = 2 if args.smoke else args.max_episodes
    effective_val_freq     = 1 if args.smoke else args.validation_freq
    effective_val_size     = 3 if args.smoke else args.validation_size

    def objective(trial: optuna.Trial) -> float:
        sampled  = suggest_hyperparameters(rep, trial, args.smoke)
        # "study_folder/trial_N" → train() will create results/study_folder/trial_N/
        run_name = f"{study_folder}/trial_{trial.number}"

        print(
            f"[PARAM][{rep_upper}] Trial {trial.number} | "
            f"episodes={effective_max_episodes} | lr={sampled['lr']:.2e} | "
            f"hidden={sampled['hidden_channels']} | layers={sampled['num_layers']} | "
            f"heads={sampled['heads']} | K={sampled['K']} | use_greedy={sampled['use_greedy']} | "
            f"tf={sampled['train_freq']}"
        )

        # NOTE: BOPO has no separate train_freq (every step is an update) and train()
        # has no val_type param (validation is always built from val/instances +
        # val/solutions) — so those two sampled/CLI values are not forwarded here.
        # "batch_size" maps to train()'s B (number of parallel sampled trajectories).
        train(
            max_episodes     = effective_max_episodes,
            new_freq         = sampled["new_freq"],
            n_cases          = sampled["n_cases"],
            mask_option      = sampled["mask_option"],
            sel_k            = sampled["sel_k"],
            B                = sampled["batch_size"],
            K                = sampled["K"],
            use_greedy       = sampled["use_greedy"],
            lr               = sampled["lr"],
            hidden_channels  = sampled["hidden_channels"],
            num_layers       = sampled["num_layers"],
            heads            = sampled["heads"],
            j_max            = sampled["j_max"],
            j_min            = sampled["j_min"],
            m_max            = sampled["m_max"],
            m_min            = sampled["m_min"],
            op_max           = sampled["op_max"],
            max_processing   = sampled["max_processing"],
            validation_freq  = effective_val_freq,
            validation_size  = effective_val_size,
            run_name         = run_name,
            representation   = rep,
        )

        q80 = last_val_q80(run_name)

        # Also read summary for ancillary attrs
        try:
            summary = read_run_summary(run_name)
        except FileNotFoundError:
            summary = {}

        trial.set_user_attr("run_name",        run_name)
        trial.set_user_attr("run_dir",         summary.get("run_dir", ""))
        trial.set_user_attr("best_difference", summary.get("best_difference"))
        trial.set_user_attr("total_runtime_sec", summary.get("total_runtime_sec"))
        trial.set_user_attr("best_val_avg_gap",  summary.get("best_validation_avg_gap"))
        return q80

    print(
        f"[PARAM] Starting Optuna study for {rep_upper} | trials={effective_trials} | "
        f"max_episodes={effective_max_episodes} | "
        f"objective=min(last_val_gap_Q80) | valdata={args.valdata}"
    )
    if not args.smoke:
        print(f"[PARAM] Critical params ({rep_upper}): {tuned_params}")
        if args.storage is None:
            print("[PARAM] WARNING: no --storage given; study is in-memory and cannot be resumed.")

    study.optimize(objective, n_trials=effective_trials, timeout=args.timeout)

    best = {
        "representation":       rep,
        "study_name":           study_name,
        "objective":            "last_val_gap_Q80",
        "critical_params":      tuned_params,
        "n_trials":             len(study.trials),
        "best_value":           float(study.best_value),
        "best_params":          study.best_trial.params,
        "best_user_attrs":      study.best_trial.user_attrs,
    }
    return best


# ── entrypoint ────────────────────────────────────────────────────────────────

def run_tuning(args: Optional[argparse.Namespace] = None) -> List[Dict]:
    """Programmatic entry point for Optuna tuning (used by both this script's
    CLI and main.py). Returns the list of per-representation result dicts."""
    if args is None:
        args = parse_args()
    args.storage = _normalize_storage_url(args.storage)
    reps = [r.lower().strip() for r in args.representations]

    print("=" * 70)
    print("[PARAM] Hyperparameter tuning runner (Optuna)")
    print(f"[PARAM] Representations : {[r.upper() for r in reps]}")
    print(f"[PARAM] Trials          : {args.trials}")
    print(f"[PARAM] Max episodes    : {args.max_episodes}")
    print(f"[PARAM] Objective       : min(last_val_gap_Q80)")
    print(f"[PARAM] Validation data : {args.valdata}")
    print(f"[PARAM] Storage         : {args.storage or 'in-memory (not persistent)'}")
    print("=" * 70)

    ensure_prerequisites(args.valdata)

    output_dir = Path("results") / "optuna"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = [tune_representation(rep, args) for rep in reps]

    timestamp = int(time.time())
    out_path = output_dir / f"tuning_summary_{timestamp}.json"
    out_path.write_text(json.dumps(all_results, indent=2))

    print("-" * 70)
    print(f"[PARAM] Tuning completed. Summary -> {out_path}")
    print("[PARAM] Ranking by last_val_Q80 (lower is better):")
    for idx, row in enumerate(sorted(all_results, key=lambda r: r["best_value"]), start=1):
        print(f"  {idx}. {row['representation'].upper()} | last_val_Q80={row['best_value']:.6f}")
    print("-" * 70)

    return all_results


if __name__ == "__main__":
    _args = parse_args()
    run_tuning(_args)
    if not _args.no_dashboard:
        open_dashboard()
