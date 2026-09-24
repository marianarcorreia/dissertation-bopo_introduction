"""Shared data-loading helpers for the Streamlit dashboard."""

import glob
import json
import os
from typing import Any, Callable, cast

import numpy as np
import pandas as pd

try:
    import streamlit as st
except Exception:  # pragma: no cover - allows non-Streamlit imports/tests
    st = None


def _cache_data(**kwargs):
    """Use Streamlit cache when available; otherwise return an identity decorator."""
    if st is None:
        def _identity(func: Callable[..., Any]) -> Callable[..., Any]:
            return func
        return _identity
    return st.cache_data(**kwargs)

RESULTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "results")
)




# ── folder scanning ──────────────────────────────────────────────────────────

# Files test.py writes directly (see test.py's --output-prefix, default "results"),
# as opposed to the fixed metric filenames OutputManager writes for training runs.
_KNOWN_METRIC_FILES = {
    "run_summary.json",
    "episode_metrics.json",
    "update_metrics.json",
    "validation_history.json",
}


def _has_benchmark(path: str) -> bool:
    return os.path.exists(os.path.join(path, "benchmark", "instance_gaps.csv"))


def _has_trial_subdirs(path: str) -> bool:
    try:
        return any(
            entry.is_dir() and entry.name.startswith("trial_")
            for entry in os.scandir(path)
        )
    except OSError:
        return False


def _test_score_files(path: str) -> list[str]:
    """Names of test.py-style `<prefix>_<folder>.json` score files directly in `path`."""
    try:
        names = os.listdir(path)
    except OSError:
        return []
    out = []
    for name in names:
        if name in _KNOWN_METRIC_FILES or not name.endswith(".json"):
            continue
        full = os.path.join(path, name)
        if not os.path.isfile(full):
            continue
        out.append(name)
    return out


def _detect_run_type(path: str) -> str:
    if os.path.exists(os.path.join(path, "episode_metrics.json")):
        return "training"
    if _has_benchmark(path):
        return "benchmark"
    if _has_trial_subdirs(path):
        return "optuna_study"
    if _test_score_files(path):
        return "test"
    return "unknown"


@_cache_data(show_spinner=False, ttl=5)
def list_all_runs() -> list[dict]:
    """Return a list of metadata dicts for every subfolder in results/, plus a
    synthetic entry for any legacy `results_*.json` test-score files sitting
    directly in results/ (predating the per-run-folder OutputManager layout)."""
    runs = []
    if not os.path.isdir(RESULTS_DIR):
        return runs

    loose_files = _test_score_files(RESULTS_DIR)
    for folder in sorted(os.listdir(RESULTS_DIR)):
        path = os.path.join(RESULTS_DIR, folder)
        if not os.path.isdir(path):
            continue
        run_type = _detect_run_type(path)
        run = {
            "name": folder,
            "path": path,
            "type": run_type,
            "has_benchmark": _has_benchmark(path),
            "has_test_scores": bool(_test_score_files(path)),
            "representation": None,
            "summary": {},
            "gnn_type": None,
            "last_modified": os.path.getmtime(path),
        }
        summary_path = os.path.join(path, "run_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                try:
                    run["summary"] = json.load(f)
                except json.JSONDecodeError:
                    pass
            run["representation"] = run["summary"].get("representation")
            run["gnn_type"] = run["summary"].get("gnn_type")
        if run_type == "optuna_study":
            run["summary"] = {
                "trial_count": sum(
                    1 for e in os.scandir(path) if e.is_dir() and e.name.startswith("trial_")
                )
            }
        runs.append(run)

    if loose_files:
        mtimes = [os.path.getmtime(os.path.join(RESULTS_DIR, f)) for f in loose_files]
        runs.append({
            "name": "(legacy top-level results/*.json)",
            "path": RESULTS_DIR,
            "type": "test",
            "has_benchmark": False,
            "has_test_scores": True,
            "representation": None,
            "summary": {"files": loose_files},
            "gnn_type": None,
            "last_modified": max(mtimes),
        })

    runs.sort(key=lambda r: r["last_modified"], reverse=True)
    return runs


@_cache_data(show_spinner=False, ttl=5)
def training_runs() -> list[dict]:
    return [r for r in list_all_runs() if r["type"] == "training"]


@_cache_data(show_spinner=False, ttl=5)
def benchmark_runs() -> list[dict]:
    """Returns all runs that have benchmark/instance_gaps.csv (training runs included)."""
    return [r for r in list_all_runs() if r["has_benchmark"]]


@_cache_data(show_spinner=False, ttl=5)
def test_runs() -> list[dict]:
    """Returns all runs carrying test.py-style `<prefix>_<folder>.json` score files."""
    return [r for r in list_all_runs() if r.get("has_test_scores")]


# ── JSON loaders ─────────────────────────────────────────────────────────────

@_cache_data(show_spinner=False, ttl=5)
def _load_json(path: str) -> list | dict:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


@_cache_data(show_spinner=False, ttl=5)
def load_episode_metrics(run_path: str) -> list[dict]:
    return cast(list[dict], _load_json(os.path.join(run_path, "episode_metrics.json")))


@_cache_data(show_spinner=False, ttl=5)
def load_validation_history(run_path: str) -> list[dict]:
    return cast(list[dict], _load_json(os.path.join(run_path, "validation_history.json")))


@_cache_data(show_spinner=False, ttl=5)
def load_update_metrics(run_path: str) -> list[dict]:
    return cast(list[dict], _load_json(os.path.join(run_path, "update_metrics.json")))


@_cache_data(show_spinner=False, ttl=5)
def load_benchmark_results(run_path: str) -> dict[str, list[dict]]:
    """
    Returns {dataset_name: [per-instance result dicts]} from benchmark/instance_gaps.csv.

    Each record contains:
      name        – instance stem
      bks         – best-known solution (optimal makespan)
      score       – model makespan for the best checkpoint (bks * (1 + gap))
      gap_percent – gap to BKS in percent (last/best checkpoint column)
      checkpoint  – stem of the checkpoint used

    The last non-null checkpoint column is used as the representative result because
    checkpoints are saved only on validation improvement, so the last one is the best.
    """
    csv_path = os.path.join(run_path, "benchmark", "instance_gaps.csv")
    if not os.path.exists(csv_path):
        return {}

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}

    fixed_cols = {"dataset", "instance", "optimal"}
    ckpt_cols = [c for c in df.columns if c not in fixed_cols]
    if not ckpt_cols:
        return {}

    out: dict[str, list[dict]] = {}
    for _, row in df.iterrows():
        dataset = str(row["dataset"])
        instance = str(row.get("instance", ""))
        optimal = row.get("optimal")
        try:
            optimal_f = float(optimal) if optimal is not None and not pd.isna(optimal) else None
        except (TypeError, ValueError):
            optimal_f = None

        # Walk checkpoint columns from last (best) to first to find a valid gap value.
        gap_val = None
        best_ckpt = None
        for col in reversed(ckpt_cols):
            v = row.get(col)
            try:
                fv = float(v)
                if not np.isnan(fv):
                    gap_val = fv
                    best_ckpt = col
                    break
            except (TypeError, ValueError):
                continue

        score = None
        if gap_val is not None and optimal_f is not None and optimal_f > 0:
            score = optimal_f * (1.0 + gap_val)

        record = {
            "name": instance,
            "bks": optimal_f,
            "score": score,
            "gap_percent": gap_val * 100.0 if gap_val is not None else None,
            "checkpoint": best_ckpt,
        }

        out.setdefault(dataset, []).append(record)

    return out


@_cache_data(show_spinner=False, ttl=5)
def load_benchmark_all_checkpoints(run_path: str) -> dict[str, pd.DataFrame]:
    """
    Returns {dataset_name: DataFrame} where each DataFrame has columns:
      instance, optimal, <checkpoint_stem1>, <checkpoint_stem2>, ...
    with gap values (decimal fractions).

    Useful for per-checkpoint comparison within a single run.
    """
    csv_path = os.path.join(run_path, "benchmark", "instance_gaps.csv")
    if not os.path.exists(csv_path):
        return {}
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}

    out = {}
    for dataset, grp in df.groupby("dataset"):
        out[str(dataset)] = grp.drop(columns=["dataset"]).reset_index(drop=True)
    return out


# ── episode-metrics → DataFrames ─────────────────────────────────────────────

def episode_df(run_path: str, label: str | None = None) -> pd.DataFrame:
    rows = load_episode_metrics(run_path)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if label:
        df["run"] = label
    return df


def validation_df(run_path: str, label: str | None = None) -> pd.DataFrame:
    rows = load_validation_history(run_path)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if label:
        df["run"] = label
    return df


def update_df(run_path: str, label: str | None = None) -> pd.DataFrame:
    rows = load_update_metrics(run_path)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if label:
        df["run"] = label
    return df


# ── benchmark DataFrame ───────────────────────────────────────────────────────

def benchmark_df(run_path: str, label: str | None = None) -> pd.DataFrame:
    all_rows = []
    for dataset, records in load_benchmark_results(run_path).items():
        for r in records:
            row = dict(r)
            row["dataset"] = dataset
            if label:
                row["run"] = label
            all_rows.append(row)
    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


def combined_benchmark_df(benchmark_run_list: list[dict]) -> pd.DataFrame:
    """Merge benchmark results from multiple runs into one DataFrame."""
    frames = []
    for run in benchmark_run_list:
        rep = run.get("representation") or run["summary"].get("representation", "unknown")
        df = benchmark_df(run["path"], label=run["name"])
        if not df.empty:
            df["representation"] = rep
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── test.py score results (results_<folder>.json format actually produced today) ──

@_cache_data(show_spinner=False, ttl=5)
def load_test_score_files(run_path: str) -> dict[str, list[dict]]:
    """Returns {dataset_name: [{'name','score','time'}, ...]} from the
    `<output-prefix>_<folder>.json` files test.py writes into a run's directory
    (or directly into results/ for runs predating the OutputManager layout).

    Note: these carry the model's raw makespan per instance, not an optimality
    gap — test.py does not currently join against a reference/optimal value.
    """
    out: dict[str, list[dict]] = {}
    for fname in _test_score_files(run_path):
        fpath = os.path.join(run_path, fname)
        try:
            with open(fpath) as f:
                data = json.load(f)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        records = data.get("results")
        if not isinstance(records, list):
            continue
        dataset = fname[:-len(".json")]
        if dataset.startswith("results_"):
            dataset = dataset[len("results_"):]
        out[dataset] = records
    return out


def _unwrap_score(value: Any) -> Any:
    """Some older test.py output wraps score as a single-element list (e.g.
    [141.0]) instead of a bare float — unwrap it so score is scalar everywhere
    and datasets can be combined in one DataFrame/chart."""
    if isinstance(value, list):
        return value[0] if len(value) == 1 else (value[0] if value else None)
    return value


def test_score_df(run_path: str, label: str | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, records in load_test_score_files(run_path).items():
        for r in records:
            row = dict(r)
            row["score"] = _unwrap_score(row.get("score"))
            row["dataset"] = dataset
            if label:
                row["run"] = label
            rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def combined_test_score_df(test_run_list: list[dict]) -> pd.DataFrame:
    """Merge test-score results from multiple runs into one DataFrame."""
    frames = []
    for run in test_run_list:
        rep = run.get("representation") or run["summary"].get("representation", "unknown")
        df = test_score_df(run["path"], label=run["name"])
        if not df.empty:
            df["representation"] = rep
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── wide → long metric reshaping (shared by the Training Metrics and Optuna
#    Trials pages, so both pages melt run/trial metric series identically) ──

def numeric_metric_columns(df: pd.DataFrame, step_col: str | None) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if step_col and col == step_col:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().any():
            cols.append(col)
    return cols


def step_column(df: pd.DataFrame) -> str:
    for candidate in ["episode", "update", "step", "iteration"]:
        if candidate in df.columns:
            return candidate
    return "_index"


def to_long_metrics(
    df: pd.DataFrame, series_name: str, source: str, series_col: str = "run"
) -> pd.DataFrame:
    """Melt a wide metrics DataFrame into long form: [series_col, source, step, metric, value].

    `series_name` is the run/trial identifier stamped onto every row so several
    runs' long frames can be concatenated and compared on one chart.
    """
    if df.empty:
        return pd.DataFrame(columns=[series_col, "source", "step", "metric", "value"])

    data = df.copy().reset_index(drop=True)
    col = step_column(data)
    if col == "_index":
        data[col] = data.index + 1

    metric_cols = numeric_metric_columns(data, col)
    if not metric_cols:
        return pd.DataFrame(columns=[series_col, "source", "step", "metric", "value"])

    for c in metric_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    melted = data.melt(id_vars=[col], value_vars=metric_cols, var_name="metric", value_name="value")
    melted = melted.dropna(subset=["value"])
    melted = melted.rename(columns={col: "step"})
    melted[series_col] = series_name
    melted["source"] = source
    return melted[[series_col, "source", "step", "metric", "value"]]
