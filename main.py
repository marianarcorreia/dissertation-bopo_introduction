
import os
import sys
import argparse
if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(__file__))
from datetime import datetime

from src.train import train
from src.utils import open_dashboard

ALL_REPRESENTATIONS = ["oo", "om", "ojm"]


def _resolve_representations(values):
    """Expand 'all' into every representation and de-duplicate, preserving order."""
    reps = []
    for v in values:
        v = v.lower().strip()
        if v == "all":
            for r in ALL_REPRESENTATIONS:
                if r not in reps:
                    reps.append(r)
        elif v not in reps:
            reps.append(v)
    return reps


def parse_args():
    parser = argparse.ArgumentParser(
        description="BOPO for FJSP — single entry point for training, testing and Optuna tuning."
    )
    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "test", "optuna"],
        help="Mode to run: 'train' (BOPO training), 'test' (evaluate saved models in --models-file "
             "against --source-folder/--folders), 'optuna' (hyperparameter tuning per representation).",
    )
    parser.add_argument(
        "--run-name",
        default="train_run",
        help="Base name for the run's output folder inside results/. In train/optuna mode with "
             "more than one --representation, each run is named '<run-name>_<representation>' so "
             "they don't overwrite each other.",
    )
    parser.add_argument(
        "--representation",
        nargs="+",
        default=["oo"],
        choices=["oo", "om", "ojm", "all"],
        help="Graph representation(s) to use: oo (operation-only), om (operation-machine), "
             "ojm (operation-job-machine). Pass several values, or 'all', to run every "
             "representation in one invocation. Used by train and optuna modes; test mode "
             "evaluates whichever models are listed in --models-file regardless of this flag.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=100,
        help="[train] Number of BOPO training steps. [optuna] Training steps per trial.",
    )
    parser.add_argument(
        "--gnn-type",
        default="gat",
        choices=["gat", "gin", "transformer"],
        help="[train] GNN backbone used by the actor: 'gat' (GATv2Conv, attention-based), "
             "'gin' (GINEConv, sum-aggregation with edge features) or 'transformer' "
             "(TransformerConv, multi-head query/key/value attention with edge features).",
    )
    test_group = parser.add_argument_group("test mode")
    test_group.add_argument("--models-file", default="models/model_params.json",
                             help="[test] Path to model_params.json.")
    test_group.add_argument("--source-folder", default="val",
                             help="[test] Base folder containing test subfolders "
                                  "(e.g. val, data/test, data/benchmarks).")
    test_group.add_argument("--folders", default="instances",
                             help="[test] Comma-separated subfolder names inside --source-folder.")
    test_group.add_argument("--output-dir", default="results",
                             help="[test] Folder where result JSON files are written.")
    test_group.add_argument("--output-prefix", default="results",
                             help="[test] Prefix used in output file names.")
    test_group.add_argument("--workers", type=int, default=0,
                             help="[test] Parallel workers. 0 means one worker per model.")

    optuna_group = parser.add_argument_group("optuna mode")
    optuna_group.add_argument("--trials", type=int, default=30,
                               help="[optuna] Number of trials per representation.")
    optuna_group.add_argument("--validation-freq", type=int, default=10,
                               help="[optuna] Validation frequency in episodes.")
    optuna_group.add_argument("--validation-size", type=int, default=30,
                               help="[optuna] Size of EACH split (validation and test) the first "
                                    "time they're generated; ignored afterwards so every trial "
                                    "stays comparable (see --rebuild-validation-set).")
    optuna_group.add_argument("--rebuild-validation-set", action="store_true",
                               help="[optuna] Force-regenerate the fixed validation/test split "
                                    "at --validation-size. Only do this deliberately - it makes "
                                    "prior runs/trials incomparable to ones made after a rebuild.")
    optuna_group.add_argument("--sampler-seed", type=int, default=42,
                               help="[optuna] Seed for the Optuna TPE sampler.")
    optuna_group.add_argument("--storage", default=None,
                               help="[optuna] Optuna storage URL, e.g. sqlite:///optuna.db, or a bare "
                                    "file path (e.g. optuna.db) which is auto-converted to a SQLite URL "
                                    "(recommended: allows resuming interrupted studies).")
    optuna_group.add_argument("--timeout", type=int, default=None,
                               help="[optuna] Optional timeout (seconds) per representation study.")
    optuna_group.add_argument("--smoke", action="store_true",
                               help="[optuna] Quick end-to-end smoke-test mode.")

    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Do not auto-launch/open the results dashboard when the run finishes.",
    )

    return parser.parse_args()


def run_train(args):
    reps = _resolve_representations(args.representation)
    multi = len(reps) > 1
    for rep in reps:
        run_name = f"{args.run_name}_{rep}" if multi else args.run_name
        print(f"[MAIN] Training representation={rep} | run_name={run_name} | gnn_type={args.gnn_type}")
        train(run_name=run_name, representation=rep, max_episodes=args.max_episodes, gnn_type=args.gnn_type)


def run_test(args):
    import test as test_cli  # sibling test.py, resolved via this script's own directory on sys.path
    test_cli.run_tests(
        run_name=args.run_name,
        models_file=args.models_file,
        source_folder=args.source_folder,
        folders=args.folders,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        workers=args.workers,
    )


def run_optuna(args):
    import param as param_cli  # sibling param.py
    reps = _resolve_representations(args.representation)
    optuna_args = argparse.Namespace(
        trials=args.trials,
        max_episodes=args.max_episodes,
        validation_freq=args.validation_freq,
        validation_size=args.validation_size,
        rebuild_validation_set=args.rebuild_validation_set,
        sampler_seed=args.sampler_seed,
        storage=args.storage,
        timeout=args.timeout,
        smoke=args.smoke,
        representations=reps,
        gnn_type=args.gnn_type,
    )
    param_cli.run_tuning(optuna_args)


_DISPATCH = {"train": run_train, "test": run_test, "optuna": run_optuna}


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print(f"[MAIN] Run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[MAIN] mode={args.mode}")
    print(f"[MAIN] run_name={args.run_name}")
    if args.mode in ("train", "optuna"):
        print(f"[MAIN] representation(s)={_resolve_representations(args.representation)}")
    if args.mode in ("train", "optuna"):
        print(f"[MAIN] gnn_type={args.gnn_type}")
    print("=" * 60)

    _DISPATCH[args.mode](args)

    print(f"[MAIN] Run finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if not args.no_dashboard:
        open_dashboard()
