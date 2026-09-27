
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
    parser.add_argument(
        "--mask-option",
        type=int,
        default=None,
        choices=[0, 1],
        help="[train/optuna] Machine ranking for the om/ojm action mask: 0 = earliest start, "
             "1 = earliest completion. Default: train() default in train mode, param.py's "
             "value in optuna mode.",
    )
    parser.add_argument(
        "--sel-k",
        type=int,
        default=None,
        help="[train/optuna] Candidate machines kept per job by the om/ojm action mask. "
             "Default: train() default in train mode, sampled by param.py in optuna mode.",
    )
    parser.add_argument(
        "--sel-k-choices",
        type=int,
        nargs="+",
        default=None,
        help="[optuna] sel_k values the om/ojm search samples from (default: param.py's "
             "DEFAULT_SEL_K_CHOICES). Ignored when --sel-k fixes sel_k.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="[train/optuna] Number of GNN layers. Default: train() default in train mode, "
             "param.py's value in optuna mode.",
    )
    parser.add_argument(
        "--logp-norm",
        default=None,
        choices=["mean", "sum"],
        help="[train/optuna] How a rollout's log-probs are combined in the BOPO loss: 'mean' "
             "(per-decision average, train() default) or 'sum' (the loss used before f1b12fe).",
    )
    parser.add_argument(
        "--exclude-greedy",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="[train/optuna] Leave the greedy rollout out of the BOPO loss pairs (train() "
             "default). --no-exclude-greedy keeps it in, as before f1b12fe. The old loss is "
             "--logp-norm sum --no-exclude-greedy.",
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


def _override_kwargs(args):
    """Only the flags the user actually passed, so train()/param.py defaults apply otherwise."""
    overrides = {"mask_option": args.mask_option, "sel_k": args.sel_k, "num_layers": args.num_layers,
                 "logp_norm": args.logp_norm, "exclude_greedy_from_loss": args.exclude_greedy}
    return {k: v for k, v in overrides.items() if v is not None}


def run_train(args):
    reps = _resolve_representations(args.representation)
    multi = len(reps) > 1
    overrides = _override_kwargs(args)
    for rep in reps:
        run_name = f"{args.run_name}_{rep}" if multi else args.run_name
        print(f"[MAIN] Training representation={rep} | run_name={run_name} | gnn_type={args.gnn_type} | overrides={overrides}")
        train(run_name=run_name, representation=rep, max_episodes=args.max_episodes, gnn_type=args.gnn_type,
              **overrides)


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
        overrides=_override_kwargs(args),
        sel_k_choices=args.sel_k_choices,
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
        print(f"[MAIN] overrides={_override_kwargs(args)}")
    print("=" * 60)

    _DISPATCH[args.mode](args)

    print(f"[MAIN] Run finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if not args.no_dashboard:
        open_dashboard()
