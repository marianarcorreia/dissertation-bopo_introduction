import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(__file__))

from src.train import BLOCKING_REPRESENTATIONS, test_model
from src.utils import OutputManager, open_dashboard


def parse_args():
    parser = argparse.ArgumentParser(description="Run model tests and save results to JSON files.")
    parser.add_argument(
        "--run-name",
        default="test_run",
        help="Name used for the run output folder inside results/.",
    )
    parser.add_argument(
        "--models-file",
        default=None,
        help="Path to model_params.json (default: models/model_params.json, or "
             "models/blocking/model_params.json when --representation selects blocking models).",
    )
    parser.add_argument(
        "--source-folder",
        default="val",
        help="Base folder containing test subfolders (example: val, data/test, data/benchmarks).",
    )
    parser.add_argument(
        "--folders",
        default=None,
        help="Comma-separated subfolder names inside source-folder (default: instances, or "
             "blocking_test_instances for blocking models).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Folder where result JSON files are written (default: results, or results/blocking "
             "for blocking models).",
    )
    parser.add_argument(
        "--output-prefix",
        default="results",
        help="Prefix used in output file names.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel workers. 0 means one worker per model.",
    )
    parser.add_argument(
        "--representation",
        nargs="+",
        default=None,
        help="Only evaluate the models of these representations (default: every model in "
             "--models-file). Blocking (ojmb, ojmd, ojm_blk) and non-blocking models cannot be mixed.",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Do not auto-launch/open the results dashboard when the run finishes.",
    )
    return parser.parse_args()


def expand_representations(representations):
    """'all' -> oo om ojm, 'blocking' -> ojmb ojmd ojm_blk; None stays None (no filter)."""
    if not representations:
        return None
    groups = {"all": ("oo", "om", "ojm"), "blocking": BLOCKING_REPRESENTATIONS}
    return {r for v in representations for r in groups.get(v.lower(), (v.lower(),))}


def default_test_paths(representations):
    """Blocking models live in models/blocking/, are tested on the blocking test instances and
    write to results/blocking/ (see src/train.py:output_paths); everything else as before."""
    wanted = expand_representations(representations)
    if wanted and wanted <= set(BLOCKING_REPRESENTATIONS):
        return {"models_file": "models/blocking/model_params.json",
                "folders": "blocking_test_instances", "output_dir": "results/blocking"}
    return {"models_file": "models/model_params.json", "folders": "instances", "output_dir": "results"}


def select_models(model_params, representations=None):
    """Models to evaluate: those of the given representations (all if None). The score of an
    instance is the best makespan over these models, so they must all solve the same problem:
    a blocking makespan and a non-blocking one cannot be compared."""
    wanted = expand_representations(representations)
    if wanted:
        model_params = [p for p in model_params if p.get("representation", "oo") in wanted]
    blocking = {p.get("representation", "oo") in BLOCKING_REPRESENTATIONS for p in model_params}
    if len(blocking) > 1:
        raise ValueError("--models-file mixes blocking (ojmb/ojmd/ojm_blk) and non-blocking models; "
                         "select one group with --representation.")
    if blocking == {True}:
        print("[TEST] Blocking models: evaluate them on blocking instances, e.g. "
              "--source-folder val --folders blocking_test_instances")
    return model_params


def multi_run_wrapper(args):
    try:
        return test_model(*args)
    except Exception as exc:
        model_name, folder, filename, models_file = args
        raise RuntimeError(
            f"multi_run_wrapper failed for model='{model_name}', folder='{folder}', "
            f"filename='{filename}', models_file='{models_file}'"
        ) from exc


def _to_scalar_score(score_value):
    if isinstance(score_value, (list, tuple)):
        return min(score_value) if len(score_value) > 0 else float("inf")
    return score_value


def _to_scalar_time(time_value):
    if isinstance(time_value, (list, tuple)):
        return min(time_value) if len(time_value) > 0 else 0.0
    return float(time_value)


def _parse_folders(raw_value):
    return [folder.strip() for folder in raw_value.split(",") if folder.strip()]


def _safe_name(value):
    return value.replace("/", "_").replace("\\", "_")


def run_tests(
    run_name="test_run",
    models_file=None,
    source_folder="val",
    folders=None,
    output_dir=None,
    output_prefix="results",
    workers=0,
    representations=None,
):
    """Programmatic entry point for testing (used by both this script's CLI and
    main.py). Returns the OutputManager's run_dir for the completed run."""
    start = time.time()
    defaults = default_test_paths(representations)
    models_file = models_file or defaults["models_file"]
    folders = folders or defaults["folders"]
    output_dir = output_dir or defaults["output_dir"]
    print(f"[TEST] models_file={models_file} | folders={folders} | output_dir={output_dir}")

    output_manager = OutputManager(output_dir=output_dir, run_name=run_name)
    print(f"[TEST] run_name={run_name}")
    print(f"[TEST] Output run folder: {output_manager.run_dir}")

    with open(models_file, "r") as infile:
        model_params = select_models(json.load(infile), representations)
    if not model_params:
        raise ValueError(f"No model in {models_file} matches representation(s) {representations}. "
                         f"Training lists its checkpoints in candidate_models/[blocking/]model_params.json; "
                         f"copy the chosen entries into {models_file} or pass --models-file.")

    folder_paths = _parse_folders(folders)
    if not folder_paths:
        raise ValueError("No folder was provided. Use --folders with at least one value.")

    for folder_path in folder_paths:
        save_results = []
        current_folder = os.path.join(source_folder, folder_path)

        if not os.path.isdir(current_folder):
            print(f"[WARN] Skipping missing folder: {current_folder}")
            continue

        filenames = []
        for file_name in os.listdir(current_folder):
            if os.path.isfile(os.path.join(current_folder, file_name)):
                filenames.append(file_name)

        filenames.sort()

        for file_name in filenames:
            # representation passed explicitly so test_model builds the right env even for
            # entries saved without a "representation" key
            models = [(v["name"], current_folder, file_name, models_file, v.get("representation", "oo"))
                      for v in model_params]
            if not models:
                continue

            n_workers = workers if workers > 0 else len(models)
            n_workers = max(1, min(n_workers, len(models)))

            with Pool(n_workers) as pool:
                res = pool.map(multi_run_wrapper, models)

            result = min([_to_scalar_score(r[0]) for r in res])
            start_time = min([_to_scalar_time(r[1]) for r in res])
            end_time = max([_to_scalar_time(r[2]) for r in res])
            wall_time = end_time - start_time

            print(file_name, result, wall_time)
            save_results.append({"score": result, "name": file_name, "time": wall_time})

        output_name = f"{output_prefix}_{_safe_name(folder_path)}.json"
        output_path = os.path.join(output_manager.run_dir, output_name)
        with open(output_path, "w") as outfile:
            json.dump({"results": save_results}, outfile)

        print(f"[OK] Saved: {output_path}")

    output_manager.save_run_summary(
        {
            "run_name": run_name,
            "run_dir": output_manager.run_dir,
            "models_file": models_file,
            "source_folder": source_folder,
            "folders": folder_paths,
            "workers": workers,
            "total_runtime_sec": float(time.time() - start),
        }
    )

    print(f"[DONE] Total elapsed time: {time.time() - start:.2f}s")
    return output_manager.run_dir


if __name__ == "__main__":
    args = parse_args()
    run_tests(
        run_name=args.run_name,
        models_file=args.models_file,
        source_folder=args.source_folder,
        folders=args.folders,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        workers=args.workers,
        representations=args.representation,
    )

    if not args.no_dashboard:
        open_dashboard()