"""
Integrity check for val/instances + val/solutions.

run_validation() (src/utils/validation_utils.py) computes each validation gap as
mk / best_objective_bound - 1, treating best_objective_bound as the TRUE optimum. That
is only valid if the solver actually PROVED optimality (lower bound == achieved
objective) for every stored solution - otherwise best_objective_bound can be a loose
lower bound, and the reported gap is inflated (the model looks worse than it is).

The stored val/solutions/*.json files don't carry an explicit solver status, so this
script re-derives the achieved makespan from each solution's own final_schedule (the
max "end" over all scheduled operations) and compares it against the stored
best_objective_bound: if they match, the bound was tight (optimal or at least
self-consistent); if the schedule's actual makespan is BIGGER than the bound, the bound
was not achieved by the stored schedule and gap_utils comparisons against it are not
using the makespan of a real feasible solution.

Usage:
    python -m src.utils.check_validation_set [--solutions-dir val/solutions]
"""

import argparse
import json
import os
from pathlib import Path


def check_solution_file(path: Path):
    with open(path, "r") as f:
        data = json.load(f)

    schedule = data.get("final_schedule", [])
    if not schedule:
        return {"file": path.name, "error": "no final_schedule entries"}

    achieved_makespan = max(entry["end"] for entry in schedule)
    bound = data.get("indicators", {}).get("best_objective_bound")

    if bound is None:
        return {"file": path.name, "error": "no indicators.best_objective_bound"}

    return {
        "file": path.name,
        "achieved_makespan": achieved_makespan,
        "best_objective_bound": bound,
        "matches": abs(achieved_makespan - bound) < 1e-6,
        "gap_if_used_as_reference": (achieved_makespan / bound - 1.0) if bound > 0 else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solutions-dir", default="val/solutions")
    args = parser.parse_args()

    solutions_dir = Path(args.solutions_dir)
    files = sorted(solutions_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No solution files found under {solutions_dir}")

    results = [check_solution_file(f) for f in files]
    errors = [r for r in results if "error" in r]
    mismatches = [r for r in results if "error" not in r and not r["matches"]]

    print(f"Checked {len(results)} solution file(s) in {solutions_dir}")
    print(f"  OK (bound matches achieved makespan): {len(results) - len(errors) - len(mismatches)}")
    print(f"  Mismatched (bound not reached by stored schedule): {len(mismatches)}")
    print(f"  Errors (missing fields): {len(errors)}")

    if mismatches:
        print("\nMismatches (these instances' reference is a loose bound, not a proven optimum):")
        for r in mismatches:
            print(f"  {r['file']}: achieved={r['achieved_makespan']} bound={r['best_objective_bound']} "
                  f"(gap if used as-is: {r['gap_if_used_as_reference']:.4f})")

    if errors:
        print("\nFiles with missing/unreadable fields:")
        for r in errors:
            print(f"  {r['file']}: {r['error']}")

    if not mismatches and not errors:
        print("\nAll validation references are self-consistent (proven-tight bounds).")


if __name__ == "__main__":
    main()
