"""What is training on this computer right now, and how far along is it?

Usage:
    python status.py                  # one snapshot
    python status.py --watch          # refresh every 60 s (Ctrl+C to quit)
    python status.py --recent-min 120 # also list logs touched in the last 2 h

Shows (1) every running training-related process (sweep launcher, run_diagnostic_job,
Optuna/param.py, main.py, the chain script, rescoring) and (2) the latest progress of each
of their logs in sweep_logs/: training step, last validation gap, Optuna trial, or
sweep/chain event. Read-only: it never starts or stops anything.
"""
import argparse
import os
import re
import time

import psutil

ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(ROOT, "sweep_logs")
# matched against the basename of each argv entry, so an inline `python -c "...optuna..."`
# query is not mistaken for a training job
WATCHED_SCRIPTS = {"run_seed_sweep.py", "run_diagnostic_job.py", "param.py", "main.py",
                   "chain_selk2.ps1", "rescore_checkpoints.py"}
PROGRESS = {
    "step": re.compile(r"\[TRAIN\] Step (\d+)/(\d+) \| (\S+)"),
    "warmstart": re.compile(r"\[WARMSTART\] step (\d+)/(\d+)"),
    "val": re.compile(r"\[VAL\]\[ep (\d+)\] avg_gap=([\d.]+)"),
    "test": re.compile(r"\[TEST\] avg_gap=([\d.]+)"),
    "trial": re.compile(r"\[PARAM\]\[\w+\] Trial (\d+) \|"),
    "event": re.compile(r"^\[(SWEEP|CHAIN) .*"),
    "error": re.compile(r"Traceback|Error:|out of memory", re.I),
}


def running_processes():
    rows = []
    for p in psutil.process_iter(["pid", "ppid", "cmdline", "create_time"]):
        argv = p.info["cmdline"] or []
        script = next((os.path.basename(a) for a in argv if os.path.basename(a) in WATCHED_SCRIPTS), None)
        if not script:
            continue
        cmd = " ".join(argv)
        # the venv's python.exe launcher spawns a child with the same command line - list it once
        try:
            if " ".join(psutil.Process(p.info["ppid"]).cmdline()) == cmd:
                continue
        except (psutil.Error, OSError):
            pass
        args = cmd.split(script, 1)[-1].strip().strip('"').strip()
        started = time.strftime("%m-%d %H:%M", time.localtime(p.info["create_time"]))
        rows.append((p.info["pid"], script, args, started))
    return rows


def latest_progress(path):
    with open(path, "rb") as fh:
        fh.seek(max(0, os.path.getsize(path) - 200_000))
        lines = fh.read().decode("utf-8", "replace").splitlines()
    found, where = {}, {}
    for i, line in enumerate(lines):
        for key, rx in PROGRESS.items():
            m = rx.search(line)
            if m:
                found[key], where[key] = m, i
    # one log can hold several trainings (Optuna trials): ignore val/test results that belong
    # to an earlier trial than the one currently shown
    trial_start = where.get("trial", -1)
    for key in ("val", "test"):
        if key in found and where[key] < trial_start:
            del found[key]
    if "test" in found and where.get("step", -1) > where["test"]:
        del found["test"]

    parts = []
    if "trial" in found:
        parts.append(f"Optuna trial {found['trial'].group(1)}")
    if "step" in found and where["step"] > trial_start:
        s = found["step"]
        parts.append(f"step {s.group(1)}/{s.group(2)} (at {s.group(3)})")
    elif "warmstart" in found and where["warmstart"] > trial_start:
        w = found["warmstart"]
        parts.append(f"warm-start {w.group(1)}/{w.group(2)}")
    if "val" in found:
        parts.append(f"last val gap {float(found['val'].group(2)):.4f} @ep {found['val'].group(1)}")
    if "test" in found:
        parts.append(f"TEST gap {float(found['test'].group(1)):.4f} (finished)")
    if "event" in found and not parts:
        parts.append(found["event"].group(0))
    if "error" in found:
        parts.append(f"!! error: {found['error'].string.strip()[:80]}")
    return " | ".join(parts) or "(no progress lines yet)"


def snapshot(recent_min):
    print(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    procs = running_processes()
    print(f"\nRunning jobs ({len(procs)}):")
    if not procs:
        print("  nothing running")
    for pid, script, args, started in procs:
        print(f"  PID {pid:<6} {script:<24} started {started}  {args[:110]}")

    # Windows does not refresh a log's modified time while it is still open for writing, so
    # the log of every running run_diagnostic_job (sweep_logs/<run-name>.log) is always shown
    running_logs = set()
    for _, script, args, _ in procs:
        m = re.search(r"--run-name\s+(\S+)", args)
        if script == "run_diagnostic_job.py" and m:
            running_logs.add(m.group(1) + ".log")
    cutoff = time.time() - recent_min * 60
    logs = sorted((f for f in os.listdir(LOGS) if f.endswith(".log") and not f.endswith(".err.log")),
                  key=lambda f: os.path.getmtime(os.path.join(LOGS, f)), reverse=True)
    active = [f for f in logs if f in running_logs or os.path.getmtime(os.path.join(LOGS, f)) >= cutoff]
    print(f"\nLogs of running jobs, or touched in the last {recent_min} min ({len(active)}):")
    for f in active:
        tag = "RUNNING " if f in running_logs else "        "
        print(f"  {tag}{f:<48} | {latest_progress(os.path.join(LOGS, f))}")


parser = argparse.ArgumentParser()
parser.add_argument("--watch", action="store_true", help="Refresh every 60 s.")
parser.add_argument("--recent-min", type=int, default=30, help="Also show logs modified within this many minutes.")
args = parser.parse_args()

if args.watch:
    try:
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            snapshot(args.recent_min)
            time.sleep(60)
    except KeyboardInterrupt:
        pass
else:
    snapshot(args.recent_min)
