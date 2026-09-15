"""Auto-launch the Streamlit results dashboard at the end of a run.

`open_dashboard()` is called from main.py / test.py / param.py once a run
finishes. It reuses an already-running dashboard server if one is listening
on the target port (the page's data refreshes on its own via the short
st.cache_data ttl in dashboard_data.py, so re-opening it is enough to see the
new run) and only spawns a new `streamlit run` process otherwise.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_ENTRY = PROJECT_ROOT / "dashboard.py"
DEFAULT_PORT = 8501


def _port_open(port: int, host: str = "localhost", timeout: float = 0.3) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        try:
            return sock.connect_ex((host, port)) == 0
        except OSError:
            return False


def open_dashboard(port: int = DEFAULT_PORT, wait_seconds: float = 60.0) -> str:
    """Ensure the dashboard is running on `port` and open it in the browser.

    Returns the dashboard URL. Never raises: a failure to start the server or
    open a browser is logged and swallowed so it can't break the caller's run.
    """
    url = f"http://localhost:{port}"

    if not DASHBOARD_ENTRY.exists():
        print(f"[DASHBOARD] {DASHBOARD_ENTRY} not found; skipping auto-launch.")
        return url

    if not _port_open(port):
        print(f"[DASHBOARD] Starting dashboard server on {url} ...")
        creationflags = 0
        kwargs = {}
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
            kwargs["creationflags"] = creationflags
        try:
            subprocess.Popen(
                [
                    sys.executable, "-m", "streamlit", "run", str(DASHBOARD_ENTRY),
                    "--server.port", str(port),
                    "--server.headless", "true",
                    "--browser.gatherUsageStats", "false",
                ],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                **kwargs,
            )
        except Exception as exc:  # pragma: no cover - best-effort auto-launch
            print(f"[DASHBOARD] Could not start Streamlit automatically ({exc}).")
            print(f"[DASHBOARD] Run manually: streamlit run {DASHBOARD_ENTRY.name}")
            return url

        deadline = time.time() + wait_seconds
        while time.time() < deadline:
            if _port_open(port):
                break
            time.sleep(0.5)
        else:
            print(
                f"[DASHBOARD] Server hasn't confirmed startup after {wait_seconds:.0f}s; "
                "opening the browser anyway (it usually finishes within a few more seconds)."
            )
    else:
        print(f"[DASHBOARD] Reusing dashboard already running at {url}")

    try:
        webbrowser.open(url)
        print(f"[DASHBOARD] Opened {url} in your browser.")
    except Exception as exc:  # pragma: no cover - best-effort auto-launch
        print(f"[DASHBOARD] Could not open a browser automatically ({exc}). Visit {url}")

    return url
