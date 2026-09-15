"""Optuna trial comparison, embedded as a page of the main dashboard.

Thin wrapper around optuna_dashboard.py so the same trial-comparison logic
is available both standalone (`streamlit run optuna_dashboard.py`) and as a
page of the combined app (`streamlit run dashboard.py`).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna_dashboard

optuna_dashboard.main()
