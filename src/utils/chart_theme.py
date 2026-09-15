"""Shared Altair theme and color helpers for all dashboard pages.

Keeping this in one place means every chart in the app uses the same
categorical color-per-run mapping, the same fonts/gridlines, and the same
"one series = one color, always" rule instead of colors being reassigned
whenever the set of selected runs changes.
"""

from __future__ import annotations

import altair as alt

# Okabe–Ito qualitative palette: colorblind-safe, used in a fixed order so a
# given run/trial/representation always maps to the same color across pages
# (identity, not rank) instead of being recolored when the selection changes.
CATEGORICAL_PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # pink
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

# Sequential single-hue ramp for magnitude-only encodings (e.g. gap heatmaps).
SEQUENTIAL_SCHEME = "blues"


def register_theme() -> None:
    """Register and enable a consistent Altair theme (call once at app start)."""

    def _theme():
        return {
            "config": {
                "background": "transparent",
                "view": {"strokeWidth": 0},
                "axis": {
                    "grid": True,
                    "gridColor": "#e5e7eb",
                    "gridOpacity": 0.6,
                    "domainColor": "#9ca3af",
                    "tickColor": "#9ca3af",
                    "labelColor": "#4b5563",
                    "titleColor": "#374151",
                    "titleFontSize": 12,
                    "labelFontSize": 11,
                },
                "legend": {
                    "labelColor": "#374151",
                    "titleColor": "#374151",
                    "labelFontSize": 11,
                    "titleFontSize": 12,
                },
                "line": {"strokeWidth": 2.5},
                "point": {"size": 40},
                "title": {"fontSize": 14, "anchor": "start", "color": "#111827"},
                "range": {"category": CATEGORICAL_PALETTE},
            }
        }

    alt.themes.register("bopo_dashboard", _theme)
    alt.themes.enable("bopo_dashboard")


def color_by(
    field: str, domain: list[str], title: str | None = None, show_legend: bool = True
) -> alt.Color:
    """A categorical color encoding pinned to a fixed domain -> fixed palette mapping.

    Passing an explicit `domain` (rather than letting Altair infer it from
    whatever subset of runs happens to be selected) is what keeps a given
    run's color stable as the user adds/removes other runs from the
    selection.
    """
    return alt.Color(
        f"{field}:N",
        title=title or field,
        scale=alt.Scale(domain=domain, range=CATEGORICAL_PALETTE),
        legend=alt.Legend(symbolType="stroke") if show_legend else None,
    )
