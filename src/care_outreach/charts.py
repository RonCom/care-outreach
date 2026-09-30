"""Shared figure style (PNG, light theme)."""
from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"static": "#52514e", "ts": "#eb6834", "ts_discount": "#2a78d6", "oracle": "#0b0b0b",
          "adaptive": "#2a78d6", "ink2": INK2, "grid_dark": "#b9b7b1",
          "logistic": "#52514e", "hgb": "#eb6834", "seqnet": "#2a78d6"}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.axisbelow": True,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    })


def save(fig, cfg, name):
    fig.savefig(cfg.reports / "figures" / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info("figure %s", name)
