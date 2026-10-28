"""Data figures for the paper, drawn only from summary.json. PDF for LaTeX, PNG for the Word file.

Palette: three slots of a categorical palette checked for colour vision deficiency and
contrast (blue, orange, aqua). Every mark also carries a direct value label, so colour
never carries meaning alone.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logging.getLogger("fontTools").setLevel(logging.WARNING)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, INK, MUTED = "#8a8a85", "#1a1a19", "#5f5e58"
COLUMN_WIDTH_IN = 3.45

SHORT_NAMES = {
    "logprob": "LogProb", "ptrue": "P(True)", "verbal": "Verbal", "semantic_entropy": "SE",
    "selfcheck_nli": "SelfCheck", "cross_family": "Cross", "metaqa": "MetaQA", "cps": "CPS",
    "f_counter": "Counter", "dfr": "DFR", "dfr_cross": "DFR+Cross",
}


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": GREY,
        "axes.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.labelcolor": INK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def build_figures(summary: dict, model_keys: list[str], out_dir: Path) -> list[Path]:
    _style()
    out_dir.mkdir(parents=True, exist_ok=True)
    models = [(k, summary["models"][k]) for k in model_keys if k in summary["models"]]
    if not models:
        return []
    written = []
    written += _save(_flip_figure(models), out_dir / "fig_flip")
    written += _save(_cost_figure(models), out_dir / "fig_cost")
    return written


def _save(figure, stem: Path) -> list[Path]:
    paths = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    figure.savefig(paths[0], bbox_inches="tight")
    figure.savefig(paths[1], bbox_inches="tight", dpi=300)
    plt.close(figure)
    return paths


def _finite(value) -> bool:
    return value is not None and value == value  # NaN is the only value not equal to itself


def _flip_figure(models):
    figure, axes = plt.subplots(1, len(models), figsize=(COLUMN_WIDTH_IN, 1.75), sharey=True, squeeze=False)
    width = 0.36
    for axis, (_, model) in zip(axes[0], models):
        for offset, column, color, label in ((-width / 2, "f_counter", BLUE, "Counterclaim"),
                                             (width / 2, "f_doubt", ORANGE, "Doubt")):
            for x, group in enumerate(("CW", "CR")):
                entry = model["flip_rates"].get(group, {}).get(column)
                if not entry or not _finite(entry.get("value")):
                    continue
                value = entry["value"]
                axis.bar(x + offset, value, width - 0.04, color=color, label=label if x == 0 else None,
                         zorder=2)
                top = value
                if _finite(entry.get("low")) and _finite(entry.get("high")):
                    axis.errorbar(x + offset, value, yerr=[[value - entry["low"]], [entry["high"] - value]],
                                  color=INK, linewidth=0.7, capsize=1.5, zorder=3)
                    top = entry["high"]
                axis.text(x + offset, top + 0.03, f"{value:.2f}", ha="center", va="bottom",
                          fontsize=6, color=INK)
        axis.set_xticks([0, 1], ["Consistently\nwrong", "Consistently\nright"])
        axis.set_title(model["display_name"], color=INK)
        axis.set_ylim(0, 1.12)
        axis.grid(axis="y", color="#e6e5e0", linewidth=0.5, zorder=0)
    axes[0][0].set_ylabel("Share of replies that flip")
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
    figure.tight_layout()
    return figure


def _cost_figure(models):
    figure, axes = plt.subplots(1, len(models), figsize=(COLUMN_WIDTH_IN, 1.9), sharey=True, squeeze=False)
    for axis, (_, model) in zip(axes[0], models):
        plotted = 0
        for key, method in model["methods"].items():
            value = method["auroc_matched"]["value"]
            if not _finite(value):
                continue
            plotted += 1
            role = method.get("role", "ours" if method["is_ours"] else "baseline")
            color = {"ours": BLUE, "variant": ORANGE}.get(role, AQUA if method["access"] == "text" else GREY)
            marker = {"ours": "D", "variant": "^"}.get(role, "o" if method["access"] == "text" else "s")
            axis.scatter(method["calls"], value, s=22, color=color, marker=marker, edgecolors="white",
                         linewidths=0.6, zorder=3)
            axis.annotate(SHORT_NAMES.get(key, key), (method["calls"], value), xytext=(3, 2),
                          textcoords="offset points", fontsize=5.5, color=INK)
        axis.axhline(0.5, color=GREY, linewidth=0.6, linestyle=(0, (2, 2)), zorder=1)
        if not plotted:
            axis.text(0.5, 0.62, "Too few items", transform=axis.transAxes, ha="center", fontsize=6.5,
                      color=MUTED)
        axis.set_xscale("log")
        axis.set_xlim(0.8, 20)
        axis.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
        axis.set_xlabel("Generations per question")
        axis.set_title(model["display_name"], color=INK)
        axis.grid(color="#e6e5e0", linewidth=0.5, zorder=0)
    axes[0][0].set_ylabel("AUROC, CW vs CR")
    legend = [plt.Line2D([], [], marker="D", linestyle="", color=BLUE, label="Ours"),
              plt.Line2D([], [], marker="^", linestyle="", color=ORANGE, label="Other pressure scores"),
              plt.Line2D([], [], marker="o", linestyle="", color=AQUA, label="Text only baseline"),
              plt.Line2D([], [], marker="s", linestyle="", color=GREY, label="Needs logits")]
    figure.legend(handles=legend, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.16))
    figure.tight_layout()
    return figure
