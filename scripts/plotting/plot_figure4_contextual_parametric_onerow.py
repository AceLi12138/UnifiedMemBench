from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

FIGURE_ROOT = Path(__file__).resolve().parents[1]
if str(FIGURE_ROOT) not in sys.path:
    sys.path.insert(0, str(FIGURE_ROOT))

from shared_plot_utils import get_figure_output_dir
from plot_figure4_radar_and_bar_alt import (
    CTX_COLOR,
    GRID_COLOR,
    OUT_DIR,
    PM_COLOR,
    SPINE_COLOR,
    TASKS,
    collect_plot_data,
    configure_style,
    export_plot_data,
)


OUT_BASENAME = "figure4_contextual_parametric_onerow"
TASK_CODES = [code for code, _ in TASKS]
YMAX = 0.7


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(axis=grid_axis, color=GRID_COLOR, linewidth=0.8, zorder=0)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.18,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def add_overall_value_labels(ax: plt.Axes) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + 0.014,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.8,
        )


def plot_overall(ax: plt.Axes, plot_data: dict) -> None:
    models = [row["display_label"] for row in plot_data["models"]]
    contextual = [row["contextual_overall"] for row in plot_data["models"]]
    parametric = [row["parametric_overall"] for row in plot_data["models"]]
    x = np.arange(len(models))
    width = 0.34

    ax.bar(x - width / 2, contextual, width=width, color=CTX_COLOR, zorder=3)
    ax.bar(x + width / 2, parametric, width=width, color=PM_COLOR, zorder=3)
    ax.set_xticks(x, models)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_ylim(0.0, YMAX)
    ax.set_yticks(np.arange(0.0, YMAX + 0.001, 0.1))
    ax.set_ylabel("QA score")
    style_axis(ax)
    add_overall_value_labels(ax)


def plot_taskwise(ax: plt.Axes, row: dict, *, show_ylabel: bool) -> None:
    contextual = [row["contextual_taskwise"][task_code] for task_code in TASK_CODES]
    parametric = [row["parametric_taskwise"][task_code] for task_code in TASK_CODES]
    x = np.arange(len(TASK_CODES))
    width = 0.34

    ax.bar(x - width / 2, contextual, width=width, color=CTX_COLOR, zorder=3)
    ax.bar(x + width / 2, parametric, width=width, color=PM_COLOR, zorder=3)
    ax.set_xticks(x, TASK_CODES)
    ax.set_ylim(0.0, YMAX)
    ax.set_yticks(np.arange(0.0, YMAX + 0.001, 0.1))
    ax.set_title(row["display_label"], pad=10, fontweight="bold")
    if show_ylabel:
        ax.set_ylabel("QA score")
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)
    style_axis(ax)


def legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=CTX_COLOR, linewidth=8, solid_capstyle="butt", label="Contextual"),
        Line2D([0], [0], color=PM_COLOR, linewidth=8, solid_capstyle="butt", label="Parametric"),
    ]


def save_figure(fig: plt.Figure) -> None:
    output_dir = get_figure_output_dir(OUT_DIR)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"{OUT_BASENAME}.{suffix}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def build_figure(plot_data: dict) -> None:
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(18.4, 4.5),
        gridspec_kw={"width_ratios": [1.35, 1.0, 1.0, 1.0]},
    )

    plot_overall(axes[0], plot_data)
    add_panel_label(axes[0], "A")

    for label, ax, row in zip(["B", "C", "D"], axes[1:], plot_data["models"]):
        plot_taskwise(ax, row, show_ylabel=(label == "B"))
        add_panel_label(ax, label)

    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
        frameon=False,
        fontsize=11,
        handlelength=2.8,
        columnspacing=1.6,
    )
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.18, top=0.82, wspace=0.38)
    save_figure(fig)


def main() -> None:
    configure_style()
    root = Path(__file__).resolve().parents[3]
    plot_data = collect_plot_data(root)
    export_plot_data(plot_data)
    build_figure(plot_data)


if __name__ == "__main__":
    main()
