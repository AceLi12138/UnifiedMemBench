from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parent
OUT_BASENAME = "figure4_contextual_parametric_results"


CONTEXTUAL_OVERALL = {
    "Gemma-4-31B": 0.688216,
    "Qwen3.5-27B": 0.667730,
    "Gemma-4-26B-A4B": 0.655618,
    "Qwen3.5-35B-A3B": 0.617014,
    "Qwen3.5-9B": 0.574283,
    "GLM-4-9B-1M": 0.430659,
    "Llama4-Scout": 0.342886,
    "Qwen2.5-7B-1M": 0.279000,
    "GLM-4.7-Flash": 0.256295,
    "Hunyuan-A13B": 0.148167,
}

OVERLAP_CONTEXTUAL = {
    "GLM-4-9B-1M": 0.430659,
    "Qwen3.5-9B": 0.574283,
    "Qwen2.5-7B-1M": 0.279000,
}

OVERLAP_PARAMETRIC = {
    "GLM-4-9B-1M": 0.182831,
    "Qwen3.5-9B": 0.162955,
    "Qwen2.5-7B-1M": 0.132602,
}

PARAMETRIC_STAGEWISE = {
    "GLM-4-9B-1M": [
        0.180085,
        0.209402,
        0.189427,
        0.192478,
        0.200873,
        0.187234,
        0.193416,
        0.146930,
        0.167382,
        0.161088,
    ],
    "Qwen3.5-9B": [
        0.154661,
        0.185897,
        0.156388,
        0.126106,
        0.144105,
        0.182979,
        0.224280,
        0.177632,
        0.137339,
        0.140167,
    ],
    "Qwen2.5-7B-1M": [
        0.103814,
        0.149573,
        0.116740,
        0.152655,
        0.126638,
        0.129787,
        0.158436,
        0.133772,
        0.141631,
        0.112971,
    ],
}

TASK_ORDER = ["IE", "MSR", "ES", "TR", "KU", "MA"]

CONTEXTUAL_TASKWISE = {
    "GLM-4-9B-1M": {
        "IE": 0.595320,
        "MSR": 0.528826,
        "ES": 0.161360,
        "TR": 0.634309,
        "KU": 0.520097,
        "MA": 0.144042,
    },
    "Qwen3.5-9B": {
        "IE": 0.601170,
        "MSR": 0.680679,
        "ES": 0.440841,
        "TR": 0.526451,
        "KU": 0.583435,
        "MA": 0.613122,
    },
    "Qwen2.5-7B-1M": {
        "IE": 0.483482,
        "MSR": 0.244681,
        "ES": 0.149371,
        "TR": 0.572162,
        "KU": 0.163216,
        "MA": 0.061086,
    },
}

PARAMETRIC_TASKWISE = {
    "GLM-4-9B-1M": {
        "IE": 0.410497,
        "MSR": 0.153725,
        "ES": 0.015105,
        "TR": 0.088904,
        "KU": 0.290290,
        "MA": 0.108761,
    },
    "Qwen3.5-9B": {
        "IE": 0.384741,
        "MSR": 0.132643,
        "ES": 0.010536,
        "TR": 0.030202,
        "KU": 0.226615,
        "MA": 0.211517,
    },
    "Qwen2.5-7B-1M": {
        "IE": 0.330985,
        "MSR": 0.109206,
        "ES": 0.013742,
        "TR": 0.012498,
        "KU": 0.199872,
        "MA": 0.099316,
    },
}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def build_panel_d_means() -> tuple[list[float], list[float]]:
    contextual_means = []
    parametric_means = []
    for task in TASK_ORDER:
        contextual_means.append(_mean([row[task] for row in CONTEXTUAL_TASKWISE.values()]))
        parametric_means.append(_mean([row[task] for row in PARAMETRIC_TASKWISE.values()]))
    return contextual_means, parametric_means


def apply_axes_style(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8c8c8c")
    ax.spines["bottom"].set_color("#8c8c8c")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", colors="#333333", length=3, width=0.8)
    if grid_axis:
        ax.grid(
            axis=grid_axis,
            color="#d8dce2",
            linewidth=0.6,
            linestyle="-",
            alpha=0.8,
            zorder=0,
        )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="top",
    )


def plot_panel_a(ax: plt.Axes) -> None:
    items = sorted(CONTEXTUAL_OVERALL.items(), key=lambda kv: kv[1], reverse=True)
    models = [k for k, _ in items]
    scores = [v for _, v in items]
    y = np.arange(len(models))
    bar_color = "#5B84B1"

    ax.barh(y, scores, color=bar_color, edgecolor="none", height=0.64, zorder=3)
    ax.set_yticks(y, labels=models)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 0.76)
    ax.set_xlabel("Task-balanced QA score")
    ax.set_title("Contextual overall ranking", pad=6)
    apply_axes_style(ax, grid_axis="x")

    for idx, score in enumerate(scores):
        ax.text(
            score + 0.01,
            idx,
            f"{score:.3f}",
            va="center",
            ha="left",
            fontsize=7.5,
            color="#2b2b2b",
        )


def plot_panel_b(ax: plt.Axes) -> None:
    order = ["Qwen3.5-9B", "GLM-4-9B-1M", "Qwen2.5-7B-1M"]
    contextual = [OVERLAP_CONTEXTUAL[m] for m in order]
    parametric = [OVERLAP_PARAMETRIC[m] for m in order]
    x = np.arange(len(order))
    width = 0.32

    contextual_color = "#4C78A8"
    parametric_color = "#D9A441"

    ax.bar(x - width / 2, contextual, width=width, color=contextual_color, edgecolor="none", label="Contextual", zorder=3)
    ax.bar(x + width / 2, parametric, width=width, color=parametric_color, edgecolor="none", label="Parametric", zorder=3)
    ax.set_xticks(x, labels=order)
    ax.set_ylabel("QA score")
    ax.set_ylim(0.0, 0.64)
    ax.set_title("Overlap models: ranking mismatch", pad=6)
    apply_axes_style(ax, grid_axis="y")
    ax.legend(frameon=False, loc="upper right", handlelength=1.2)


def plot_panel_c(ax: plt.Axes) -> None:
    stages = np.arange(1, 11)
    colors = {
        "GLM-4-9B-1M": "#4C78A8",
        "Qwen3.5-9B": "#F58518",
        "Qwen2.5-7B-1M": "#54A24B",
    }
    markers = {
        "GLM-4-9B-1M": "o",
        "Qwen3.5-9B": "s",
        "Qwen2.5-7B-1M": "^",
    }

    for model, values in PARAMETRIC_STAGEWISE.items():
        ax.plot(
            stages,
            values,
            label=model,
            color=colors[model],
            marker=markers[model],
            linewidth=1.7,
            markersize=4.5,
            zorder=3,
        )

    ax.set_xlim(1, 10)
    ax.set_xticks(stages)
    ax.set_ylim(0.08, 0.24)
    ax.set_xlabel("Stage")
    ax.set_ylabel("Held-out current-stage QA score")
    ax.set_title("Parametric current-stage recall", pad=6)
    apply_axes_style(ax, grid_axis="y")
    ax.legend(frameon=False, loc="upper right", fontsize=7.0, handlelength=2.0)


def plot_panel_d(ax: plt.Axes) -> None:
    contextual_means, parametric_means = build_panel_d_means()
    x = np.arange(len(TASK_ORDER))
    width = 0.36
    contextual_color = "#4C78A8"
    parametric_color = "#D9A441"

    ax.bar(
        x - width / 2,
        contextual_means,
        width=width,
        color=contextual_color,
        edgecolor="none",
        label="Contextual",
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        parametric_means,
        width=width,
        color=parametric_color,
        edgecolor="none",
        label="Parametric",
        zorder=3,
    )

    ax.set_xticks(x, labels=TASK_ORDER)
    ax.set_ylabel("Mean QA score")
    ax.set_ylim(0.0, 0.66)
    ax.set_title("Task-wise contextual vs parametric", pad=6)
    apply_axes_style(ax, grid_axis="y")
    ax.legend(frameon=False, loc="upper right", handlelength=1.2)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.3,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.75))
    fig.subplots_adjust(left=0.11, right=0.985, top=0.93, bottom=0.16, wspace=0.35, hspace=0.42)

    plot_panel_a(axes[0, 0])
    plot_panel_b(axes[0, 1])
    plot_panel_c(axes[1, 0])
    plot_panel_d(axes[1, 1])

    for label, ax in zip(["A", "B", "C", "D"], axes.flat):
        add_panel_label(ax, label)

    fig.text(
        0.5,
        0.04,
        "IE: Information Extraction   MSR: Multi-session Reasoning   ES: Event Summarization   "
        "TR: Temporal Reasoning   KU: Knowledge Updating   MA: Memory Arbitration",
        ha="center",
        va="center",
        fontsize=7.0,
        color="#444444",
    )

    pdf_path = OUT_DIR / f"{OUT_BASENAME}.pdf"
    svg_path = OUT_DIR / f"{OUT_BASENAME}.svg"
    png_path = OUT_DIR / f"{OUT_BASENAME}.png"

    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=400)
    plt.close(fig)

    print(pdf_path)
    print(svg_path)
    print(png_path)


if __name__ == "__main__":
    main()
