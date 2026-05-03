from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "report"

MAIN_OUTPUT_STEM = "figure6_retention_main"
APP_FINAL_RETENTION_STEM = "figure6_appendix_taskwise_final_retention_heatmap"
APP_FORGETTING_STEM = "figure6_appendix_taskwise_forgetting_heatmap"
APP_BY_LAG_STEM = "figure6_appendix_taskwise_retention_by_lag"
MAIN_DATA_CSV = ROOT / "figure6_main_plot_data.csv"
APPENDIX_DATA_CSV = ROOT / "figure6_appendix_plot_data.csv"

MODEL_ORDER = ["GLM-4-9B-1M", "Qwen3.5-9B", "Qwen2.5-7B-1M"]
TASK_ORDER = ["IE", "MSR", "ES", "TR", "KU", "MA"]
MODEL_COLORS = {
    "GLM-4-9B-1M": "#1f77b4",
    "Qwen3.5-9B": "#ff7f0e",
    "Qwen2.5-7B-1M": "#2ca02c",
}
MODEL_MARKERS = {
    "GLM-4-9B-1M": "o",
    "Qwen3.5-9B": "s",
    "Qwen2.5-7B-1M": "^",
}
BAR_COLORS = {
    "current": "#4c78a8",
    "final_old": "#f58518",
}
EXPECTED_SUMMARY = {
    "GLM-4-9B-1M": {"current_stage_mean": 0.1779, "final_old_mean": 0.0966},
    "Qwen3.5-9B": {"current_stage_mean": 0.1660, "final_old_mean": 0.1131},
    "Qwen2.5-7B-1M": {"current_stage_mean": 0.1276, "final_old_mean": 0.0985},
}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _to_int(value: Any) -> int:
    return int(str(value))


def _to_float(value: Any) -> float:
    return float(str(value))


def _float_or_nan(value: Any) -> float:
    text = str(value).strip()
    if not text:
        return float("nan")
    return float(text)


def _write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in (".pdf", ".svg", ".png"):
        out_path = ROOT / f"{stem}{suffix}"
        fig.savefig(
            out_path,
            dpi=220 if suffix == ".png" else None,
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(fig)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
        }
    )


def load_main_protocol_data() -> Dict[str, Any]:
    overall_rows = [
        row
        for row in _read_csv(REPORT_DIR / "retention_overall_matrix_long.csv")
        if row["split_type"] == "held_out_character"
    ]
    old_curve_rows = [
        row
        for row in _read_csv(REPORT_DIR / "retention_old_stage_curve.csv")
        if row["split_type"] == "held_out_character" and row["old_stage_mean_score"]
    ]
    current_rows = [
        row
        for row in _read_csv(REPORT_DIR / "retention_current_stage_curve.csv")
        if row["split_type"] == "held_out_character" and row["current_stage_score"]
    ]
    final_old_rows = [
        row
        for row in _read_csv(REPORT_DIR / "retention_final_old_stage_summary.csv")
        if row["split_type"] == "held_out_character"
    ]
    taskwise_final_rows = [
        row
        for row in _read_csv(REPORT_DIR / "retention_taskwise_final_summary.csv")
        if row["split_type"] == "held_out_character"
    ]
    taskwise_forgetting_rows = [
        row
        for row in _read_csv(REPORT_DIR / "retention_taskwise_forgetting_summary.csv")
        if row["split_type"] == "held_out_character"
    ]
    taskwise_lag_rows = [
        row
        for row in _read_csv(REPORT_DIR / "retention_taskwise_by_lag.csv")
        if row["split_type"] == "held_out_character" and row["mean_score"]
    ]

    matrices: Dict[str, np.ndarray] = {}
    for model_name in MODEL_ORDER:
        matrix = np.full((10, 10), np.nan, dtype=float)
        for row in overall_rows:
            if row["model_name"] != model_name:
                continue
            c = _to_int(row["checkpoint_stage"]) - 1
            e = _to_int(row["eval_stage"]) - 1
            matrix[c, e] = _to_float(row["overall_score"])
        matrices[model_name] = matrix

    old_curve: Dict[str, Dict[str, List[float]]] = {}
    for model_name in MODEL_ORDER:
        xs: List[int] = []
        ys: List[float] = []
        for row in old_curve_rows:
            if row["model_name"] != model_name:
                continue
            stage = _to_int(row["checkpoint_stage"])
            if stage < 2:
                continue
            xs.append(stage)
            ys.append(_to_float(row["old_stage_mean_score"]))
        old_curve[model_name] = {"x": xs, "y": ys}

    current_means: Dict[str, float] = {}
    by_model_current: Dict[str, List[float]] = defaultdict(list)
    for row in current_rows:
        by_model_current[row["model_name"]].append(_to_float(row["current_stage_score"]))
    for model_name in MODEL_ORDER:
        vals = by_model_current[model_name]
        current_means[model_name] = sum(vals) / len(vals)

    final_old_means: Dict[str, float] = {}
    for row in final_old_rows:
        final_old_means[row["model_name"]] = _to_float(row["final_old_stage_mean_score"])

    for model_name, expected in EXPECTED_SUMMARY.items():
        if abs(current_means[model_name] - expected["current_stage_mean"]) > 1e-3:
            raise ValueError(
                f"Current-stage mean mismatch for {model_name}: "
                f"{current_means[model_name]:.4f} vs expected {expected['current_stage_mean']:.4f}"
            )
        if abs(final_old_means[model_name] - expected["final_old_mean"]) > 1e-3:
            raise ValueError(
                f"Final old-stage mean mismatch for {model_name}: "
                f"{final_old_means[model_name]:.4f} vs expected {expected['final_old_mean']:.4f}"
            )

    return {
        "overall_rows": overall_rows,
        "matrices": matrices,
        "old_curve": old_curve,
        "current_means": current_means,
        "final_old_means": final_old_means,
        "taskwise_final_rows": taskwise_final_rows,
        "taskwise_forgetting_rows": taskwise_forgetting_rows,
        "taskwise_lag_rows": taskwise_lag_rows,
    }


def write_plot_data_csvs(data: Dict[str, Any]) -> None:
    main_rows: List[Dict[str, Any]] = []
    for row in data["overall_rows"]:
        main_rows.append(
            {
                "panel": "A",
                "model_name": row["model_name"],
                "metric_name": "overall_equal_weighted_score",
                "checkpoint_stage": row["checkpoint_stage"],
                "eval_stage": row["eval_stage"],
                "stage_relation": row["stage_relation"],
                "value": row["overall_score"],
            }
        )
    for model_name in MODEL_ORDER:
        xs = data["old_curve"][model_name]["x"]
        ys = data["old_curve"][model_name]["y"]
        for checkpoint_stage, value in zip(xs, ys):
            main_rows.append(
                {
                    "panel": "B",
                    "model_name": model_name,
                    "metric_name": "old_stage_mean_score",
                    "checkpoint_stage": checkpoint_stage,
                    "eval_stage": "",
                    "stage_relation": "old",
                    "value": f"{value:.6f}",
                }
            )
    for model_name in MODEL_ORDER:
        main_rows.extend(
            [
                {
                    "panel": "C",
                    "model_name": model_name,
                    "metric_name": "mean_current_stage_diagonal_score",
                    "checkpoint_stage": "",
                    "eval_stage": "",
                    "stage_relation": "current",
                    "value": f"{data['current_means'][model_name]:.6f}",
                },
                {
                    "panel": "C",
                    "model_name": model_name,
                    "metric_name": "final_old_stage_retention_mean",
                    "checkpoint_stage": "",
                    "eval_stage": "",
                    "stage_relation": "old",
                    "value": f"{data['final_old_means'][model_name]:.6f}",
                },
            ]
        )
    _write_csv(
        MAIN_DATA_CSV,
        ["panel", "model_name", "metric_name", "checkpoint_stage", "eval_stage", "stage_relation", "value"],
        main_rows,
    )

    appendix_rows: List[Dict[str, Any]] = []
    for row in data["taskwise_final_rows"]:
        appendix_rows.append(
            {
                "panel": "appendix_final_retention_heatmap",
                "model_name": row["model_name"],
                "task_type": row["task_type"],
                "checkpoint_stage": row["final_checkpoint_stage"],
                "lag": "",
                "metric_name": "final_old_stage_task_mean_score",
                "value": row["final_old_stage_task_mean_score"],
            }
        )
    for row in data["taskwise_forgetting_rows"]:
        appendix_rows.extend(
            [
                {
                    "panel": "appendix_forgetting_heatmap",
                    "model_name": row["model_name"],
                    "task_type": row["task_type"],
                    "checkpoint_stage": "",
                    "lag": "",
                    "metric_name": "mean_forgetting_abs",
                    "value": row["mean_forgetting_abs"],
                },
                {
                    "panel": "appendix_forgetting_heatmap",
                    "model_name": row["model_name"],
                    "task_type": row["task_type"],
                    "checkpoint_stage": "",
                    "lag": "",
                    "metric_name": "mean_normalized_retention",
                    "value": row["mean_normalized_retention"],
                },
            ]
        )
    for row in data["taskwise_lag_rows"]:
        appendix_rows.append(
            {
                "panel": "appendix_retention_by_lag",
                "model_name": row["model_name"],
                "task_type": row["task_type"],
                "checkpoint_stage": "",
                "lag": row["lag"],
                "metric_name": "mean_score",
                "value": row["mean_score"],
            }
        )
    _write_csv(
        APPENDIX_DATA_CSV,
        ["panel", "model_name", "task_type", "checkpoint_stage", "lag", "metric_name", "value"],
        appendix_rows,
    )


def _annotate_panel(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False,
    )


def plot_main_figure(data: Dict[str, Any]) -> None:
    cmap = plt.cm.YlGnBu.copy()
    cmap.set_bad("#efefef")
    vmin = 0.0
    vmax = 0.25

    fig = plt.figure(figsize=(7.4, 5.2))
    gs = fig.add_gridspec(
        2,
        7,
        width_ratios=[1, 1, 1, 1, 1, 1, 0.12],
        height_ratios=[1.08, 0.92],
        wspace=0.35,
        hspace=0.55,
    )

    heatmap_axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
    ]
    cax = fig.add_subplot(gs[0, 6])
    ax_b = fig.add_subplot(gs[1, 0:4])
    ax_c = fig.add_subplot(gs[1, 4:7])

    im = None
    for idx, (ax, model_name) in enumerate(zip(heatmap_axes, MODEL_ORDER)):
        matrix = np.ma.masked_invalid(data["matrices"][model_name])
        im = ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(model_name, pad=5)
        ax.set_xticks(range(10))
        ax.set_xticklabels([f"S{i}" for i in range(1, 11)])
        ax.set_yticks(range(10))
        if idx == 0:
            ax.set_yticklabels([f"C{i}" for i in range(1, 11)])
            ax.set_ylabel("Checkpoint stage")
            _annotate_panel(ax, "A")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Eval stage")
        ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.7, alpha=0.95)
        ax.tick_params(which="minor", bottom=False, left=False)

    colorbar = fig.colorbar(im, cax=cax)
    colorbar.set_label("Overall QA score")
    colorbar.set_ticks(np.arange(0.0, 0.26, 0.05))

    ax_b.grid(axis="y")
    ax_b.set_title("Old-stage retention over updates", pad=4)
    ax_b.set_xlabel("Checkpoint stage")
    ax_b.set_ylabel("Mean old-stage score")
    ax_b.set_xticks(range(2, 11))
    ax_b.set_ylim(0.0, 0.18)
    _annotate_panel(ax_b, "B")
    for model_name in MODEL_ORDER:
        ax_b.plot(
            data["old_curve"][model_name]["x"],
            data["old_curve"][model_name]["y"],
            color=MODEL_COLORS[model_name],
            marker=MODEL_MARKERS[model_name],
            linewidth=1.6,
            markersize=4.2,
            label=model_name,
        )
    ax_b.legend(loc="upper right", frameon=False)

    x = np.arange(len(MODEL_ORDER))
    width = 0.34
    current_vals = [data["current_means"][model_name] for model_name in MODEL_ORDER]
    final_vals = [data["final_old_means"][model_name] for model_name in MODEL_ORDER]
    bars_current = ax_c.bar(
        x - width / 2,
        current_vals,
        width=width,
        color=BAR_COLORS["current"],
        label="Mean current-stage diagonal",
    )
    bars_final = ax_c.bar(
        x + width / 2,
        final_vals,
        width=width,
        color=BAR_COLORS["final_old"],
        label="Final old-stage mean",
    )
    ax_c.grid(axis="y")
    ax_c.set_title("Write-in vs. retention", pad=4)
    ax_c.set_ylabel("Overall QA score")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(MODEL_ORDER, rotation=12, ha="right")
    ax_c.set_ylim(0.0, 0.22)
    ax_c.legend(loc="upper right", frameon=False)
    _annotate_panel(ax_c, "C")
    for bars in (bars_current, bars_final):
        for bar in bars:
            height = bar.get_height()
            ax_c.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.004,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    fig.subplots_adjust(left=0.08, right=0.96, top=0.95, bottom=0.12)
    _save_figure(fig, MAIN_OUTPUT_STEM)


def _task_matrix(rows: List[Dict[str, str]], value_key: str) -> np.ndarray:
    lookup = {(row["model_name"], row["task_type"]): _float_or_nan(row[value_key]) for row in rows}
    matrix = np.full((len(MODEL_ORDER), len(TASK_ORDER)), np.nan, dtype=float)
    for i, model_name in enumerate(MODEL_ORDER):
        for j, task_code in enumerate(TASK_ORDER):
            matrix[i, j] = lookup[(model_name, task_code)]
    return matrix


def plot_appendix_heatmap(
    rows: List[Dict[str, str]],
    value_key: str,
    stem: str,
    colorbar_label: str,
    cmap_name: str,
    vmax: float,
) -> None:
    matrix = _task_matrix(rows, value_key)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#efefef")
    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax)
    ax.set_xticks(range(len(TASK_ORDER)))
    ax.set_xticklabels(TASK_ORDER)
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_ORDER)
    ax.set_xlabel("Task type")
    ax.set_ylabel("Model")
    ax.set_xticks(np.arange(-0.5, len(TASK_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MODEL_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7, alpha=0.95)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.95)
    cbar.set_label(colorbar_label)
    fig.subplots_adjust(left=0.20, right=0.95, bottom=0.22, top=0.96)
    _save_figure(fig, stem)


def plot_appendix_retention_by_lag(rows: List[Dict[str, str]]) -> None:
    by_task_model: Dict[tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_task_model[(row["task_type"], row["model_name"])].append(row)

    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.6), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, task_code in zip(axes, TASK_ORDER):
        ax.grid(axis="y")
        ax.set_title(task_code, pad=3)
        ax.set_ylim(0.0, 0.30)
        for model_name in MODEL_ORDER:
            task_rows = sorted(
                by_task_model.get((task_code, model_name), []),
                key=lambda row: _to_int(row["lag"]),
            )
            xs = [_to_int(row["lag"]) for row in task_rows]
            ys = [_to_float(row["mean_score"]) for row in task_rows]
            ax.plot(
                xs,
                ys,
                color=MODEL_COLORS[model_name],
                marker=MODEL_MARKERS[model_name],
                linewidth=1.4,
                markersize=3.6,
                label=model_name,
            )
    axes[0].set_ylabel("Mean old-stage score")
    axes[3].set_ylabel("Mean old-stage score")
    for ax in axes[3:]:
        ax.set_xlabel("Lag")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.86, wspace=0.28, hspace=0.40)
    _save_figure(fig, APP_BY_LAG_STEM)


def main() -> None:
    _style()
    data = load_main_protocol_data()
    write_plot_data_csvs(data)
    plot_main_figure(data)
    plot_appendix_heatmap(
        data["taskwise_final_rows"],
        value_key="final_old_stage_task_mean_score",
        stem=APP_FINAL_RETENTION_STEM,
        colorbar_label="Final old-stage task score",
        cmap_name="YlGnBu",
        vmax=0.30,
    )
    plot_appendix_heatmap(
        data["taskwise_forgetting_rows"],
        value_key="mean_forgetting_abs",
        stem=APP_FORGETTING_STEM,
        colorbar_label="Mean forgetting amount",
        cmap_name="OrRd",
        vmax=0.22,
    )
    plot_appendix_retention_by_lag(data["taskwise_lag_rows"])


if __name__ == "__main__":
    main()
