from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

FIGURE_ROOT = Path(__file__).resolve().parents[1]
if str(FIGURE_ROOT) not in sys.path:
    sys.path.insert(0, str(FIGURE_ROOT))

from shared_plot_utils import apply_publication_rcparams, get_display_label, get_figure_output_dir


OUT_DIR = Path(__file__).resolve().parent

TASKS = [
    ("IE", "Information Extraction"),
    ("MSR", "Multi-session Reasoning"),
    ("ES", "Event Summarization"),
    ("TR", "Temporal Reasoning"),
    ("KU", "Knowledge Updating"),
    ("MA", "Memory Arbitration"),
]
TASK_CODES = [code for code, _ in TASKS]
TASK_NAME_TO_CODE = {name: code for code, name in TASKS}

MODELS = [
    {
        "name": "Qwen3.5-9B",
        "contextual_scores": Path(
            "dialogue_training/project/outputs/eval_context_parallel/"
            "qwen35_full_220k_nothinking/scores.json"
        ),
        "parametric_summary": Path(
            "dialogue_training/project_entity_split_sw_natural_header_qa_upweight24/"
            "outputs/curves_stagewise_qwen35_9b_nothink/unseen_forgetting_summary.json"
        ),
        "parametric_taskwise": Path(
            "dialogue_training/project_entity_split_sw_natural_header_qa_upweight24/"
            "outputs/taskwise_stagewise_unseen_analysis_qwen35_9b_nothink/"
            "taskwise_unseen_diagonal_scores.csv"
        ),
    },
    {
        "name": "GLM-4-9B-1M",
        "contextual_scores": Path(
            "dialogue_training/project/outputs/eval_context_parallel/"
            "glm_vllm_full_8gpu_220k_1024_mimo_judge_gpu90_2026-03-24/scores.json"
        ),
        "parametric_summary": Path(
            "dialogue_training/project_entity_split_sw_natural_header_qa_upweight24/"
            "outputs/curves_stagewise_final/unseen_forgetting_summary.json"
        ),
        "parametric_taskwise": Path(
            "dialogue_training/project_entity_split_sw_natural_header_qa_upweight24/"
            "outputs/taskwise_stagewise_unseen_analysis_glm_qa_upweight24/"
            "taskwise_unseen_diagonal_scores.csv"
        ),
        "stage1_override": Path(
            "dialogue_training/project_entity_split_sw_natural_header_qa_upweight24/"
            "outputs/eval_memory_baseline/"
            "glm4_entitysplit_sw_nh_qa_upweight24_stage_01_fullptqa_e1_lr5e6_4567_"
            "unseen_vllm_mimo_seq/scores.json"
        ),
    },
    {
        "name": "Qwen2.5-7B-1M",
        "contextual_scores": Path(
            "dialogue_training/project/outputs/eval_context_parallel/"
            "qwen_vllm_custom_full_8gpu_220k_1024_mimo_2026-03-26/scores.json"
        ),
        "parametric_summary": Path(
            "dialogue_training/project_entity_split_sw_natural_header_qa_upweight24/"
            "outputs/curves_stagewise_qwen25_7b_1m/unseen_forgetting_summary.json"
        ),
        "parametric_taskwise": Path(
            "dialogue_training/project_entity_split_sw_natural_header_qa_upweight24/"
            "outputs/taskwise_stagewise_unseen_analysis_qwen25_7b_1m/"
            "taskwise_unseen_diagonal_scores.csv"
        ),
    },
]

CTX_COLOR = "#4C78A8"
PM_COLOR = "#F28E2B"
GRID_COLOR = "#D9DEE7"
TEXT_COLOR = "#222222"
SPINE_COLOR = "#9098A1"
RADIAL_MAX = 0.7


def configure_style() -> None:
    apply_publication_rcparams(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.edgecolor": SPINE_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "svg.fonttype": "none",
        }
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_contextual_scores(path: Path) -> tuple[float, dict[str, float]]:
    data = read_json(path)
    taskwise = {
        TASK_NAME_TO_CODE[task_name]: task_data["avg_final_score"]
        for task_name, task_data in data["scores_by_task_type"].items()
    }
    return data["overall_equal_weighted_score"], taskwise


def read_parametric_stagewise(summary_path: Path, stage1_override_path: Path | None = None) -> list[float]:
    data = read_json(summary_path)
    checkpoint_stages = data["checkpoint_stages"]
    dialogue_stages = data["dialogue_stages"]
    matrix = data["matrix"]

    diag_by_stage: dict[int, float] = {}
    for row_idx, checkpoint_stage in enumerate(checkpoint_stages):
        col_idx = dialogue_stages.index(checkpoint_stage)
        value = matrix[row_idx][col_idx]
        if value is None:
            raise ValueError(f"Missing current-stage value for stage {checkpoint_stage} in {summary_path}")
        diag_by_stage[int(checkpoint_stage)] = float(value)

    if stage1_override_path is not None:
        diag_by_stage[1] = float(read_json(stage1_override_path)["avg_final_score"])

    ordered_stages = sorted(diag_by_stage)
    if ordered_stages != list(range(1, 11)):
        raise ValueError(f"Expected stages 1-10, found {ordered_stages} from {summary_path}")
    return [diag_by_stage[stage] for stage in ordered_stages]


def read_parametric_taskwise(path: Path) -> dict[str, float]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return {
        code: float(np.mean([float(row[code]) for row in rows]))
        for code in TASK_CODES
    }


def collect_plot_data(root: Path) -> dict:
    records = []
    for model_cfg in MODELS:
        contextual_overall, contextual_taskwise = read_contextual_scores(root / model_cfg["contextual_scores"])
        parametric_stagewise = read_parametric_stagewise(
            root / model_cfg["parametric_summary"],
            root / model_cfg["stage1_override"] if "stage1_override" in model_cfg else None,
        )
        parametric_taskwise = read_parametric_taskwise(root / model_cfg["parametric_taskwise"])
        records.append(
            {
                "model": model_cfg["name"],
                "display_label": get_display_label(model_cfg["name"]),
                "sources": {
                    "contextual_scores": str(model_cfg["contextual_scores"]),
                    "parametric_summary": str(model_cfg["parametric_summary"]),
                    "parametric_taskwise": str(model_cfg["parametric_taskwise"]),
                    "stage1_override": str(model_cfg["stage1_override"])
                    if "stage1_override" in model_cfg
                    else None,
                },
                "contextual_overall": contextual_overall,
                "parametric_overall": float(np.mean(parametric_stagewise)),
                "parametric_stagewise_current": parametric_stagewise,
                "contextual_taskwise": contextual_taskwise,
                "parametric_taskwise": parametric_taskwise,
            }
        )

    return {
        "figure_title": "Figure 4. Contextual and Parametric Memory Comparison",
        "task_order": [{"code": code, "name": name} for code, name in TASKS],
        "models": records,
    }


def export_plot_data(plot_data: dict) -> None:
    (OUT_DIR / "figure4_plot_data.json").write_text(json.dumps(plot_data, indent=2))

    with (OUT_DIR / "figure4_overall_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "display_label", "contextual_overall", "parametric_overall"],
        )
        writer.writeheader()
        for row in plot_data["models"]:
            writer.writerow(
                {
                    "model": row["model"],
                    "display_label": row["display_label"],
                    "contextual_overall": f"{row['contextual_overall']:.6f}",
                    "parametric_overall": f"{row['parametric_overall']:.6f}",
                }
            )

    with (OUT_DIR / "figure4_taskwise_profiles.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "display_label", "task_code", "task_name", "contextual_score", "parametric_score"],
        )
        writer.writeheader()
        for row in plot_data["models"]:
            for task_code, task_name in TASKS:
                writer.writerow(
                    {
                        "model": row["model"],
                        "display_label": row["display_label"],
                        "task_code": task_code,
                        "task_name": task_name,
                        "contextual_score": f"{row['contextual_taskwise'][task_code]:.6f}",
                        "parametric_score": f"{row['parametric_taskwise'][task_code]:.6f}",
                    }
                )


def style_cartesian_axis(ax: plt.Axes, grid_axis: str) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(axis=grid_axis, color=GRID_COLOR, linewidth=0.8, zorder=0)


def add_bar_value_labels(ax: plt.Axes) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + 0.012,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_overall_panel(ax: plt.Axes, plot_data: dict) -> None:
    models = [row["display_label"] for row in plot_data["models"]]
    contextual = [row["contextual_overall"] for row in plot_data["models"]]
    parametric = [row["parametric_overall"] for row in plot_data["models"]]
    x = np.arange(len(models))
    width = 0.33

    ax.bar(x - width / 2, contextual, width=width, color=CTX_COLOR, label="Contextual", zorder=3)
    ax.bar(x + width / 2, parametric, width=width, color=PM_COLOR, label="Parametric", zorder=3)
    ax.set_xticks(x, models)
    ax.set_ylim(0.0, 0.66)
    ax.set_ylabel("QA score")
    style_cartesian_axis(ax, grid_axis="y")
    add_bar_value_labels(ax)


def radar_angles() -> np.ndarray:
    base = np.linspace(0, 2 * np.pi, len(TASK_CODES), endpoint=False)
    return np.concatenate([base, [base[0]]])


def close_values(values: list[float]) -> np.ndarray:
    return np.asarray(values + [values[0]])


def style_radar_axis(ax: plt.Axes) -> None:
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(radar_angles()[:-1], TASK_CODES)
    ax.set_ylim(0.0, RADIAL_MAX)
    ax.set_yticks([0.2, 0.4, 0.6])
    ax.set_yticklabels(["0.2", "0.4", "0.6"])
    ax.set_rlabel_position(92)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.spines["polar"].set_color(SPINE_COLOR)
    ax.spines["polar"].set_linewidth(0.9)


def plot_radar_panel(ax: plt.Axes, row: dict) -> None:
    contextual = [row["contextual_taskwise"][task_code] for task_code in TASK_CODES]
    parametric = [row["parametric_taskwise"][task_code] for task_code in TASK_CODES]
    angles = radar_angles()

    ax.plot(angles, close_values(contextual), color=CTX_COLOR, linewidth=2.1)
    ax.fill(angles, close_values(contextual), color=CTX_COLOR, alpha=0.14)
    ax.plot(angles, close_values(parametric), color=PM_COLOR, linewidth=2.1, linestyle="--")
    ax.fill(angles, close_values(parametric), color=PM_COLOR, alpha=0.12)
    style_radar_axis(ax)
    ax.set_title(row["display_label"], pad=18, fontweight="bold")


def plot_bar_profile_panel(ax: plt.Axes, row: dict) -> None:
    contextual = [row["contextual_taskwise"][task_code] for task_code in TASK_CODES]
    parametric = [row["parametric_taskwise"][task_code] for task_code in TASK_CODES]
    x = np.arange(len(TASK_CODES))
    width = 0.35

    ax.bar(x - width / 2, contextual, width=width, color=CTX_COLOR, zorder=3)
    ax.bar(x + width / 2, parametric, width=width, color=PM_COLOR, zorder=3)
    ax.set_xticks(x, TASK_CODES)
    ax.set_ylim(0.0, RADIAL_MAX)
    ax.set_title(row["display_label"], pad=10, fontweight="bold")
    style_cartesian_axis(ax, grid_axis="y")


def figure_legend() -> list[Line2D]:
    return [
        Line2D([0], [0], color=CTX_COLOR, linewidth=2.4, label="Contextual"),
        Line2D([0], [0], color=PM_COLOR, linewidth=2.4, linestyle="--", label="Parametric"),
    ]


def save_figure(fig: plt.Figure, basename: str) -> None:
    output_dir = get_figure_output_dir(OUT_DIR)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"{basename}.{suffix}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def build_radar_figure(plot_data: dict) -> None:
    fig = plt.figure(figsize=(12.8, 8.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.95, 1.3], hspace=0.42, wspace=0.38)

    overall_ax = fig.add_subplot(gs[0, :])
    plot_overall_panel(overall_ax, plot_data)

    radar_axes = [fig.add_subplot(gs[1, idx], projection="polar") for idx in range(3)]
    for ax, row in zip(radar_axes, plot_data["models"]):
        plot_radar_panel(ax, row)

    fig.text(0.015, 0.935, "A", fontsize=15, fontweight="bold")
    fig.text(0.015, 0.47, "B", fontsize=15, fontweight="bold")
    fig.legend(
        handles=figure_legend(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
    )
    save_figure(fig, "figure4_radar")


def build_bar_alt_figure(plot_data: dict) -> None:
    fig = plt.figure(figsize=(13.2, 8.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.95, 1.25], hspace=0.4, wspace=0.28)

    overall_ax = fig.add_subplot(gs[0, :])
    plot_overall_panel(overall_ax, plot_data)

    task_axes = [fig.add_subplot(gs[1, idx]) for idx in range(3)]
    for ax, row in zip(task_axes, plot_data["models"]):
        plot_bar_profile_panel(ax, row)

    task_axes[0].set_ylabel("QA score")

    fig.text(0.015, 0.935, "A", fontsize=15, fontweight="bold")
    fig.text(0.015, 0.47, "B", fontsize=15, fontweight="bold")
    fig.legend(
        handles=figure_legend(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
    )
    save_figure(fig, "figure4_bar_alt")


def main() -> None:
    configure_style()
    root = Path(__file__).resolve().parents[3]
    plot_data = collect_plot_data(root)
    export_plot_data(plot_data)
    build_radar_figure(plot_data)
    build_bar_alt_figure(plot_data)


if __name__ == "__main__":
    main()
