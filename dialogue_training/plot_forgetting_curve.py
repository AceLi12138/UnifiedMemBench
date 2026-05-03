from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from dialogue_training.core import build_forgetting_summary  # type: ignore
else:
    from dialogue_training.core import build_forgetting_summary


def _write_csv(path: Path, header: List[str], rows: Iterable[Iterable[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def _matrix_to_rows(summary: dict) -> List[List[object]]:
    rows: List[List[object]] = []
    for checkpoint_stage, line in zip(summary["checkpoint_stages"], summary["matrix"]):
        row: List[object] = [checkpoint_stage]
        row.extend("" if value is None else f"{value:.6f}" for value in line)
        rows.append(row)
    return rows


def _plot_overall_curve(summary: dict, output_path: Path) -> None:
    xs = [item["checkpoint_stage"] for item in summary["overall_curve"]]
    ys = [item["overall_avg_final_score"] for item in summary["overall_curve"]]
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker="o", linewidth=2)
    plt.xlabel("Checkpoint Stage")
    plt.ylabel("Overall QA Score")
    plt.title("Overall QA Curve")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _plot_stage_lines(summary: dict, output_path: Path) -> None:
    xs = summary["checkpoint_stages"]
    plt.figure(figsize=(10, 6))
    for idx, dialogue_stage in enumerate(summary["dialogue_stages"]):
        ys: List[Optional[float]] = [row[idx] for row in summary["matrix"]]
        filtered_xs = [x for x, y in zip(xs, ys) if y is not None]
        filtered_ys = [y for y in ys if y is not None]
        if not filtered_xs:
            continue
        plt.plot(filtered_xs, filtered_ys, marker="o", linewidth=1.5, label=f"Stage {dialogue_stage:02d}")
    plt.xlabel("Checkpoint Stage")
    plt.ylabel("Avg QA Score")
    plt.title("Forgetting Curves by Dialogue Stage")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _plot_heatmap(summary: dict, output_path: Path) -> None:
    matrix = [
        [float("nan") if value is None else float(value) for value in row]
        for row in summary["matrix"]
    ]
    plt.figure(figsize=(10, 6))
    plt.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    plt.colorbar(label="Avg QA Score")
    plt.xticks(range(len(summary["dialogue_stages"])), [f"{stage:02d}" for stage in summary["dialogue_stages"]])
    plt.yticks(range(len(summary["checkpoint_stages"])), [f"{stage:02d}" for stage in summary["checkpoint_stages"]])
    plt.xlabel("Origin Dialogue Stage")
    plt.ylabel("Checkpoint Stage")
    plt.title("Retention Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def export_curves(results_root: Path, output_dir: Path) -> dict:
    summary = build_forgetting_summary(results_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "forgetting_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(
        output_dir / "forgetting_matrix.csv",
        ["checkpoint_stage"] + [f"dialogue_stage_{stage:02d}" for stage in summary["dialogue_stages"]],
        _matrix_to_rows(summary),
    )
    _write_csv(
        output_dir / "overall_qa_curve.csv",
        ["checkpoint_stage", "overall_avg_final_score", "num_samples"],
        [
            [
                item["checkpoint_stage"],
                f"{item['overall_avg_final_score']:.6f}",
                item["num_samples"],
            ]
            for item in summary["overall_curve"]
        ],
    )

    task_rows: List[List[object]] = []
    for task_type, items in sorted(summary["per_task_overall_curve"].items()):
        for item in items:
            task_rows.append(
                [
                    task_type,
                    item["checkpoint_stage"],
                    f"{item['avg_final_score']:.6f}",
                    item["num_samples"],
                ]
            )
    _write_csv(
        output_dir / "task_qa_curve.csv",
        ["task_type", "checkpoint_stage", "avg_final_score", "num_samples"],
        task_rows,
    )

    if summary["overall_curve"]:
        _plot_overall_curve(summary, output_dir / "overall_qa_curve.png")
    if summary["matrix"]:
        _plot_stage_lines(summary, output_dir / "forgetting_curve_by_stage.png")
        _plot_heatmap(summary, output_dir / "retention_heatmap.png")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate stage-wise eval outputs and plot forgetting curves."
    )
    parser.add_argument(
        "--results_root",
        type=Path,
        default=Path("dialogue_training/project/outputs/eval"),
        help="Directory containing checkpoint_stage_XX subdirectories.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("dialogue_training/project/outputs/curves"),
        help="Where to write CSV/JSON/PNG outputs.",
    )
    args = parser.parse_args()

    summary = export_curves(args.results_root, args.output_dir)
    print(json.dumps({"checkpoint_stages": summary["checkpoint_stages"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
