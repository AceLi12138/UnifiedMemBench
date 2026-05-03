from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


TASK_SHORT = {
    "Information Extraction": "IE",
    "Knowledge Updating": "KU",
    "Memory Arbitration": "MA",
    "Temporal Reasoning": "TR",
    "Multi-session Reasoning": "MSR",
    "Event Summarization": "ES",
}
TASK_ORDER = [
    "Information Extraction",
    "Knowledge Updating",
    "Memory Arbitration",
    "Temporal Reasoning",
    "Multi-session Reasoning",
    "Event Summarization",
]
TASK_COLORS = {
    "IE": "#2166ac",
    "KU": "#1b9e77",
    "MA": "#d95f02",
    "TR": "#7570b3",
    "MSR": "#e7298a",
    "ES": "#666666",
}


@dataclass(frozen=True)
class TaskScore:
    checkpoint_stage: int
    eval_stage: int
    task_type: str
    task_short: str
    count: int
    avg_final_score: float
    retention_ratio: Optional[float]
    retention_delta: Optional[float]
    overall_equal_weighted_score: Optional[float]
    avg_final_overall: Optional[float]
    scores_path: Path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate_from_scores(path: Path) -> dict:
    data = _read_json(path)
    return data.get("aggregate", data)


def _float_or_none(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_path(
    eval_root: Path,
    checkpoint_stage: int,
    eval_stage: int,
    split: str,
) -> Optional[Path]:
    checkpoint_dir = eval_root / f"checkpoint_stage_{checkpoint_stage:02d}"
    if not checkpoint_dir.exists():
        return None
    prefix = f"stage_{eval_stage:02d}_{split}"
    candidates = sorted(checkpoint_dir.glob(f"{prefix}*/scores.json"))
    return candidates[0] if candidates else None


def _available_checkpoint_stages(eval_root: Path) -> List[int]:
    stages: List[int] = []
    if not eval_root.exists():
        return stages
    pattern = re.compile(r"checkpoint_stage_(\d+)$")
    for path in eval_root.iterdir():
        if not path.is_dir():
            continue
        match = pattern.match(path.name)
        if match:
            stages.append(int(match.group(1)))
    return sorted(stages)


def collect_task_scores(
    eval_root: Path,
    split: str,
    stage01_scores_json: Optional[Path],
    min_stage: int,
    max_stage: int,
) -> Tuple[List[TaskScore], List[str]]:
    raw_records: List[Tuple[int, int, Path, dict]] = []
    missing: List[str] = []

    for checkpoint_stage in range(min_stage, max_stage + 1):
        for eval_stage in range(1, checkpoint_stage + 1):
            path = _score_path(eval_root, checkpoint_stage, eval_stage, split)
            if path is None and checkpoint_stage == 1 and eval_stage == 1 and stage01_scores_json:
                path = stage01_scores_json
            if path is None:
                missing.append(f"checkpoint_stage_{checkpoint_stage:02d}/stage_{eval_stage:02d}_{split}")
                continue
            raw_records.append((checkpoint_stage, eval_stage, path, _aggregate_from_scores(path)))

    diagonal: Dict[Tuple[int, str], float] = {}
    for checkpoint_stage, eval_stage, _path, aggregate in raw_records:
        if checkpoint_stage != eval_stage:
            continue
        for task_type, task_stats in aggregate.get("scores_by_task_type", {}).items():
            score = _float_or_none(task_stats.get("avg_final_score"))
            if score is not None:
                diagonal[(eval_stage, task_type)] = score

    records: List[TaskScore] = []
    for checkpoint_stage, eval_stage, path, aggregate in raw_records:
        overall_equal = _float_or_none(aggregate.get("overall_equal_weighted_score"))
        overall_avg = _float_or_none(aggregate.get("avg_final_score"))
        for task_type in TASK_ORDER:
            task_stats = aggregate.get("scores_by_task_type", {}).get(task_type)
            if not task_stats:
                continue
            score = _float_or_none(task_stats.get("avg_final_score"))
            if score is None:
                continue
            diag_score = diagonal.get((eval_stage, task_type))
            retention_ratio = None
            retention_delta = None
            if diag_score is not None:
                retention_delta = score - diag_score
                if diag_score > 0:
                    retention_ratio = score / diag_score
            records.append(
                TaskScore(
                    checkpoint_stage=checkpoint_stage,
                    eval_stage=eval_stage,
                    task_type=task_type,
                    task_short=TASK_SHORT.get(task_type, task_type),
                    count=int(task_stats.get("count", 0)),
                    avg_final_score=score,
                    retention_ratio=retention_ratio,
                    retention_delta=retention_delta,
                    overall_equal_weighted_score=overall_equal,
                    avg_final_overall=overall_avg,
                    scores_path=path,
                )
            )
    return records, missing


def _fmt(value: Optional[float], digits: int = 6) -> str:
    return "" if value is None or math.isnan(value) else f"{value:.{digits}f}"


def _write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(value) for value in values]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _by_task_eval_checkpoint(records: Sequence[TaskScore]) -> Dict[Tuple[str, int, int], TaskScore]:
    return {(r.task_short, r.checkpoint_stage, r.eval_stage): r for r in records}


def _write_long_csv(records: Sequence[TaskScore], output_dir: Path, prefix: str) -> None:
    _write_csv(
        output_dir / f"{prefix}_long.csv",
        [
            "checkpoint_stage",
            "eval_stage",
            "task_type",
            "task_short",
            "count",
            "avg_final_score",
            "retention_ratio",
            "retention_delta",
            "overall_equal_weighted_score",
            "avg_final_overall",
            "scores_path",
        ],
        [
            [
                r.checkpoint_stage,
                r.eval_stage,
                r.task_type,
                r.task_short,
                r.count,
                _fmt(r.avg_final_score),
                _fmt(r.retention_ratio),
                _fmt(r.retention_delta),
                _fmt(r.overall_equal_weighted_score),
                _fmt(r.avg_final_overall),
                str(r.scores_path),
            ]
            for r in records
        ],
    )


def _write_matrices(records: Sequence[TaskScore], output_dir: Path, prefix: str) -> None:
    max_stage = max((r.checkpoint_stage for r in records), default=0)
    lookup = _by_task_eval_checkpoint(records)
    header = ["task_type", "checkpoint_stage"] + [f"eval_stage_{stage:02d}" for stage in range(1, max_stage + 1)]

    def matrix_rows(field: str) -> List[List[object]]:
        rows: List[List[object]] = []
        for task_type in TASK_ORDER:
            task_short = TASK_SHORT[task_type]
            for checkpoint_stage in range(1, max_stage + 1):
                row: List[object] = [task_type, checkpoint_stage]
                for eval_stage in range(1, max_stage + 1):
                    record = lookup.get((task_short, checkpoint_stage, eval_stage))
                    value = getattr(record, field) if record is not None else None
                    row.append(_fmt(value))
                rows.append(row)
        return rows

    _write_csv(output_dir / f"{prefix}_avg_final_score_matrices.csv", header, matrix_rows("avg_final_score"))
    _write_csv(output_dir / f"{prefix}_retention_ratio_matrices.csv", header, matrix_rows("retention_ratio"))
    _write_csv(output_dir / f"{prefix}_retention_delta_matrices.csv", header, matrix_rows("retention_delta"))


def _write_summary_csvs(records: Sequence[TaskScore], output_dir: Path, prefix: str) -> None:
    max_stage = max((r.checkpoint_stage for r in records), default=0)
    lookup = _by_task_eval_checkpoint(records)

    diagonal_rows: List[List[object]] = []
    for stage in range(1, max_stage + 1):
        scores: List[float] = []
        row: List[object] = [stage]
        for task_type in TASK_ORDER:
            record = lookup.get((TASK_SHORT[task_type], stage, stage))
            value = record.avg_final_score if record is not None else None
            if value is not None:
                scores.append(value)
            row.append(_fmt(value))
        row.append(_fmt(_mean(scores)))
        diagonal_rows.append(row)
    _write_csv(
        output_dir / f"{prefix}_diagonal_scores.csv",
        ["stage"] + list(TASK_SHORT.values()) + ["task_mean"],
        diagonal_rows,
    )

    checkpoint_rows: List[List[object]] = []
    for eval_stage in range(1, max_stage + 1):
        score_values: List[float] = []
        retention_values: List[float] = []
        row: List[object] = [eval_stage]
        for task_type in TASK_ORDER:
            record = lookup.get((TASK_SHORT[task_type], max_stage, eval_stage))
            value = record.avg_final_score if record is not None else None
            if value is not None:
                score_values.append(value)
            row.append(_fmt(value))
        row.append(_fmt(_mean(score_values)))
        for task_type in TASK_ORDER:
            record = lookup.get((TASK_SHORT[task_type], max_stage, eval_stage))
            value = record.retention_ratio if record is not None else None
            if value is not None:
                retention_values.append(value)
            row.append(_fmt(value))
        row.append(_fmt(_mean(retention_values)))
        checkpoint_rows.append(row)
    _write_csv(
        output_dir / f"{prefix}_checkpoint{max_stage:02d}_by_stage.csv",
        ["eval_stage"]
        + [f"{short}_score" for short in TASK_SHORT.values()]
        + ["task_mean_score"]
        + [f"{short}_retention" for short in TASK_SHORT.values()]
        + ["task_mean_retention_available"],
        checkpoint_rows,
    )

    by_lag_rows: List[List[object]] = []
    for task_type in TASK_ORDER:
        task_short = TASK_SHORT[task_type]
        for lag in range(0, max_stage):
            task_records = [
                r for r in records if r.task_short == task_short and r.checkpoint_stage - r.eval_stage == lag
            ]
            score_mean = _mean(r.avg_final_score for r in task_records)
            retention_mean = _mean(
                r.retention_ratio for r in task_records if r.retention_ratio is not None
            )
            by_lag_rows.append(
                [
                    task_type,
                    task_short,
                    lag,
                    len(task_records),
                    _fmt(score_mean),
                    _fmt(retention_mean),
                ]
            )
    _write_csv(
        output_dir / f"{prefix}_by_lag.csv",
        ["task_type", "task_short", "lag", "num_points", "mean_score", "mean_retention_ratio"],
        by_lag_rows,
    )


def _svg_escape(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _blend_hex(low: str, high: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    lo = tuple(int(low[i : i + 2], 16) for i in (1, 3, 5))
    hi = tuple(int(high[i : i + 2], 16) for i in (1, 3, 5))
    vals = [round(lo[i] + (hi[i] - lo[i]) * t) for i in range(3)]
    return "#" + "".join(f"{v:02x}" for v in vals)


def _write_svg(path: Path, width: int, height: int, body: Sequence[str]) -> None:
    path.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="white"/>',
                '<style>text{font-family:Arial, Helvetica, sans-serif; fill:#222;} .small{font-size:12px;} .label{font-size:14px;} .title{font-size:18px; font-weight:700;} .axis{stroke:#333; stroke-width:1;} .grid{stroke:#d8d8d8; stroke-width:1;} .panel{font-size:14px; font-weight:700;}</style>',
                *body,
                "</svg>",
            ]
        ),
        encoding="utf-8",
    )


def _line_path(points: Sequence[Tuple[float, float]]) -> str:
    if not points:
        return ""
    first, *rest = points
    parts = [f"M {first[0]:.2f} {first[1]:.2f}"]
    parts.extend(f"L {x:.2f} {y:.2f}" for x, y in rest)
    return " ".join(parts)


def _plot_retention_by_lag(records: Sequence[TaskScore], output_dir: Path, prefix: str) -> None:
    max_stage = max((r.checkpoint_stage for r in records), default=0)
    lag_values = list(range(max_stage))
    series: Dict[str, List[Optional[float]]] = {}
    for task_type in TASK_ORDER:
        task_short = TASK_SHORT[task_type]
        ys: List[Optional[float]] = []
        for lag in lag_values:
            values = [
                r.retention_ratio
                for r in records
                if r.task_short == task_short and r.checkpoint_stage - r.eval_stage == lag and r.retention_ratio is not None
            ]
            ys.append(_mean(values))
        series[task_short] = ys

    width, height = 920, 560
    left, right, top, bottom = 80, 220, 56, 74
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_max = max([1.0] + [value for ys in series.values() for value in ys if value is not None])
    y_max = math.ceil(y_max * 10) / 10

    def x_pos(lag: int) -> float:
        return left + (lag / max(1, max_stage - 1)) * plot_w

    def y_pos(value: float) -> float:
        return top + (1.0 - value / y_max) * plot_h

    body: List[str] = [
        '<text x="460" y="30" text-anchor="middle" class="title">Task-wise Unseen Retention by Lag</text>',
    ]
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = top + (1.0 - frac) * plot_h
        value = frac * y_max
        body.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" class="grid"/>')
        body.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" class="small">{value:.2f}</text>')
    for lag in lag_values:
        x = x_pos(lag)
        body.append(f'<text x="{x:.2f}" y="{top + plot_h + 28}" text-anchor="middle" class="small">{lag}</text>')
    body.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>')
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>')
    body.append(f'<text x="{left + plot_w / 2:.2f}" y="{height - 22}" text-anchor="middle" class="label">Lag = checkpoint stage - evaluated stage</text>')
    body.append(f'<text x="22" y="{top + plot_h / 2:.2f}" text-anchor="middle" transform="rotate(-90 22 {top + plot_h / 2:.2f})" class="label">Retention ratio</text>')
    for task_short, ys in series.items():
        points = [(x_pos(lag), y_pos(value)) for lag, value in zip(lag_values, ys) if value is not None]
        color = TASK_COLORS[task_short]
        body.append(f'<path d="{_line_path(points)}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        for x, y in points:
            body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{color}"/>')
    for idx, task_short in enumerate(TASK_SHORT.values()):
        y = top + 22 + idx * 25
        color = TASK_COLORS[task_short]
        body.append(f'<line x1="{left + plot_w + 38}" y1="{y}" x2="{left + plot_w + 68}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        body.append(f'<text x="{left + plot_w + 78}" y="{y + 4}" class="small">{task_short}</text>')
    _write_svg(output_dir / f"{prefix}_mean_retention_by_lag.svg", width, height, body)


def _plot_score_heatmaps(records: Sequence[TaskScore], output_dir: Path, prefix: str, vmax: float) -> None:
    max_stage = max((r.checkpoint_stage for r in records), default=0)
    lookup = _by_task_eval_checkpoint(records)
    width, height = 1080, 740
    margin_x, margin_y = 70, 70
    panel_w, panel_h = 310, 285
    cell = 21
    body: List[str] = [
        f'<text x="{width / 2:.0f}" y="32" text-anchor="middle" class="title">Task-wise Unseen Score Heatmaps</text>',
        f'<text x="{width / 2:.0f}" y="54" text-anchor="middle" class="small">Cell = avg_final_score; color scale clipped at {vmax:.2f}</text>',
    ]
    for idx, task_type in enumerate(TASK_ORDER):
        task_short = TASK_SHORT[task_type]
        col, row = idx % 3, idx // 3
        ox = margin_x + col * panel_w
        oy = margin_y + row * panel_h
        body.append(f'<text x="{ox + 105}" y="{oy - 10}" text-anchor="middle" class="panel">{task_short}</text>')
        for checkpoint_stage in range(1, max_stage + 1):
            for eval_stage in range(1, max_stage + 1):
                x = ox + 44 + (eval_stage - 1) * cell
                y = oy + 18 + (checkpoint_stage - 1) * cell
                record = lookup.get((task_short, checkpoint_stage, eval_stage))
                if record is None:
                    color = "#f2f2f2"
                else:
                    color = _blend_hex("#f7fbff", "#08306b", record.avg_final_score / vmax)
                body.append(f'<rect x="{x}" y="{y}" width="{cell - 1}" height="{cell - 1}" fill="{color}"/>')
        for stage in range(1, max_stage + 1):
            x = ox + 44 + (stage - 1) * cell + cell / 2
            y = oy + 18 + (stage - 1) * cell + cell / 2
            body.append(f'<text x="{x:.1f}" y="{oy + 12}" text-anchor="middle" class="small">{stage}</text>')
            body.append(f'<text x="{ox + 35}" y="{y + 4:.1f}" text-anchor="end" class="small">{stage}</text>')
        body.append(f'<text x="{ox + 44 + max_stage * cell / 2:.1f}" y="{oy + 18 + max_stage * cell + 28}" text-anchor="middle" class="small">eval stage</text>')
        body.append(f'<text x="{ox + 8}" y="{oy + 18 + max_stage * cell / 2:.1f}" text-anchor="middle" transform="rotate(-90 {ox + 8} {oy + 18 + max_stage * cell / 2:.1f})" class="small">checkpoint</text>')

    legend_x, legend_y, legend_w, legend_h = 810, 690, 190, 12
    for i in range(legend_w):
        color = _blend_hex("#f7fbff", "#08306b", i / max(1, legend_w - 1))
        body.append(f'<rect x="{legend_x + i}" y="{legend_y}" width="1" height="{legend_h}" fill="{color}"/>')
    body.append(f'<text x="{legend_x}" y="{legend_y + 30}" text-anchor="middle" class="small">0</text>')
    body.append(f'<text x="{legend_x + legend_w}" y="{legend_y + 30}" text-anchor="middle" class="small">{vmax:.2f}</text>')
    _write_svg(output_dir / f"{prefix}_score_heatmaps.svg", width, height, body)


def _plot_checkpoint_history(records: Sequence[TaskScore], output_dir: Path, prefix: str) -> None:
    max_stage = max((r.checkpoint_stage for r in records), default=0)
    lookup = _by_task_eval_checkpoint(records)
    width, height = 920, 560
    left, right, top, bottom = 80, 220, 56, 74
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_score = max(
        [0.3]
        + [
            lookup[(TASK_SHORT[task], max_stage, eval_stage)].avg_final_score
            for task in TASK_ORDER
            for eval_stage in range(1, max_stage + 1)
            if (TASK_SHORT[task], max_stage, eval_stage) in lookup
        ]
    )
    y_max = math.ceil(max_score * 10) / 10

    def x_pos(stage: int) -> float:
        return left + ((stage - 1) / max(1, max_stage - 1)) * plot_w

    def y_pos(value: float) -> float:
        return top + (1.0 - value / y_max) * plot_h

    body: List[str] = [
        f'<text x="{width / 2:.0f}" y="30" text-anchor="middle" class="title">Checkpoint {max_stage:02d}: Historical Unseen Scores by Task</text>',
    ]
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = top + (1.0 - frac) * plot_h
        value = frac * y_max
        body.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" class="grid"/>')
        body.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" class="small">{value:.2f}</text>')
    for stage in range(1, max_stage + 1):
        x = x_pos(stage)
        body.append(f'<text x="{x:.2f}" y="{top + plot_h + 28}" text-anchor="middle" class="small">{stage}</text>')
    body.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>')
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>')
    body.append(f'<text x="{left + plot_w / 2:.2f}" y="{height - 22}" text-anchor="middle" class="label">Evaluated historical stage</text>')
    body.append(f'<text x="22" y="{top + plot_h / 2:.2f}" text-anchor="middle" transform="rotate(-90 22 {top + plot_h / 2:.2f})" class="label">avg_final_score</text>')
    for task_type in TASK_ORDER:
        task_short = TASK_SHORT[task_type]
        points = []
        for eval_stage in range(1, max_stage + 1):
            record = lookup.get((task_short, max_stage, eval_stage))
            if record is not None:
                points.append((x_pos(eval_stage), y_pos(record.avg_final_score)))
        color = TASK_COLORS[task_short]
        body.append(f'<path d="{_line_path(points)}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        for x, y in points:
            body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{color}"/>')
    for idx, task_short in enumerate(TASK_SHORT.values()):
        y = top + 22 + idx * 25
        color = TASK_COLORS[task_short]
        body.append(f'<line x1="{left + plot_w + 38}" y1="{y}" x2="{left + plot_w + 68}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        body.append(f'<text x="{left + plot_w + 78}" y="{y + 4}" class="small">{task_short}</text>')
    _write_svg(output_dir / f"{prefix}_checkpoint{max_stage:02d}_history_by_task.svg", width, height, body)


def write_plots(records: Sequence[TaskScore], output_dir: Path, prefix: str, heatmap_vmax: float) -> None:
    _plot_retention_by_lag(records, output_dir, prefix)
    _plot_score_heatmaps(records, output_dir, prefix, heatmap_vmax)
    _plot_checkpoint_history(records, output_dir, prefix)


def write_report(
    records: Sequence[TaskScore],
    missing: Sequence[str],
    output_dir: Path,
    record_path: Path,
    prefix: str,
    eval_root: Path,
    stage01_scores_json: Optional[Path],
    split: str,
    model_label: str,
) -> None:
    max_stage = max((r.checkpoint_stage for r in records), default=0)
    lookup = _by_task_eval_checkpoint(records)
    lines: List[str] = [
        f"# {model_label} {split} task-wise stagewise analysis",
        "",
        "## Data Sources",
        "",
        f"- stagewise eval root: `{eval_root}`",
    ]
    if stage01_scores_json:
        lines.append(f"- stage 01 baseline scores: `{stage01_scores_json}`")
    lines.extend(
        [
            f"- split: `{split}`",
            "- Metric: `scores_by_task_type[task].avg_final_score` for each task.",
            "",
            "## Output Files",
            "",
        ]
    )
    for name in [
        f"{prefix}_long.csv",
        f"{prefix}_avg_final_score_matrices.csv",
        f"{prefix}_retention_ratio_matrices.csv",
        f"{prefix}_retention_delta_matrices.csv",
        f"{prefix}_diagonal_scores.csv",
        f"{prefix}_checkpoint{max_stage:02d}_by_stage.csv",
        f"{prefix}_by_lag.csv",
        f"{prefix}_mean_retention_by_lag.svg",
        f"{prefix}_score_heatmaps.svg",
        f"{prefix}_checkpoint{max_stage:02d}_history_by_task.svg",
    ]:
        lines.append(f"- `{output_dir / name}`")

    lines.extend(
        [
            "",
            "## Task Abbreviations",
            "",
        ]
    )
    for task_type in TASK_ORDER:
        lines.append(f"- `{TASK_SHORT[task_type]}` = {task_type}")

    lines.extend(
        [
            "",
            "## Diagonal Mean and Later Retention by Task",
            "",
            "| task | diagonal mean | checkpoint final stage mean | lag>=3 mean retention |",
            "|---|---:|---:|---:|",
        ]
    )
    for task_type in TASK_ORDER:
        short = TASK_SHORT[task_type]
        diagonal = [
            r.avg_final_score
            for r in records
            if r.task_short == short and r.checkpoint_stage == r.eval_stage
        ]
        final_scores = [
            r.avg_final_score for r in records if r.task_short == short and r.checkpoint_stage == max_stage
        ]
        lag3 = [
            r.retention_ratio
            for r in records
            if r.task_short == short and r.checkpoint_stage - r.eval_stage >= 3 and r.retention_ratio is not None
        ]
        lines.append(
            f"| {task_type} | {_fmt(_mean(diagonal), 4)} | {_fmt(_mean(final_scores), 4)} | {_fmt(_mean(lag3), 4)} |"
        )

    lines.extend(
        [
            "",
            f"## Checkpoint {max_stage:02d} Task-wise Mean Scores on Historical Stages",
            "",
            "| eval stage | IE | KU | MA | TR | MSR | ES | task mean |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for eval_stage in range(1, max_stage + 1):
        vals: List[Optional[float]] = []
        for task_type in TASK_ORDER:
            record = lookup.get((TASK_SHORT[task_type], max_stage, eval_stage))
            vals.append(record.avg_final_score if record else None)
        task_mean = _mean(v for v in vals if v is not None)
        lines.append(
            f"| {eval_stage} | "
            + " | ".join(_fmt(v, 4) for v in vals)
            + f" | {_fmt(task_mean, 4)} |"
        )

    lines.extend(["", "## Current Observations", ""])
    lines.append(f"- After task-wise decomposition, forgetting for `{model_label}` is not a uniform global decline; it differs substantially by task type.")
    lines.append("- `IE` has the highest absolute score and is the easiest task family to write into and retain parametrically.")
    lines.append("- `KU` has high diagonal scores, but its final-checkpoint mean drops substantially, suggesting knowledge-updating tasks are more easily overwritten across stages.")
    lines.append("- `MA` and `TR` have low and volatile scores, which would be hidden by an overall mean alone.")
    lines.append("- `ES` has a low diagonal baseline; occasional high retention ratios are mainly caused by a low denominator and should not be interpreted as strong summarization retention.")
    if missing:
        lines.extend(["", "## Missing Files", ""])
        for item in missing:
            lines.append(f"- `{item}`")
    else:
        lines.extend(["", "## Missing Files", "", "- No missing files found."])

    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate task-wise scores from stagewise memory eval outputs."
    )
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--stage01-scores-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record-path", type=Path, default=None)
    parser.add_argument("--split", choices=["seen", "unseen"], default="unseen")
    parser.add_argument("--min-stage", type=int, default=1)
    parser.add_argument("--max-stage", type=int, default=None)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--heatmap-vmax", type=float, default=0.5)
    parser.add_argument("--model-label", type=str, default="GLM mainline")
    args = parser.parse_args()

    available = _available_checkpoint_stages(args.eval_root)
    max_stage = args.max_stage if args.max_stage is not None else (max(available) if available else 1)
    if args.stage01_scores_json and max_stage < 1:
        max_stage = 1
    if args.min_stage < 1 or max_stage < args.min_stage:
        raise ValueError(f"Invalid stage range: {args.min_stage}..{max_stage}")

    prefix = args.prefix or f"taskwise_{args.split}"
    records, missing = collect_task_scores(
        eval_root=args.eval_root,
        split=args.split,
        stage01_scores_json=args.stage01_scores_json,
        min_stage=args.min_stage,
        max_stage=max_stage,
    )
    if not records:
        raise RuntimeError("No task-wise records found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_long_csv(records, args.output_dir, prefix)
    _write_matrices(records, args.output_dir, prefix)
    _write_summary_csvs(records, args.output_dir, prefix)
    write_plots(records, args.output_dir, prefix, args.heatmap_vmax)
    if args.record_path:
        write_report(
            records=records,
            missing=missing,
            output_dir=args.output_dir,
            record_path=args.record_path,
            prefix=prefix,
            eval_root=args.eval_root,
            stage01_scores_json=args.stage01_scores_json,
            split=args.split,
            model_label=args.model_label,
        )

    result = {
        "num_records": len(records),
        "missing_count": len(missing),
        "output_dir": str(args.output_dir),
        "record_path": str(args.record_path) if args.record_path else None,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if missing:
        print(json.dumps({"missing": list(missing)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
