#!/usr/bin/env python3
"""Compute old-stage retention-rate trajectories from saved stagewise results.

This script intentionally uses only Python standard-library modules so it can run
in the current analysis environment without installing plotting dependencies.
It reads the held-out-character retention matrices already produced for Figure 6,
computes old-stage retention ratios, and writes CSV/Markdown plus SVG/PDF/PNG
figures.
"""

from __future__ import annotations

import csv
import math
import statistics
import struct
import zlib
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

OVERALL_MATRIX = ROOT / "paper/details/retention_overall_matrix_long.csv"
TASKWISE_MATRIX = ROOT / "paper/details/retention_taskwise_matrix_long.csv"
PROTOCOL_INVENTORY = ROOT / "paper/details/retention_protocol_inventory.csv"

MODEL_MAP = {
    "GLM-4-9B-1M": "GLM-4-9B-Chat-1M",
    "GLM-4-9B-Chat-1M": "GLM-4-9B-Chat-1M",
    "Qwen3.5-9B": "Qwen3.5-9B",
    "Qwen2.5-7B-1M": "Qwen2.5-7B-Instruct-1M",
    "Qwen2.5-7B-Instruct-1M": "Qwen2.5-7B-Instruct-1M",
}
MODELS = ["GLM-4-9B-Chat-1M", "Qwen3.5-9B", "Qwen2.5-7B-Instruct-1M"]
TASKS = ["IE", "MSR", "ES", "TR", "KU", "MA"]
ALL_TASKS = ["overall"] + TASKS
SPLIT = "held_out_character"

MODEL_COLORS = {
    "GLM-4-9B-Chat-1M": "#2563eb",
    "Qwen3.5-9B": "#dc2626",
    "Qwen2.5-7B-Instruct-1M": "#059669",
}
TASK_COLORS = {
    "IE": "#2563eb",
    "MSR": "#dc2626",
    "ES": "#7c3aed",
    "TR": "#f59e0b",
    "KU": "#059669",
    "MA": "#6b7280",
}


def fnum(value: float | None, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NaN"
    return f"{value:.{digits}f}"


def parse_int(value: str) -> int:
    return int(str(value).strip())


def read_scores() -> tuple[dict[tuple[str, int, int, str], float], dict[str, object]]:
    if not OVERALL_MATRIX.exists() or not TASKWISE_MATRIX.exists():
        missing = [str(p) for p in [OVERALL_MATRIX, TASKWISE_MATRIX] if not p.exists()]
        raise FileNotFoundError("Missing priority retention matrix input(s): " + ", ".join(missing))

    scores: dict[tuple[str, int, int, str], float] = {}
    source_files: dict[str, set[str]] = {"overall": set(), "taskwise": set()}
    split_values: set[str] = set()

    with OVERALL_MATRIX.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            split = row.get("split_type", "")
            split_values.add(split)
            if split != SPLIT:
                continue
            model = MODEL_MAP.get(row["model_name"], row["model_name"])
            c = parse_int(row["checkpoint_stage"])
            s = parse_int(row["eval_stage"])
            scores[(model, c, s, "overall")] = float(row["overall_score"])
            if row.get("source_file"):
                source_files["overall"].add(row["source_file"])

    with TASKWISE_MATRIX.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            split = row.get("split_type", "")
            split_values.add(split)
            if split != SPLIT:
                continue
            model = MODEL_MAP.get(row["model_name"], row["model_name"])
            c = parse_int(row["checkpoint_stage"])
            s = parse_int(row["eval_stage"])
            task = row["task_type"]
            scores[(model, c, s, task)] = float(row["task_score"])
            source_files["taskwise"].add(str(TASKWISE_MATRIX.relative_to(ROOT)))

    meta = {
        "input_priority_used": "retention_overall_matrix_long.csv + retention_taskwise_matrix_long.csv",
        "overall_matrix": str(OVERALL_MATRIX.relative_to(ROOT)),
        "taskwise_matrix": str(TASKWISE_MATRIX.relative_to(ROOT)),
        "protocol_inventory": str(PROTOCOL_INVENTORY.relative_to(ROOT)) if PROTOCOL_INVENTORY.exists() else "MISSING",
        "split_values_seen": sorted(split_values),
        "overall_source_files": sorted(source_files["overall"]),
    }
    return scores, meta


def build_cellwise(scores: dict[tuple[str, int, int, str], float]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in MODELS:
        for c in range(2, 11):
            for s in range(1, c):
                for task in ALL_TASKS:
                    baseline = scores.get((model, s, s, task))
                    old_score = scores.get((model, c, s, task))
                    notes: list[str] = []
                    denominator_valid = baseline is not None and baseline > 0
                    if baseline is None:
                        notes.append("baseline_diagonal_missing")
                    elif baseline <= 0:
                        notes.append("baseline_diagonal_zero_or_nonpositive")
                    else:
                        if baseline < 0.03:
                            notes.append("baseline_below_0.03")
                        if baseline < 0.05:
                            notes.append("baseline_below_0.05")
                    if old_score is None:
                        notes.append("old_stage_score_missing")
                    if denominator_valid and old_score is not None:
                        retention_rate = old_score / baseline  # type: ignore[operator]
                        if retention_rate > 3:
                            notes.append("retention_rate_gt_3")
                        elif retention_rate > 2:
                            notes.append("retention_rate_gt_2")
                    else:
                        retention_rate = math.nan
                    rows.append(
                        {
                            "model": model,
                            "checkpoint_stage": c,
                            "eval_stage": s,
                            "task_type": task,
                            "baseline_diagonal_score": baseline,
                            "old_stage_score": old_score,
                            "retention_rate": retention_rate,
                            "split": SPLIT,
                            "denominator_valid": "yes" if denominator_valid else "no",
                            "notes": ";".join(notes) if notes else "",
                        }
                    )
    return rows


def valid_rate(row: dict[str, object], min_baseline: float | None = None) -> bool:
    rate = row["retention_rate"]
    baseline = row["baseline_diagonal_score"]
    if not isinstance(rate, float) or math.isnan(rate):
        return False
    if baseline is None or not isinstance(baseline, float):
        return False
    if min_baseline is not None and baseline < min_baseline:
        return False
    return True


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def summarize_overall_by_model(rows: list[dict[str, object]], min_baseline: float | None = None) -> list[dict[str, object]]:
    out = []
    for model in MODELS:
        for c in range(2, 11):
            candidates = [r for r in rows if r["model"] == model and r["checkpoint_stage"] == c and r["task_type"] == "overall"]
            vals = [r["retention_rate"] for r in candidates if valid_rate(r, min_baseline)]  # type: ignore[list-item]
            mean, std = mean_std(vals)  # type: ignore[arg-type]
            out.append(
                {
                    "model": model,
                    "checkpoint_stage": c,
                    "mean_retention_rate": mean,
                    "std_retention_rate": std,
                    "num_old_stages": c - 1,
                    "num_valid_old_stages": len(vals),
                }
            )
    return out


def summarize_taskwise_mean(rows: list[dict[str, object]], min_baseline: float | None = None) -> list[dict[str, object]]:
    out = []
    for task in TASKS:
        for c in range(2, 11):
            candidates = [r for r in rows if r["task_type"] == task and r["checkpoint_stage"] == c]
            vals = [r["retention_rate"] for r in candidates if valid_rate(r, min_baseline)]  # type: ignore[list-item]
            mean, std = mean_std(vals)  # type: ignore[arg-type]
            out.append(
                {
                    "task_type": task,
                    "checkpoint_stage": c,
                    "mean_retention_rate": mean,
                    "std_retention_rate": std,
                    "num_model_stage_cells": len(candidates),
                    "num_valid_cells": len(vals),
                }
            )
    return out


def summarize_taskwise_by_model(rows: list[dict[str, object]], min_baseline: float | None = None) -> list[dict[str, object]]:
    out = []
    for model in MODELS:
        for task in TASKS:
            for c in range(2, 11):
                candidates = [
                    r for r in rows if r["model"] == model and r["task_type"] == task and r["checkpoint_stage"] == c
                ]
                vals = [r["retention_rate"] for r in candidates if valid_rate(r, min_baseline)]  # type: ignore[list-item]
                mean, std = mean_std(vals)  # type: ignore[arg-type]
                out.append(
                    {
                        "model": model,
                        "task_type": task,
                        "checkpoint_stage": c,
                        "mean_retention_rate": mean,
                        "std_retention_rate": std,
                        "num_old_stages": c - 1,
                        "num_valid_old_stages": len(vals),
                    }
                )
    return out


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            clean = {}
            for field in fieldnames:
                value = row.get(field)
                if isinstance(value, float):
                    clean[field] = fnum(value)
                elif value is None:
                    clean[field] = "NaN"
                else:
                    clean[field] = value
            writer.writerow(clean)


def filtered_rows(rows: list[dict[str, object]], threshold: float = 0.05) -> list[dict[str, object]]:
    out = []
    for row in rows:
        baseline = row["baseline_diagonal_score"]
        if isinstance(baseline, float) and baseline >= threshold:
            out.append(row)
    return out


def slope(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def collect_series(rows: list[dict[str, object]], key_field: str, key_values: list[str]) -> dict[str, list[tuple[int, float | None]]]:
    series: dict[str, list[tuple[int, float | None]]] = {}
    for key in key_values:
        points = []
        for c in range(2, 11):
            matches = [r for r in rows if r[key_field] == key and r["checkpoint_stage"] == c]
            value = matches[0].get("mean_retention_rate") if matches else None
            points.append((c, value if isinstance(value, float) else None))
        series[key] = points
    return series


def y_domain(series_list: list[dict[str, list[tuple[int, float | None]]]]) -> tuple[float, float]:
    vals = []
    for series in series_list:
        for points in series.values():
            vals.extend([v for _, v in points if isinstance(v, float) and not math.isnan(v)])
    vals.append(1.0)
    if not vals:
        return 0.0, 1.0
    ymin = min(0.0, min(vals))
    ymax = max(vals)
    pad = max(0.08, (ymax - ymin) * 0.12)
    return ymin, ymax + pad


def render_svg_line_figure(
    path: Path,
    overall_rows: list[dict[str, object]],
    task_rows: list[dict[str, object]],
    note: str,
) -> None:
    width, height = 1200, 520
    panel_w, panel_h = 500, 340
    left_a, top = 80, 65
    left_b = 670
    bottom = top + panel_h
    stage_ticks = list(range(2, 11))
    overall_series = collect_series(overall_rows, "model", MODELS)
    task_series = collect_series(task_rows, "task_type", TASKS)
    ymin, ymax = y_domain([overall_series, task_series])

    def px(stage: int, left: int) -> float:
        return left + (stage - 2) / 8.0 * panel_w

    def py(value: float) -> float:
        return bottom - (value - ymin) / (ymax - ymin) * panel_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:'Times New Roman',Times,serif;fill:#111827}.axis{stroke:#111827;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}.line{fill:none;stroke-width:2.2}.legend{font-size:13px}.small{font-size:12px}.label{font-size:14px}.title{font-size:17px;font-weight:700}</style>",
        '<rect x="0" y="0" width="1200" height="520" fill="white"/>',
    ]

    def panel(left: int, title: str) -> None:
        parts.append(f'<text class="title" x="{left + 8}" y="{top + 22}">{xml_escape(title)}</text>')
        for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
            yval = ymin + (ymax - ymin) * frac
            y = py(yval)
            parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + panel_w}" y2="{y:.2f}"/>')
            parts.append(
                f'<text class="small" x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">{yval:.1f}</text>'
            )
        y_one = py(1.0)
        parts.append(
            f'<line x1="{left}" y1="{y_one:.2f}" x2="{left + panel_w}" y2="{y_one:.2f}" stroke="#374151" stroke-dasharray="6,5" stroke-width="1.2"/>'
        )
        parts.append(f'<text class="small" x="{left + panel_w - 2}" y="{y_one - 6:.2f}" text-anchor="end">No loss vs. write-in</text>')
        parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>')
        parts.append(f'<line class="axis" x1="{left}" y1="{bottom}" x2="{left + panel_w}" y2="{bottom}"/>')
        for stage in stage_ticks:
            x = px(stage, left)
            parts.append(f'<line class="axis" x1="{x:.2f}" y1="{bottom}" x2="{x:.2f}" y2="{bottom + 5}"/>')
            parts.append(f'<text class="small" x="{x:.2f}" y="{bottom + 22}" text-anchor="middle">C{stage}</text>')
        parts.append(f'<text class="label" x="{left + panel_w / 2}" y="{bottom + 48}" text-anchor="middle">Checkpoint stage</text>')
        parts.append(
            f'<text class="label" x="{left - 56}" y="{top + panel_h / 2}" text-anchor="middle" transform="rotate(-90 {left - 56} {top + panel_h / 2})">Mean retention rate</text>'
        )

    def draw_series(left: int, series: dict[str, list[tuple[int, float | None]]], colors: dict[str, str]) -> None:
        for name, points in series.items():
            valid = [(px(c, left), py(v), c, v) for c, v in points if isinstance(v, float) and not math.isnan(v)]
            if len(valid) >= 2:
                d = " ".join(("M" if i == 0 else "L") + f"{x:.2f},{y:.2f}" for i, (x, y, _, _) in enumerate(valid))
                parts.append(f'<path class="line" d="{d}" stroke="{colors[name]}"/>')
            for x, y, _, _ in valid:
                parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.4" fill="{colors[name]}"/>')

    def legend(left: int, y: int, labels: list[str], colors: dict[str, str], columns: int = 1) -> None:
        col_w = 165 if columns > 1 else 260
        for i, label in enumerate(labels):
            row = i // columns
            col = i % columns
            x = left + col * col_w
            yy = y + row * 22
            parts.append(f'<line x1="{x}" y1="{yy}" x2="{x + 22}" y2="{yy}" stroke="{colors[label]}" stroke-width="2.4"/>')
            parts.append(f'<circle cx="{x + 11}" cy="{yy}" r="3" fill="{colors[label]}"/>')
            parts.append(f'<text class="legend" x="{x + 30}" y="{yy + 4}">{xml_escape(label)}</text>')

    panel(left_a, "A. Overall old-stage retention rate")
    panel(left_b, "B. Task-wise retention rate")
    draw_series(left_a, overall_series, MODEL_COLORS)
    draw_series(left_b, task_series, TASK_COLORS)
    legend(left_a, 455, MODELS, MODEL_COLORS, columns=1)
    legend(left_b, 455, TASKS, TASK_COLORS, columns=3)
    parts.append(f'<text class="small" x="80" y="508">{xml_escape(note)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_pdf_line_figure(
    path: Path,
    overall_rows: list[dict[str, object]],
    task_rows: list[dict[str, object]],
    note: str,
) -> None:
    width, height = 864, 374
    sx, sy = width / 1200.0, height / 520.0
    panel_w, panel_h = 500 * sx, 340 * sy
    left_a, top = 80 * sx, 65 * sy
    left_b = 670 * sx
    bottom = top + panel_h
    stage_ticks = list(range(2, 11))
    overall_series = collect_series(overall_rows, "model", MODELS)
    task_series = collect_series(task_rows, "task_type", TASKS)
    ymin, ymax = y_domain([overall_series, task_series])

    def rgb(hex_color: str) -> tuple[float, float, float]:
        h = hex_color.lstrip("#")
        return int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255

    def px(stage: int, left: float) -> float:
        return left + (stage - 2) / 8.0 * panel_w

    def py(value: float) -> float:
        return bottom - (value - ymin) / (ymax - ymin) * panel_h

    def yp(y: float) -> float:
        return height - y

    cmds: list[str] = []

    def stroke_line(x1: float, y1: float, x2: float, y2: float, color: str = "#111827", w: float = 1.0, dash: str = "") -> None:
        r, g, b = rgb(color)
        cmds.append(f"{r:.3f} {g:.3f} {b:.3f} RG {w:.2f} w {dash} {x1:.2f} {yp(y1):.2f} m {x2:.2f} {yp(y2):.2f} l S [] 0 d")

    def text(x: float, y: float, s: str, size: float = 9, bold: bool = False, anchor: str = "left", rotate: bool = False) -> None:
        width_est = len(s) * size * 0.48
        tx = x - width_est / 2 if anchor == "middle" else x - width_est if anchor == "end" else x
        font = "F2" if bold else "F1"
        if rotate:
            cmds.append(f"BT /{font} {size:.1f} Tf 0 1 -1 0 {tx:.2f} {yp(y):.2f} Tm ({pdf_escape(s)}) Tj ET")
        else:
            cmds.append(f"BT /{font} {size:.1f} Tf {tx:.2f} {yp(y):.2f} Td ({pdf_escape(s)}) Tj ET")

    def marker(x: float, y: float, color: str) -> None:
        r, g, b = rgb(color)
        size = 3.8
        cmds.append(f"{r:.3f} {g:.3f} {b:.3f} rg {x-size/2:.2f} {yp(y)-size/2:.2f} {size:.2f} {size:.2f} re f")

    def panel(left: float, title: str) -> None:
        text(left + 8 * sx, top + 22 * sy, title, size=12, bold=True)
        for frac in [0, 0.25, 0.5, 0.75, 1.0]:
            yval = ymin + (ymax - ymin) * frac
            y = py(yval)
            stroke_line(left, y, left + panel_w, y, "#e5e7eb", 0.6)
            text(left - 10 * sx, y + 4 * sy, f"{yval:.1f}", size=7, anchor="end")
        y_one = py(1.0)
        stroke_line(left, y_one, left + panel_w, y_one, "#374151", 0.8, "[4 4] 0 d")
        text(left + panel_w - 2, y_one - 6 * sy, "No loss vs. write-in", size=7, anchor="end")
        stroke_line(left, top, left, bottom, "#111827", 0.8)
        stroke_line(left, bottom, left + panel_w, bottom, "#111827", 0.8)
        for stage in stage_ticks:
            x = px(stage, left)
            stroke_line(x, bottom, x, bottom + 5 * sy, "#111827", 0.8)
            text(x, bottom + 22 * sy, f"C{stage}", size=7, anchor="middle")
        text(left + panel_w / 2, bottom + 48 * sy, "Checkpoint stage", size=9, anchor="middle")
        text(left - 56 * sx, top + panel_h / 2, "Mean retention rate", size=9, anchor="middle", rotate=True)

    def draw_series(left: float, series: dict[str, list[tuple[int, float | None]]], colors: dict[str, str]) -> None:
        for name, points in series.items():
            valid = [(px(c, left), py(v)) for c, v in points if isinstance(v, float) and not math.isnan(v)]
            if len(valid) >= 2:
                r, g, b = rgb(colors[name])
                seg = [f"{r:.3f} {g:.3f} {b:.3f} RG 1.6 w"]
                for i, (x, y) in enumerate(valid):
                    seg.append(f"{x:.2f} {yp(y):.2f} {'m' if i == 0 else 'l'}")
                seg.append("S")
                cmds.append(" ".join(seg))
            for x, y in valid:
                marker(x, y, colors[name])

    def legend(left: float, y: float, labels: list[str], colors: dict[str, str], columns: int) -> None:
        col_w = 165 * sx if columns > 1 else 260 * sx
        for i, label in enumerate(labels):
            row = i // columns
            col = i % columns
            x = left + col * col_w
            yy = y + row * 22 * sy
            stroke_line(x, yy, x + 22 * sx, yy, colors[label], 1.6)
            marker(x + 11 * sx, yy, colors[label])
            text(x + 30 * sx, yy + 4 * sy, label, size=7.5)

    # white background
    cmds.append(f"1 1 1 rg 0 0 {width} {height} re f")
    panel(left_a, "A. Overall old-stage retention rate")
    panel(left_b, "B. Task-wise retention rate")
    draw_series(left_a, overall_series, MODEL_COLORS)
    draw_series(left_b, task_series, TASK_COLORS)
    legend(left_a, 455 * sy, MODELS, MODEL_COLORS, 1)
    legend(left_b, 455 * sy, TASKS, TASK_COLORS, 3)
    text(80 * sx, 508 * sy, note, size=7)

    stream = "\n".join(cmds).encode("latin-1", errors="replace")
    objects: list[bytes] = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objects.append(
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width:.2f} {height:.2f}] /Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> /Contents 4 0 R >>".encode()
    )
    objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf.extend(f"{i} 0 obj\n".encode())
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode())
    for off in offsets[1:]:
        pdf.extend(f"{off:010d} 00000 n \n".encode())
    pdf.extend(f"trailer << /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    path.write_bytes(bytes(pdf))


# 5x7 bitmap font. Lowercase is rendered as uppercase to keep the PNG generator small.
FONT = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "J": ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "/": ["00001", "00010", "00010", "00100", "01000", "01000", "10000"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
}


def hex_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


class Canvas:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.pixels = bytearray([255, 255, 255] * width * height)

    def set_px(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            i = (y * self.width + x) * 3
            self.pixels[i : i + 3] = bytes(color)

    def line(self, x1: float, y1: float, x2: float, y2: float, color: tuple[int, int, int], w: int = 2) -> None:
        x1i, y1i, x2i, y2i = map(lambda v: int(round(v)), [x1, y1, x2, y2])
        dx, dy = abs(x2i - x1i), -abs(y2i - y1i)
        sx = 1 if x1i < x2i else -1
        sy = 1 if y1i < y2i else -1
        err = dx + dy
        x, y = x1i, y1i
        while True:
            for ox in range(-w // 2, w // 2 + 1):
                for oy in range(-w // 2, w // 2 + 1):
                    self.set_px(x + ox, y + oy, color)
            if x == x2i and y == y2i:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy

    def rect(self, x: int, y: int, w: int, h: int, color: tuple[int, int, int]) -> None:
        for yy in range(y, y + h):
            for xx in range(x, x + w):
                self.set_px(xx, yy, color)

    def circle(self, cx: float, cy: float, r: int, color: tuple[int, int, int]) -> None:
        cxi, cyi = int(round(cx)), int(round(cy))
        for y in range(cyi - r, cyi + r + 1):
            for x in range(cxi - r, cxi + r + 1):
                if (x - cxi) ** 2 + (y - cyi) ** 2 <= r * r:
                    self.set_px(x, y, color)

    def text(self, x: int, y: int, text: str, color: tuple[int, int, int], scale: int = 2) -> None:
        cursor = x
        for ch in text.upper():
            pattern = FONT.get(ch, FONT[" "])
            for row, bits in enumerate(pattern):
                for col, bit in enumerate(bits):
                    if bit == "1":
                        self.rect(cursor + col * scale, y + row * scale, scale, scale, color)
            cursor += 6 * scale

    def write_png(self, path: Path) -> None:
        def chunk(tag: bytes, data: bytes) -> bytes:
            return struct.pack("!I", len(data)) + tag + data + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)

        raw = bytearray()
        stride = self.width * 3
        for y in range(self.height):
            raw.append(0)
            raw.extend(self.pixels[y * stride : (y + 1) * stride])
        png = bytearray(b"\x89PNG\r\n\x1a\n")
        png.extend(chunk(b"IHDR", struct.pack("!IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)))
        png.extend(chunk(b"IDAT", zlib.compress(bytes(raw), 9)))
        png.extend(chunk(b"IEND", b""))
        path.write_bytes(bytes(png))


def write_png_line_figure(
    path: Path,
    overall_rows: list[dict[str, object]],
    task_rows: list[dict[str, object]],
    note: str,
) -> None:
    width, height = 2400, 1040
    scale = 2.0
    c = Canvas(width, height)
    panel_w, panel_h = 500 * scale, 340 * scale
    left_a, top = 80 * scale, 65 * scale
    left_b = 670 * scale
    bottom = top + panel_h
    stage_ticks = list(range(2, 11))
    overall_series = collect_series(overall_rows, "model", MODELS)
    task_series = collect_series(task_rows, "task_type", TASKS)
    ymin, ymax = y_domain([overall_series, task_series])

    def px(stage: int, left: float) -> float:
        return left + (stage - 2) / 8.0 * panel_w

    def py(value: float) -> float:
        return bottom - (value - ymin) / (ymax - ymin) * panel_h

    def panel(left: float, title: str) -> None:
        c.text(int(left + 16), int(top + 18), title, (17, 24, 39), 3)
        for frac in [0, 0.25, 0.5, 0.75, 1.0]:
            yval = ymin + (ymax - ymin) * frac
            y = py(yval)
            c.line(left, y, left + panel_w, y, (229, 231, 235), 2)
            c.text(int(left - 90), int(y - 9), f"{yval:.1f}", (17, 24, 39), 2)
        y_one = py(1.0)
        c.line(left, y_one, left + panel_w, y_one, (55, 65, 81), 2)
        c.text(int(left + panel_w - 330), int(y_one - 32), "NO LOSS VS. WRITE-IN", (55, 65, 81), 2)
        c.line(left, top, left, bottom, (17, 24, 39), 2)
        c.line(left, bottom, left + panel_w, bottom, (17, 24, 39), 2)
        for stage in stage_ticks:
            x = px(stage, left)
            c.line(x, bottom, x, bottom + 10, (17, 24, 39), 2)
            c.text(int(x - 18), int(bottom + 22), f"C{stage}", (17, 24, 39), 2)
        c.text(int(left + panel_w / 2 - 115), int(bottom + 78), "CHECKPOINT STAGE", (17, 24, 39), 2)
        c.text(int(left - 135), int(top + panel_h / 2 - 20), "RETENTION", (17, 24, 39), 2)

    def draw_series(left: float, series: dict[str, list[tuple[int, float | None]]], colors: dict[str, str]) -> None:
        for name, points in series.items():
            color = hex_rgb(colors[name])
            valid = [(px(s, left), py(v)) for s, v in points if isinstance(v, float) and not math.isnan(v)]
            for (x1, y1), (x2, y2) in zip(valid, valid[1:]):
                c.line(x1, y1, x2, y2, color, 4)
            for x, y in valid:
                c.circle(x, y, 6, color)

    def legend(left: float, y: float, labels: list[str], colors: dict[str, str], columns: int) -> None:
        col_w = 350 if columns > 1 else 520
        for i, label in enumerate(labels):
            row = i // columns
            col = i % columns
            x = left + col * col_w
            yy = y + row * 45
            color = hex_rgb(colors[label])
            c.line(x, yy, x + 45, yy, color, 4)
            c.circle(x + 22, yy, 6, color)
            c.text(int(x + 60), int(yy - 11), label, (17, 24, 39), 2)

    panel(left_a, "A. OVERALL OLD-STAGE RETENTION RATE")
    panel(left_b, "B. TASK-WISE RETENTION RATE")
    draw_series(left_a, overall_series, MODEL_COLORS)
    draw_series(left_b, task_series, TASK_COLORS)
    legend(left_a, 455 * scale, MODELS, MODEL_COLORS, 1)
    legend(left_b, 455 * scale, TASKS, TASK_COLORS, 3)
    c.text(160, 1010, note, (17, 24, 39), 2)
    c.write_png(path)


def render_svg_raw_ab_three_b(
    path: Path,
    overall_rows: list[dict[str, object]],
    task_by_model_rows: list[dict[str, object]],
    note: str,
) -> None:
    width, height = 1300, 760
    a_left, a_top, a_w, a_h = 80, 105, 480, 420
    b_left, b_top, b_w, b_h, b_gap = 665, 105, 560, 130, 55
    stages = list(range(2, 11))
    overall_series = collect_series(overall_rows, "model", MODELS)
    task_series_by_model = {
        model: collect_series([r for r in task_by_model_rows if r["model"] == model], "task_type", TASKS)
        for model in MODELS
    }
    a_ymin, a_ymax = y_domain([overall_series])
    b_ymin, b_ymax = y_domain(list(task_series_by_model.values()))

    def x_at(stage: int, left: float, panel_w: float) -> float:
        return left + (stage - 2) / 8.0 * panel_w

    def y_at(value: float, top: float, panel_h: float, ymin: float, ymax: float) -> float:
        return top + panel_h - (value - ymin) / (ymax - ymin) * panel_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:'Times New Roman',Times,serif;fill:#111827}.axis{stroke:#111827;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}.line{fill:none;stroke-width:2.1}.legend{font-size:13px}.small{font-size:12px}.label{font-size:14px}.title{font-size:17px;font-weight:700}.subtitle{font-size:14px;font-weight:700}</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
    ]

    def axes(left: float, top: float, panel_w: float, panel_h: float, ymin: float, ymax: float, title: str, xticks: bool, ylabel: bool) -> None:
        bottom = top + panel_h
        parts.append(f'<text class="title" x="{left + 8}" y="{top + 22}">{xml_escape(title)}</text>')
        for frac in [0, 0.25, 0.5, 0.75, 1.0]:
            yval = ymin + (ymax - ymin) * frac
            y = y_at(yval, top, panel_h, ymin, ymax)
            parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + panel_w}" y2="{y:.2f}"/>')
            parts.append(f'<text class="small" x="{left - 8}" y="{y + 4:.2f}" text-anchor="end">{yval:.1f}</text>')
        y_one = y_at(1.0, top, panel_h, ymin, ymax)
        if top <= y_one <= bottom:
            parts.append(f'<line x1="{left}" y1="{y_one:.2f}" x2="{left + panel_w}" y2="{y_one:.2f}" stroke="#374151" stroke-dasharray="6,5" stroke-width="1.2"/>')
        parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>')
        parts.append(f'<line class="axis" x1="{left}" y1="{bottom}" x2="{left + panel_w}" y2="{bottom}"/>')
        for stage in stages:
            x = x_at(stage, left, panel_w)
            parts.append(f'<line class="axis" x1="{x:.2f}" y1="{bottom}" x2="{x:.2f}" y2="{bottom + 4}"/>')
            if xticks:
                parts.append(f'<text class="small" x="{x:.2f}" y="{bottom + 20}" text-anchor="middle">C{stage}</text>')
        if xticks:
            parts.append(f'<text class="label" x="{left + panel_w / 2}" y="{bottom + 44}" text-anchor="middle">Checkpoint stage</text>')
        if ylabel:
            parts.append(
                f'<text class="label" x="{left - 56}" y="{top + panel_h / 2}" text-anchor="middle" transform="rotate(-90 {left - 56} {top + panel_h / 2})">Mean retention rate</text>'
            )

    def draw(left: float, top: float, panel_w: float, panel_h: float, ymin: float, ymax: float, series: dict[str, list[tuple[int, float | None]]], colors: dict[str, str]) -> None:
        for name, points in series.items():
            valid = [
                (x_at(c, left, panel_w), y_at(v, top, panel_h, ymin, ymax))
                for c, v in points
                if isinstance(v, float) and not math.isnan(v)
            ]
            if len(valid) >= 2:
                d = " ".join(("M" if i == 0 else "L") + f"{x:.2f},{y:.2f}" for i, (x, y) in enumerate(valid))
                parts.append(f'<path class="line" d="{d}" stroke="{colors[name]}"/>')
            for x, y in valid:
                parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.1" fill="{colors[name]}"/>')

    axes(a_left, a_top, a_w, a_h, a_ymin, a_ymax, "A. Overall old-stage retention rate", True, True)
    draw(a_left, a_top, a_w, a_h, a_ymin, a_ymax, overall_series, MODEL_COLORS)
    for i, model in enumerate(MODELS):
        top = b_top + i * (b_h + b_gap)
        title = f"B{i + 1}. {model}" if i == 0 else model
        axes(b_left, top, b_w, b_h, b_ymin, b_ymax, title, i == len(MODELS) - 1, i == 1)
        draw(b_left, top, b_w, b_h, b_ymin, b_ymax, task_series_by_model[model], TASK_COLORS)
    parts.append(f'<text class="small" x="{b_left + b_w - 4}" y="{b_top + 8}" text-anchor="end">B. Task-wise retention rate by model</text>')

    # Legends.
    for i, model in enumerate(MODELS):
        x, y = a_left + i * 170, 610
        parts.append(f'<line x1="{x}" y1="{y}" x2="{x + 22}" y2="{y}" stroke="{MODEL_COLORS[model]}" stroke-width="2.4"/>')
        parts.append(f'<circle cx="{x + 11}" cy="{y}" r="3" fill="{MODEL_COLORS[model]}"/>')
        parts.append(f'<text class="legend" x="{x + 30}" y="{y + 4}">{xml_escape(model)}</text>')
    for i, task in enumerate(TASKS):
        x, y = b_left + (i % 3) * 150, 610 + (i // 3) * 24
        parts.append(f'<line x1="{x}" y1="{y}" x2="{x + 22}" y2="{y}" stroke="{TASK_COLORS[task]}" stroke-width="2.4"/>')
        parts.append(f'<circle cx="{x + 11}" cy="{y}" r="3" fill="{TASK_COLORS[task]}"/>')
        parts.append(f'<text class="legend" x="{x + 30}" y="{y + 4}">{task}</text>')
    parts.append(f'<text class="small" x="80" y="725">{xml_escape(note)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_pdf_raw_ab_three_b(
    path: Path,
    overall_rows: list[dict[str, object]],
    task_by_model_rows: list[dict[str, object]],
    note: str,
) -> None:
    width, height = 936, 547
    sx, sy = width / 1300.0, height / 760.0
    a_left, a_top, a_w, a_h = 80 * sx, 105 * sy, 480 * sx, 420 * sy
    b_left, b_top, b_w, b_h, b_gap = 665 * sx, 105 * sy, 560 * sx, 130 * sy, 55 * sy
    stages = list(range(2, 11))
    overall_series = collect_series(overall_rows, "model", MODELS)
    task_series_by_model = {
        model: collect_series([r for r in task_by_model_rows if r["model"] == model], "task_type", TASKS)
        for model in MODELS
    }
    a_ymin, a_ymax = y_domain([overall_series])
    b_ymin, b_ymax = y_domain(list(task_series_by_model.values()))
    cmds: list[str] = [f"1 1 1 rg 0 0 {width} {height} re f"]

    def rgb(hex_color: str) -> tuple[float, float, float]:
        h = hex_color.lstrip("#")
        return int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255

    def yp(y: float) -> float:
        return height - y

    def x_at(stage: int, left: float, panel_w: float) -> float:
        return left + (stage - 2) / 8.0 * panel_w

    def y_at(value: float, top: float, panel_h: float, ymin: float, ymax: float) -> float:
        return top + panel_h - (value - ymin) / (ymax - ymin) * panel_h

    def stroke_line(x1: float, y1: float, x2: float, y2: float, color: str = "#111827", w: float = 1.0, dash: str = "") -> None:
        r, g, b = rgb(color)
        cmds.append(f"{r:.3f} {g:.3f} {b:.3f} RG {w:.2f} w {dash} {x1:.2f} {yp(y1):.2f} m {x2:.2f} {yp(y2):.2f} l S [] 0 d")

    def text(x: float, y: float, s: str, size: float = 8.0, bold: bool = False, anchor: str = "left") -> None:
        width_est = len(s) * size * 0.48
        tx = x - width_est / 2 if anchor == "middle" else x - width_est if anchor == "end" else x
        font = "F2" if bold else "F1"
        cmds.append(f"BT /{font} {size:.1f} Tf {tx:.2f} {yp(y):.2f} Td ({pdf_escape(s)}) Tj ET")

    def marker(x: float, y: float, color: str) -> None:
        r, g, b = rgb(color)
        size = 3.0
        cmds.append(f"{r:.3f} {g:.3f} {b:.3f} rg {x-size/2:.2f} {yp(y)-size/2:.2f} {size:.2f} {size:.2f} re f")

    def axes(left: float, top: float, panel_w: float, panel_h: float, ymin: float, ymax: float, title: str, xticks: bool, ylabel: bool) -> None:
        bottom = top + panel_h
        text(left + 8 * sx, top + 22 * sy, title, 10, True)
        for frac in [0, 0.25, 0.5, 0.75, 1.0]:
            yval = ymin + (ymax - ymin) * frac
            y = y_at(yval, top, panel_h, ymin, ymax)
            stroke_line(left, y, left + panel_w, y, "#e5e7eb", 0.55)
            text(left - 8 * sx, y + 4 * sy, f"{yval:.1f}", 6, anchor="end")
        y_one = y_at(1.0, top, panel_h, ymin, ymax)
        if top <= y_one <= bottom:
            stroke_line(left, y_one, left + panel_w, y_one, "#374151", 0.8, "[4 4] 0 d")
        stroke_line(left, top, left, bottom, "#111827", 0.8)
        stroke_line(left, bottom, left + panel_w, bottom, "#111827", 0.8)
        for stage in stages:
            x = x_at(stage, left, panel_w)
            stroke_line(x, bottom, x, bottom + 4 * sy, "#111827", 0.7)
            if xticks:
                text(x, bottom + 20 * sy, f"C{stage}", 6, anchor="middle")
        if xticks:
            text(left + panel_w / 2, bottom + 44 * sy, "Checkpoint stage", 8, anchor="middle")
        if ylabel:
            text(left - 56 * sx, top + panel_h / 2, "Mean retention rate", 8, anchor="middle")

    def draw(left: float, top: float, panel_w: float, panel_h: float, ymin: float, ymax: float, series: dict[str, list[tuple[int, float | None]]], colors: dict[str, str]) -> None:
        for name, points in series.items():
            valid = [
                (x_at(c, left, panel_w), y_at(v, top, panel_h, ymin, ymax))
                for c, v in points
                if isinstance(v, float) and not math.isnan(v)
            ]
            if len(valid) >= 2:
                r, g, b = rgb(colors[name])
                seg = [f"{r:.3f} {g:.3f} {b:.3f} RG 1.35 w"]
                for i, (x, y) in enumerate(valid):
                    seg.append(f"{x:.2f} {yp(y):.2f} {'m' if i == 0 else 'l'}")
                seg.append("S")
                cmds.append(" ".join(seg))
            for x, y in valid:
                marker(x, y, colors[name])

    axes(a_left, a_top, a_w, a_h, a_ymin, a_ymax, "A. Overall old-stage retention rate", True, True)
    draw(a_left, a_top, a_w, a_h, a_ymin, a_ymax, overall_series, MODEL_COLORS)
    for i, model in enumerate(MODELS):
        top = b_top + i * (b_h + b_gap)
        title = f"B{i + 1}. {model}" if i == 0 else model
        axes(b_left, top, b_w, b_h, b_ymin, b_ymax, title, i == len(MODELS) - 1, i == 1)
        draw(b_left, top, b_w, b_h, b_ymin, b_ymax, task_series_by_model[model], TASK_COLORS)
    text(b_left + b_w, b_top + 8 * sy, "B. Task-wise retention rate by model", 7, anchor="end")
    for i, model in enumerate(MODELS):
        x, y = a_left + i * 170 * sx, 610 * sy
        stroke_line(x, y, x + 22 * sx, y, MODEL_COLORS[model], 1.5)
        marker(x + 11 * sx, y, MODEL_COLORS[model])
        text(x + 30 * sx, y + 4 * sy, model, 7)
    for i, task in enumerate(TASKS):
        x, y = b_left + (i % 3) * 150 * sx, 610 * sy + (i // 3) * 24 * sy
        stroke_line(x, y, x + 22 * sx, y, TASK_COLORS[task], 1.5)
        marker(x + 11 * sx, y, TASK_COLORS[task])
        text(x + 30 * sx, y + 4 * sy, task, 7)
    text(80 * sx, 725 * sy, note, 7)

    stream = "\n".join(cmds).encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] /Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> /Contents 4 0 R >>".encode(),
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf.extend(f"{i} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode())
    for off in offsets[1:]:
        pdf.extend(f"{off:010d} 00000 n \n".encode())
    pdf.extend(f"trailer << /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    path.write_bytes(bytes(pdf))


def write_png_raw_ab_three_b(
    path: Path,
    overall_rows: list[dict[str, object]],
    task_by_model_rows: list[dict[str, object]],
    note: str,
) -> None:
    width, height = 2600, 1520
    scale = 2.0
    c = Canvas(width, height)
    a_left, a_top, a_w, a_h = 80 * scale, 105 * scale, 480 * scale, 420 * scale
    b_left, b_top, b_w, b_h, b_gap = 665 * scale, 105 * scale, 560 * scale, 130 * scale, 55 * scale
    stages = list(range(2, 11))
    overall_series = collect_series(overall_rows, "model", MODELS)
    task_series_by_model = {
        model: collect_series([r for r in task_by_model_rows if r["model"] == model], "task_type", TASKS)
        for model in MODELS
    }
    a_ymin, a_ymax = y_domain([overall_series])
    b_ymin, b_ymax = y_domain(list(task_series_by_model.values()))

    def x_at(stage: int, left: float, panel_w: float) -> float:
        return left + (stage - 2) / 8.0 * panel_w

    def y_at(value: float, top: float, panel_h: float, ymin: float, ymax: float) -> float:
        return top + panel_h - (value - ymin) / (ymax - ymin) * panel_h

    def axes(left: float, top: float, panel_w: float, panel_h: float, ymin: float, ymax: float, title: str, xticks: bool, ylabel: bool) -> None:
        bottom = top + panel_h
        c.text(int(left + 16), int(top + 18), title, (17, 24, 39), 3)
        for frac in [0, 0.25, 0.5, 0.75, 1.0]:
            yval = ymin + (ymax - ymin) * frac
            y = y_at(yval, top, panel_h, ymin, ymax)
            c.line(left, y, left + panel_w, y, (229, 231, 235), 2)
            c.text(int(left - 90), int(y - 9), f"{yval:.1f}", (17, 24, 39), 2)
        y_one = y_at(1.0, top, panel_h, ymin, ymax)
        if top <= y_one <= bottom:
            c.line(left, y_one, left + panel_w, y_one, (55, 65, 81), 2)
        c.line(left, top, left, bottom, (17, 24, 39), 2)
        c.line(left, bottom, left + panel_w, bottom, (17, 24, 39), 2)
        for stage in stages:
            x = x_at(stage, left, panel_w)
            c.line(x, bottom, x, bottom + 8, (17, 24, 39), 2)
            if xticks:
                c.text(int(x - 18), int(bottom + 18), f"C{stage}", (17, 24, 39), 2)
        if xticks:
            c.text(int(left + panel_w / 2 - 115), int(bottom + 72), "CHECKPOINT STAGE", (17, 24, 39), 2)
        if ylabel:
            c.text(int(left - 135), int(top + panel_h / 2 - 20), "RETENTION", (17, 24, 39), 2)

    def draw(left: float, top: float, panel_w: float, panel_h: float, ymin: float, ymax: float, series: dict[str, list[tuple[int, float | None]]], colors: dict[str, str]) -> None:
        for name, points in series.items():
            color = hex_rgb(colors[name])
            valid = [
                (x_at(s, left, panel_w), y_at(v, top, panel_h, ymin, ymax))
                for s, v in points
                if isinstance(v, float) and not math.isnan(v)
            ]
            for (x1, y1), (x2, y2) in zip(valid, valid[1:]):
                c.line(x1, y1, x2, y2, color, 4)
            for x, y in valid:
                c.circle(x, y, 5, color)

    axes(a_left, a_top, a_w, a_h, a_ymin, a_ymax, "A. OVERALL OLD-STAGE RETENTION RATE", True, True)
    draw(a_left, a_top, a_w, a_h, a_ymin, a_ymax, overall_series, MODEL_COLORS)
    for i, model in enumerate(MODELS):
        top = b_top + i * (b_h + b_gap)
        title = f"B{i + 1}. {model}" if i == 0 else model
        axes(b_left, top, b_w, b_h, b_ymin, b_ymax, title, i == len(MODELS) - 1, i == 1)
        draw(b_left, top, b_w, b_h, b_ymin, b_ymax, task_series_by_model[model], TASK_COLORS)
    c.text(int(b_left + b_w - 510), int(b_top - 30), "B. TASK-WISE RETENTION RATE BY MODEL", (17, 24, 39), 2)
    for i, model in enumerate(MODELS):
        x, y = a_left + i * 330, 610 * scale
        color = hex_rgb(MODEL_COLORS[model])
        c.line(x, y, x + 45, y, color, 4)
        c.circle(x + 22, y, 5, color)
        c.text(int(x + 60), int(y - 11), model, (17, 24, 39), 2)
    for i, task in enumerate(TASKS):
        x, y = b_left + (i % 3) * 300, 610 * scale + (i // 3) * 48
        color = hex_rgb(TASK_COLORS[task])
        c.line(x, y, x + 45, y, color, 4)
        c.circle(x + 22, y, 5, color)
        c.text(int(x + 60), int(y - 11), task, (17, 24, 39), 2)
    c.text(160, 1450, note, (17, 24, 39), 2)
    c.write_png(path)


def render_smallmultiples_svg(path: Path, by_model_rows: list[dict[str, object]], note: str) -> None:
    width, height = 1200, 860
    panel_w, panel_h = 980, 190
    left, top0 = 110, 70
    gap = 80
    ymin, ymax = y_domain([collect_series([r for r in by_model_rows if r["model"] == m], "task_type", TASKS) for m in MODELS])

    def px(stage: int) -> float:
        return left + (stage - 2) / 8.0 * panel_w

    def py(value: float, top: int) -> float:
        bottom = top + panel_h
        return bottom - (value - ymin) / (ymax - ymin) * panel_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:'Times New Roman',Times,serif;fill:#111827}.axis{stroke:#111827;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}.line{fill:none;stroke-width:2.1}.legend{font-size:13px}.small{font-size:12px}.title{font-size:17px;font-weight:700}</style>",
        '<rect x="0" y="0" width="1200" height="860" fill="white"/>',
    ]
    for idx, model in enumerate(MODELS):
        top = top0 + idx * (panel_h + gap)
        bottom = top + panel_h
        parts.append(f'<text class="title" x="{left}" y="{top - 22}">{xml_escape(model)}</text>')
        for frac in [0, 0.5, 1.0]:
            yval = ymin + (ymax - ymin) * frac
            y = py(yval, top)
            parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + panel_w}" y2="{y:.2f}"/>')
            parts.append(f'<text class="small" x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">{yval:.1f}</text>')
        y_one = py(1.0, top)
        parts.append(f'<line x1="{left}" y1="{y_one:.2f}" x2="{left + panel_w}" y2="{y_one:.2f}" stroke="#374151" stroke-dasharray="6,5" stroke-width="1.2"/>')
        parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>')
        parts.append(f'<line class="axis" x1="{left}" y1="{bottom}" x2="{left + panel_w}" y2="{bottom}"/>')
        for stage in range(2, 11):
            x = px(stage)
            parts.append(f'<text class="small" x="{x:.2f}" y="{bottom + 18}" text-anchor="middle">C{stage}</text>')
        series = collect_series([r for r in by_model_rows if r["model"] == model], "task_type", TASKS)
        for task, points in series.items():
            valid = [(px(c), py(v, top), c, v) for c, v in points if isinstance(v, float) and not math.isnan(v)]
            if len(valid) >= 2:
                d = " ".join(("M" if i == 0 else "L") + f"{x:.2f},{y:.2f}" for i, (x, y, _, _) in enumerate(valid))
                parts.append(f'<path class="line" d="{d}" stroke="{TASK_COLORS[task]}"/>')
            for x, y, _, _ in valid:
                parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.0" fill="{TASK_COLORS[task]}"/>')
    legend_y = 820
    for i, task in enumerate(TASKS):
        x = 210 + i * 135
        parts.append(f'<line x1="{x}" y1="{legend_y}" x2="{x + 22}" y2="{legend_y}" stroke="{TASK_COLORS[task]}" stroke-width="2.4"/>')
        parts.append(f'<text class="legend" x="{x + 30}" y="{legend_y + 4}">{task}</text>')
    parts.append(f'<text class="small" x="110" y="850">{xml_escape(note)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_pdf_smallmultiples(path: Path, by_model_rows: list[dict[str, object]], note: str) -> None:
    width, height = 864, 620
    sx, sy = width / 1200.0, height / 860.0
    panel_w, panel_h = 980 * sx, 190 * sy
    left, top0 = 110 * sx, 70 * sy
    gap = 80 * sy
    ymin, ymax = y_domain([collect_series([r for r in by_model_rows if r["model"] == m], "task_type", TASKS) for m in MODELS])

    def rgb(hex_color: str) -> tuple[float, float, float]:
        h = hex_color.lstrip("#")
        return int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255

    def px(stage: int) -> float:
        return left + (stage - 2) / 8.0 * panel_w

    def py(value: float, top: float) -> float:
        bottom = top + panel_h
        return bottom - (value - ymin) / (ymax - ymin) * panel_h

    def yp(y: float) -> float:
        return height - y

    cmds: list[str] = [f"1 1 1 rg 0 0 {width} {height} re f"]

    def stroke_line(x1: float, y1: float, x2: float, y2: float, color: str = "#111827", w: float = 1.0, dash: str = "") -> None:
        r, g, b = rgb(color)
        cmds.append(f"{r:.3f} {g:.3f} {b:.3f} RG {w:.2f} w {dash} {x1:.2f} {yp(y1):.2f} m {x2:.2f} {yp(y2):.2f} l S [] 0 d")

    def text(x: float, y: float, s: str, size: float = 9, bold: bool = False, anchor: str = "left") -> None:
        width_est = len(s) * size * 0.48
        tx = x - width_est / 2 if anchor == "middle" else x - width_est if anchor == "end" else x
        font = "F2" if bold else "F1"
        cmds.append(f"BT /{font} {size:.1f} Tf {tx:.2f} {yp(y):.2f} Td ({pdf_escape(s)}) Tj ET")

    def marker(x: float, y: float, color: str) -> None:
        r, g, b = rgb(color)
        size = 3.2
        cmds.append(f"{r:.3f} {g:.3f} {b:.3f} rg {x-size/2:.2f} {yp(y)-size/2:.2f} {size:.2f} {size:.2f} re f")

    for idx, model in enumerate(MODELS):
        top = top0 + idx * (panel_h + gap)
        bottom = top + panel_h
        text(left, top - 22 * sy, model, 12, True)
        for frac in [0, 0.5, 1.0]:
            yval = ymin + (ymax - ymin) * frac
            y = py(yval, top)
            stroke_line(left, y, left + panel_w, y, "#e5e7eb", 0.6)
            text(left - 10 * sx, y + 4 * sy, f"{yval:.1f}", 7, anchor="end")
        y_one = py(1.0, top)
        stroke_line(left, y_one, left + panel_w, y_one, "#374151", 0.8, "[4 4] 0 d")
        stroke_line(left, top, left, bottom, "#111827", 0.8)
        stroke_line(left, bottom, left + panel_w, bottom, "#111827", 0.8)
        for stage in range(2, 11):
            x = px(stage)
            text(x, bottom + 18 * sy, f"C{stage}", 7, anchor="middle")
        series = collect_series([r for r in by_model_rows if r["model"] == model], "task_type", TASKS)
        for task, points in series.items():
            valid = [(px(c), py(v, top)) for c, v in points if isinstance(v, float) and not math.isnan(v)]
            if len(valid) >= 2:
                r, g, b = rgb(TASK_COLORS[task])
                seg = [f"{r:.3f} {g:.3f} {b:.3f} RG 1.4 w"]
                for i, (x, y) in enumerate(valid):
                    seg.append(f"{x:.2f} {yp(y):.2f} {'m' if i == 0 else 'l'}")
                seg.append("S")
                cmds.append(" ".join(seg))
            for x, y in valid:
                marker(x, y, TASK_COLORS[task])
    legend_y = 820 * sy
    for i, task in enumerate(TASKS):
        x = (210 + i * 135) * sx
        stroke_line(x, legend_y, x + 22 * sx, legend_y, TASK_COLORS[task], 1.6)
        text(x + 30 * sx, legend_y + 4 * sy, task, 8)
    text(110 * sx, 850 * sy, note, 7)

    stream = "\n".join(cmds).encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] /Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> /Contents 4 0 R >>".encode(),
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf.extend(f"{i} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode())
    for off in offsets[1:]:
        pdf.extend(f"{off:010d} 00000 n \n".encode())
    pdf.extend(f"trailer << /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    path.write_bytes(bytes(pdf))


def write_png_smallmultiples(path: Path, by_model_rows: list[dict[str, object]], note: str) -> None:
    width, height = 2400, 1720
    scale = 2.0
    c = Canvas(width, height)
    panel_w, panel_h = 980 * scale, 190 * scale
    left, top0 = 110 * scale, 70 * scale
    gap = 80 * scale
    ymin, ymax = y_domain([collect_series([r for r in by_model_rows if r["model"] == m], "task_type", TASKS) for m in MODELS])

    def px(stage: int) -> float:
        return left + (stage - 2) / 8.0 * panel_w

    def py(value: float, top: float) -> float:
        bottom = top + panel_h
        return bottom - (value - ymin) / (ymax - ymin) * panel_h

    for idx, model in enumerate(MODELS):
        top = top0 + idx * (panel_h + gap)
        bottom = top + panel_h
        c.text(int(left), int(top - 44), model, (17, 24, 39), 3)
        for frac in [0, 0.5, 1.0]:
            yval = ymin + (ymax - ymin) * frac
            y = py(yval, top)
            c.line(left, y, left + panel_w, y, (229, 231, 235), 2)
            c.text(int(left - 90), int(y - 9), f"{yval:.1f}", (17, 24, 39), 2)
        y_one = py(1.0, top)
        c.line(left, y_one, left + panel_w, y_one, (55, 65, 81), 2)
        c.line(left, top, left, bottom, (17, 24, 39), 2)
        c.line(left, bottom, left + panel_w, bottom, (17, 24, 39), 2)
        for stage in range(2, 11):
            x = px(stage)
            c.text(int(x - 18), int(bottom + 22), f"C{stage}", (17, 24, 39), 2)
        series = collect_series([r for r in by_model_rows if r["model"] == model], "task_type", TASKS)
        for task, points in series.items():
            color = hex_rgb(TASK_COLORS[task])
            valid = [(px(s), py(v, top)) for s, v in points if isinstance(v, float) and not math.isnan(v)]
            for (x1, y1), (x2, y2) in zip(valid, valid[1:]):
                c.line(x1, y1, x2, y2, color, 4)
            for x, y in valid:
                c.circle(x, y, 5, color)
    legend_y = int(820 * scale)
    for i, task in enumerate(TASKS):
        x = int((210 + i * 135) * scale)
        color = hex_rgb(TASK_COLORS[task])
        c.line(x, legend_y, x + 45, legend_y, color, 4)
        c.text(x + 60, legend_y - 11, task, (17, 24, 39), 2)
    c.text(int(110 * scale), int(850 * scale), note, (17, 24, 39), 2)
    c.write_png(path)


def write_denominator_report(rows: list[dict[str, object]]) -> None:
    diagonal: dict[tuple[str, str, int], float] = {}
    for row in rows:
        model, task, stage = row["model"], row["task_type"], row["eval_stage"]
        if row["checkpoint_stage"] == row["eval_stage"]:
            continue
        # Fill from repeated old cells' baseline; same baseline appears multiple times.
        b = row["baseline_diagonal_score"]
        if isinstance(b, float):
            diagonal[(str(model), str(task), int(stage))] = b

    # Prefer direct unique baseline extraction from cellwise rows.
    by_model_task: dict[tuple[str, str], dict[int, float]] = {}
    for row in rows:
        b = row["baseline_diagonal_score"]
        if isinstance(b, float):
            by_model_task.setdefault((str(row["model"]), str(row["task_type"])), {})[int(row["eval_stage"])] = b

    lines = ["# Retention Rate Denominator Stability Report", ""]
    lines.append("Input denominator is the current-stage diagonal write-in score `B(m,s,t)` from the held-out-character split.")
    lines.append("")
    lines.append("## Baseline diagonal score ranges")
    lines.append("")
    lines.append("| model | task_type | min | max | stages_with_denominator | count_<=0 | count_<0.03 | count_<0.05 |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    low_03, low_05, nonpos = [], [], []
    for model in MODELS:
        for task in ALL_TASKS:
            values_by_stage = by_model_task.get((model, task), {})
            vals = [values_by_stage[s] for s in sorted(values_by_stage)]
            count_nonpos = sum(v <= 0 for v in vals)
            count_03 = sum(0 < v < 0.03 for v in vals)
            count_05 = sum(0 < v < 0.05 for v in vals)
            for s, v in values_by_stage.items():
                if v <= 0:
                    nonpos.append((model, task, s, v))
                if 0 < v < 0.03:
                    low_03.append((model, task, s, v))
                if 0 < v < 0.05:
                    low_05.append((model, task, s, v))
            lines.append(
                f"| {model} | {task} | {min(vals) if vals else float('nan'):.6f} | {max(vals) if vals else float('nan'):.6f} | {len(vals)} | {count_nonpos} | {count_03} | {count_05} |"
            )
    lines.append("")
    lines.append("## Denominator warnings")
    lines.append("")
    lines.append(f"- denominator <= 0 cells: {len(nonpos)} unique model/task/stage denominators")
    lines.append(f"- 0 < denominator < 0.03 cells: {len(low_03)} unique model/task/stage denominators")
    lines.append(f"- 0 < denominator < 0.05 cells: {len(low_05)} unique model/task/stage denominators")
    if nonpos:
        lines.append("- <=0 denominators: " + "; ".join(f"{m}/{t}/S{s}={v:.6f}" for m, t, s, v in nonpos[:40]))
    if low_05:
        lines.append("- <0.05 denominators: " + "; ".join(f"{m}/{t}/S{s}={v:.6f}" for m, t, s, v in low_05[:60]))

    valid_rates = [r for r in rows if valid_rate(r)]
    gt2 = [r for r in valid_rates if r["retention_rate"] > 2]  # type: ignore[operator]
    gt3 = [r for r in valid_rates if r["retention_rate"] > 3]  # type: ignore[operator]
    lines.append("")
    lines.append("## Raw retention-rate anomalies")
    lines.append("")
    lines.append(f"- raw retention_rate > 2: {len(gt2)} cells")
    lines.append(f"- raw retention_rate > 3: {len(gt3)} cells")
    if gt2:
        examples = []
        for r in gt2[:30]:
            examples.append(
                f"{r['model']}/{r['task_type']}/C{r['checkpoint_stage']}/S{r['eval_stage']}="
                f"{r['retention_rate']:.3f} (B={r['baseline_diagonal_score']:.6f})"
            )
        lines.append("- Examples: " + "; ".join(examples))
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    if low_05 or gt2:
        lines.append(
            "Use `filtered005` for the appendix task-wise trajectory because ES/TR/MA include zero or tiny write-in denominators, and raw ratios contain >2 or >3 artifacts. The overall trajectory is stable because all overall denominators are >= 0.05."
        )
    else:
        lines.append("Raw retention-rate trajectories are stable under the denominator checks; filtered005 is a sensitivity analysis.")
    (OUT / "retention_rate_denominator_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def md_table(rows: list[dict[str, object]], cols: list[str], max_rows: int = 20) -> str:
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows[:max_rows]:
        cells = []
        for col in cols:
            val = row.get(col, "")
            if val is None:
                cells.append("NaN")
            elif isinstance(val, float):
                cells.append(fnum(val))
            else:
                cells.append(str(val))
        out.append("| " + " | ".join(cells) + " |")
    if len(rows) > max_rows:
        out.append(f"\nShowing first {max_rows} of {len(rows)} rows.\n")
    return "\n".join(out)


def write_summary_report(
    meta: dict[str, object],
    cellwise: list[dict[str, object]],
    overall_raw: list[dict[str, object]],
    overall_filtered: list[dict[str, object]],
    task_raw: list[dict[str, object]],
    task_filtered: list[dict[str, object]],
) -> None:
    def model_global(rows: list[dict[str, object]]) -> dict[str, float]:
        out = {}
        for model in MODELS:
            vals = [r["mean_retention_rate"] for r in rows if r["model"] == model and isinstance(r["mean_retention_rate"], float)]
            out[model] = statistics.mean(vals) if vals else math.nan
        return out

    def task_global(rows: list[dict[str, object]]) -> dict[str, float]:
        out = {}
        for task in TASKS:
            vals = [r["mean_retention_rate"] for r in rows if r["task_type"] == task and isinstance(r["mean_retention_rate"], float)]
            out[task] = statistics.mean(vals) if vals else math.nan
        return out

    model_means = model_global(overall_filtered)
    cell_weighted_model_means: dict[str, float] = {}
    for model in MODELS:
        vals = [
            r["retention_rate"]
            for r in cellwise
            if r["model"] == model and r["task_type"] == "overall" and valid_rate(r, 0.05)
        ]
        cell_weighted_model_means[model] = statistics.mean(vals) if vals else math.nan  # type: ignore[arg-type]
    best_model = max(model_means, key=lambda m: model_means[m])
    task_means = task_global(task_filtered)
    best_task = max(task_means, key=lambda t: task_means[t])
    worst_task = min(task_means, key=lambda t: task_means[t])

    slopes = {}
    for model in MODELS:
        pts = [(r["checkpoint_stage"], r["mean_retention_rate"]) for r in overall_filtered if r["model"] == model]
        xs = [float(x) for x, y in pts if isinstance(y, float)]
        ys = [float(y) for x, y in pts if isinstance(y, float)]
        slopes[model] = slope(xs, ys)

    raw_gt2 = sum(1 for r in cellwise if valid_rate(r) and r["retention_rate"] > 2)  # type: ignore[operator]
    raw_gt3 = sum(1 for r in cellwise if valid_rate(r) and r["retention_rate"] > 3)  # type: ignore[operator]
    dropped_by_task = {}
    for task in TASKS:
        raw_candidates = [r for r in cellwise if r["task_type"] == task]
        filtered_candidates = [r for r in raw_candidates if valid_rate(r, 0.05)]
        dropped_by_task[task] = len(raw_candidates) - len(filtered_candidates)

    lines = [
        "# Retention Rate Analysis Report",
        "",
        "## Data source and split",
        "",
        f"- Input priority used: `{meta['input_priority_used']}`.",
        f"- Overall matrix: `{meta['overall_matrix']}`.",
        f"- Taskwise matrix: `{meta['taskwise_matrix']}`.",
        f"- Protocol inventory: `{meta['protocol_inventory']}`.",
        f"- Split filter: `{SPLIT}` only. This corresponds to the held-out-character / unseen evaluation split used by the main stagewise protocol.",
        f"- Split values observed in source matrices: `{meta['split_values_seen']}`.",
        "- No model inference was run; all values come from saved retention matrix CSVs.",
        "",
        "## Main answers",
        "",
        f"1. Highest mean old-stage retention rate: `{best_model}` under the recommended `filtered005` overall trajectory (mean over C2-C10 panel points = {model_means[best_model]:.3f}; cell-weighted old-stage mean = {cell_weighted_model_means[best_model]:.3f}).",
        "2. Retention-rate trend with checkpoint stage: slopes on filtered overall trajectories are "
        + ", ".join(f"{m}: {slopes[m]:.4f}" for m in MODELS)
        + ". Negative slopes indicate decline with later checkpoints; near-zero/positive slopes indicate no monotonic decline under this normalized ratio.",
        f"3. Best retained task type under filtered task-wise means: `{best_task}` ({task_means[best_task]:.3f}); worst: `{worst_task}` ({task_means[worst_task]:.3f}).",
        "4. Unreliable task ratios: ES/TR/MA have zero or tiny current-stage write-in denominators; ES is especially unstable. See `retention_rate_denominator_report.md`.",
        "5. Replacement rationale: this analysis is better suited than current-stage trajectories for the appendix because it measures memory retained for old stages after later training, normalized by each stage's original write-in score.",
        "6. Recommended appendix version: use `filtered005` for the main appendix figure. Raw outputs are included as sensitivity files because raw task-wise ratios contain denominator-driven spikes.",
        "",
        "## Filtered overall retention by model",
        "",
        md_table(overall_filtered, ["model", "checkpoint_stage", "mean_retention_rate", "std_retention_rate", "num_valid_old_stages"], 40),
        "",
        "## Filtered task-wise mean retention",
        "",
        md_table(task_filtered, ["task_type", "checkpoint_stage", "mean_retention_rate", "std_retention_rate", "num_valid_cells"], 60),
        "",
        "## Filter sensitivity",
        "",
        f"- Raw cellwise old-stage rows: {len(cellwise)}.",
        f"- Raw valid ratio cells: {sum(1 for r in cellwise if valid_rate(r))}.",
        f"- Filtered005 valid ratio cells: {sum(1 for r in cellwise if valid_rate(r, 0.05))}.",
        f"- Raw retention_rate > 2 cells: {raw_gt2}; > 3 cells: {raw_gt3}.",
        "- Dropped cells by task under filtered005: " + ", ".join(f"{t}: {n}" for t, n in dropped_by_task.items()) + ".",
        "",
        "## Output files",
        "",
        "- Cell-level CSVs: `retention_rate_cellwise.csv`, `retention_rate_cellwise_filtered005.csv`.",
        "- Aggregates: `retention_rate_overall_by_model*.csv`, `retention_rate_taskwise_mean*.csv`, `retention_rate_taskwise_by_model*.csv`.",
        "- Figures: `figure_appendix_retention_rate_AB.{pdf,png,svg}` (filtered005 primary), `figure_appendix_retention_rate_AB_raw.{pdf,png,svg}` (raw sensitivity), and optional small-multiple taskwise-by-model figure.",
    ]
    (OUT / "retention_rate_analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    scores, meta = read_scores()
    cellwise = build_cellwise(scores)
    cellwise_filtered = filtered_rows(cellwise, 0.05)

    overall_raw = summarize_overall_by_model(cellwise)
    task_mean_raw = summarize_taskwise_mean(cellwise)
    task_by_model_raw = summarize_taskwise_by_model(cellwise)
    overall_filtered = summarize_overall_by_model(cellwise, 0.05)
    task_mean_filtered = summarize_taskwise_mean(cellwise, 0.05)
    task_by_model_filtered = summarize_taskwise_by_model(cellwise, 0.05)

    write_csv(
        OUT / "retention_rate_cellwise.csv",
        cellwise,
        [
            "model",
            "checkpoint_stage",
            "eval_stage",
            "task_type",
            "baseline_diagonal_score",
            "old_stage_score",
            "retention_rate",
            "split",
            "denominator_valid",
            "notes",
        ],
    )
    write_csv(
        OUT / "retention_rate_overall_by_model.csv",
        overall_raw,
        ["model", "checkpoint_stage", "mean_retention_rate", "std_retention_rate", "num_old_stages", "num_valid_old_stages"],
    )
    write_csv(
        OUT / "retention_rate_taskwise_mean.csv",
        task_mean_raw,
        ["task_type", "checkpoint_stage", "mean_retention_rate", "std_retention_rate", "num_model_stage_cells", "num_valid_cells"],
    )
    write_csv(
        OUT / "retention_rate_taskwise_by_model.csv",
        task_by_model_raw,
        ["model", "task_type", "checkpoint_stage", "mean_retention_rate", "std_retention_rate", "num_old_stages", "num_valid_old_stages"],
    )
    write_csv(
        OUT / "retention_rate_cellwise_filtered005.csv",
        cellwise_filtered,
        [
            "model",
            "checkpoint_stage",
            "eval_stage",
            "task_type",
            "baseline_diagonal_score",
            "old_stage_score",
            "retention_rate",
            "split",
            "denominator_valid",
            "notes",
        ],
    )
    write_csv(
        OUT / "retention_rate_overall_by_model_filtered005.csv",
        overall_filtered,
        ["model", "checkpoint_stage", "mean_retention_rate", "std_retention_rate", "num_old_stages", "num_valid_old_stages"],
    )
    write_csv(
        OUT / "retention_rate_taskwise_mean_filtered005.csv",
        task_mean_filtered,
        ["task_type", "checkpoint_stage", "mean_retention_rate", "std_retention_rate", "num_model_stage_cells", "num_valid_cells"],
    )
    write_csv(
        OUT / "retention_rate_taskwise_by_model_filtered005.csv",
        task_by_model_filtered,
        ["model", "task_type", "checkpoint_stage", "mean_retention_rate", "std_retention_rate", "num_old_stages", "num_valid_old_stages"],
    )

    write_denominator_report(cellwise)
    write_summary_report(meta, cellwise, overall_raw, overall_filtered, task_mean_raw, task_mean_filtered)

    note_filtered = "Filtered005: cells with baseline diagonal score < 0.05 are excluded from means."
    note_raw = "Raw ratios: no baseline filter; denominator-driven spikes are possible."
    render_svg_line_figure(OUT / "figure_appendix_retention_rate_AB.svg", overall_filtered, task_mean_filtered, note_filtered)
    write_pdf_line_figure(OUT / "figure_appendix_retention_rate_AB.pdf", overall_filtered, task_mean_filtered, note_filtered)
    write_png_line_figure(OUT / "figure_appendix_retention_rate_AB.png", overall_filtered, task_mean_filtered, note_filtered)
    render_svg_raw_ab_three_b(OUT / "figure_appendix_retention_rate_AB_raw.svg", overall_raw, task_by_model_raw, note_raw)
    write_pdf_raw_ab_three_b(OUT / "figure_appendix_retention_rate_AB_raw.pdf", overall_raw, task_by_model_raw, note_raw)
    write_png_raw_ab_three_b(OUT / "figure_appendix_retention_rate_AB_raw.png", overall_raw, task_by_model_raw, note_raw)

    # Optional companion: the SVG contains the full small-multiple plot. PDF is a lightweight pointer
    # because the main appendix replacement is the AB figure above.
    render_smallmultiples_svg(
        OUT / "figure_appendix_retention_rate_taskwise_by_model_smallmultiples.svg",
        task_by_model_filtered,
        note_filtered,
    )
    write_pdf_smallmultiples(
        OUT / "figure_appendix_retention_rate_taskwise_by_model_smallmultiples.pdf",
        task_by_model_filtered,
        note_filtered,
    )
    write_png_smallmultiples(
        OUT / "figure_appendix_retention_rate_taskwise_by_model_smallmultiples.png",
        task_by_model_filtered,
        note_filtered,
    )

    print("wrote retention-rate analysis to", OUT)
    print("cellwise_rows", len(cellwise), "filtered_rows", len(cellwise_filtered))
    print("valid_raw", sum(1 for r in cellwise if valid_rate(r)), "valid_filtered005", sum(1 for r in cellwise if valid_rate(r, 0.05)))


if __name__ == "__main__":
    main()
