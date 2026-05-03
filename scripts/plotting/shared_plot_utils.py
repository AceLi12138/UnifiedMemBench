from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib


PAPER_ROOT = Path(__file__).resolve().parents[1]
if str(PAPER_ROOT) not in sys.path:
    sys.path.insert(0, str(PAPER_ROOT))

from model_audit.model_name_registry import get_model_record  # type: ignore


def apply_publication_rcparams(extra: Dict[str, Any] | None = None) -> None:
    rc = {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "Nimbus Roman No9 L",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "text.usetex": False,
    }
    if extra:
        rc.update(extra)
    matplotlib.rcParams.update(rc)


def get_figure_output_dir(default_dir: Path) -> Path:
    override = os.environ.get("UMB_FIGURE_OUTPUT_DIR", "").strip()
    out_dir = Path(override) if override else Path(default_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def get_display_label(short_label: str) -> str:
    return str(get_model_record(short_label)["display_label"])


def get_official_name(short_label: str) -> str:
    return str(get_model_record(short_label)["official_model_name"])


def wrap_display_label(short_label: str) -> str:
    label = get_display_label(short_label)
    wrapped = {
        "GLM-4-9B-Chat-1M": "GLM-4-9B-\nChat-1M",
        "Qwen2.5-7B-Instruct-1M": "Qwen2.5-7B-\nInstruct-1M",
        "Llama-4-Scout": "Llama-4-\nScout",
        "Llama-4-Scout-17B-16E-Instruct": "Llama-4-Scout\n17B-16E-Instruct",
        "Hunyuan-A13B-Instruct": "Hunyuan-A13B-\nInstruct",
        "Hunyuan-A13B-Instruct-256k": "Hunyuan-A13B-\nInstruct-256k",
        "Hunyuan-A13B-Instruct-256k-nothink": "Hunyuan-A13B-\nInstruct-256k",
        "Gemma-4-26B-A4B-it": "Gemma-4-26B-\nA4B-it",
        "Ministral-3-8B-Instruct-2512": "Ministral-3-8B-\nInstruct-2512",
    }
    return wrapped.get(label, label)
