from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_EVAL_DIR = Path(__file__).resolve().parent.parent / "dialogue_gen_api" / "evaluation"


def _load_eval_modules():
    if str(_EVAL_DIR) not in sys.path:
        sys.path.insert(0, str(_EVAL_DIR))

    import metrics  # type: ignore
    import run_eval  # type: ignore

    return metrics, run_eval


def load_memory_eval_jsonl(dataset_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(dataset_path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_first_json_object(text: str) -> Optional[str]:
    raw = str(text or "").strip()
    if not raw:
        return None

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(raw):
        if ch != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(raw[idx:])
        except Exception:
            continue
        if isinstance(parsed, dict):
            return json.dumps(parsed, ensure_ascii=False)
    return None


def score_result_rows(
    result_rows: List[Dict[str, Any]],
    judge_llm: Any = None,
    semantic_workers: int = 10,
    task_scoring_scheme: str = "A",
    judge_votes: int = 2,
    judge_tiebreak: bool = True,
    binary_fallback_judge: bool = True,
    scoring_config_path: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not result_rows:
        return [], {}

    metrics, run_eval = _load_eval_modules()

    extracted_json_responses = [_extract_first_json_object(row.get("response", "")) for row in result_rows]
    responses = [extracted or row["response"] for row, extracted in zip(result_rows, extracted_json_responses)]
    references = [row["reference"] for row in result_rows]
    metadatas = [row.get("metadata", {}) for row in result_rows]

    evaluated_rows, _ = metrics.evaluate_batch(
        responses,
        references,
        metadatas=metadatas,
        judge_llm=judge_llm,
        compute_semantic=False,
        semantic_workers=semantic_workers,
        eval_profile="umb_tasklight_v1",
        task_scoring_scheme=task_scoring_scheme,
        judge_votes=judge_votes,
        judge_tiebreak=judge_tiebreak,
        binary_fallback_judge=binary_fallback_judge,
        scoring_config_path=scoring_config_path,
    )

    enriched: List[Dict[str, Any]] = []
    for base, row, scoring_response, extracted_json in zip(
        result_rows,
        evaluated_rows,
        responses,
        extracted_json_responses,
    ):
        merged = dict(base)
        merged["scoring_response"] = scoring_response
        merged["extracted_json_response"] = extracted_json
        merged["scores"] = row.get("scores", {})
        merged["parsed_output"] = row.get("parsed_output")
        merged["parse_ok"] = row.get("parse_ok")
        merged["rule_signals"] = row.get("rule_signals")
        merged["judge_band"] = row.get("judge_band")
        merged["judge_score"] = row.get("judge_score")
        merged["final_score"] = row.get("final_score", row.get("scores", {}).get("final_score"))
        merged["score_source"] = row.get("score_source")
        merged["judge_meta"] = row.get("judge_meta", {})
        enriched.append(merged)

    aggregate = run_eval.compute_aggregate_scores(enriched)
    return enriched, aggregate
