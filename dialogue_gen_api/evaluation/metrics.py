"""
UMB evaluation metrics.

Profiles:
- legacy_v2: original task-specific metrics (component_recall/llm_judge/etc.)
- umb_tasklight_v1: lightweight scheme A with strict JSON parsing and per-task scoring
"""

from __future__ import annotations

import asyncio
import json
import re
import statistics
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from rapidfuzz import fuzz
except ImportError:
    import difflib

    class _FuzzFallback:
        @staticmethod
        def partial_ratio(a: str, b: str) -> float:
            s1 = str(a or "")
            s2 = str(b or "")
            if not s1 or not s2:
                return 0.0
            shorter, longer = (s1, s2) if len(s1) <= len(s2) else (s2, s1)
            window = len(shorter)
            best = 0.0
            for i in range(0, len(longer) - window + 1):
                chunk = longer[i:i + window]
                ratio = difflib.SequenceMatcher(None, shorter, chunk).ratio()
                if ratio > best:
                    best = ratio
            return best * 100.0

    fuzz = _FuzzFallback()


ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
BINARY_TASKS = {
    "Information Extraction",
    "Temporal Reasoning",
    "Knowledge Updating",
}
THREE_LEVEL_TASKS = {
    "Multi-session Reasoning",
    "Event Summarization",
    "Memory Arbitration",
}
ALL_TASKS = BINARY_TASKS | THREE_LEVEL_TASKS
SCORE_TO_BAND = {
    1.0: "correct",
    0.5: "partial",
    0.0: "wrong",
}
FOCUS_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "them",
    "there",
    "they",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}
FOCUS_SCAFFOLD_TOKENS = {
    "action",
    "answer",
    "author",
    "authors",
    "behavior",
    "business",
    "city",
    "completed",
    "correction",
    "current",
    "date",
    "decision",
    "destination",
    "difference",
    "dog",
    "gift",
    "include",
    "includes",
    "initiated",
    "item",
    "items",
    "learned",
    "location",
    "method",
    "name",
    "nature",
    "new",
    "object",
    "project",
    "quality",
    "relationship",
    "report",
    "requirement",
    "result",
    "rule",
    "shift",
    "specific",
    "status",
    "tradition",
    "types",
}
UNSPECIFIED_MARKERS = {
    "does not specify",
    "did not specify",
    "not specified",
    "not mention",
    "does not mention",
    "did not mention",
    "not provided",
    "not given",
    "unknown",
    "unclear",
}
SLOT_SCAFFOLD_TOKENS = {
    "art",
    "form",
    "trip",
    "travel",
    "planned",
    "plan",
    "going",
    "with",
    "item",
    "items",
    "object",
    "objects",
    "type",
    "types",
    "state",
    "status",
    "latest",
    "current",
}


def _mean(values: List[float]) -> float:
    return float(statistics.fmean(values))


def _std(values: List[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def ensure_nltk_data():
    """Download required NLTK data if not available."""
    try:
        import nltk
    except ImportError:
        return

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


ensure_nltk_data()


@lru_cache(maxsize=4)
def load_task_scoring_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    path = (
        Path(config_path)
        if config_path
        else Path(__file__).parent / "task_scoring_v1.json"
    )
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            pass

    return {
        "profile": "umb_tasklight_v1",
        "task_scoring_scheme": "A",
        "tasks": {
            "Information Extraction": {
                "score_mode": "binary",
                "required_fields": ["answer", "evidence_snippets"],
            },
            "Temporal Reasoning": {
                "score_mode": "binary",
                "required_fields": ["final_answer", "days", "start_date", "end_date"],
            },
            "Knowledge Updating": {
                "score_mode": "binary",
                "required_fields": ["latest_state", "as_of_time", "deprecated_state"],
            },
            "Multi-session Reasoning": {
                "score_mode": "three_level",
                "required_fields": ["event_chain", "final_outcome"],
            },
            "Event Summarization": {
                "score_mode": "three_level",
                "required_fields": ["time_span", "key_turning_points", "summary"],
            },
            "Memory Arbitration": {
                "score_mode": "three_level",
                "required_fields": [
                    "premise_verdict",
                    "premise_error",
                    "corrected_facts",
                    "final_answer",
                ],
            },
        },
    }


def _required_fields(task_type: str, cfg: Dict[str, Any]) -> List[str]:
    tasks = cfg.get("tasks", {})
    task_cfg = tasks.get(task_type, {})
    fields = task_cfg.get("required_fields", [])
    if isinstance(fields, list):
        return [str(x) for x in fields]
    return []


def _score_mode(task_type: str, cfg: Dict[str, Any]) -> str:
    tasks = cfg.get("tasks", {})
    task_cfg = tasks.get(task_type, {})
    mode = str(task_cfg.get("score_mode", "binary")).strip().lower()
    if mode in {"binary", "three_level"}:
        return mode
    if task_type in THREE_LEVEL_TASKS:
        return "three_level"
    return "binary"


def _strip_code_fence(text: str) -> str:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def _safe_json_loads(text: str) -> Tuple[Optional[Any], Optional[str]]:
    cleaned = _strip_code_fence(text)
    if not cleaned:
        return None, "empty_response"
    try:
        return json.loads(cleaned), None
    except Exception:
        return None, "invalid_json"


def _parse_task_output(
    response_text: str,
    task_type: str,
    cfg: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], bool, List[str], str]:
    parsed, err = _safe_json_loads(response_text)
    required = _required_fields(task_type, cfg)
    if parsed is None or not isinstance(parsed, dict):
        return None, False, required, err or "invalid_json"

    missing = [field for field in required if field not in parsed]
    if missing:
        return parsed, False, missing, "missing_required_fields"
    return parsed, True, [], ""


def _normalize_text(text: Any) -> str:
    t = str(text or "").lower()
    t = re.sub(r"[^a-z0-9\s\-:]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _soft_required_defaults(task_type: str) -> Dict[str, Any]:
    if task_type == "Information Extraction":
        return {"evidence_snippets": []}
    if task_type == "Knowledge Updating":
        return {"as_of_time": "", "deprecated_state": ""}
    return {}


def _apply_soft_required_fields(
    task_type: str,
    parsed_output: Optional[Dict[str, Any]],
    parse_ok: bool,
    missing_fields: List[str],
    parse_error: str,
) -> Tuple[Optional[Dict[str, Any]], bool, List[str], List[str], str]:
    if parse_ok or not isinstance(parsed_output, dict):
        return parsed_output, parse_ok, missing_fields, [], parse_error

    defaults = _soft_required_defaults(task_type)
    if not defaults or parse_error != "missing_required_fields":
        return parsed_output, parse_ok, missing_fields, [], parse_error

    soft_missing: List[str] = []
    hard_missing: List[str] = []
    for field in missing_fields:
        if field in defaults:
            parsed_output.setdefault(field, defaults[field])
            soft_missing.append(field)
        else:
            hard_missing.append(field)

    if hard_missing:
        return parsed_output, False, hard_missing, soft_missing, parse_error
    return parsed_output, True, [], soft_missing, ""


def _simplify_focus_token(token: str) -> str:
    t = str(token or "").strip()
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 4 and t.endswith("es"):
        return t[:-2]
    if len(t) > 4 and t.endswith("s"):
        return t[:-1]
    return t


def _focus_tokens(text: Any) -> List[str]:
    raw = _normalize_text(text)
    tokens: List[str] = []
    for token in raw.split():
        token = _simplify_focus_token(token)
        if len(token) <= 1 or token.isdigit():
            continue
        if token in FOCUS_STOPWORDS or token in FOCUS_SCAFFOLD_TOKENS:
            continue
        tokens.append(token)
    return tokens


def _query_focus_kind(query: str) -> Optional[str]:
    normalized = _normalize_text(query)
    if " name " in f" {normalized} " or normalized.startswith("who "):
        return "name"
    if "job title" in normalized or "current title" in normalized:
        return "title"
    if "primary job" in normalized or "current job" in normalized or " job " in f" {normalized} ":
        return "job"
    if "current status" in normalized or "status of" in normalized:
        return "status"
    if (
        "where " in f" {normalized} "
        or "which city" in normalized
        or "what city" in normalized
        or "specific space" in normalized
        or "what space" in normalized
        or "what location" in normalized
    ):
        return "location"
    if "what method" in normalized or ("how did" in normalized and "learn" in normalized):
        return "method"
    if "what rule" in normalized:
        return "rule"
    if (
        "what object" in normalized
        or "what instrument" in normalized
        or "what item" in normalized
        or "what specific types of items" in normalized
        or "what project" in normalized
    ):
        return "object"
    return None


def _required_slot_count(query: str) -> int:
    normalized = _normalize_text(query)
    padded = f" {normalized} "
    if any(marker in padded for marker in (" what two ", " which two ", " two specific ", " two names ")):
        return 2
    if " and who " in padded or " and what " in padded or " and which " in padded:
        return 2
    if "for whom" in normalized:
        return 2
    if "what are the names of" in normalized or "who are the names of" in normalized:
        return 2
    if re.search(r"\bwhat .+?, and .+\b", normalized):
        return 2
    return 1


def _matched_component_count(
    response_text: str,
    components: List[str],
) -> int:
    if not response_text or not components:
        return 0
    response_tokens = set(_focus_tokens(response_text))
    count = 0
    for comp in components:
        comp_tokens = {
            token
            for token in _focus_tokens(comp)
            if token not in SLOT_SCAFFOLD_TOKENS
        }
        if not comp_tokens:
            continue
        overlap = len(comp_tokens & response_tokens) / max(len(comp_tokens), 1)
        if overlap >= 0.5 or (len(comp_tokens) == 1 and overlap >= 1.0):
            count += 1
    return count


def _extract_value_candidates(text: Any, query_kind: Optional[str] = None) -> List[str]:
    raw = _normalize_text(text)
    if not raw:
        return []

    candidates: List[str] = []

    def add_candidate(value: str) -> None:
        cleaned = value.strip()
        if cleaned:
            candidates.append(cleaned)

    if not query_kind:
        add_candidate(raw)

    if ":" in raw:
        left, right = raw.split(":", 1)
        if right.strip() and len(left.split()) <= 8:
            add_candidate(right.strip())

    if query_kind == "name":
        for pattern in (
            r"\bname (?:is|was|remains)?\s*['\"]?(.+?)['\"]?(?:[.;]|$)",
            r"\bnamed\s+['\"]?(.+?)['\"]?(?:[.;]|$)",
            r"\bcalled\s+['\"]?(.+?)['\"]?(?:[.;]|$)",
        ):
            match = re.search(pattern, raw)
            if match:
                add_candidate(match.group(1))
    elif query_kind == "title":
        for pattern in (
            r"\b(?:job title|title)\s+(?:is|was|remains)\s+(?:as\s+)?(.+?)(?:[.;]|$)",
            r"\b(?:is|was|became|remains|currently is|is currently|is now)\s+(?:a\s+|an\s+)?(.+?)(?:[.;]|$)",
        ):
            match = re.search(pattern, raw)
            if match:
                add_candidate(match.group(1))
    elif query_kind == "job":
        for pattern in (
            r"\b(?:primary|current)?\s*job\s+(?:is|was|remains)\s+(?:as\s+)?(.+?)(?:[.;]|$)",
            r"\b(?:works|worked|serves|served)\s+as\s+(.+?)(?:[.;]|$)",
            r"\b(?:is|was|became|remains|currently is|is currently|is now)\s+(?:a\s+|an\s+)?(.+?)(?:[.;]|$)",
        ):
            match = re.search(pattern, raw)
            if match:
                add_candidate(match.group(1))
    elif query_kind == "status":
        for pattern in (
            r"\bstatus\s+(?:is|was|remains)\s+(.+?)(?:[.;]|$)",
            r"\b(?:is|was|became|remains|currently is|is currently|is now)\s+(.+?)(?:[.;]|$)",
        ):
            match = re.search(pattern, raw)
            if match:
                add_candidate(match.group(1))
    elif query_kind == "location":
        for pattern in (
            r"\blocation (?:is|was) (.+)$",
            r"\bdestination (?:is|was) (.+)$",
            r"\b(?:traveled|travelled|went|moved) to (.+)$",
            r"\bset up .* in (.+)$",
            r"\bconvert(?:ed)? .* (?:into|in) (.+)$",
            r"\b(?:located|based|situated)\s+(?:at|in)\s+(.+?)(?:[.;]|$)",
            r"\blives?\s+in\s+(.+?)(?:[.;]|$)",
            r"\bis in\s+(.+?)(?:[.;]|$)",
        ):
            match = re.search(pattern, raw)
            if match:
                add_candidate(match.group(1))
    elif query_kind == "method":
        for pattern in (r"\bused (.+)$", r"\bfollowing (.+)$", r"\blearned to (.+)$"):
            match = re.search(pattern, raw)
            if match:
                add_candidate(match.group(1))
    elif query_kind == "rule":
        match = re.search(r"\brule (?:is|was) (.+)$", raw)
        if match:
            add_candidate(match.group(1))
    elif query_kind == "object":
        for pattern in (
            r"\bit (?:is|was) (.+)$",
            r"\bproduced (.+)$",
            r"\borganized (.+)$",
            r"\bcompleted (.+)$",
        ):
            match = re.search(pattern, raw)
            if match:
                add_candidate(match.group(1))

    if not candidates and not query_kind:
        add_candidate(raw)

    deduped: List[str] = []
    seen = set()
    for item in candidates:
        cleaned = item.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            deduped.append(cleaned)
    return deduped


def _answer_focus_metrics(
    answer: str,
    query: str,
    reference: str,
    components: List[str],
) -> Dict[str, float]:
    query_kind = _query_focus_kind(query)
    query_terms = set(_focus_tokens(query))
    answer_terms = set(_focus_tokens(answer)) - query_terms
    candidate_texts: List[str] = []
    for text in [reference] + [str(x) for x in components]:
        candidate_texts.extend(_extract_value_candidates(text, query_kind=query_kind))
    if not candidate_texts:
        for text in [reference] + [str(x) for x in components]:
            candidate_texts.extend(_extract_value_candidates(text, query_kind=None))

    best_candidate_ratio = 0.0
    best_focus_recall = 0.0
    for candidate in candidate_texts:
        best_candidate_ratio = max(best_candidate_ratio, _fuzzy_ratio(answer, candidate))
        candidate_terms = set(_focus_tokens(candidate)) - query_terms
        if answer_terms and candidate_terms:
            overlap = len(answer_terms & candidate_terms) / max(len(candidate_terms), 1)
            if overlap > best_focus_recall:
                best_focus_recall = overlap

    return {
        "answer_candidate_ratio": best_candidate_ratio,
        "focus_term_recall": best_focus_recall,
        "answer_novel_token_count": float(len(answer_terms)),
    }


def _token_overlap_metrics(answer: str, candidate: str) -> Dict[str, float]:
    answer_tokens = set(_focus_tokens(answer))
    candidate_tokens = set(_focus_tokens(candidate))
    if not answer_tokens or not candidate_tokens:
        return {
            "token_recall": 0.0,
            "token_precision": 0.0,
            "token_f1": 0.0,
        }
    overlap = answer_tokens & candidate_tokens
    recall = len(overlap) / max(len(candidate_tokens), 1)
    precision = len(overlap) / max(len(answer_tokens), 1)
    if recall + precision == 0.0:
        f1 = 0.0
    else:
        f1 = (2.0 * recall * precision) / (recall + precision)
    return {
        "token_recall": recall,
        "token_precision": precision,
        "token_f1": f1,
    }


def _has_unspecified_marker(text: Any) -> bool:
    normalized = _normalize_text(text)
    return any(marker in normalized for marker in UNSPECIFIED_MARKERS)


def _best_value_match_metrics(
    answer: str,
    query: str,
    reference: str,
    components: List[str],
) -> Dict[str, Any]:
    query_kind = _query_focus_kind(query)
    candidate_texts: List[str] = []
    for text in [reference] + [str(x) for x in components]:
        candidate_texts.extend(_extract_value_candidates(text, query_kind=query_kind))
    if not candidate_texts:
        for text in [reference] + [str(x) for x in components]:
            candidate_texts.extend(_extract_value_candidates(text, query_kind=None))

    answer_candidates = _extract_value_candidates(answer, query_kind=query_kind)
    if not answer_candidates:
        answer_candidates = [answer]

    best = {
        "query_kind": query_kind or "",
        "value_candidate_ratio": 0.0,
        "value_token_recall": 0.0,
        "value_token_precision": 0.0,
        "value_token_f1": 0.0,
        "best_value_candidate": "",
        "best_value_answer": "",
    }

    for answer_candidate in answer_candidates:
        for candidate in candidate_texts:
            ratio = _fuzzy_ratio(answer_candidate, candidate)
            token_metrics = _token_overlap_metrics(answer_candidate, candidate)
            score = (
                ratio * 0.55
                + token_metrics["token_f1"] * 0.25
                + token_metrics["token_recall"] * 0.20
            )
            current = (
                best["value_candidate_ratio"] * 0.55
                + best["value_token_f1"] * 0.25
                + best["value_token_recall"] * 0.20
            )
            if score > current:
                best.update(
                    {
                        "value_candidate_ratio": ratio,
                        "value_token_recall": token_metrics["token_recall"],
                        "value_token_precision": token_metrics["token_precision"],
                        "value_token_f1": token_metrics["token_f1"],
                        "best_value_candidate": candidate,
                        "best_value_answer": answer_candidate,
                    }
                )
    return best


def _to_int(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    s = str(value or "").strip()
    if not s:
        return None
    m = re.search(r"-?\d+", s)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def _extract_days(text: str) -> Optional[int]:
    no_dates = ISO_DATE_RE.sub(" ", text or "")
    return _to_int(no_dates)


def _fuzzy_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return float(fuzz.partial_ratio(_normalize_text(a), _normalize_text(b))) / 100.0


def _extract_years(text: Any) -> List[str]:
    return YEAR_RE.findall(str(text or ""))


def _segment_reference_stats(segments: List[Any], reference: str) -> Dict[str, float]:
    clean_segments = [str(x).strip() for x in segments if str(x).strip()]
    if not clean_segments:
        return {
            "segment_reference_avg": 0.0,
            "segment_reference_max": 0.0,
        }
    ratios = [_fuzzy_ratio(segment, reference) for segment in clean_segments]
    return {
        "segment_reference_avg": _mean(ratios),
        "segment_reference_max": max(ratios),
    }


def _is_parenthesized_fragment(text: Any) -> bool:
    stripped = str(text or "").strip()
    return len(stripped) >= 2 and stripped.startswith("(") and stripped.endswith(")")


def _has_template_artifact(text: Any) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    markers = (
        "for the query",
        "from the dialogue",
        "event chain",
        "event_chain",
        "final outcome",
        "final_outcome",
        "key turning point",
        "key_turning_points",
        "summary for the query",
        "answer for the query",
    )
    return any(marker in normalized for marker in markers)


def _template_artifact_ratio(segments: List[Any]) -> float:
    clean_segments = [str(x).strip() for x in segments if str(x).strip()]
    if not clean_segments:
        return 0.0
    flagged = [
        segment
        for segment in clean_segments
        if _has_template_artifact(segment) or _is_parenthesized_fragment(segment)
    ]
    return len(flagged) / len(clean_segments)


def _should_skip_tasklight_judge(
    task_type: str,
    rule_signals: Dict[str, Any],
) -> Optional[str]:
    if task_type == "Multi-session Reasoning":
        template_artifact_ratio = float(rule_signals.get("template_artifact_ratio", 0.0) or 0.0)
        component_recall = float(rule_signals.get("component_recall", 0.0) or 0.0)
        if template_artifact_ratio >= 0.60 and component_recall <= 0.05:
            return "template_artifact_low_recall"
    return None


def _temporal_aux_scores(
    response: str,
    reference: str,
    parse_ok: bool,
    final_score: float,
) -> Dict[str, float]:
    semantic_score = max(float(final_score), eval_temporal_exact_match(response, reference))
    return {
        "format_compliant": 1.0 if parse_ok else 0.0,
        "semantic_exact_correct": float(semantic_score),
        "temporal_exact_match": float(semantic_score),
    }


def eval_component_recall(
    response_text: str,
    components: List[str],
    threshold: int = 70,
) -> float:
    if not components:
        return 0.0
    match_count = 0
    for comp in components:
        score = fuzz.partial_ratio(
            _normalize_text(comp),
            _normalize_text(response_text),
        )
        if score >= threshold:
            match_count += 1
    return match_count / max(len(components), 1)


def _msr_alignment_units(chain: List[Any], outcome: str) -> Tuple[List[str], List[Tuple[float, str]]]:
    clean_chain = [str(x).strip() for x in chain if str(x).strip()]
    units: List[Tuple[float, str]] = []
    for idx, segment in enumerate(clean_chain):
        units.append((float(idx), segment))
    for idx in range(len(clean_chain) - 1):
        units.append((idx + 0.5, f"{clean_chain[idx]} {clean_chain[idx + 1]}"))
    for idx in range(len(clean_chain) - 2):
        units.append(
            (
                idx + 0.75,
                f"{clean_chain[idx]} {clean_chain[idx + 1]} {clean_chain[idx + 2]}",
            )
        )
    outcome_text = str(outcome or "").strip()
    if outcome_text:
        units.append((float(len(clean_chain)), outcome_text))
    return clean_chain, units


_MSR_ANCHOR_STOPWORDS = {
    "this",
    "that",
    "these",
    "those",
    "led",
    "lead",
    "leading",
    "where",
    "which",
    "years",
    "year",
    "later",
    "after",
    "before",
    "into",
    "from",
    "through",
    "allowed",
    "allowing",
    "success",
    "coupled",
    "him",
    "her",
}


def _msr_anchor_tokens(text: str) -> List[str]:
    tokens: List[str] = []
    for raw_token in _focus_tokens(text):
        if raw_token in _MSR_ANCHOR_STOPWORDS or ISO_DATE_RE.fullmatch(raw_token):
            continue
        token = raw_token
        if len(token) > 5 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 4 and token.endswith("ed"):
            token = token[:-2]
        token = _simplify_focus_token(token)
        if len(token) <= 1 or token in _MSR_ANCHOR_STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _msr_token_matches(component_token: str, unit_token: str) -> bool:
    if component_token == unit_token:
        return True
    if min(len(component_token), len(unit_token)) >= 4 and (
        component_token.startswith(unit_token) or unit_token.startswith(component_token)
    ):
        return True
    return float(fuzz.partial_ratio(component_token, unit_token)) / 100.0 >= 0.72


def _msr_soft_token_recall(component: str, unit_text: str) -> float:
    component_tokens = _msr_anchor_tokens(component)
    unit_tokens = _msr_anchor_tokens(unit_text)
    if not component_tokens or not unit_tokens:
        return 0.0
    matched = 0
    for component_token in component_tokens:
        if any(_msr_token_matches(component_token, unit_token) for unit_token in unit_tokens):
            matched += 1
    return matched / max(len(component_tokens), 1)


def _msr_year_recall(component: str, unit_text: str) -> float:
    component_years = set(_extract_years(component))
    unit_years = set(_extract_years(unit_text))
    if not component_years:
        return 0.0
    return len(component_years & unit_years) / max(len(component_years), 1)


def _msr_component_match_score(component: str, unit_text: str) -> float:
    fuzzy_ratio = _fuzzy_ratio(component, unit_text)
    token_metrics = _token_overlap_metrics(component, unit_text)
    soft_token_recall = _msr_soft_token_recall(component, unit_text)
    component_dates = set(ISO_DATE_RE.findall(component or ""))
    unit_dates = set(ISO_DATE_RE.findall(unit_text or ""))
    date_overlap = (
        len(component_dates & unit_dates) / max(len(component_dates), 1)
        if component_dates
        else 0.0
    )
    year_recall = _msr_year_recall(component, unit_text)
    anchor_overlap = max(date_overlap, year_recall)
    return max(
        fuzzy_ratio,
        fuzzy_ratio * 0.30
        + token_metrics["token_f1"] * 0.10
        + soft_token_recall * 0.40
        + anchor_overlap * 0.20,
        fuzzy_ratio * 0.25
        + token_metrics["token_recall"] * 0.10
        + soft_token_recall * 0.45
        + anchor_overlap * 0.20,
    )


def _msr_component_coverage(
    chain: List[Any],
    outcome: str,
    components: List[str],
) -> Dict[str, Any]:
    clean_chain, units = _msr_alignment_units(chain, outcome)
    if not components:
        return {
            "bridge_component_recall": 0.0,
            "endpoint_coverage": 0.0,
            "ordered_component_hits": 0.0,
            "matched_component_positions": [],
            "component_match_scores": [],
            "last_matched_chain_index": -1,
            "clean_chain": clean_chain,
        }

    matched_positions: List[float] = []
    component_match_scores: List[float] = []
    matched_flags: List[bool] = []
    last_matched_chain_index = -1
    hit_threshold = 0.55

    for component in components:
        best_score = 0.0
        best_position: Optional[float] = None
        for position, unit_text in units:
            score = _msr_component_match_score(component, unit_text)
            if score > best_score:
                best_score = score
                best_position = position
        component_match_scores.append(best_score)
        matched = best_score >= hit_threshold and best_position is not None
        matched_flags.append(bool(matched))
        if matched and best_position is not None:
            matched_positions.append(float(best_position))
            if best_position < len(clean_chain):
                last_matched_chain_index = max(last_matched_chain_index, int(best_position))

    bridge_component_recall = sum(1 for flag in matched_flags if flag) / max(len(components), 1)
    endpoint_indices = [0]
    if len(components) > 1:
        endpoint_indices.append(len(components) - 1)
    endpoint_hits = sum(1 for idx in endpoint_indices if matched_flags[idx])
    endpoint_coverage = endpoint_hits / max(len(endpoint_indices), 1)

    if len(matched_positions) <= 1:
        ordered_component_hits = 1.0 if matched_positions else 0.0
    else:
        ordered_pairs = sum(
            1 for left, right in zip(matched_positions, matched_positions[1:]) if right >= left
        )
        ordered_component_hits = ordered_pairs / max(len(matched_positions) - 1, 1)

    return {
        "bridge_component_recall": bridge_component_recall,
        "endpoint_coverage": endpoint_coverage,
        "ordered_component_hits": ordered_component_hits,
        "matched_component_positions": matched_positions,
        "component_match_scores": component_match_scores,
        "last_matched_chain_index": last_matched_chain_index,
        "clean_chain": clean_chain,
    }



def _es_component_coverage(
    points: List[Any],
    summary: str,
    components: List[str],
) -> Dict[str, Any]:
    clean_points = [str(x).strip() for x in points if str(x).strip()]
    summary_text = str(summary or "").strip()
    point_units: List[Tuple[float, str]] = [
        (float(idx), point) for idx, point in enumerate(clean_points)
    ]
    units: List[Tuple[float, str]] = list(point_units)
    if summary_text:
        units.append((float(len(clean_points)), summary_text))

    if not components:
        return {
            "anchor_component_recall": 0.0,
            "point_anchor_recall": 0.0,
            "summary_anchor_recall": 0.0,
            "summary_only_anchor_recall": 0.0,
            "endpoint_coverage": 0.0,
            "ordered_component_hits": 0.0,
            "component_match_scores": [],
            "point_component_scores": [],
            "summary_component_scores": [],
            "clean_points": clean_points,
        }

    hit_threshold = 0.58
    matched_positions: List[float] = []
    ordered_point_positions: List[float] = []
    component_match_scores: List[float] = []
    point_component_scores: List[float] = []
    summary_component_scores: List[float] = []
    matched_flags: List[bool] = []
    point_matched_flags: List[bool] = []
    summary_matched_flags: List[bool] = []

    for component in components:
        best_score = 0.0
        best_position: Optional[float] = None
        for position, unit_text in units:
            score = _msr_component_match_score(component, unit_text)
            if score > best_score:
                best_score = score
                best_position = position

        best_point_score = 0.0
        best_point_position: Optional[float] = None
        for position, point_text in point_units:
            score = _msr_component_match_score(component, point_text)
            if score > best_point_score:
                best_point_score = score
                best_point_position = position

        summary_score = _msr_component_match_score(component, summary_text) if summary_text else 0.0

        component_match_scores.append(best_score)
        point_component_scores.append(best_point_score)
        summary_component_scores.append(summary_score)

        matched = best_score >= hit_threshold and best_position is not None
        point_matched = best_point_score >= hit_threshold and best_point_position is not None
        summary_matched = summary_score >= hit_threshold

        matched_flags.append(bool(matched))
        point_matched_flags.append(bool(point_matched))
        summary_matched_flags.append(bool(summary_matched))

        if matched and best_position is not None:
            matched_positions.append(float(best_position))
        if point_matched and best_point_position is not None:
            ordered_point_positions.append(float(best_point_position))

    anchor_component_recall = sum(1 for flag in matched_flags if flag) / max(len(components), 1)
    point_anchor_recall = sum(1 for flag in point_matched_flags if flag) / max(len(components), 1)
    summary_anchor_recall = sum(1 for flag in summary_matched_flags if flag) / max(len(components), 1)
    summary_only_anchor_recall = (
        sum(1 for matched, point_matched in zip(matched_flags, point_matched_flags) if matched and not point_matched)
        / max(len(components), 1)
    )

    endpoint_indices = [0]
    if len(components) > 1:
        endpoint_indices.append(len(components) - 1)
    endpoint_hits = sum(1 for idx in endpoint_indices if matched_flags[idx])
    endpoint_coverage = endpoint_hits / max(len(endpoint_indices), 1)

    if len(ordered_point_positions) <= 1:
        ordered_component_hits = 1.0 if ordered_point_positions else 0.0
    else:
        ordered_pairs = sum(
            1 for left, right in zip(ordered_point_positions, ordered_point_positions[1:]) if right >= left
        )
        ordered_component_hits = ordered_pairs / max(len(ordered_point_positions) - 1, 1)

    return {
        "anchor_component_recall": anchor_component_recall,
        "point_anchor_recall": point_anchor_recall,
        "summary_anchor_recall": summary_anchor_recall,
        "summary_only_anchor_recall": summary_only_anchor_recall,
        "endpoint_coverage": endpoint_coverage,
        "ordered_component_hits": ordered_component_hits,
        "component_match_scores": component_match_scores,
        "point_component_scores": point_component_scores,
        "summary_component_scores": summary_component_scores,
        "matched_component_positions": matched_positions,
        "clean_points": clean_points,
    }

def eval_temporal_exact_match(response_text: str, gt_text: str) -> float:
    gt_days = _extract_days(gt_text)
    pred_days = _extract_days(response_text)
    gt_dates = set(ISO_DATE_RE.findall(gt_text or ""))
    pred_dates = set(ISO_DATE_RE.findall(response_text or ""))
    if gt_days is not None:
        return 1.0 if pred_days == gt_days else 0.0
    if gt_dates:
        return 1.0 if gt_dates.issubset(pred_dates) else 0.0
    return 0.0


def _binary_information_extraction(
    parsed: Dict[str, Any],
    query: str,
    reference: str,
    components: List[str],
) -> Tuple[Optional[float], Dict[str, Any], bool]:
    answer = str(parsed.get("answer", ""))
    snippets = parsed.get("evidence_snippets", [])
    if not isinstance(snippets, list):
        snippets = [snippets] if snippets else []
    merged = " ".join([answer] + [str(x) for x in snippets])
    focus_metrics = _answer_focus_metrics(answer, query, reference, components)
    value_metrics = _best_value_match_metrics(answer, query, reference, components)
    reference_unspecified = _has_unspecified_marker(reference)
    answer_unspecified = _has_unspecified_marker(answer)
    signals: Dict[str, Any] = {
        "rule_name": "info_extraction_binary",
        "component_recall": eval_component_recall(merged, components),
        "answer_reference_ratio": _fuzzy_ratio(answer, reference),
        "reference_unspecified": reference_unspecified,
        "answer_unspecified": answer_unspecified,
        **focus_metrics,
        **value_metrics,
    }
    ratio = signals["answer_reference_ratio"]
    candidate_ratio = signals["answer_candidate_ratio"]
    focus_recall = signals["focus_term_recall"]
    has_novel_terms = signals["answer_novel_token_count"] > 0
    value_ratio = signals["value_candidate_ratio"]
    value_recall = signals["value_token_recall"]
    value_precision = signals["value_token_precision"]
    query_kind = signals["query_kind"]
    required_slots = _required_slot_count(query)
    matched_slots = _matched_component_count(answer, components)
    signals["required_slots"] = required_slots
    signals["matched_slots"] = matched_slots

    if reference_unspecified:
        return (1.0 if answer_unspecified else 0.0), signals, True

    if required_slots > 1:
        if matched_slots < required_slots:
            return 0.0, signals, True
        if ratio >= 0.55 or candidate_ratio >= 0.55 or value_ratio >= 0.55:
            return 1.0, signals, True
        return None, signals, False

    if query_kind == "name":
        if (
            (value_recall >= 0.999 and value_precision >= 0.80)
            or (value_ratio >= 0.97 and signals["value_token_f1"] >= 0.80 and value_precision >= 0.70)
        ):
            return 1.0, signals, True
        if value_recall <= 0.50 or value_precision < 0.60:
            return 0.0, signals, True
        return None, signals, False

    if query_kind == "location":
        if value_ratio >= 0.82 or (value_recall >= 0.50 and value_precision >= 0.95):
            return 1.0, signals, True
        if ratio <= 0.20 and candidate_ratio <= 0.35 and focus_recall == 0.0:
            return 0.0, signals, True
        return None, signals, False

    # Treat the direct answer match as the primary signal and use query-aware
    # focus overlap as auxiliary support for short factual answers.
    if ratio >= 0.75 or value_ratio >= 0.82:
        return 1.0, signals, True
    if has_novel_terms and (candidate_ratio >= 0.75 or focus_recall >= 0.75):
        return 1.0, signals, True
    if ratio <= 0.20 and candidate_ratio <= 0.35 and focus_recall == 0.0:
        return 0.0, signals, True
    return None, signals, False


def _binary_temporal_reasoning(
    parsed: Dict[str, Any],
    reference: str,
) -> Tuple[Optional[float], Dict[str, Any], bool]:
    final_answer = str(parsed.get("final_answer", ""))
    pred_days = _to_int(parsed.get("days"))
    if pred_days is None:
        pred_days = _extract_days(final_answer)
    ref_days = _extract_days(reference)
    start_date = str(parsed.get("start_date", "")).strip()
    end_date = str(parsed.get("end_date", "")).strip()

    date_legal = bool(ISO_DATE_RE.fullmatch(start_date)) and bool(
        ISO_DATE_RE.fullmatch(end_date)
    )
    final_answer_dates = set(ISO_DATE_RE.findall(final_answer))
    final_answer_days = _extract_days(final_answer) if final_answer else None
    answer_days_consistent = final_answer_days is None or pred_days is None or final_answer_days == pred_days
    answer_dates_consistent = not final_answer_dates or final_answer_dates.issubset({start_date, end_date})
    date_consistency = bool(final_answer) and answer_days_consistent and answer_dates_consistent
    ref_dates = set(ISO_DATE_RE.findall(reference or ""))
    pred_dates = {start_date, end_date}
    dates_match_reference = True if not ref_dates else ref_dates.issubset(pred_dates)
    day_match = ref_days is not None and pred_days == ref_days

    signals = {
        "rule_name": "temporal_binary",
        "pred_days": pred_days,
        "ref_days": ref_days,
        "date_legal": date_legal,
        "date_consistency": date_consistency,
        "dates_match_reference": dates_match_reference,
        "day_match": day_match,
    }
    score = (
        1.0
        if day_match and date_legal and date_consistency and dates_match_reference
        else 0.0
    )
    return score, signals, True


def _binary_knowledge_updating(
    parsed: Dict[str, Any],
    query: str,
    reference: str,
    components: List[str],
) -> Tuple[Optional[float], Dict[str, Any], bool]:
    latest_state = str(parsed.get("latest_state", ""))
    as_of_time = str(parsed.get("as_of_time", ""))
    deprecated_state = str(parsed.get("deprecated_state", ""))
    merged_latest = " ".join([latest_state, as_of_time]).strip()
    focus_metrics = _answer_focus_metrics(latest_state, query, reference, components)
    value_metrics = _best_value_match_metrics(latest_state, query, reference, components)
    reference_unspecified = _has_unspecified_marker(reference)
    answer_unspecified = _has_unspecified_marker(latest_state)
    latest_vs_reference = _fuzzy_ratio(latest_state, reference)
    latest_component_recall = eval_component_recall(latest_state, components)
    as_of_time_ratio = _fuzzy_ratio(as_of_time, reference)
    deprecated_vs_reference = _fuzzy_ratio(deprecated_state, reference)

    signals = {
        "rule_name": "knowledge_updating_binary",
        "latest_vs_reference": latest_vs_reference,
        "latest_component_recall": latest_component_recall,
        "as_of_time_ratio": as_of_time_ratio,
        "deprecated_vs_reference": deprecated_vs_reference,
        "reference_unspecified": reference_unspecified,
        "answer_unspecified": answer_unspecified,
        **focus_metrics,
        **value_metrics,
    }

    candidate_ratio = signals["answer_candidate_ratio"]
    focus_recall = signals["focus_term_recall"]
    novel_terms = signals["answer_novel_token_count"]
    value_ratio = signals["value_candidate_ratio"]
    value_recall = signals["value_token_recall"]
    value_precision = signals["value_token_precision"]
    query_kind = signals["query_kind"]
    required_slots = _required_slot_count(query)
    matched_slots = _matched_component_count(latest_state, components)
    signals["required_slots"] = required_slots
    signals["matched_slots"] = matched_slots

    if reference_unspecified:
        return (1.0 if answer_unspecified else 0.0), signals, True

    if deprecated_state and deprecated_vs_reference >= 0.75 and latest_vs_reference < 0.55:
        return 0.0, signals, True

    if required_slots > 1:
        if matched_slots < required_slots:
            return 0.0, signals, True
        if (
            latest_component_recall >= 0.50
            or value_ratio >= 0.55
            or candidate_ratio >= 0.55
            or latest_vs_reference >= 0.55
        ):
            return 1.0, signals, True
        signals["low_confidence_reason"] = "multi_slot_partial_match"
        return None, signals, False

    if query_kind == "name":
        if value_ratio >= 0.90 or value_recall >= 0.999:
            return 1.0, signals, True
        if value_ratio <= 0.55 and value_recall <= 0.25 and value_precision <= 0.55:
            return 0.0, signals, True
        signals["low_confidence_reason"] = "name_partial_match"
        return None, signals, False

    if query_kind in {"job", "title"}:
        if value_ratio >= 0.85 or (value_recall >= 0.75 and value_precision >= 0.75):
            return 1.0, signals, True
        if value_ratio <= 0.40 and value_recall <= 0.25 and focus_recall == 0.0:
            return 0.0, signals, True
        signals["low_confidence_reason"] = "job_title_partial_match"
        return None, signals, False

    if query_kind == "location":
        if value_ratio >= 0.80 or (
            value_recall >= 0.50 and value_precision >= 0.60 and focus_recall > 0.0
        ):
            return 1.0, signals, True
        if novel_terms >= 3.0 and focus_recall == 0.0 and latest_component_recall == 0.0:
            return 0.0, signals, True
        if value_ratio <= 0.20 and candidate_ratio <= 0.35 and focus_recall == 0.0:
            return 0.0, signals, True
        signals["low_confidence_reason"] = "location_partial_match"
        return None, signals, False

    severe_value_precision_mismatch = (
        value_ratio >= 0.80
        and value_precision < 0.25
        and novel_terms >= 5.0
        and latest_component_recall < 0.25
    )
    signals["severe_value_precision_mismatch"] = severe_value_precision_mismatch
    if severe_value_precision_mismatch:
        return 0.0, signals, True

    if latest_component_recall >= 0.50:
        return 1.0, signals, True
    if value_ratio >= 0.80 and (focus_recall >= 0.25 or novel_terms <= 2.0):
        return 1.0, signals, True
    if candidate_ratio >= 0.85 and focus_recall >= 0.60 and novel_terms <= 2.0:
        return 1.0, signals, True

    if latest_vs_reference <= 0.20 and candidate_ratio <= 0.35 and focus_recall == 0.0:
        return 0.0, signals, True
    if (
        latest_vs_reference <= 0.40
        and candidate_ratio <= 0.40
        and value_ratio <= 0.40
        and focus_recall == 0.0
        and value_recall == 0.0
        and latest_component_recall == 0.0
    ):
        return 0.0, signals, True
    if novel_terms >= 3.0 and focus_recall == 0.0 and latest_component_recall == 0.0:
        return 0.0, signals, True
    signals["low_confidence_reason"] = "generic_partial_match"
    return None, signals, False


def _three_level_multisession(
    parsed: Dict[str, Any],
    reference: str,
    components: List[str],
) -> Tuple[float, Dict[str, Any], bool]:
    chain = parsed.get("event_chain", [])
    if not isinstance(chain, list):
        chain = [chain] if chain else []
    outcome = str(parsed.get("final_outcome", ""))
    merged = " ".join([str(x) for x in chain] + [outcome])
    recall = eval_component_recall(merged, components)
    coverage = _msr_component_coverage(chain, outcome, components)
    clean_chain = coverage["clean_chain"]
    chain_len = len(clean_chain)
    reference_stats = _segment_reference_stats([*clean_chain, outcome], reference)
    outcome_ratio = _fuzzy_ratio(outcome, reference)
    template_artifact_ratio = _template_artifact_ratio([*clean_chain, outcome])
    parenthesized_chain_ratio = (
        sum(1 for segment in clean_chain if segment.startswith("(") and segment.endswith(")"))
        / max(chain_len, 1)
        if chain_len
        else 0.0
    )

    segment_relevance_scores: List[float] = []
    segment_year_coverages: List[float] = []
    for segment in clean_chain:
        component_match = max(
            (_msr_component_match_score(component, segment) for component in components),
            default=0.0,
        )
        segment_year_coverage = max(
            (_msr_year_recall(component, segment) for component in components),
            default=0.0,
        )
        segment_relevance_scores.append(max(component_match, _fuzzy_ratio(segment, reference)))
        segment_year_coverages.append(segment_year_coverage)
    extraneous_event_ratio = (
        sum(1 for score in segment_relevance_scores if score < 0.45) / max(chain_len, 1)
        if chain_len
        else 0.0
    )
    last_matched_chain_index = int(coverage["last_matched_chain_index"])
    tail_scores = (
        segment_relevance_scores[last_matched_chain_index + 1 :]
        if 0 <= last_matched_chain_index < chain_len - 1
        else ([] if last_matched_chain_index >= chain_len - 1 else segment_relevance_scores)
    )
    tail_year_coverages = (
        segment_year_coverages[last_matched_chain_index + 1 :]
        if 0 <= last_matched_chain_index < chain_len - 1
        else ([] if last_matched_chain_index >= chain_len - 1 else segment_year_coverages)
    )
    tail_extraneous_ratio = (
        sum(
            1
            for score, year_coverage in zip(tail_scores, tail_year_coverages)
            if score < 0.45 or (year_coverage == 0.0 and score < 0.60)
        )
        / max(len(tail_scores), 1)
        if tail_scores
        else 0.0
    )

    signals = {
        "rule_name": "multisession_three_level",
        "component_recall": recall,
        "bridge_component_recall": coverage["bridge_component_recall"],
        "endpoint_coverage": coverage["endpoint_coverage"],
        "ordered_component_hits": coverage["ordered_component_hits"],
        "extraneous_event_ratio": extraneous_event_ratio,
        "tail_extraneous_ratio": tail_extraneous_ratio,
        "chain_len": chain_len,
        **reference_stats,
        "outcome_reference_ratio": outcome_ratio,
        "template_artifact_ratio": template_artifact_ratio,
        "parenthesized_chain_ratio": parenthesized_chain_ratio,
    }
    if template_artifact_ratio >= 0.60 and recall <= 0.05 and parenthesized_chain_ratio >= 0.75:
        return 0.0, signals, True
    if template_artifact_ratio >= 0.60 and recall <= 0.05 and coverage["bridge_component_recall"] <= 0.15:
        return 0.0, signals, True
    if (
        chain_len >= 2
        and outcome_ratio >= 0.50
        and coverage["endpoint_coverage"] >= 1.0
        and coverage["bridge_component_recall"] >= 0.75
        and coverage["ordered_component_hits"] >= 0.66
        and extraneous_event_ratio <= 0.25
        and tail_extraneous_ratio == 0.0
    ):
        return 1.0, signals, True
    if (
        reference_stats["segment_reference_max"] <= 0.20
        and outcome_ratio <= 0.20
        and coverage["bridge_component_recall"] <= 0.15
    ):
        return 0.0, signals, True
    partial_floor = (
        chain_len >= 2
        and coverage["endpoint_coverage"] >= 1.0
        and coverage["bridge_component_recall"] >= 0.75
        and coverage["ordered_component_hits"] >= 0.66
        and reference_stats["segment_reference_max"] >= 0.55
        and extraneous_event_ratio <= 0.50
    )
    if partial_floor:
        signals["judge_floor_score"] = 0.5
        signals["judge_floor_reason"] = "msr_bridge_and_endpoint_preserved"
        return 0.5, signals, False
    if chain_len >= 1 and (
        coverage["bridge_component_recall"] >= 0.25
        or (
            reference_stats["segment_reference_avg"] >= 0.50
            and reference_stats["segment_reference_max"] >= 0.70
            and outcome_ratio >= 0.35
        )
    ):
        return 0.5, signals, False
    return 0.0, signals, True


def _three_level_summarization(
    parsed: Dict[str, Any],
    query: str,
    reference: str,
    components: List[str],
) -> Tuple[float, Dict[str, Any], bool]:
    span = str(parsed.get("time_span", ""))
    points = parsed.get("key_turning_points", [])
    if not isinstance(points, list):
        points = [points] if points else []
    summary = str(parsed.get("summary", ""))
    merged = " ".join([span, summary] + [str(x) for x in points])
    recall = eval_component_recall(merged, components)
    coverage = _es_component_coverage(points, summary, components)
    clean_points = coverage["clean_points"]
    point_count = len(clean_points)
    reference_stats = _segment_reference_stats([summary, *clean_points], reference)
    summary_ratio = _fuzzy_ratio(summary, reference)
    span_ratio = _fuzzy_ratio(span, f"{query} {reference}")
    expected_years = set(_extract_years(query) or _extract_years(reference))
    predicted_years = set(_extract_years(span))
    has_year_overlap = (
        True
        if not expected_years or not predicted_years
        else bool(expected_years & predicted_years)
    )
    has_out_of_scope_years = bool(predicted_years - expected_years) if expected_years else False
    signals = {
        "rule_name": "event_summarization_three_level",
        "component_recall": recall,
        "anchor_component_recall": coverage["anchor_component_recall"],
        "point_anchor_recall": coverage["point_anchor_recall"],
        "summary_anchor_recall": coverage["summary_anchor_recall"],
        "summary_only_anchor_recall": coverage["summary_only_anchor_recall"],
        "endpoint_coverage": coverage["endpoint_coverage"],
        "ordered_component_hits": coverage["ordered_component_hits"],
        "point_count": point_count,
        **reference_stats,
        "summary_reference_ratio": summary_ratio,
        "time_span_reference_ratio": span_ratio,
        "expected_years": sorted(expected_years),
        "predicted_years": sorted(predicted_years),
        "has_year_overlap": has_year_overlap,
        "has_out_of_scope_years": has_out_of_scope_years,
    }

    if not summary.strip() or point_count == 0:
        return 0.0, signals, True

    strong_full = (
        point_count >= 2
        and not has_out_of_scope_years
        and has_year_overlap
        and coverage["anchor_component_recall"] >= 0.85
        and coverage["point_anchor_recall"] >= 0.75
        and coverage["summary_only_anchor_recall"] <= 0.15
        and (
            (
                summary_ratio >= 0.65
                and reference_stats["segment_reference_avg"] >= 0.55
            )
            or (
                reference_stats["segment_reference_avg"] >= 0.60
                and reference_stats["segment_reference_max"] >= 0.80
            )
            or (
                reference_stats["segment_reference_max"] >= 0.75
                and coverage["point_anchor_recall"] >= 1.0
            )
        )
    )
    if strong_full:
        return 1.0, signals, True

    if (
        expected_years
        and predicted_years
        and not has_year_overlap
        and summary_ratio <= 0.20
        and coverage["anchor_component_recall"] <= 0.15
        and coverage["point_anchor_recall"] <= 0.15
    ):
        return 0.0, signals, True

    partial_floor = (
        point_count >= 1
        and not has_out_of_scope_years
        and has_year_overlap
        and coverage["anchor_component_recall"] >= 0.75
        and coverage["point_anchor_recall"] >= 0.50
        and coverage["endpoint_coverage"] >= 1.0
    )
    if partial_floor:
        signals["judge_floor_score"] = 0.5
        signals["judge_floor_reason"] = "event_summary_anchors_preserved"
        return 0.5, signals, False

    if point_count >= 1 and not has_out_of_scope_years and (
        coverage["anchor_component_recall"] >= 0.25
        or coverage["point_anchor_recall"] >= 0.25
        or (
            summary_ratio >= 0.45
            and reference_stats["segment_reference_max"] >= 0.65
        )
        or (
            reference_stats["segment_reference_avg"] >= 0.50
            and reference_stats["segment_reference_max"] >= 0.70
        )
    ):
        return 0.5, signals, False
    return 0.0, signals, True


def _expected_premise_verdict(reference: str) -> Optional[str]:
    normalized = _normalize_text(reference)
    incorrect_markers = [
        "premise is incorrect",
        "this is incorrect",
        "contains a false premise",
        "false premise",
        "query contains a slight timing error",
        "this is false",
        "this is wrong",
        "query is incorrect",
    ]
    if any(marker in normalized for marker in incorrect_markers):
        return "incorrect"
    if "premise is correct" in normalized or "query is correct" in normalized:
        return "correct"
    return None


def _three_level_memory_arbitration(
    parsed: Dict[str, Any],
    reference: str,
    components: List[str],
) -> Tuple[float, Dict[str, Any], bool]:
    verdict = str(parsed.get("premise_verdict", "")).strip().lower()
    error_text = str(parsed.get("premise_error", ""))
    corrected = parsed.get("corrected_facts", [])
    if not isinstance(corrected, list):
        corrected = [corrected] if corrected else []
    final_answer = str(parsed.get("final_answer", ""))
    corrected_text = " ".join(str(x) for x in corrected if str(x).strip())
    merged = " ".join([error_text, final_answer] + [str(x) for x in corrected])
    recall = eval_component_recall(merged, components)
    corrected_recall = eval_component_recall(corrected_text, components)
    final_answer_ratio = _fuzzy_ratio(final_answer, reference)
    verdict_wrong = verdict in {"incorrect", "wrong", "false", "invalid"}
    verdict_correct = verdict in {"correct", "true", "valid"}
    expected_verdict = _expected_premise_verdict(reference)
    has_correction = bool(error_text.strip()) or any(str(x).strip() for x in corrected)
    verdict_matches = (
        expected_verdict is None
        or (expected_verdict == "incorrect" and verdict_wrong)
        or (expected_verdict == "correct" and verdict_correct)
    )
    signals = {
        "rule_name": "memory_arbitration_three_level",
        "component_recall": recall,
        "corrected_recall": corrected_recall,
        "verdict_wrong": verdict_wrong,
        "verdict_correct": verdict_correct,
        "final_answer_ratio": final_answer_ratio,
        "expected_verdict": expected_verdict,
        "verdict_matches": verdict_matches,
        "has_correction": has_correction,
    }

    if expected_verdict == "incorrect":
        if (
            verdict_wrong
            and has_correction
            and corrected_recall >= 0.30
            and final_answer_ratio >= 0.45
        ):
            return 1.0, signals, True
        if (
            verdict_wrong
            and has_correction
            and (
                corrected_recall >= 0.15
                or (final_answer_ratio >= 0.35 and recall >= 0.20)
            )
        ):
            return 0.5, signals, False
        return 0.0, signals, True

    if expected_verdict == "correct":
        if verdict_correct and (final_answer_ratio >= 0.45 or recall >= 0.55):
            return 1.0, signals, True
        if verdict_correct and (final_answer_ratio >= 0.35 or recall >= 0.30):
            return 0.5, signals, False
        return 0.0, signals, True

    if verdict_wrong and corrected_recall >= 0.30 and final_answer_ratio >= 0.45:
        return 1.0, signals, True
    if verdict_wrong and corrected_recall >= 0.15 and (final_answer_ratio >= 0.35 or recall >= 0.25):
        return 0.5, signals, False
    if verdict_correct and (final_answer_ratio >= 0.35 or recall >= 0.25):
        return 0.5, signals, False
    return 0.0, signals, True


def _band_from_score(score: float, allow_partial: bool) -> str:
    s = float(score)
    if not allow_partial:
        return "correct" if s >= 0.5 else "wrong"
    if s >= 0.75:
        return "correct"
    if s <= 0.25:
        return "wrong"
    return "partial"


def _judge_prompt(
    task_type: str,
    query: str,
    reference: str,
    parsed_output: Dict[str, Any],
    allow_partial: bool,
    answer_components: Optional[List[str]] = None,
) -> str:
    allowed = "correct/wrong/partial" if allow_partial else "correct/wrong"
    judge_payload = parsed_output
    task_guidance = ""
    if task_type == "Information Extraction":
        judge_payload = {"answer": parsed_output.get("answer", "")}
        task_guidance = (
            "- Focus on whether the answer itself identifies the queried fact; "
            "missing evidence snippets should not outweigh a correct answer.\n"
        )
    elif task_type == "Knowledge Updating":
        task_guidance = (
            "- Focus on whether latest_state captures the latest truth; "
            "as_of_time and deprecated_state are auxiliary unless they contradict it.\n"
        )
    elif task_type == "Memory Arbitration":
        task_guidance = (
            "- Focus on whether premise_verdict matches the reference and whether "
            "the final answer follows the corrected facts.\n"
            "- Verdict alone is not enough for full credit.\n"
            "- Partial credit requires the correct verdict plus at least one materially correct correction.\n"
            "- If corrected_facts contain invented or unsupported corrected facts, score at most partial.\n"
            "- If corrected_facts introduce the wrong person, date, location, organization, or object, score wrong rather than partial.\n"
            "- If corrected_facts are clearly fabricated or materially contradict the reference, score wrong.\n"
        )
    elif task_type == "Multi-session Reasoning":
        task_guidance = (
            "- Focus on whether event_chain and final_outcome reconstruct the queried causal path; key causal anchors are listed below when available.\n"
            "- A full-credit answer must preserve most of the key milestones in the causal chain.\n"
            "- Partial credit requires the correct start/end state plus at least one correctly preserved interior bridge event or turning point.\n"
            "- Compressing adjacent steps is acceptable and should not by itself cause a wrong score.\n"
            "- one extra non-contradictory event, or one approximate substitution for a bridge event, should usually score partial rather than wrong.\n"
            "- Mild ordering noise should usually score partial rather than wrong when the main causal anchors are still preserved.\n"
            "- If the answer only repeats the endpoints and preserves no interior bridge event, score wrong.\n"
        )
    elif task_type == "Event Summarization":
        task_guidance = (
            "- Focus on whether time_span, key_turning_points, and summary capture the asked period, "
            "major themes, and overall arc; key summary anchors are listed below when available.\n"
            "- Theme-level similarity alone is not enough for full credit.\n"
            "- If the answer replaces a central hobby, milestone, or turning point with a different one, score at most partial.\n"
            "- If summary is empty or empty turning points are provided, score wrong.\n"
            "- If time_span overlaps but the answer gives only a generic arc/theme without retaining any central milestone, score wrong rather than partial.\n"
            "- If key_turning_points are generic or entirely replaced with different events, score wrong unless at least one central milestone is correctly retained.\n"
            "- one omitted or compressed milestone should usually score partial rather than wrong when the overall arc and major anchors are still preserved.\n"
            "- Partial credit requires a non-empty summary plus at least one correct turning point or factual anchor.\n"
            "- If the answer substitutes an unrelated theme, hobby, project, or belief system for the reference milestones, score wrong rather than partial.\n"
        )
    components_section = ""
    clean_components = [str(x).strip() for x in (answer_components or []) if str(x).strip()]
    if task_type == "Multi-session Reasoning" and clean_components:
        components_lines = "\n".join(f"- {item}" for item in clean_components)
        components_section = f"[Key Causal Anchors]\n{components_lines}\n\n"
    elif task_type == "Event Summarization" and clean_components:
        components_lines = "\n".join(f"- {item}" for item in clean_components)
        components_section = f"[Key Summary Anchors]\n{components_lines}\n\n"

    return (
        "You are an impartial grader for benchmark QA.\n"
        f"Task type: {task_type}\n\n"
        f"[Question]\n{query}\n\n"
        f"[Reference Answer]\n{reference}\n\n"
        f"{components_section}"
        "[Model Structured Output JSON]\n"
        f"{json.dumps(judge_payload, ensure_ascii=False, indent=2)}\n\n"
        f"Output ONLY valid JSON with keys: band, score, reason.\n"
        f"- band must be one of: {allowed}\n"
        "- score must be 1 or 0 for binary tasks, and 1/0.5/0 for three-level tasks.\n"
        "- Extra non-contradictory details should not be penalized; focus on whether the core answer is correct.\n"
        f"{task_guidance}"
        "- reason should be one concise sentence.\n"
    )


def _parse_judge_vote(raw_text: str, allow_partial: bool) -> Optional[Dict[str, Any]]:
    vote, _ = _parse_judge_vote_diagnostic(raw_text, allow_partial=allow_partial)
    return vote


def _preview_text(text: str, limit: int = 240) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _parse_judge_vote_diagnostic(
    raw_text: str,
    allow_partial: bool,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    payload, parse_error = _safe_json_loads(raw_text)
    if isinstance(payload, dict):
        band = str(payload.get("band", "")).strip().lower()
        score = payload.get("score")
        reason = str(payload.get("reason", "")).strip()
    else:
        lowered = (raw_text or "").lower()
        reason = ""
        if "correct" in lowered:
            band = "correct"
        elif "partial" in lowered:
            band = "partial"
        elif "wrong" in lowered:
            band = "wrong"
        else:
            band = ""
        score = None
        m = re.search(r"\b(?:1(?:\.0)?|0(?:\.5)?)\b", lowered)
        if m:
            score = float(m.group(0))

    if band not in {"correct", "partial", "wrong"}:
        failure_type = "invalid_json" if parse_error in {"invalid_json", "empty_response"} else "invalid_vote_schema"
        return None, {
            "failure_type": failure_type,
            "detail": parse_error or "missing_or_invalid_band",
            "raw_preview": _preview_text(raw_text),
        }
    if not allow_partial and band == "partial":
        band = "wrong"

    if score is None:
        score = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}[band]
    else:
        try:
            score = float(score)
        except Exception:
            return None, {
                "failure_type": "invalid_vote_schema",
                "detail": "non_numeric_score",
                "raw_preview": _preview_text(raw_text),
            }

    if not allow_partial and score not in {0.0, 1.0}:
        score = 1.0 if score >= 0.5 else 0.0
    if allow_partial and score not in {0.0, 0.5, 1.0}:
        score = 1.0 if score >= 0.75 else (0.0 if score <= 0.25 else 0.5)

    return {
        "band": band if allow_partial else ("correct" if score >= 0.5 else "wrong"),
        "score": score,
        "reason": reason,
    }, None


def _judge_once(
    task_type: str,
    query: str,
    reference: str,
    parsed_output: Dict[str, Any],
    judge_llm,
    allow_partial: bool,
    answer_components: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if not judge_llm:
        return {
            "ok": False,
            "vote": None,
            "failure": {
                "failure_type": "other_exception",
                "detail": "judge_llm_unavailable",
            },
        }
    prompt = _judge_prompt(
        task_type=task_type,
        query=query,
        reference=reference,
        parsed_output=parsed_output,
        allow_partial=allow_partial,
        answer_components=answer_components,
    )
    try:
        completion = asyncio.run(judge_llm.inference(prompt, max_tokens=128))
    except Exception as exc:
        return {
            "ok": False,
            "vote": None,
            "failure": {
                "failure_type": "api_exception",
                "detail": "inference_error",
                "exception_type": type(exc).__name__,
                "exception_message": _preview_text(str(exc), limit=180),
            },
        }
    try:
        decoded = judge_llm.decode(completion)
        vote, failure = _parse_judge_vote_diagnostic(decoded, allow_partial=allow_partial)
        if vote is not None:
            return {
                "ok": True,
                "vote": vote,
                "failure": None,
            }
        return {
            "ok": False,
            "vote": None,
            "failure": failure
            or {
                "failure_type": "other_exception",
                "detail": "judge_vote_parse_failed",
                "raw_preview": _preview_text(decoded),
            },
        }
    except Exception as exc:
        return {
            "ok": False,
            "vote": None,
            "failure": {
                "failure_type": "other_exception",
                "detail": "decode_error",
                "exception_type": type(exc).__name__,
                "exception_message": _preview_text(str(exc), limit=180),
            },
        }


def _judge_with_votes(
    task_type: str,
    query: str,
    reference: str,
    parsed_output: Dict[str, Any],
    judge_llm,
    allow_partial: bool,
    judge_votes: int,
    judge_tiebreak: bool,
    answer_components: Optional[List[str]] = None,
) -> Dict[str, Any]:
    votes: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    base_votes = max(1, int(judge_votes))
    for _ in range(base_votes):
        result = _judge_once(
            task_type=task_type,
            query=query,
            reference=reference,
            parsed_output=parsed_output,
            judge_llm=judge_llm,
            allow_partial=allow_partial,
            answer_components=answer_components,
        )
        if result.get("ok") and result.get("vote") is not None:
            votes.append(result["vote"])
        elif isinstance(result.get("failure"), dict):
            failures.append(result["failure"])

    disagreement = False
    used_tiebreak = False
    if len(votes) >= 2 and votes[0]["score"] != votes[1]["score"]:
        disagreement = True
        if judge_tiebreak:
            extra_result = _judge_once(
                task_type=task_type,
                query=query,
                reference=reference,
                parsed_output=parsed_output,
                judge_llm=judge_llm,
                allow_partial=allow_partial,
                answer_components=answer_components,
            )
            used_tiebreak = True
            if extra_result.get("ok") and extra_result.get("vote") is not None:
                votes.append(extra_result["vote"])
            elif isinstance(extra_result.get("failure"), dict):
                failures.append(extra_result["failure"])

    failure_counts: Dict[str, int] = {}
    for failure in failures:
        failure_type = str(failure.get("failure_type", "other_exception")).strip() or "other_exception"
        failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1

    if not votes:
        return {
            "score": None,
            "band": None,
            "votes": [],
            "disagreement": False,
            "used_tiebreak": used_tiebreak,
            "attempted_votes": base_votes + (1 if used_tiebreak else 0),
            "successful_votes": 0,
            "failure_counts": failure_counts,
            "failures": failures,
        }

    score_values = sorted(v["score"] for v in votes)
    mid = len(score_values) // 2
    final_score = float(score_values[mid])
    final_band = _band_from_score(final_score, allow_partial=allow_partial)
    return {
        "score": final_score,
        "band": final_band,
        "votes": votes,
        "disagreement": disagreement,
        "used_tiebreak": used_tiebreak,
        "attempted_votes": base_votes + (1 if used_tiebreak else 0),
        "successful_votes": len(votes),
        "failure_counts": failure_counts,
        "failures": failures,
    }


def evaluate_single_sample_tasklight(
    response: str,
    reference: str,
    metadata: Dict[str, Any],
    judge_llm=None,
    task_scoring_scheme: str = "A",
    judge_votes: int = 2,
    judge_tiebreak: bool = True,
    binary_fallback_judge: bool = True,
    scoring_config_path: Optional[str] = None,
) -> Dict[str, Any]:
    _ = task_scoring_scheme  # scheme A is currently fixed by config
    task_type = metadata.get("task_type", "Unknown")
    query = metadata.get("query", "")
    components = metadata.get("answer_components", []) or []
    cfg = load_task_scoring_config(scoring_config_path)
    score_mode = _score_mode(task_type, cfg)

    parsed_output, parse_ok, missing_fields, parse_error = _parse_task_output(
        response_text=response,
        task_type=task_type,
        cfg=cfg,
    )
    soft_missing_fields: List[str] = []
    parsed_output, parse_ok, missing_fields, soft_missing_fields, parse_error = _apply_soft_required_fields(
        task_type=task_type,
        parsed_output=parsed_output,
        parse_ok=parse_ok,
        missing_fields=missing_fields,
        parse_error=parse_error,
    )

    rule_signals: Dict[str, Any] = {
        "task_type": task_type,
        "score_mode": score_mode,
        "missing_fields": soft_missing_fields + missing_fields,
        "parse_error": parse_error,
        "format_compliant": parse_ok,
    }
    judge_band: Optional[str] = None
    judge_score: Optional[float] = None
    judge_meta: Dict[str, Any] = {
        "votes": [],
        "disagreement": False,
        "used_tiebreak": False,
        "attempted_votes": 0,
        "successful_votes": 0,
        "failure_counts": {},
        "failures": [],
    }

    if not parse_ok or parsed_output is None:
        scores: Dict[str, Optional[float]] = {"final_score": 0.0}
        if task_type == "Temporal Reasoning":
            scores["structured_final_score"] = 0.0
            scores.update(_temporal_aux_scores(response, reference, parse_ok=False, final_score=0.0))
            scores["final_score"] = float(scores["semantic_exact_correct"])
        return {
            "scores": scores,
            "parsed_output": parsed_output,
            "parse_ok": False,
            "rule_signals": rule_signals,
            "judge_band": None,
            "judge_score": None,
            "final_score": float(scores["final_score"]),
            "score_source": "rule",
            "judge_meta": judge_meta,
        }

    score_source = "rule"
    final_score = 0.0
    rule_high_confidence = False

    if score_mode == "binary":
        if task_type == "Information Extraction":
            rule_score, task_signals, rule_high_confidence = _binary_information_extraction(
                parsed_output, query, reference, components
            )
        elif task_type == "Temporal Reasoning":
            rule_score, task_signals, rule_high_confidence = _binary_temporal_reasoning(
                parsed_output, reference
            )
        elif task_type == "Knowledge Updating":
            rule_score, task_signals, rule_high_confidence = _binary_knowledge_updating(
                parsed_output, query, reference, components
            )
        else:
            ratio = _fuzzy_ratio(json.dumps(parsed_output, ensure_ascii=False), reference)
            rule_score = 1.0 if ratio >= 0.8 else 0.0
            task_signals = {"fallback_ratio": ratio}
            rule_high_confidence = True

        rule_signals.update(task_signals)
        if rule_score is None and binary_fallback_judge and judge_llm:
            judge_meta = _judge_with_votes(
                task_type=task_type,
                query=query,
                reference=reference,
                parsed_output=parsed_output,
                judge_llm=judge_llm,
                allow_partial=False,
                judge_votes=judge_votes,
                judge_tiebreak=judge_tiebreak,
                answer_components=components,
            )
            if judge_meta["score"] is not None:
                judge_score = 1.0 if float(judge_meta["score"]) >= 0.5 else 0.0
                judge_band = "correct" if judge_score >= 0.5 else "wrong"
                final_score = judge_score
                score_source = "judge"
            else:
                final_score = 0.0
        else:
            final_score = float(rule_score if rule_score is not None else 0.0)

        scores: Dict[str, Optional[float]] = {"final_score": final_score}
        if task_type == "Temporal Reasoning":
            structured_final_score = float(final_score)
            scores["structured_final_score"] = structured_final_score
            scores.update(_temporal_aux_scores(response, reference, parse_ok=True, final_score=final_score))
            final_score = float(scores["semantic_exact_correct"])
            scores["final_score"] = final_score
        if "component_recall" in rule_signals:
            scores["component_recall"] = float(rule_signals["component_recall"])

    else:
        if task_type == "Multi-session Reasoning":
            rule_score, task_signals, rule_high_confidence = _three_level_multisession(
                parsed_output, reference, components
            )
        elif task_type == "Event Summarization":
            rule_score, task_signals, rule_high_confidence = _three_level_summarization(
                parsed_output, query, reference, components
            )
        elif task_type == "Memory Arbitration":
            rule_score, task_signals, rule_high_confidence = _three_level_memory_arbitration(
                parsed_output, reference, components
            )
        else:
            ratio = _fuzzy_ratio(json.dumps(parsed_output, ensure_ascii=False), reference)
            rule_score = 1.0 if ratio >= 0.8 else (0.5 if ratio >= 0.4 else 0.0)
            task_signals = {"fallback_ratio": ratio}
            rule_high_confidence = ratio >= 0.8 or ratio <= 0.2

        rule_signals.update(task_signals)
        judge_skip_reason = _should_skip_tasklight_judge(task_type, rule_signals)
        if judge_skip_reason is not None:
            rule_signals["judge_guard"] = judge_skip_reason
        if judge_llm and judge_skip_reason is None:
            judge_meta = _judge_with_votes(
                task_type=task_type,
                query=query,
                reference=reference,
                parsed_output=parsed_output,
                judge_llm=judge_llm,
                allow_partial=True,
                judge_votes=judge_votes,
                judge_tiebreak=judge_tiebreak,
                answer_components=components,
            )
            if judge_meta["score"] is not None:
                judge_score = float(judge_meta["score"])
                judge_band = str(judge_meta["band"])
                final_score = judge_score
                score_source = "judge"
                judge_floor_score = rule_signals.get("judge_floor_score")
                if judge_floor_score is not None and final_score < float(judge_floor_score):
                    final_score = float(judge_floor_score)
                    score_source = "rule_floor"
                    rule_signals["judge_floor_applied"] = True
            else:
                final_score = float(rule_score)
        else:
            final_score = float(rule_score)

        scores = {"final_score": final_score}
        if judge_score is not None:
            scores["llm_judge"] = judge_score
        if "component_recall" in rule_signals:
            scores["component_recall"] = float(rule_signals["component_recall"])

    return {
        "scores": scores,
        "parsed_output": parsed_output,
        "parse_ok": True,
        "rule_signals": rule_signals,
        "judge_band": judge_band,
        "judge_score": judge_score,
        "final_score": float(scores.get("final_score", 0.0)),
        "score_source": score_source,
        "judge_meta": judge_meta,
    }


def eval_llm_judge(
    response_text: str,
    reference: str,
    task_type: str,
    query: str,
    llm_client,
) -> Optional[float]:
    if not llm_client:
        return None
    allow_partial = task_type in THREE_LEVEL_TASKS
    payload = {"answer": response_text}
    judged = _judge_with_votes(
        task_type=task_type,
        query=query,
        reference=reference,
        parsed_output=payload,
        judge_llm=llm_client,
        allow_partial=allow_partial,
        judge_votes=1,
        judge_tiebreak=False,
    )
    if judged["score"] is None:
        return None
    return float(judged["score"])


def evaluate_single_sample_v2(
    response: str,
    reference: str,
    metadata: Dict[str, Any],
    compute_semantic: bool = True,
    client=None,
    judge_llm=None,
) -> Dict[str, Optional[float]]:
    _ = (compute_semantic, client)
    scores: Dict[str, Optional[float]] = {}
    task_type = metadata.get("task_type", "Unknown")
    components = metadata.get("answer_components", [])
    query = metadata.get("query", "")

    if task_type == "Temporal Reasoning":
        scores["temporal_exact_match"] = eval_temporal_exact_match(response, reference)
        if components:
            scores["component_recall"] = eval_component_recall(response, components)
    elif task_type == "Information Extraction":
        if components:
            scores["component_recall"] = eval_component_recall(response, components)
        if judge_llm:
            scores["llm_judge"] = eval_llm_judge(
                response, reference, task_type, query, judge_llm
            )
    elif task_type in {"Multi-session Reasoning", "Event Summarization", "Knowledge Updating", "Memory Arbitration"}:
        if judge_llm:
            scores["llm_judge"] = eval_llm_judge(
                response, reference, task_type, query, judge_llm
            )
        if task_type != "Memory Arbitration" and components:
            scores["component_recall"] = eval_component_recall(response, components)
    else:
        if components:
            scores["component_recall"] = eval_component_recall(response, components)
    return scores


def evaluate_batch(
    responses: List[str],
    references: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    compute_semantic: bool = True,
    semantic_workers: int = 10,
    cpu_workers: int = None,
    judge_llm=None,
    eval_profile: str = "umb_tasklight_v1",
    task_scoring_scheme: str = "A",
    judge_votes: int = 2,
    judge_tiebreak: bool = True,
    binary_fallback_judge: bool = True,
    scoring_config_path: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    _ = (compute_semantic, cpu_workers)
    if len(responses) != len(references):
        raise ValueError("responses and references length mismatch")

    n = len(responses)
    if metadatas is None:
        metadatas = [{} for _ in range(n)]

    workers = semantic_workers or 10
    profile = str(eval_profile or "").strip()

    if profile != "umb_tasklight_v1":
        def _process_legacy(idx: int) -> Dict[str, Optional[float]]:
            return evaluate_single_sample_v2(
                response=responses[idx],
                reference=references[idx],
                metadata=metadatas[idx],
                compute_semantic=compute_semantic,
                judge_llm=judge_llm,
            )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            individual_scores = list(executor.map(_process_legacy, range(n)))

        aggregate_scores: Dict[str, float] = {}
        all_keys = set()
        for sample in individual_scores:
            all_keys.update(sample.keys())
        for key in all_keys:
            values = [s[key] for s in individual_scores if s.get(key) is not None]
            if values:
                aggregate_scores[f"avg_{key}"] = _mean(values)
                aggregate_scores[f"std_{key}"] = _std(values)
                aggregate_scores[f"min_{key}"] = float(min(values))
                aggregate_scores[f"max_{key}"] = float(max(values))
                aggregate_scores[f"valid_count_{key}"] = len(values)
                aggregate_scores[f"failed_count_{key}"] = len(individual_scores) - len(values)
        return individual_scores, aggregate_scores

    def _process_tasklight(idx: int) -> Dict[str, Any]:
        return evaluate_single_sample_tasklight(
            response=responses[idx],
            reference=references[idx],
            metadata=metadatas[idx],
            judge_llm=judge_llm,
            task_scoring_scheme=task_scoring_scheme,
            judge_votes=judge_votes,
            judge_tiebreak=judge_tiebreak,
            binary_fallback_judge=binary_fallback_judge,
            scoring_config_path=scoring_config_path,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(_process_tasklight, range(n)))

    aggregate_scores: Dict[str, float] = {}
    score_keys = set()
    for row in rows:
        score_keys.update((row.get("scores") or {}).keys())
    for key in score_keys:
        values = [
            row["scores"][key]
            for row in rows
            if isinstance(row.get("scores"), dict) and row["scores"].get(key) is not None
        ]
        if values:
            aggregate_scores[f"avg_{key}"] = _mean(values)
            aggregate_scores[f"std_{key}"] = _std(values)
            aggregate_scores[f"min_{key}"] = float(min(values))
            aggregate_scores[f"max_{key}"] = float(max(values))
            aggregate_scores[f"valid_count_{key}"] = len(values)
            aggregate_scores[f"failed_count_{key}"] = len(rows) - len(values)

    return rows, aggregate_scores
