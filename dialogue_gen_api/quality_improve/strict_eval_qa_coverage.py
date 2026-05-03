#!/usr/bin/env python3
"""
Strict QA coverage audit with task-type-aware component verification.

Functionality:
1. Determine which answer_components must appear in the dialogue and which are inferred results.
2. Use rule-based pre-screening over dates, numbers, and phrases.
3. Verify uncertain components with an LLM using focused local evidence windows.
4. Export a detailed audit report.

Usage:
    python strict_eval_qa_coverage.py \
        --input /path/to/dialogues.json \
        --output eval_report.json \
        --use_llm              # Enable LLM verification.
        --concurrency 10       # Number of concurrent workers.
        --limit 5              # Evaluate only the first N dialogues for debugging.
"""

import os
import sys
import json
import re
import argparse
import asyncio
import time
import random
import hashlib
from threading import Lock
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]

# Load .env
try:
    from dotenv import load_dotenv
    env_path = SCRIPT_DIR / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()
except ImportError:
    pass

# Import LLM client
try:
    from llm import get_llm, LLM
except ImportError:
    sys.path.insert(0, str(SCRIPT_DIR))
    from llm import get_llm, LLM


# ============================================================
# Data structures
# ============================================================

@dataclass
class ComponentVerification:
    """Verification result for one answer component."""
    task_idx: int
    comp_idx: int
    task_type: str
    component_text: str
    should_verify: bool           # Whether this component needs verification; inferred results are False.
    component_role: str = ""      # EVIDENCE_REQUIRED / INFERRED_RESULT
    component_role_source: str = ""  # llm_vote / heuristic_rule / fallback
    component_role_confidence: str = ""
    component_role_votes: Dict[str, Any] = field(default_factory=dict)
    component_role_lock_hit: bool = False
    component_role_lock_key: str = ""
    query: str = ""
    answer_components: List[str] = field(default_factory=list)
    target_component_idx: int = -1
    rule_status: str = ""         # PRESENT / SUSPICIOUS / MISSING
    rule_confidence: str = ""     # high / medium / low
    rule_reason_code: str = ""    # Auditable rule code.
    hard_status: str = ""         # PASS / HARD_MISSING
    hard_initial_status: str = ""  # Initial hard-constraint result.
    hard_gate_applied: bool = False
    hard_gate_result: str = ""     # present_override / missing_confirmed / pending
    hard_missing_facts: List[str] = field(default_factory=list)
    llm_status: str = ""          # PRESENT / MISSING / SKIPPED / PENDING / ERROR
    final_status: str = ""        # PRESENT / MISSING / PENDING_LLM / SKIPPED
    decision_stage: str = ""      # skipped / hard / rule / llm_pending / llm_rechecked
    decision_reason_code: str = ""
    llm_vote_meta: Dict[str, Any] = field(default_factory=dict)
    required_slots: List[str] = field(default_factory=list)
    slot_verdicts: Dict[str, bool] = field(default_factory=dict)
    evidence_turns: List[int] = field(default_factory=list)
    date_equivalent_match_count: int = 0
    hard_number_filtered: List[str] = field(default_factory=list)
    constraint_spec: Dict[str, Any] = field(default_factory=dict)
    constraint_source: str = ""    # llm_object / fallback_rule
    hard_object_reason: str = ""
    match_policy: Dict[str, str] = field(default_factory=dict)
    constraint_validation_errors: List[str] = field(default_factory=list)
    constraint_anchor_hits: Dict[str, Any] = field(default_factory=dict)
    constraint_degraded_to_soft: bool = False
    matched_turns: List[int] = field(default_factory=list)  # Matched turn indices.
    retrieval_centers_used: int = 0
    retrieval_route_slot_hits: int = 0
    retrieval_route_date_hits: int = 0
    retrieval_route_semantic_hits: int = 0
    retrieval_overlap_ratio: float = 0.0
    decision_complete: bool = True
    repair_ready: bool = False
    decision_confidence: str = ""
    reason: str = ""


@dataclass
class ConstraintSpec:
    hard_slots: List[str] = field(default_factory=list)
    soft_slots: List[str] = field(default_factory=list)
    forbidden_hard_slots: List[str] = field(default_factory=list)
    slot_values: Dict[str, List[str]] = field(default_factory=dict)
    match_policy: Dict[str, str] = field(default_factory=dict)
    rationale: str = ""


@dataclass
class DialogueEvalResult:
    """Evaluation result for one dialogue."""
    dialogue_idx: int
    character: str
    dialogue_id: str
    total_components: int = 0
    skipped_inference: int = 0    # Reasoning-only components skipped by design.
    verifiable: int = 0           # Components that require verification.
    present: int = 0
    missing: int = 0
    hard_missing_count: int = 0
    hard_missing_initial_count: int = 0
    hard_missing_llm_overridden_present: int = 0
    hard_missing_llm_confirmed_missing: int = 0
    hard_missing_llm_pending: int = 0
    regression_guard_triggered_count: int = 0
    llm_rechecked_present_count: int = 0
    revote_trigger_count: int = 0
    number_label_filtered_count: int = 0
    llm_no_evidence_reject_count: int = 0
    llm_object_hard_count: int = 0
    llm_object_soft_count: int = 0
    hard_object_conflict_count: int = 0
    constraint_validation_reject_count: int = 0
    hard_slot_degraded_count: int = 0
    date_equivalent_match_count: int = 0
    unfinished_components_count: int = 0
    failed_components_for_rerun: int = 0
    retry_rounds_total: int = 0
    components_retried_count: int = 0
    max_rounds_single_component: int = 0
    network_retry_events: int = 0
    component_role_lock_hits: int = 0
    component_role_lock_misses: int = 0
    constraint_llm_success_tasks: int = 0
    constraint_fallback_tasks: int = 0
    constraint_partial_parse_rejects: int = 0
    constraint_retry_count: int = 0
    retrieval_route_slot_hits: int = 0
    retrieval_route_date_hits: int = 0
    retrieval_route_semantic_hits: int = 0
    retrieval_overlap_ratio_sum: float = 0.0
    retrieval_overlap_ratio_samples: int = 0
    focused_to_global_escalations: int = 0
    focused_strong_yes_shortcuts: int = 0
    global_final_present: int = 0
    global_final_missing: int = 0
    vote_target_valid_votes_focused: int = 0
    vote_target_valid_votes_global: int = 0
    vote_majority_threshold_focused: int = 0
    vote_majority_threshold_global: int = 0
    details: List[ComponentVerification] = field(default_factory=list)


# ============================================================
# Component role classification
# ============================================================

def should_verify_component(task_type: str, component: str) -> bool:
    """Return whether a component should explicitly appear in the dialogue."""
    comp_lower = component.lower().strip()

    if task_type == "Temporal Reasoning":
        # "Calculation: 188 days" is an inferred result and should be skipped.
        if comp_lower.startswith("calculation"):
            return False

    if task_type == "Memory Arbitration":
        # "Correction: He was emotionally absent" is a reasoning judgment and should be skipped.
        if comp_lower.startswith("correction"):
            return False
        # "Fundamental difference: From 'fixing' to 'connecting'" is an inferred conclusion.
        if comp_lower.startswith("fundamental difference"):
            return False

    return True


def _extract_years(text: str) -> List[int]:
    years: List[int] = []
    for match in re.findall(r'\b(19\d{2}|20\d{2}|21\d{2})\b', text or ""):
        try:
            years.append(int(match))
        except ValueError:
            continue
    return years


def _component_content_tokens(text: str) -> List[str]:
    lowered = (text or "").lower()
    lowered = re.sub(r'\b(19\d{2}|20\d{2}|21\d{2})\b', ' ', lowered)
    lowered = re.sub(
        r'\b(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|'
        r'sep|sept|september|oct|october|nov|november|dec|december)\b',
        ' ',
        lowered,
    )
    lowered = re.sub(r'[^a-z]+', ' ', lowered)
    stop = {
        "period", "duration", "range", "timespan", "time", "span", "window",
        "from", "to", "through", "between", "and", "of", "the", "a", "an",
        "late", "early", "mid", "start", "end", "key", "events", "event",
        "outcome", "turning", "point", "result", "summary", "phase",
    }
    return [w for w in lowered.split() if len(w) > 2 and w not in stop]


def _is_period_like_component(text: str) -> bool:
    lowered = (text or "").lower().strip()
    if re.search(r'\b(period|duration|time\s*span|timespan|range|window|phase)\b', lowered):
        return True
    if re.search(r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}\s*(?:-|–|—|to|through)\s*(?:[a-z]{3,9}\s+\d{4}|\d{4})\b', lowered):
        return True
    if re.search(r'\b\d{4}-\d{2}(?:-\d{2})?\s*(?:-|–|—|to|through)\s*\d{4}-\d{2}(?:-\d{2})?\b', lowered):
        return True
    return False


def _is_period_only_component(text: str) -> bool:
    if not _is_period_like_component(text):
        return False
    # Period-only components are usually covered by sibling event components.
    return len(_component_content_tokens(text)) <= 2


def _period_component_covered_by_siblings(components: List[str], target_idx: int) -> bool:
    if target_idx < 0 or target_idx >= len(components):
        return False
    target = components[target_idx]
    if not _is_period_only_component(target):
        return False

    target_years = set(_extract_years(target))
    sibling_years: set[int] = set()
    substantive_sibling = False
    for idx, comp in enumerate(components):
        if idx == target_idx:
            continue
        sibling_years.update(_extract_years(comp))
        if len(_component_content_tokens(comp)) >= 2:
            substantive_sibling = True

    if not substantive_sibling:
        return False
    if target_years and target_years.issubset(sibling_years):
        return True
    return False


def _apply_component_role_guards(
    task_type: str,
    components: List[str],
    decisions: Dict[int, Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    if not components or not decisions:
        return decisions

    guarded = {
        idx: dict(info) for idx, info in decisions.items()
    }
    for idx in range(len(components)):
        info = guarded.get(idx)
        if not info:
            continue
        if info.get("role") != "EVIDENCE_REQUIRED":
            continue

        if _period_component_covered_by_siblings(components, idx):
            src = str(info.get("source", "")).strip()
            info["role"] = "COVERED_BY_OTHER_COMPONENT"
            info["should_verify"] = False
            info["source"] = f"{src}+period_coverage_guard" if src else "period_coverage_guard"
            if info.get("confidence") == "high":
                info["confidence"] = "medium"
            guarded[idx] = info

    return guarded


def _build_component_role_lock_key(task_type: str, query: str, components: List[str]) -> str:
    payload = {
        "task_type": task_type or "",
        "query": query or "",
        "components": list(components or []),
    }
    return hashlib.sha1(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _load_component_role_lock(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    lock_path = Path(path)
    if not lock_path.exists():
        return {}
    try:
        with open(lock_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _save_component_role_lock(path: str, lock_store: Dict[str, Any]) -> None:
    if not path:
        return
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as f:
        json.dump(lock_store, f, ensure_ascii=False, indent=2)


def _load_constraint_cache(path: str) -> Dict[str, Any]:
    """Load the constraint_spec cache from disk."""
    if not path:
        return {}
    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _save_constraint_cache(path: str, cache: Dict[str, Any]) -> None:
    """Save the constraint_spec cache to disk."""
    if not path:
        return
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Drop internal fields that are not JSON-serializable.
    serializable = {}
    for k, v in cache.items():
        if isinstance(v, dict):
            clean = {}
            for vk, vv in v.items():
                if isinstance(vv, dict):
                    clean_inner = {ik: iv for ik, iv in vv.items() if not isinstance(iv, set)}
                    clean[vk] = clean_inner
                elif not isinstance(vv, set):
                    clean[vk] = vv
            serializable[k] = clean
        elif not isinstance(v, set):
            serializable[k] = v
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def _extract_locked_roles(record: Any, num_components: int) -> Dict[int, str]:
    if not isinstance(record, dict):
        return {}
    decisions_raw = record.get("decisions", record)
    if not isinstance(decisions_raw, dict):
        return {}
    parsed: Dict[int, str] = {}
    for idx in range(num_components):
        item = decisions_raw.get(str(idx))
        if item is None:
            item = decisions_raw.get(idx)
        role = ""
        if isinstance(item, dict):
            role = str(item.get("role", "")).strip().upper()
        elif isinstance(item, str):
            role = item.strip().upper()
        if role in {
            "EVIDENCE_REQUIRED",
            "INFERRED_RESULT",
            "COVERED_BY_OTHER_COMPONENT",
            "NON_ESSENTIAL",
        }:
            parsed[idx] = role
    return parsed if len(parsed) == num_components else {}


def build_component_role_prompt(task_type: str, query: str, components: List[str]) -> str:
    lines = [f"{idx}. {comp}" for idx, comp in enumerate(components)]
    components_text = "\n".join(lines) if lines else "(none)"
    return f"""You are a QA answer-component role classifier.

Task type:
{task_type}

Question:
{query}

Answer components:
{components_text}

Classify each component into exactly one role:
- EVIDENCE_REQUIRED: this fact should be explicitly present in dialogue content.
- INFERRED_RESULT: this is a reasoning/calculation/comparison result that should be derived from dialogue evidence, not explicitly injected.
- COVERED_BY_OTHER_COMPONENT: this component is redundant because other components already cover the same information for this QA.
- NON_ESSENTIAL: this component is background/stylistic and not required to answer the QA faithfully.

Rules:
1. Use a leave-one-out test for each component: if the QA can still be answered faithfully from remaining components, do NOT mark it EVIDENCE_REQUIRED.
2. Temporal duration/count final answers are usually INFERRED_RESULT if derivable from other components.
3. A period/range-only component (e.g., "Period: Nov 2023 - Feb 2025") should be COVERED_BY_OTHER_COMPONENT when sibling components already contain dated events/outcomes spanning that range.
4. Date/entity/event anchors are EVIDENCE_REQUIRED only when they add unique factual information not already covered by siblings.
5. Return one role for every component index.

Output JSON only:
{{
  "components": [
    {{"index": 0, "role": "EVIDENCE_REQUIRED", "reason": "..."}},
    {{"index": 1, "role": "INFERRED_RESULT", "reason": "..."}}
  ]
}}"""


def _parse_component_role_response(response: str, num_components: int) -> Dict[int, str]:
    if not response:
        return {}
    role_map: Dict[int, str] = {}
    role_alias = {
        "EVIDENCE_REQUIRED": "EVIDENCE_REQUIRED",
        "EVIDENCE": "EVIDENCE_REQUIRED",
        "EXPLICIT": "EVIDENCE_REQUIRED",
        "MUST_APPEAR": "EVIDENCE_REQUIRED",
        "INFERRED_RESULT": "INFERRED_RESULT",
        "INFERRED": "INFERRED_RESULT",
        "DERIVED": "INFERRED_RESULT",
        "COMPUTED": "INFERRED_RESULT",
        "REASONING_ONLY": "INFERRED_RESULT",
        "COVERED_BY_OTHER_COMPONENT": "COVERED_BY_OTHER_COMPONENT",
        "COVERED_BY_OTHERS": "COVERED_BY_OTHER_COMPONENT",
        "REDUNDANT": "COVERED_BY_OTHER_COMPONENT",
        "SUBSUMED": "COVERED_BY_OTHER_COMPONENT",
        "NON_ESSENTIAL": "NON_ESSENTIAL",
        "OPTIONAL": "NON_ESSENTIAL",
        "NOT_REQUIRED": "NON_ESSENTIAL",
    }
    for candidate in _extract_json_candidates(response):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        items = []
        if isinstance(payload, dict):
            if isinstance(payload.get("components"), list):
                items = payload.get("components", [])
            elif isinstance(payload.get("items"), list):
                items = payload.get("items", [])
        elif isinstance(payload, list):
            items = payload

        if not isinstance(items, list):
            continue

        parsed: Dict[int, str] = {}
        for i, raw in enumerate(items):
            if not isinstance(raw, dict):
                continue
            idx = raw.get("index", i)
            try:
                idx = int(idx)
            except (TypeError, ValueError):
                continue
            if idx < 0 or idx >= num_components:
                continue

            role = str(raw.get("role", "")).strip().upper()
            if not role and "should_verify" in raw:
                role = "EVIDENCE_REQUIRED" if _to_bool(raw.get("should_verify")) else "INFERRED_RESULT"
            normalized = role_alias.get(role)
            if normalized is None:
                continue
            parsed[idx] = normalized

        if len(parsed) == num_components:
            return parsed
        role_map = parsed

    return role_map if len(role_map) == num_components else {}


def classify_component_roles(
    task_type: str,
    query: str,
    components: List[str],
    llm: Optional[LLM] = None,
    use_llm: bool = False,
    target_valid_votes: int = 3,
    max_vote_attempts: int = 8,
    role_lock_store: Optional[Dict[str, Any]] = None,
    role_lock_mode: str = "read_write",
    role_lock_mutex: Optional[Lock] = None,
) -> Dict[int, Dict[str, Any]]:
    decisions: Dict[int, Dict[str, Any]] = {}
    if not components:
        return decisions

    all_roles = [
        "EVIDENCE_REQUIRED",
        "INFERRED_RESULT",
        "COVERED_BY_OTHER_COMPONENT",
        "NON_ESSENTIAL",
    ]

    def heuristic(idx: int, source: str, confidence: str = "low") -> Dict[str, Any]:
        comp = components[idx]
        sv = should_verify_component(task_type, comp)
        return {
            "role": "EVIDENCE_REQUIRED" if sv else "INFERRED_RESULT",
            "should_verify": sv,
            "source": source,
            "confidence": confidence,
        }

    role_lock_mode = (role_lock_mode or "read_write").strip().lower()
    if role_lock_mode not in {"read_write", "read_only", "off"}:
        role_lock_mode = "read_write"
    lock_enabled = role_lock_mode != "off"
    lock_key = _build_component_role_lock_key(task_type, query, components) if lock_enabled else ""

    def _attach_lock_meta(
        items: Dict[int, Dict[str, Any]],
        lock_hit: bool,
    ) -> Dict[int, Dict[str, Any]]:
        for idx in range(len(components)):
            item = items.get(idx)
            if not item:
                continue
            item["lock_hit"] = bool(lock_hit)
            item["lock_key"] = lock_key
            item["lock_checked"] = bool(lock_enabled and lock_key)
        return items

    if lock_enabled and role_lock_store is not None and lock_key:
        if role_lock_mutex is not None:
            with role_lock_mutex:
                locked_record = role_lock_store.get(lock_key)
        else:
            locked_record = role_lock_store.get(lock_key)
        locked_roles = _extract_locked_roles(locked_record, len(components))
        if locked_roles:
            for idx in range(len(components)):
                role = locked_roles[idx]
                should_verify = role == "EVIDENCE_REQUIRED"
                decisions[idx] = {
                    "role": role,
                    "should_verify": should_verify,
                    "source": "lock",
                    "confidence": "high",
                    "votes": {
                        "target_valid_votes": 0,
                        "max_vote_attempts": 0,
                        "attempts_used": 0,
                        "valid_votes": 0,
                        "role_counts": {r: 0 for r in all_roles},
                    },
                }
            guarded = _apply_component_role_guards(task_type, components, decisions)
            return _attach_lock_meta(guarded, lock_hit=True)

    if not use_llm or llm is None:
        for idx in range(len(components)):
            base = heuristic(idx, source="heuristic_rule", confidence="medium")
            base["votes"] = {
                "target_valid_votes": 0,
                "max_vote_attempts": 0,
                "attempts_used": 0,
                "valid_votes": 0,
                "role_counts": {role: 0 for role in all_roles},
            }
            decisions[idx] = base
        guarded = _apply_component_role_guards(task_type, components, decisions)
        return _attach_lock_meta(guarded, lock_hit=False)

    prompt = build_component_role_prompt(task_type, query, components)
    n = len(components)
    role_votes = [{role: 0 for role in all_roles} for _ in range(n)]
    attempts_used = 0
    valid_votes = 0
    target_valid_votes = max(1, int(target_valid_votes))
    max_vote_attempts = max(1, int(max_vote_attempts))

    while valid_votes < target_valid_votes and attempts_used < max_vote_attempts:
        attempts_used += 1
        response = _run_llm_once_sync(llm, prompt, max_tokens=1200)
        parsed = _parse_component_role_response(response or "", n)
        if len(parsed) != n:
            continue
        valid_votes += 1
        for idx in range(n):
            role = parsed.get(idx, "EVIDENCE_REQUIRED")
            if role not in role_votes[idx]:
                role = "EVIDENCE_REQUIRED"
            role_votes[idx][role] += 1

    for idx in range(n):
        role_count = dict(role_votes[idx])
        ev = role_count.get("EVIDENCE_REQUIRED", 0)
        iv = role_count.get("INFERRED_RESULT", 0)
        cv = role_count.get("COVERED_BY_OTHER_COMPONENT", 0)
        nv = role_count.get("NON_ESSENTIAL", 0)
        vote_meta = {
            "target_valid_votes": target_valid_votes,
            "max_vote_attempts": max_vote_attempts,
            "attempts_used": attempts_used,
            "valid_votes": valid_votes,
            "evidence_votes": ev,
            "inferred_votes": iv,
            "covered_votes": cv,
            "non_essential_votes": nv,
            "role_counts": role_count,
        }
        if valid_votes == 0:
            base = heuristic(idx, source="heuristic_no_valid_votes", confidence="low")
            base["votes"] = vote_meta
            decisions[idx] = base
            continue

        top_count = max(role_count.values()) if role_count else 0
        top_roles = [r for r, c in role_count.items() if c == top_count and c > 0]
        if len(top_roles) != 1:
            base = heuristic(idx, source="llm_vote_tie_heuristic_fallback", confidence="low")
            base["votes"] = vote_meta
            decisions[idx] = base
            continue

        role = top_roles[0]
        should_verify = role == "EVIDENCE_REQUIRED"
        source = "llm_vote"
        confidence = "high" if top_count == valid_votes else "medium"
        decisions[idx] = {
            "role": role,
            "should_verify": should_verify,
            "source": source,
            "confidence": confidence,
            "votes": vote_meta,
        }

    guarded = _apply_component_role_guards(task_type, components, decisions)
    guarded = _attach_lock_meta(guarded, lock_hit=False)

    if lock_enabled and role_lock_mode == "read_write" and role_lock_store is not None and lock_key:
        lock_payload = {
            "task_type": task_type or "",
            "query": query or "",
            "components": list(components or []),
            "decisions": {
                str(idx): {
                    "role": str(guarded[idx].get("role", "EVIDENCE_REQUIRED")),
                    "should_verify": bool(guarded[idx].get("should_verify", True)),
                }
                for idx in range(len(components))
                if idx in guarded
            },
        }
        if role_lock_mutex is not None:
            with role_lock_mutex:
                role_lock_store[lock_key] = lock_payload
        else:
            role_lock_store[lock_key] = lock_payload

    return guarded


# ============================================================
# Date extraction utilities
# ============================================================

# Month name mapping
MONTH_NAMES = {
    'january': '01', 'february': '02', 'march': '03', 'april': '04',
    'may': '05', 'june': '06', 'july': '07', 'august': '08',
    'september': '09', 'october': '10', 'november': '11', 'december': '12',
    'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
    'jun': '06', 'jul': '07', 'aug': '08', 'sep': '09',
    'oct': '10', 'nov': '11', 'dec': '12'
}

MONTH_NUM_TO_NAME = {
    '01': 'january', '02': 'february', '03': 'march', '04': 'april',
    '05': 'may', '06': 'june', '07': 'july', '08': 'august',
    '09': 'september', '10': 'october', '11': 'november', '12': 'december'
}


def extract_dates_from_text(text: str) -> List[str]:
    """Extract dates in YYYY-MM-DD format from text."""
    return re.findall(r'\d{4}-\d{2}-\d{2}', text)


def generate_date_variants(date_str: str) -> List[str]:
    """
    Generate search variants from a YYYY-MM-DD date:
    2020-03-15 -> ["2020-03-15", "march 15", "march 15th", "march 2020", "15th of march"]
    """
    variants = [date_str]
    try:
        parts = date_str.split('-')
        year, month, day = parts[0], parts[1], parts[2]
        month_name = MONTH_NUM_TO_NAME.get(month, '')
        day_int = int(day)

        if month_name:
            # "March 15, 2020"  /  "March 15th, 2020"
            variants.append(f"{month_name} {day_int}")
            variants.append(f"{month_name} {day_int}th")
            variants.append(f"{month_name} {day_int}st" if day_int == 1 or day_int == 21 or day_int == 31 else "")
            variants.append(f"{month_name} {day_int}nd" if day_int == 2 or day_int == 22 else "")
            variants.append(f"{month_name} {day_int}rd" if day_int == 3 or day_int == 23 else "")
            # "March 2020" (looser match)
            variants.append(f"{month_name} {year}")
            # "March of 2020"
            variants.append(f"{month_name} of {year}")

        # Drop empty strings
        variants = [v for v in variants if v]
    except (ValueError, IndexError):
        pass

    return variants


def search_date_in_dialogue(date_str: str, dialogue: List[Dict]) -> List[int]:
    """Search a date in dialogue turns and return matched turn indices."""
    variants = generate_date_variants(date_str)
    matched_turns = []

    for i, turn in enumerate(dialogue):
        content_lower = turn['content'].lower()
        for variant in variants:
            if variant.lower() in content_lower:
                matched_turns.append(i)
                break  # Count each turn once

    return matched_turns


def extract_date_ranges_from_text(text: str) -> List[Tuple[str, str]]:
    """Extract date ranges such as 2023-01-20 to 2023-08-18."""
    ranges = []
    pattern = re.compile(r'(\d{4}-\d{2}-\d{2})\s*(?:to|~|-)\s*(\d{4}-\d{2}-\d{2})')
    for m in pattern.finditer(text):
        ranges.append((m.group(1), m.group(2)))
    return ranges


def extract_month_year_refs(text: str) -> List[str]:
    """Extract Month Year references such as Jun 2024."""
    return re.findall(
        r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}\b',
        text,
        flags=re.IGNORECASE,
    )


def infer_query_intents(query: str) -> Dict[str, bool]:
    q = (query or "").lower()
    return {
        "count_intent": bool(re.search(r'\b(how many|number of|exactly|count)\b', q)),
        "temporal_intent": bool(re.search(r'\b(duration|interval|elapsed|between|days|months|years|when)\b', q)),
        "status_intent": bool(re.search(r'\b(as of|currently|current|latest|status|is this correct)\b', q)),
    }


def get_task_judge_profile(task_type: str, query: str, task_profile: str = "qa_task_v1") -> Dict[str, Any]:
    """
    Task-aware judge profile configuration (v3).
    """
    intents = infer_query_intents(query)

    profile = {
        "enforce_iso_date": False,
        "enforce_number": False,
        "month_year_soft": True,
        "required_slots_default": [],
    }

    if task_type == "Temporal Reasoning":
        profile["enforce_iso_date"] = True
        profile["enforce_number"] = True
    elif task_type == "Multi-session Reasoning":
        profile["enforce_iso_date"] = True
        profile["enforce_number"] = intents["count_intent"]
    elif task_type == "Event Summarization":
        profile["enforce_iso_date"] = True
        profile["enforce_number"] = intents["count_intent"]
    elif task_type == "Information Extraction":
        profile["enforce_iso_date"] = intents["temporal_intent"]
        profile["enforce_number"] = intents["count_intent"]
    elif task_type == "Knowledge Updating":
        profile["enforce_iso_date"] = intents["status_intent"] or intents["temporal_intent"]
        profile["enforce_number"] = intents["count_intent"]
    elif task_type == "Memory Arbitration":
        profile["enforce_iso_date"] = intents["temporal_intent"] or intents["status_intent"]
        profile["enforce_number"] = intents["count_intent"]

    return profile


CONSTRAINT_SLOT_ORDER = ["date", "number", "entity", "event", "status", "relation"]
CONSTRAINT_SLOT_NAMES = set(CONSTRAINT_SLOT_ORDER)


def _dedupe_str_list(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    out = []
    seen = set()
    for v in values:
        s = str(v).strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _normalize_slot_name(value: Any) -> str:
    s = str(value or "").strip().lower()
    alias = {
        "time": "date",
        "datetime": "date",
        "temporal": "date",
        "count": "number",
        "duration": "number",
        "quantity": "number",
        "name": "entity",
        "person": "entity",
        "location": "entity",
        "place": "entity",
        "fact": "event",
        "summary": "event",
    }
    s = alias.get(s, s)
    if s not in CONSTRAINT_SLOT_NAMES:
        return ""
    return s


def _slot_sort_key(slot: str) -> Tuple[int, str]:
    try:
        return (CONSTRAINT_SLOT_ORDER.index(slot), slot)
    except ValueError:
        return (len(CONSTRAINT_SLOT_ORDER), slot)


def _sort_slots(slots: List[str]) -> List[str]:
    return sorted(_dedupe_str_list(slots), key=_slot_sort_key)


def _sort_value_list(values: List[str]) -> List[str]:
    deduped = _dedupe_str_list(values)
    return sorted(deduped, key=lambda x: (x.lower(), x))


def _normalize_constraint_spec(raw: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    raw = raw or {}
    hard_slots = [_normalize_slot_name(x) for x in _dedupe_str_list(raw.get("hard_slots", []))]
    hard_slots = [x for x in hard_slots if x]
    soft_slots = [_normalize_slot_name(x) for x in _dedupe_str_list(raw.get("soft_slots", []))]
    soft_slots = [x for x in soft_slots if x]
    forbidden = [_normalize_slot_name(x) for x in _dedupe_str_list(raw.get("forbidden_hard_slots", []))]
    forbidden = [x for x in forbidden if x]

    conflict_count = len(set(hard_slots) & set(forbidden))
    hard_slots = [x for x in hard_slots if x not in forbidden]
    soft_slots = [x for x in soft_slots if x not in hard_slots and x not in forbidden]
    hard_slots = _sort_slots(hard_slots)
    soft_slots = _sort_slots(soft_slots)
    forbidden = _sort_slots(forbidden)

    slot_values_raw = raw.get("slot_values", {})
    slot_values: Dict[str, List[str]] = {}
    slot_anchors: Dict[str, Dict[str, List[str]]] = {}
    if isinstance(slot_values_raw, dict):
        for k, v in slot_values_raw.items():
            nk = _normalize_slot_name(k)
            if not nk:
                continue
            values: List[str] = []
            anchors_for_slot: Dict[str, List[str]] = {}

            candidates = v if isinstance(v, list) else [v]
            for item in candidates:
                value = ""
                anchors: List[str] = []
                if isinstance(item, dict):
                    value = str(
                        item.get("value", item.get("text", item.get("slot_value", "")))
                    ).strip()
                    anchors = _dedupe_str_list(
                        item.get("anchor_text", item.get("anchor", item.get("anchors", [])))
                    )
                else:
                    value = str(item).strip()

                if not value:
                    continue
                values.append(value)
                if anchors:
                    anchors_for_slot.setdefault(value, [])
                    for a in anchors:
                        if a not in anchors_for_slot[value]:
                            anchors_for_slot[value].append(a)

            deduped_values = _dedupe_str_list(values)
            if deduped_values:
                slot_values[nk] = _sort_value_list(deduped_values)
            if anchors_for_slot:
                normalized_anchors: Dict[str, List[str]] = {}
                for value_key in sorted(anchors_for_slot.keys(), key=lambda x: (x.lower(), x)):
                    normalized_anchors[value_key] = _sort_value_list(anchors_for_slot[value_key])
                slot_anchors[nk] = normalized_anchors

    # Compatibility: merge a standalone slot_anchors field if the model returns one.
    slot_anchors_raw = raw.get("slot_anchors", {})
    if isinstance(slot_anchors_raw, dict):
        for k, mapping in slot_anchors_raw.items():
            nk = _normalize_slot_name(k)
            if not nk or not isinstance(mapping, dict):
                continue
            slot_anchors.setdefault(nk, {})
            for value, anchors in mapping.items():
                value_s = str(value).strip()
                if not value_s:
                    continue
                anchors_list = _dedupe_str_list(anchors)
                if not anchors_list:
                    continue
                slot_anchors[nk].setdefault(value_s, [])
                for a in anchors_list:
                    if a not in slot_anchors[nk][value_s]:
                        slot_anchors[nk][value_s].append(a)
        for slot in list(slot_anchors.keys()):
            mapping = slot_anchors.get(slot, {})
            normalized_mapping: Dict[str, List[str]] = {}
            for value_key in sorted(mapping.keys(), key=lambda x: (x.lower(), x)):
                normalized_mapping[value_key] = _sort_value_list(mapping[value_key])
            slot_anchors[slot] = normalized_mapping

    match_policy_raw = raw.get("match_policy", {})
    match_policy: Dict[str, str] = {}
    if isinstance(match_policy_raw, dict):
        for k, v in match_policy_raw.items():
            nk = _normalize_slot_name(k)
            if not nk:
                continue
            pv = str(v).strip().lower()
            match_policy[nk] = pv if pv in {"exact", "fuzzy", "window"} else "fuzzy"

    for slot in hard_slots:
        match_policy.setdefault(slot, "exact")
    for slot in soft_slots:
        match_policy.setdefault(slot, "fuzzy")
    match_policy = {k: match_policy[k] for k in sorted(match_policy.keys(), key=_slot_sort_key)}
    slot_values = {k: slot_values[k] for k in sorted(slot_values.keys(), key=_slot_sort_key)}
    slot_anchors = {k: slot_anchors[k] for k in sorted(slot_anchors.keys(), key=_slot_sort_key)}

    return {
        "hard_slots": hard_slots,
        "soft_slots": soft_slots,
        "forbidden_hard_slots": forbidden,
        "slot_values": slot_values,
        "slot_anchors": slot_anchors,
        "match_policy": match_policy,
        "rationale": str(raw.get("rationale", "")).strip(),
    }, conflict_count


def _extract_json_candidates(text: str) -> List[str]:
    stripped = (text or "").strip()
    if not stripped:
        return []
    stripped = re.sub(r'^\s*```(?:json)?\s*', '', stripped, flags=re.IGNORECASE)
    stripped = re.sub(r'\s*```\s*$', '', stripped)
    out = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        out.append(stripped[start:end + 1])
    return out


def _build_date_query_variants(value: str) -> List[str]:
    """
    Build equivalent search variants for date values, including YYYY-MM and Month Year matches.
    """
    s = str(value or "").strip().lower()
    if not s:
        return []

    variants = [s]

    # Full ISO date: reuse the existing variants.
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}', s):
        for v in generate_date_variants(s):
            vv = str(v).strip().lower()
            if vv and vv not in variants:
                variants.append(vv)
        return variants

    # Year-month: support March 2020 / Mar 2020 / March of 2020.
    m = re.fullmatch(r'(\d{4})-(\d{2})', s)
    if m:
        year, month = m.group(1), m.group(2)
        month_name = MONTH_NUM_TO_NAME.get(month, "")
        if month_name:
            for v in [
                f"{month_name} {year}",
                f"{month_name[:3]} {year}",
                f"{month_name} of {year}",
                f"{month_name}, {year}",
            ]:
                if v not in variants:
                    variants.append(v)
        return variants

    # early/mid/late + year
    m = re.fullmatch(r'(early|mid|late)\s+(\d{4})', s)
    if m:
        qual, year = m.group(1), m.group(2)
        for v in [f"{qual} {year}", f"{qual} in {year}", f"{qual}-{year}"]:
            if v not in variants:
                variants.append(v)
        return variants

    # Year only
    if re.fullmatch(r'\d{4}', s):
        return variants

    return variants


def _extract_month_year_pair(value: str) -> Optional[Tuple[str, str]]:
    """
    Extract (year, month_num) from a date-like value.
    Supports:
    - YYYY-MM
    - Feb 2022 / February 2022
    """
    s = str(value or "").strip().lower()
    if not s:
        return None

    iso = re.fullmatch(r'(\d{4})-(\d{2})', s)
    if iso:
        return iso.group(1), iso.group(2)

    txt = re.fullmatch(r'([a-z]{3,9})\s+(\d{4})', s)
    if txt:
        month_token = txt.group(1)
        month_num = MONTH_NAMES.get(month_token)
        if month_num:
            return txt.group(2), month_num
    return None


def _build_month_year_regex(year: str, month_num: str) -> Optional[re.Pattern]:
    month_name = MONTH_NUM_TO_NAME.get(month_num)
    if not month_name:
        return None
    month_alias = [month_name, month_name[:3]]
    month_alias = [re.escape(x.lower()) for x in month_alias if x]
    if not month_alias:
        return None
    pattern = (
        rf'\b(?:{"|".join(month_alias)})\s+'
        rf'(?:\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*|\s+))?{re.escape(year)}\b'
    )
    return re.compile(pattern, flags=re.IGNORECASE)


def _split_date_range_value(value: str) -> Optional[Tuple[str, str]]:
    """
    Split a date-range value such as:
    - 2023-11 to 2025-02
    - Nov 2023 - Feb 2025
    """
    s = str(value or "").strip()
    if not s:
        return None
    if re.fullmatch(r'\d{4}-\d{2}', s):
        return None
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}', s):
        return None
    m = re.search(
        r'(.+?)\s*(?:-|–|—|to|through|until)\s*(.+)',
        s,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    left = m.group(1).strip(" ,;")
    right = m.group(2).strip(" ,;")
    if not left or not right:
        return None
    return left, right


def _search_date_turns_with_meta(value: str, dialogue: List[Dict]) -> Tuple[List[int], bool]:
    """
    Search date evidence and return (matched_turns, used_equivalent_match).
    """
    variants = _build_date_query_variants(value)
    if not variants:
        return [], False
    target = variants[0]
    equivalent_variants = variants[1:]

    turns: List[int] = []
    used_equivalent = False
    month_year_pair = _extract_month_year_pair(value)
    month_year_regex = None
    if month_year_pair is not None:
        month_year_regex = _build_month_year_regex(month_year_pair[0], month_year_pair[1])

    for i, turn in enumerate(dialogue):
        content = str(turn.get("content", "")).lower()
        if target and target in content:
            turns.append(i)
            continue
        for v in equivalent_variants:
            if v and v in content:
                turns.append(i)
                used_equivalent = True
                break
        else:
            if month_year_regex is not None and month_year_regex.search(content):
                turns.append(i)
                used_equivalent = True
    return sorted(set(turns)), used_equivalent


def _value_is_anchored_to_component(
    slot: str,
    value: str,
    component_text: str,
    anchors: Optional[List[str]] = None,
) -> bool:
    """
    Check whether a slot value can be anchored back to the target component.
    """
    comp_lower = str(component_text or "").lower()
    value_s = str(value or "").strip()
    value_lower = value_s.lower()
    anchors = _dedupe_str_list(anchors or [])

    if not value_s:
        return False

    # Prefer direct value anchoring.
    if slot == "date":
        for v in _build_date_query_variants(value_s):
            if v and v in comp_lower:
                return True
    elif slot == "number":
        if re.search(rf'\b{re.escape(value_lower)}\b', comp_lower):
            return True
    else:
        if value_lower in comp_lower:
            return True

    # Fall back to anchor_text.
    for a in anchors:
        if a.lower() in comp_lower:
            return True
    return False


def _looks_like_month_sequence(values: List[str]) -> bool:
    """
    Heuristically detect batch month injection, e.g. 2020-01 ... 2020-12.
    """
    ym = []
    for v in values:
        m = re.fullmatch(r'(\d{4})-(\d{2})', str(v).strip())
        if m:
            ym.append((m.group(1), m.group(2)))
    if len(ym) < 4:
        return False
    by_year: Dict[str, set] = {}
    for y, m in ym:
        by_year.setdefault(y, set()).add(m)
    return any(len(months) >= 4 for months in by_year.values())


def validate_constraint_spec_against_component(
    spec: Dict[str, Any],
    component: str,
    query: str,
    task_type: str,
) -> Dict[str, Any]:
    """
    Constraint-object guardrail: prevent cross-component contamination and keep hard slots anchorable.
    """
    normalized, _ = _normalize_constraint_spec(spec or {})
    hard_slots = list(normalized.get("hard_slots", []))
    soft_slots = list(normalized.get("soft_slots", []))
    slot_values = dict(normalized.get("slot_values", {}) or {})
    slot_anchors = dict(normalized.get("slot_anchors", {}) or {})

    errors: List[str] = []
    anchor_hits: Dict[str, Any] = {}
    degraded_to_soft = False

    # 1) Hard slots must be anchorable; otherwise downgrade them.
    for slot in list(hard_slots):
        values = list(slot_values.get(slot, []) or [])
        if not values:
            errors.append(f"{slot}:empty_hard_slot")
            hard_slots.remove(slot)
            if slot not in soft_slots:
                soft_slots.append(slot)
            degraded_to_soft = True
            continue

        kept: List[str] = []
        dropped: List[str] = []
        for value in values:
            anchors = (slot_anchors.get(slot, {}) or {}).get(value, [])
            if _value_is_anchored_to_component(slot, value, component, anchors):
                kept.append(value)
            else:
                dropped.append(value)

        anchor_hits[slot] = {"kept": kept, "dropped": dropped}
        if dropped:
            errors.append(f"{slot}:unanchored_values={dropped}")
        if not kept:
            hard_slots.remove(slot)
            if slot not in soft_slots:
                soft_slots.append(slot)
            slot_values[slot] = []
            degraded_to_soft = True
        else:
            slot_values[slot] = _sort_value_list(kept)

    # 2) Block batch month injection.
    date_values = list(slot_values.get("date", []) or [])
    if date_values and _looks_like_month_sequence(date_values):
        anchored = [
            v for v in date_values
            if _value_is_anchored_to_component(
                "date", v, component, (slot_anchors.get("date", {}) or {}).get(v, [])
            )
        ]
        if len(anchored) <= 1:
            errors.append("date:batch_month_injection")
            slot_values["date"] = anchored
            if "date" in hard_slots:
                hard_slots.remove("date")
                if "date" not in soft_slots:
                    soft_slots.append("date")
                degraded_to_soft = True

    normalized["hard_slots"] = _sort_slots(hard_slots)
    normalized["soft_slots"] = _sort_slots(soft_slots)
    normalized["slot_values"] = {
        k: _sort_value_list(v)
        for k, v in sorted(slot_values.items(), key=lambda item: _slot_sort_key(item[0]))
    }
    normalized["constraint_validation_errors"] = errors
    normalized["constraint_anchor_hits"] = anchor_hits
    normalized["constraint_degraded_to_soft"] = degraded_to_soft
    return normalized


def build_constraint_extraction_prompt(
    task_type: str,
    query: str,
    all_components: List[str],
) -> str:
    task_guidance = {
        "Temporal Reasoning": (
            "If a component is only an event date anchor, prefer hard date and avoid hard number.\n"
            "Make number hard only when that component itself contains required quantitative evidence "
            "(e.g. exact elapsed days/months/years required by query)."
        ),
        "Multi-session Reasoning": (
            "Prioritize event chain and cross-session linkage in soft slots.\n"
            "Use hard date only for explicit date-slot components."
        ),
        "Event Summarization": (
            "Summary meaning and key events are primary; time is usually soft unless explicit date-slot component."
        ),
        "Information Extraction": (
            "Explicit slot values should be hard where possible (entity/number/date)."
        ),
        "Knowledge Updating": (
            "Current/as-of status facts are priority; date hard only when query or component requires exact temporal anchor."
        ),
        "Memory Arbitration": (
            "Corrected core fact can be hard; background narrative remains soft."
        ),
    }

    components_text = "\n".join([f"{i}. {c}" for i, c in enumerate(all_components or [])])
    return f"""You are a strict QA constraint object extractor.

Task type: {task_type}
Query: {query}

All answer components:
{components_text}

Task-specific guidance:
{task_guidance.get(task_type, "Infer hard-vs-soft slots conservatively.")}

Core objective:
- Extract a minimal-sufficient verification set for each component.
- "Minimal-sufficient" means: as few clues as possible, but DO NOT omit any clue
  that is necessary to correctly answer the QA intent.

Global rules:
- Decide each component independently. Do not copy values from other components.
- Prefer values grounded in the target component text. If query intent requires a shared anchor,
  keep it explicit in rationale.
- Do not expand a coarse time reference into fabricated series (for example, never expand one month
  into many month values).
- Use hard slots only when exact presence is necessary for correctness; otherwise use soft.
- If uncertain between hard and soft, choose soft.

For EACH component, decide:
1) hard_slots: exact-match required slots.
2) soft_slots: semantic/fuzzy slots.
3) forbidden_hard_slots: slots that must NOT be hard for this component.
4) slot_values: concrete values to check for each slot.
   Each value MUST include anchor_text copied from the target component text.
5) match_policy: exact/fuzzy/window.

Return JSON only:
{{
  "components": [
    {{
      "index": 0,
      "hard_slots": ["date"],
      "soft_slots": ["event"],
      "forbidden_hard_slots": ["number"],
      "slot_values": {{
        "date": [{{"value": "2020-03-15", "anchor_text": "on 2020-03-15"}}],
        "number": [{{"value": "56", "anchor_text": "56 days"}}],
        "entity": [],
        "event": [{{"value": "accepted apprenticeship", "anchor_text": "accepted apprenticeship"}}],
        "status": [],
        "relation": []
      }},
      "match_policy": {{"date": "exact", "event": "fuzzy"}},
      "rationale": "brief reason"
    }}
  ]
}}"""


def _parse_constraint_specs_response(
    response: str,
    num_components: int,
) -> Dict[int, Dict[str, Any]]:
    parsed_map: Dict[int, Dict[str, Any]] = {}
    for candidate in _extract_json_candidates(response):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        items = []
        if isinstance(payload, dict):
            if isinstance(payload.get("components"), list):
                items = payload["components"]
            elif isinstance(payload.get("items"), list):
                items = payload["items"]
        elif isinstance(payload, list):
            items = payload

        if not isinstance(items, list):
            continue

        for i, raw in enumerate(items):
            if not isinstance(raw, dict):
                continue
            idx = raw.get("index", i)
            try:
                idx = int(idx)
            except (TypeError, ValueError):
                continue
            if idx < 0 or idx >= num_components:
                continue
            spec, _conflicts = _normalize_constraint_spec(raw)
            parsed_map[idx] = spec

        if parsed_map:
            return parsed_map

    return {}


def _run_llm_once_sync(
    llm: Optional[LLM],
    prompt: str,
    max_tokens: int = 900,
    temperature: Optional[float] = None,
) -> Optional[str]:
    async def _call_once() -> Any:
        try:
            return await llm.inference(prompt, max_tokens=max_tokens, temperature=temperature)
        except TypeError:
            # Compatibility with old test stubs/implementations that do not accept temperature.
            return await llm.inference(prompt, max_tokens=max_tokens)

    if llm is None:
        return None
    try:
        completion = asyncio.run(_call_once())
        return llm.decode(completion)
    except RuntimeError:
        # fallback when loop is already running
        loop = asyncio.new_event_loop()
        try:
            completion = loop.run_until_complete(_call_once())
            return llm.decode(completion)
        finally:
            loop.close()
    except Exception:
        return None


def _infer_entity_values(component: str) -> List[str]:
    candidates = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b', component or "")
    blocked = {"Event", "Start", "End", "Date", "Duration", "Calculation", "Result"}
    out = []
    seen = set()
    for c in candidates:
        if c in blocked:
            continue
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out[:5]


def build_fallback_constraint_spec(
    task_type: str,
    query: str,
    component: str,
) -> Dict[str, Any]:
    intents = infer_query_intents(query)
    dates = extract_dates_from_text(component)
    month_year_dates = extract_month_year_refs(component)
    relative_year_dates = re.findall(r'\b(?:early|mid|late)\s+\d{4}\b', component, flags=re.IGNORECASE)
    date_values = _dedupe_str_list(dates + month_year_dates + relative_year_dates)
    numbers = extract_numbers_without_dates(component)
    keywords, _phrases = extract_key_phrases(component)
    entities = _infer_entity_values(component)
    component_lower = (component or "").lower()

    hard_slots: List[str] = []
    soft_slots: List[str] = []
    forbidden_hard_slots: List[str] = []

    has_explicit_date_field = bool(
        re.search(r'\b(start\s*date|end\s*date|date)\b\s*[:=]', component_lower)
        or re.search(r'\d{4}-\d{2}-\d{2}', component_lower)
    )
    has_explicit_number_field = bool(
        re.search(r'\b(duration|count|number|amount|total|days?|months?|years?)\b\s*[:=]?', component_lower)
    )
    has_explicit_entity_field = bool(
        re.search(r'\b(name|person|location|place|city|organization|company|school|project)\b\s*[:=]', component_lower)
    )

    if task_type == "Temporal Reasoning":
        if date_values and (has_explicit_date_field or intents["temporal_intent"]):
            hard_slots.append("date")
        elif date_values:
            soft_slots.append("date")
        if intents["count_intent"] and numbers and has_explicit_number_field:
            hard_slots.append("number")
        elif numbers and intents["count_intent"]:
            soft_slots.append("number")
        soft_slots.append("event")
    elif task_type == "Information Extraction":
        if date_values and has_explicit_date_field:
            hard_slots.append("date")
        elif date_values:
            soft_slots.append("date")
        if numbers and intents["count_intent"] and has_explicit_number_field:
            hard_slots.append("number")
        elif numbers and intents["count_intent"]:
            soft_slots.append("number")
        if entities and has_explicit_entity_field:
            hard_slots.append("entity")
        elif entities:
            soft_slots.append("entity")
    elif task_type == "Knowledge Updating":
        if date_values and intents["status_intent"] and has_explicit_date_field:
            hard_slots.append("date")
        soft_slots.extend(["status", "event", "date"])
    elif task_type == "Memory Arbitration":
        if date_values and intents["temporal_intent"] and has_explicit_date_field:
            hard_slots.append("date")
        if numbers and intents["count_intent"] and has_explicit_number_field:
            hard_slots.append("number")
        soft_slots.extend(["event", "relation", "date"])
    elif task_type == "Multi-session Reasoning":
        # Handle multi-session reasoning conservatively: soft by default, hard only with explicit date fields.
        if date_values and has_explicit_date_field:
            hard_slots.append("date")
        elif date_values:
            soft_slots.append("date")
        soft_slots.extend(["event", "relation"])
    else:  # Event Summarization / Unknown
        soft_slots.extend(["event", "date"])

    slot_values: Dict[str, List[str]] = {}
    if date_values:
        slot_values["date"] = date_values
    if numbers:
        slot_values["number"] = numbers
    if entities:
        slot_values["entity"] = entities
    if keywords:
        slot_values["event"] = keywords[:6]

    raw = {
        "hard_slots": hard_slots,
        "soft_slots": soft_slots,
        "forbidden_hard_slots": forbidden_hard_slots,
        "slot_values": slot_values,
        "match_policy": {},
        "rationale": "fallback_rule_profile",
    }
    spec, _conflicts = _normalize_constraint_spec(raw)
    return validate_constraint_spec_against_component(spec, component, query, task_type)


def extract_constraint_spec(
    task_type: str,
    query: str,
    all_components: List[str],
    target_component: str,
    target_component_idx: int,
    llm: Optional[LLM] = None,
    use_llm: bool = False,
    cache: Optional[Dict[str, Any]] = None,
    task_stats: Optional[Dict[str, Any]] = None,
    task_instance_key: Optional[str] = None,
    max_retry_attempts: int = 3,
) -> Tuple[Dict[str, Any], str]:
    if task_stats is not None:
        task_stats.setdefault("constraint_llm_success_tasks", 0)
        task_stats.setdefault("constraint_fallback_tasks", 0)
        task_stats.setdefault("constraint_partial_parse_rejects", 0)
        task_stats.setdefault("constraint_retry_count", 0)

    def _build_task_key() -> str:
        payload = {
            "task_type": task_type,
            "query": query,
            "components": all_components,
        }
        return hashlib.sha1(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _is_usable_spec(spec: Dict[str, Any]) -> bool:
        if not isinstance(spec, dict):
            return False
        hard_slots = _dedupe_str_list(spec.get("hard_slots", []))
        soft_slots = _dedupe_str_list(spec.get("soft_slots", []))
        slot_values = spec.get("slot_values", {}) or {}
        if not isinstance(slot_values, dict):
            slot_values = {}
        has_values = any(bool(_dedupe_str_list(v)) for v in slot_values.values())
        return bool(hard_slots or soft_slots or has_values)

    def _build_task_fallback_specs() -> Dict[int, Dict[str, Any]]:
        specs: Dict[int, Dict[str, Any]] = {}
        for idx, comp_text in enumerate(all_components):
            fallback = build_fallback_constraint_spec(task_type, query, comp_text)
            specs[idx] = validate_constraint_spec_against_component(
                fallback,
                comp_text,
                query,
                task_type,
            )
        return specs

    def _record_task_stats(task_key: str, meta: Dict[str, Any]) -> None:
        if task_stats is None:
            return
        counter_key = task_instance_key or task_key
        seen = task_stats.setdefault("_constraint_counted_tasks", set())
        if counter_key in seen:
            return
        seen.add(counter_key)
        if meta.get("source") == "llm_object":
            task_stats["constraint_llm_success_tasks"] = int(
                task_stats.get("constraint_llm_success_tasks", 0)
            ) + 1
        else:
            task_stats["constraint_fallback_tasks"] = int(
                task_stats.get("constraint_fallback_tasks", 0)
            ) + 1
        if meta.get("partial_parse_rejected"):
            task_stats["constraint_partial_parse_rejects"] = int(
                task_stats.get("constraint_partial_parse_rejects", 0)
            ) + 1
        task_stats["constraint_retry_count"] = int(
            task_stats.get("constraint_retry_count", 0)
        ) + int(meta.get("retry_count", 0))

    def _unpack_cache_entry(entry: Any) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
        if not isinstance(entry, dict):
            return {}, {}
        if "__specs__" in entry:
            specs = entry.get("__specs__", {})
            meta = entry.get("__meta__", {})
            if isinstance(specs, dict):
                typed_specs = {int(k): v for k, v in specs.items() if str(k).isdigit() or isinstance(k, int)}
            else:
                typed_specs = {}
            return typed_specs, meta if isinstance(meta, dict) else {}
        typed_specs: Dict[int, Dict[str, Any]] = {}
        for k, v in entry.items():
            if isinstance(k, int):
                typed_specs[k] = v
            elif isinstance(k, str) and k.isdigit():
                typed_specs[int(k)] = v
        inferred_source = "llm_object" if typed_specs else "fallback_rule"
        return typed_specs, {"source": inferred_source, "retry_count": 0, "partial_parse_rejected": False}

    def _store_cache_entry(task_key: str, specs: Dict[int, Dict[str, Any]], meta: Dict[str, Any]) -> None:
        if cache is None:
            return
        cache[task_key] = {
            "__specs__": specs,
            "__meta__": {
                "source": meta.get("source", "fallback_rule"),
                "retry_count": int(meta.get("retry_count", 0)),
                "partial_parse_rejected": bool(meta.get("partial_parse_rejected", False)),
            },
        }

    key = _build_task_key()

    if cache is not None and key in cache:
        per_task, meta = _unpack_cache_entry(cache[key])
        _record_task_stats(key, meta)
        if target_component_idx in per_task:
            return per_task[target_component_idx], str(meta.get("source", "llm_object"))

    per_task_specs: Dict[int, Dict[str, Any]] = {}
    task_meta: Dict[str, Any] = {
        "source": "fallback_rule",
        "retry_count": 0,
        "partial_parse_rejected": False,
    }
    if use_llm and llm is not None:
        prompt = build_constraint_extraction_prompt(task_type, query, all_components)
        max_retry_attempts = max(1, int(max_retry_attempts))
        attempts_used = 0
        partial_rejected = False
        for _ in range(max_retry_attempts):
            attempts_used += 1
            response = _run_llm_once_sync(llm, prompt, max_tokens=1200, temperature=0.0)
            if not response:
                continue
            parsed = _parse_constraint_specs_response(response, len(all_components))
            if not parsed:
                continue
            if len(parsed) != len(all_components) or any(i not in parsed for i in range(len(all_components))):
                partial_rejected = True
                continue
            candidate_specs: Dict[int, Dict[str, Any]] = {}
            all_usable = True
            for i in range(len(all_components)):
                comp_text = all_components[i] if 0 <= i < len(all_components) else ""
                validated = validate_constraint_spec_against_component(
                    spec=parsed.get(i, {}),
                    component=comp_text,
                    query=query,
                    task_type=task_type,
                )
                if not _is_usable_spec(validated):
                    all_usable = False
                candidate_specs[i] = validated
            if not all_usable:
                partial_rejected = True
                continue
            per_task_specs = candidate_specs
            task_meta = {
                "source": "llm_object",
                "retry_count": max(0, attempts_used - 1),
                "partial_parse_rejected": False,
            }
            break
        else:
            task_meta = {
                "source": "fallback_rule",
                "retry_count": max(0, attempts_used - 1),
                "partial_parse_rejected": partial_rejected,
            }

    if not per_task_specs:
        per_task_specs = _build_task_fallback_specs()
    _store_cache_entry(key, per_task_specs, task_meta)
    _record_task_stats(key, task_meta)

    if target_component_idx in per_task_specs:
        return per_task_specs[target_component_idx], str(task_meta.get("source", "fallback_rule"))

    fallback = build_fallback_constraint_spec(task_type, query, target_component)
    return validate_constraint_spec_against_component(fallback, target_component, query, task_type), "fallback_rule"


def extract_numbers_without_dates(
    text: str,
    hard_number_mode: str = "contextual_strict",
    return_filtered: bool = False,
) -> Any:
    """
    Extract semantic numbers, excluding date numbers and structural ordinal numbers.
    Also normalizes thousands separators.
    """
    working = text or ""
    filtered_numbers: List[str] = []

    # Remove ISO dates.
    working = re.sub(r'\d{4}-\d{2}-\d{2}', ' ', working)

    # Remove Month Year references; they are soft anchors, not bare numeric hard locks.
    working = re.sub(
        r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}\b',
        ' ',
        working,
        flags=re.IGNORECASE,
    )

    # Normalize thousands separators: 15,000 -> 15000, avoiding number:000.
    def _comma_repl(m: re.Match) -> str:
        return m.group(0).replace(',', '')
    working = re.sub(r'\b\d{1,3}(?:,\d{3})+\b', _comma_repl, working)

    # Filter structural ordinal numbers such as Event 1 / Step 2.
    label_num_pattern = re.compile(
        r'\b(event|step|phase|ingredient|origin|escalation|intermediate\s+step|part|chapter)\s+(\d+)\b',
        flags=re.IGNORECASE,
    )
    for m in label_num_pattern.finditer(working):
        filtered_numbers.append(m.group(2))
    working = label_num_pattern.sub(r'\1 ', working)

    numbers = re.findall(r'\b\d+\b', working)
    semantic_numbers: List[str] = []
    for n in numbers:
        # Prevent trailing-block noise from incorrectly split values such as 15,000.
        if n == "000":
            filtered_numbers.append(n)
            continue
        semantic_numbers.append(str(int(n)) if n.isdigit() else n)

    deduped = sorted(set(semantic_numbers))
    if return_filtered:
        return deduped, sorted(set(filtered_numbers))
    return deduped


def _search_text_turns(value: str, dialogue: List[Dict]) -> List[int]:
    target = str(value or "").strip().lower()
    if not target:
        return []
    turns = []
    for i, turn in enumerate(dialogue):
        if target in turn.get("content", "").lower():
            turns.append(i)
    return turns


def _normalize_numeric_token(value: str) -> str:
    token = str(value or "").strip().lower()
    if not token:
        return ""
    token = token.replace("$", "")
    token = token.replace(",", "")
    token = token.replace(" ", "")
    if re.fullmatch(r'[+-]?\d+(?:\.\d+)?', token):
        if "." in token:
            token = token.rstrip("0").rstrip(".")
        if token.startswith("+"):
            token = token[1:]
        if re.fullmatch(r'[+-]?\d+', token):
            negative = token.startswith("-")
            digits = token[1:] if negative else token
            digits = digits.lstrip("0") or "0"
            token = f"-{digits}" if negative else digits
    return token


def _search_number_turns(value: str, dialogue: List[Dict]) -> List[int]:
    s = str(value or "").strip()
    if not s:
        return []
    target_norm = _normalize_numeric_token(s)
    if not target_norm:
        return []
    text_pattern = re.compile(rf'\b{re.escape(s.lower())}\b')
    numeric_pattern = re.compile(r'[$]?\d[\d,]*(?:\.\d+)?')
    turns = []
    for i, turn in enumerate(dialogue):
        content = turn.get("content", "")
        text = content.lower()
        if text_pattern.search(text):
            turns.append(i)
            continue

        matched = False
        for m in numeric_pattern.finditer(content):
            candidate_norm = _normalize_numeric_token(m.group(0))
            if candidate_norm == target_norm:
                matched = True
                break
        if matched:
            turns.append(i)
    return turns


def _search_date_turns(value: str, dialogue: List[Dict]) -> List[int]:
    turns, _used_equivalent = _search_date_turns_with_meta(value, dialogue)
    return turns


def evaluate_hard_facts(
    component: str,
    dialogue: List[Dict],
    dialogue_text_lower: str,
    task_type: str = "",
    query: str = "",
    task_profile: str = "qa_task_v1",
    hard_number_mode: str = "contextual_strict",
    return_meta: bool = False,
    constraint_spec: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Strict hard constraints:
    1) Dates must be matched, including both endpoints of a date range.
    2) Independent numbers must be matched.
    """
    missing: List[str] = []
    required_slots: List[str] = []
    slot_verdicts: Dict[str, bool] = {}
    hard_meta: Dict[str, Any] = {}
    hard_meta["constraint_validation_errors"] = _dedupe_str_list(
        (constraint_spec or {}).get("constraint_validation_errors", [])
    )
    hard_meta["constraint_anchor_hits"] = (
        (constraint_spec or {}).get("constraint_anchor_hits", {})
        if isinstance((constraint_spec or {}).get("constraint_anchor_hits", {}), dict)
        else {}
    )
    hard_meta["constraint_degraded_to_soft"] = bool(
        (constraint_spec or {}).get("constraint_degraded_to_soft", False)
    )
    hard_meta["date_equivalent_match_count"] = 0
    hard_meta["hard_slot_degraded_count"] = 0

    profile = get_task_judge_profile(task_type, query, task_profile=task_profile)
    normalized_spec: Dict[str, Any] = {}
    spec_conflicts = 0
    if constraint_spec:
        normalized_spec, spec_conflicts = _normalize_constraint_spec(constraint_spec)
    hard_meta["spec_conflicts"] = spec_conflicts

    if normalized_spec:
        hard_slots = list(normalized_spec.get("hard_slots", []))
        forbidden_slots = set(normalized_spec.get("forbidden_hard_slots", []))
        hard_slots = [slot for slot in hard_slots if slot not in forbidden_slots]
        slot_values = normalized_spec.get("slot_values", {}) or {}
        match_policy = normalized_spec.get("match_policy", {}) or {}

        hard_meta["match_policy"] = match_policy
        hard_meta["constraint_rationale"] = normalized_spec.get("rationale", "")

        # Compatibility with old audit fields: still count filtered structural numbers.
        _numbers_from_component, filtered = extract_numbers_without_dates(
            component,
            hard_number_mode=hard_number_mode,
            return_filtered=True,
        )
        hard_meta["hard_number_filtered"] = filtered
        degraded_slots: List[str] = []

        for slot in hard_slots:
            values = _dedupe_str_list(slot_values.get(slot, []))
            if not values:
                hard_meta["spec_conflicts"] = hard_meta.get("spec_conflicts", 0) + 1
                degraded_slots.append(slot)
                slot_verdicts[slot] = True
                continue

            required_slots.append(slot)
            slot_missing = []
            slot_hits = []
            for value in values:
                if slot == "date":
                    range_pair = _split_date_range_value(value)
                    if range_pair:
                        start_value, end_value = range_pair
                        start_turns, start_equiv = _search_date_turns_with_meta(start_value, dialogue)
                        end_turns, end_equiv = _search_date_turns_with_meta(end_value, dialogue)
                        turns = sorted(set(start_turns + end_turns))
                        if start_turns and end_turns:
                            if start_equiv:
                                hard_meta["date_equivalent_match_count"] = (
                                    int(hard_meta.get("date_equivalent_match_count", 0)) + 1
                                )
                            if end_equiv:
                                hard_meta["date_equivalent_match_count"] = (
                                    int(hard_meta.get("date_equivalent_match_count", 0)) + 1
                                )
                        else:
                            turns = []
                            if not start_turns:
                                slot_missing.append(f"date_range_start:{start_value}")
                            if not end_turns:
                                slot_missing.append(f"date_range_end:{end_value}")
                    else:
                        turns, used_equiv = _search_date_turns_with_meta(value, dialogue)
                        if turns and used_equiv:
                            hard_meta["date_equivalent_match_count"] = (
                                int(hard_meta.get("date_equivalent_match_count", 0)) + 1
                            )
                elif slot == "number":
                    turns = _search_number_turns(value, dialogue)
                else:
                    turns = _search_text_turns(value, dialogue)
                if turns:
                    slot_hits.extend(turns)
                else:
                    if slot != "date" or not _split_date_range_value(value):
                        slot_missing.append(f"{slot}:{value}")

            if slot == "date":
                hard_meta["date_evidence_turns"] = sorted(set(slot_hits))

            missing.extend(slot_missing)
            slot_verdicts[slot] = len(slot_missing) == 0

        if degraded_slots:
            hard_meta["constraint_degraded_to_soft"] = True
            hard_meta["hard_slot_degraded_count"] = len(degraded_slots)
            hard_meta["degraded_slots"] = degraded_slots

        for slot in CONSTRAINT_SLOT_NAMES:
            slot_verdicts.setdefault(slot, True)
    else:
        # Dates, including date ranges.
        dates = extract_dates_from_text(component)
        if dates and profile["enforce_iso_date"]:
            required_slots.append("date")
            date_missing = []
            date_hits = []
            for date in dates:
                turns, used_equiv = _search_date_turns_with_meta(date, dialogue)
                if not turns:
                    date_missing.append(f"date:{date}")
                else:
                    date_hits.extend(turns)
                    if used_equiv:
                        hard_meta["date_equivalent_match_count"] = (
                            int(hard_meta.get("date_equivalent_match_count", 0)) + 1
                        )
            missing.extend(date_missing)
            slot_verdicts["date"] = len(date_missing) == 0
            hard_meta["date_evidence_turns"] = sorted(set(date_hits))
        else:
            slot_verdicts["date"] = True

        # Independent numbers.
        numbers, filtered = extract_numbers_without_dates(
            component,
            hard_number_mode=hard_number_mode,
            return_filtered=True,
        )
        hard_meta["hard_number_filtered"] = filtered

        enforce_number = profile["enforce_number"] and len(numbers) > 0
        if enforce_number:
            required_slots.append("number")
            number_missing = []
            for n in numbers:
                if not _search_number_turns(n, dialogue):
                    number_missing.append(f"number:{n}")
            missing.extend(number_missing)
            slot_verdicts["number"] = len(number_missing) == 0
        else:
            slot_verdicts["number"] = True

    status = "PASS" if not missing else "HARD_MISSING"
    hard_meta["required_slots"] = required_slots
    hard_meta["slot_verdicts"] = slot_verdicts
    hard_meta["task_profile"] = task_profile
    hard_meta["hard_number_mode"] = hard_number_mode

    if return_meta:
        return status, missing, hard_meta
    return status, missing


# ============================================================
# Keyword/phrase extraction and search
# ============================================================

def extract_key_phrases(component: str) -> List[str]:
    """
    Extract key phrases from component text:
    - meaningful words longer than 4 characters
    - contiguous 2+ word phrases, such as "graphic design" or "reading chair"
    - proper nouns with capitalized initial letters
    """
    # Remove dates and leading labels.
    text = re.sub(r'\d{4}-\d{2}-\d{2}', '', component)
    text = re.sub(r'^(Start date|End date|Calculation|Correction|Tradition|Location|'
                  r'Date|Requirement|Status|Project|Quality|Action|Shift|'
                  r'New Status|Behavior|Gift|Dates|Result|Mindset shift):\s*',
                  '', text, flags=re.IGNORECASE)

    # Extract meaningful words (>3 characters, excluding stop words).
    stop_words = {'this', 'that', 'with', 'from', 'they', 'them', 'their',
                  'have', 'been', 'were', 'will', 'would', 'could', 'should',
                  'about', 'after', 'before', 'during', 'between', 'into',
                  'through', 'being', 'each', 'which', 'what', 'when', 'where',
                  'than', 'then', 'also', 'just', 'more', 'most', 'some',
                  'very', 'much', 'such', 'only', 'over', 'same', 'other',
                  'part', 'feeling', 'sense'}
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    keywords = [w for w in words if len(w) > 3 and w.lower() not in stop_words]

    # Extract contiguous 2-4 word phrases.
    phrases = []
    clean_words = text.split()
    for length in [4, 3, 2]:
        for i in range(len(clean_words) - length + 1):
            phrase = ' '.join(clean_words[i:i+length])
            # Remove punctuation.
            phrase = re.sub(r'[.,;:!?()\[\]{}"\']', '', phrase).strip()
            if len(phrase) > 8:  # Phrase must be at least 8 characters.
                phrases.append(phrase)

    return keywords, phrases


def search_keywords_in_window(keywords: List[str], dialogue: List[Dict],
                               window_center: int, window_size: int = 10) -> Tuple[int, int]:
    """Search keywords in a window and return (hits, total)."""
    start = max(0, window_center - window_size)
    end = min(len(dialogue), window_center + window_size)

    window_text = ' '.join(t['content'].lower() for t in dialogue[start:end])
    found = sum(1 for kw in keywords if kw.lower() in window_text)
    return found, len(keywords)


def collect_date_evidence_turns(component: str, dialogue: List[Dict], cap: int = 16) -> List[int]:
    """
    Collect date-anchor evidence turns and adjacent context.
    """
    dates = extract_dates_from_text(component)
    if not dates:
        return []

    selected = set()
    for date in dates:
        for turn_idx in search_date_in_dialogue(date, dialogue):
            for offset in (-1, 0, 1):
                idx = turn_idx + offset
                if 0 <= idx < len(dialogue):
                    selected.add(idx)
            if len(selected) >= cap:
                break
        if len(selected) >= cap:
            break

    return sorted(selected)[:cap]


# ============================================================
# Rule-based pre-screening by task type
# ============================================================

def rule_check_component(
    task_type: str,
    component: str,
    dialogue: List[Dict],
    dialogue_text_lower: str,
    constraint_spec: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[int], str, str, str]:
    """
    Rule-based pre-screening for one component.
    Returns: (status, matched_turns, reason, confidence, reason_code)
    status: PRESENT / SUSPICIOUS / MISSING
    """
    # --- Extract dates ---
    dates = extract_dates_from_text(component)

    # --- Extract keywords and phrases ---
    keywords, phrases = extract_key_phrases(component)

    # === Strategy 0: prioritize soft slots from constraint_spec ===
    if constraint_spec:
        spec, _conflicts = _normalize_constraint_spec(constraint_spec)
        soft_slots = spec.get("soft_slots", [])
        slot_values = spec.get("slot_values", {}) or {}
        if soft_slots:
            total = 0
            matched = 0
            matched_turns: List[int] = []
            for slot in soft_slots:
                values = _dedupe_str_list(slot_values.get(slot, []))
                if not values:
                    if slot == "event":
                        values = keywords[:6]
                    elif slot == "entity":
                        values = _infer_entity_values(component)
                    elif slot == "date":
                        values = dates
                if not values:
                    continue
                for value in values:
                    total += 1
                    if slot == "date":
                        turns = _search_date_turns(value, dialogue)
                    elif slot == "number":
                        turns = _search_number_turns(value, dialogue)
                    else:
                        turns = _search_text_turns(value, dialogue)
                    if turns:
                        matched += 1
                        matched_turns.extend(turns)

            if total > 0:
                ratio = matched / total
                merged_turns = sorted(set(matched_turns))
                if ratio >= 0.8:
                    return "PRESENT", merged_turns, (
                        f"soft slots matched {matched}/{total}"
                    ), "high", "soft_slot_match_high"
                if ratio >= 0.4:
                    return "SUSPICIOUS", merged_turns, (
                        f"soft slots partially matched {matched}/{total}"
                    ), "low", "soft_slot_partial"
                return "SUSPICIOUS", merged_turns, (
                    f"soft slots sparsely matched {matched}/{total}"
                ), "low", "soft_slot_sparse"

    # === Strategy 1: date verification (Temporal / Multi-session / Memory Arbitration) ===
    if dates:
        all_date_turns = []
        missing_dates = []
        for date in dates:
            turns = search_date_in_dialogue(date, dialogue)
            if turns:
                all_date_turns.extend(turns)
            else:
                missing_dates.append(date)

        if missing_dates:
            return "MISSING", [], f"dates {missing_dates} not found in dialogue", "high", "date_missing"

        # Date exists; check whether the event description appears nearby.
        if keywords and all_date_turns:
            # Search keywords in windows near the date positions.
            best_ratio = 0
            best_local_hits = 0
            for turn_idx in all_date_turns:
                found, total = search_keywords_in_window(keywords, dialogue, turn_idx, 10)
                ratio = found / max(total, 1)
                best_ratio = max(best_ratio, ratio)
                best_local_hits = max(best_local_hits, found)

            if best_ratio >= 0.8:
                return "PRESENT", all_date_turns, f"date and {best_ratio:.0%} of keywords co-occur in nearby window", "high", "date_hit_keyword_high"
            elif best_ratio >= 0.5:
                return "PRESENT", all_date_turns, f"date and {best_ratio:.0%} of keywords co-occur in nearby window", "medium", "date_hit_keyword_medium"
            elif best_local_hits >= 2:
                # Strong date anchor plus at least two local keywords avoids over-penalizing window-ratio noise.
                return "PRESENT", all_date_turns, (
                    f"date anchor exists and {best_local_hits} local keywords matched"
                ), "medium", "date_anchor_local_hit"
            else:
                return "SUSPICIOUS", all_date_turns, f"date exists but only {best_ratio:.0%} of keywords are nearby", "low", "date_hit_keyword_low"
        else:
            return "PRESENT", all_date_turns, "date found in dialogue", "medium", "date_hit_no_keywords"

    # === Strategy 2: exact phrase matching ===
    if phrases:
        for phrase in phrases:
            if phrase.lower() in dialogue_text_lower:
                # Locate where the phrase appears.
                matched = []
                for i, t in enumerate(dialogue):
                    if phrase.lower() in t['content'].lower():
                        matched.append(i)
                return "PRESENT", matched, f"phrase \"{phrase}\" exactly matched", "high", "phrase_match_strong"

    # === Strategy 3: keyword coverage ===
    if keywords:
        found_globally = [kw for kw in keywords if kw.lower() in dialogue_text_lower]
        ratio = len(found_globally) / len(keywords)

        if ratio >= 0.8:
            # Most keywords are found; verify they co-occur in a reasonable window.
            # Find the densest keyword-hit location.
            best_window_turn = -1
            best_window_ratio = 0
            for i in range(0, len(dialogue), 5):
                found, total = search_keywords_in_window(keywords, dialogue, i, 8)
                r = found / max(total, 1)
                if r > best_window_ratio:
                    best_window_ratio = r
                    best_window_turn = i

            if best_window_ratio >= 0.6:
                return "PRESENT", [best_window_turn], f"{ratio:.0%} of keywords matched; best window {best_window_ratio:.0%}", "high", "keyword_dense_window_high"
            if best_window_ratio >= 0.5:
                return "PRESENT", [best_window_turn], f"{ratio:.0%} of keywords matched; best window {best_window_ratio:.0%}", "medium", "keyword_dense_window_medium"
            else:
                return "SUSPICIOUS", [best_window_turn] if best_window_turn >= 0 else [], \
                    f"{ratio:.0%} of keywords matched but are scattered; best window only {best_window_ratio:.0%}", "medium", "keyword_dense_scattered"

        elif ratio >= 0.4:
            return "SUSPICIOUS", [], f"only {ratio:.0%} ({len(found_globally)}/{len(keywords)}) of keywords matched", "low", "keyword_sparse"
        else:
            return "MISSING", [], f"only {ratio:.0%} ({len(found_globally)}/{len(keywords)}) of keywords matched", "low", "keyword_very_sparse"

    # === Strategy 4: Event Summarization - summary-style language ===
    if task_type == "Event Summarization":
        # Summary components usually do not appear verbatim and need LLM verification.
        # First check whether the time span is mentioned.
        time_refs = re.findall(r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4})', component)
        if time_refs:
            found_time = any(tr.lower() in dialogue_text_lower for tr in time_refs)
            if found_time:
                return "SUSPICIOUS", [], f"time references {time_refs} appear in dialogue; semantic LLM verification needed", "medium", "event_time_found_needs_semantic"
            else:
                return "MISSING", [], f"time references {time_refs} not found in dialogue", "medium", "event_time_missing"
        return "SUSPICIOUS", [], "summary-style component; semantic LLM verification needed", "low", "event_summary_needs_semantic"

    # Default: no extractable structured evidence.
    return "SUSPICIOUS", [], "cannot be judged by rules; LLM verification needed", "low", "unknown_needs_semantic"


# ============================================================
# LLM semantic verification
# ============================================================

def _task_retrieval_radius(task_type: str) -> int:
    task = (task_type or "").strip()
    if task in {"Event Summarization", "Multi-session Reasoning"}:
        return 4
    if task in {"Temporal Reasoning", "Memory Arbitration"}:
        return 3
    return 2


def _parse_retrieval_route_weights(raw: Any) -> Dict[str, float]:
    default = {"slot": 3.0, "date": 2.0, "semantic": 1.0}
    if isinstance(raw, dict):
        parsed = {}
        for k in ("slot", "date", "semantic"):
            try:
                parsed[k] = max(0.0, float(raw.get(k, default[k])))
            except Exception:
                parsed[k] = default[k]
        if parsed["slot"] == 0 and parsed["date"] == 0 and parsed["semantic"] == 0:
            return default
        return parsed

    text = str(raw or "").strip()
    if not text:
        return default
    parsed = dict(default)
    for part in text.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip().lower()
        if key not in parsed:
            continue
        try:
            parsed[key] = max(0.0, float(value.strip()))
        except Exception:
            continue
    if parsed["slot"] == 0 and parsed["date"] == 0 and parsed["semantic"] == 0:
        return default
    return parsed


def extract_relevant_turns(
    component: str,
    dialogue: List[Dict],
    matched_turns: List[int],
    max_turns: int = 30,
    task_type: str = "",
    constraint_spec: Optional[Dict[str, Any]] = None,
    retrieval_min_centers: int = 5,
    retrieval_route_weights: Optional[Dict[str, float]] = None,
    return_meta: bool = False,
) -> Any:
    """
    Evidence-block extraction:
    1) Prefer slot_values/date/entity/event hits.
    2) Then use date anchors and entity phrases.
    3) Fall back to keywords.
    4) Always keep matched_turns.
    """
    total_turns = len(dialogue)
    if total_turns == 0 or max_turns <= 0:
        empty_result: List[Tuple[int, str]] = []
        if return_meta:
            return empty_result, {"centers_used": 0}
        return empty_result

    spec = constraint_spec or {}
    slot_values = spec.get("slot_values", {}) if isinstance(spec, dict) else {}
    match_policy = spec.get("match_policy", {}) if isinstance(spec, dict) else {}
    if not isinstance(slot_values, dict):
        slot_values = {}
    if not isinstance(match_policy, dict):
        match_policy = {}

    route_weights = _parse_retrieval_route_weights(retrieval_route_weights)
    slot_weight = float(route_weights.get("slot", 3.0))
    date_weight = float(route_weights.get("date", 2.0))
    semantic_weight = float(route_weights.get("semantic", 1.0))

    stop_words = {
        'this', 'that', 'with', 'from', 'they', 'them', 'have', 'been',
        'were', 'will', 'about', 'after', 'before', 'during', 'between',
        'their', 'which', 'event', 'date', 'these', 'those', 'would',
        'could', 'should', 'into', 'through', 'being', 'each', 'what',
        'when', 'where', 'than', 'then', 'also', 'just', 'more', 'most',
        'some', 'very', 'much', 'such', 'only', 'over', 'same', 'other',
        'social', 'context', 'timeline', 'states', 'directly'
    }
    keywords, phrases = extract_key_phrases(component)
    keywords = [w.lower() for w in keywords if w.lower() not in stop_words]
    phrases = [p.lower().strip() for p in phrases if str(p).strip()]

    date_values = _dedupe_str_list(slot_values.get("date", [])) or extract_dates_from_text(component)
    entity_values = _dedupe_str_list(slot_values.get("entity", []))
    event_values = _dedupe_str_list(slot_values.get("event", []))
    if not entity_values:
        entity_values = [p for p in re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', component) if len(p) > 2][:6]

    date_variants = []
    for date_value in date_values:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(date_value)):
            date_variants.extend(v.lower() for v in generate_date_variants(str(date_value)))
        else:
            date_variants.append(str(date_value).lower())
    date_variants = list(dict.fromkeys(date_variants))

    def _token_overlap_match(value: str, content_lower: str) -> bool:
        value_tokens = [
            t for t in re.findall(r"[a-z0-9]+", value.lower())
            if len(t) >= 3 and t not in stop_words
        ]
        if not value_tokens:
            return value.lower() in content_lower
        hits = sum(1 for t in value_tokens if re.search(rf"\b{re.escape(t)}\b", content_lower))
        need = max(1, int((len(value_tokens) * 0.6) + 0.5))
        return hits >= need

    slot_score_map: Dict[int, float] = {}
    date_score_map: Dict[int, float] = {}
    semantic_score_map: Dict[int, float] = {}

    slot_hit_turns: set[int] = set()
    date_hit_turns: set[int] = set()
    semantic_hit_turns: set[int] = set()

    for ti, turn in enumerate(dialogue):
        content_lower = turn.get("content", "").lower()

        slot_score = 0.0
        for slot, values_raw in slot_values.items():
            if _normalize_slot_name(slot) == "date":
                continue
            values = _dedupe_str_list(values_raw)
            if not values:
                continue
            policy = str(match_policy.get(slot, "fuzzy")).lower()
            for value in values:
                value_l = str(value).strip().lower()
                if not value_l:
                    continue
                matched = False
                if policy == "exact":
                    matched = value_l in content_lower
                else:
                    matched = _token_overlap_match(value_l, content_lower)
                if matched:
                    slot_score += 1.0
        if slot_score > 0:
            slot_score_map[ti] = slot_score
            slot_hit_turns.add(ti)

        date_score = 0.0
        for dv in date_variants:
            if dv and dv in content_lower:
                date_score += 1.0
                break
        if date_score > 0:
            date_score_map[ti] = date_score
            date_hit_turns.add(ti)

        semantic_score = 0.0
        phrase_hits = sum(1 for phrase in phrases if phrase and phrase in content_lower)
        if phrase_hits > 0:
            semantic_score += min(phrase_hits, 3) * 2.0
        kw_hits = 0
        for kw in keywords:
            if kw in content_lower:
                kw_hits += 1
        if kw_hits > 0:
            semantic_score += min(kw_hits, 6) * 0.5
        if semantic_score > 0:
            semantic_score_map[ti] = semantic_score
            semantic_hit_turns.add(ti)

    def _normalize_scores(score_map: Dict[int, float]) -> Dict[int, float]:
        if not score_map:
            return {}
        max_score = max(score_map.values())
        if max_score <= 0:
            return {}
        return {idx: (val / max_score) for idx, val in score_map.items()}

    slot_norm = _normalize_scores(slot_score_map)
    date_norm = _normalize_scores(date_score_map)
    semantic_norm = _normalize_scores(semantic_score_map)

    scores: Dict[int, float] = {}
    for ti in range(total_turns):
        s_slot = slot_norm.get(ti, 0.0)
        s_date = date_norm.get(ti, 0.0)
        s_sem = semantic_norm.get(ti, 0.0)
        score = (slot_weight * s_slot) + (date_weight * s_date) + (semantic_weight * s_sem)
        if score > 0:
            scores[ti] = score

    for mt in matched_turns:
        if 0 <= mt < total_turns:
            scores[mt] = max(scores.get(mt, 0.0), (slot_weight + date_weight + semantic_weight))

    if not scores:
        sample_points = [total_turns // 5, total_turns // 2, total_turns * 4 // 5]
        scores = {p: 1.0 for p in sample_points if 0 <= p < total_turns}

    radius = _task_retrieval_radius(task_type)
    candidate_centers = sorted(
        scores.items(),
        key=lambda item: (
            -item[1],
            -slot_norm.get(item[0], 0.0),
            -date_norm.get(item[0], 0.0),
            -semantic_norm.get(item[0], 0.0),
            item[0],
        ),
    )

    selected: set[int] = set()
    # Force keeping rule-derived evidence anchors first.
    for mt in matched_turns:
        if 0 <= mt < total_turns:
            for off in range(-1, 2):
                idx = mt + off
                if 0 <= idx < total_turns:
                    selected.add(idx)
    if len(selected) > max_turns:
        selected = set(sorted(selected)[:max_turns])

    retrieval_min_centers = max(1, int(retrieval_min_centers))
    forced_centers = [center for center, _ in candidate_centers[:retrieval_min_centers]]
    centers_used: List[int] = []

    for center in forced_centers:
        if len(selected) >= max_turns:
            break
        if center not in centers_used:
            centers_used.append(center)
        start = max(0, center - radius)
        end = min(total_turns - 1, center + radius)
        for idx in range(start, end + 1):
            if len(selected) >= max_turns:
                break
            selected.add(idx)

    for center, _score in candidate_centers:
        if len(selected) >= max_turns:
            break
        if center not in centers_used:
            centers_used.append(center)
        start = max(0, center - radius)
        end = min(total_turns - 1, center + radius)
        for idx in range(start, end + 1):
            if len(selected) >= max_turns:
                break
            selected.add(idx)

    sorted_indices = sorted(selected)[:max_turns]
    result: List[Tuple[int, str]] = []
    for idx in sorted_indices:
        role = "User" if dialogue[idx].get('role') == 'user' else "Character"
        content = dialogue[idx].get('content', '')[:300]
        result.append((idx, f"[Turn {idx}] {role}: {content}"))

    selected_set = set(sorted_indices)
    slot_hits = sum(1 for idx in selected_set if idx in slot_hit_turns)
    date_hits = sum(1 for idx in selected_set if idx in date_hit_turns)
    semantic_hits = sum(1 for idx in selected_set if idx in semantic_hit_turns)
    overlap_hits = 0
    for idx in selected_set:
        route_hit_count = int(idx in slot_hit_turns) + int(idx in date_hit_turns) + int(idx in semantic_hit_turns)
        if route_hit_count >= 2:
            overlap_hits += 1
    overlap_ratio = float(overlap_hits) / max(len(selected_set), 1)

    if return_meta:
        return result, {
            "centers_used": len(set(centers_used)),
            "route_slot_hits": slot_hits,
            "route_date_hits": date_hits,
            "route_semantic_hits": semantic_hits,
            "overlap_ratio": overlap_ratio,
        }
    return result


def build_llm_verify_prompt(
    component: str,
    task_type: str,
    dialogue: List[Dict],
    target_turns: List[int],
    query: str = "",
    all_components: Optional[List[str]] = None,
    target_component_idx: int = -1,
    required_slots: Optional[List[str]] = None,
    candidate_evidence_turns: Optional[List[int]] = None,
    constraint_spec: Optional[Dict[str, Any]] = None,
    retrieval_max_turns: int = 30,
    retrieval_min_centers: int = 5,
    retrieval_route_weights: Optional[Dict[str, float]] = None,
    hard_case: bool = False,
    return_meta: bool = False,
) -> Any:
    """
    Global extraction strategy: extract turns containing relevant keywords/dates
    from the whole dialogue and send them to the LLM for semantic matching.
    
    This does not depend on a fixed window; the LLM sees all places where the
    dialogue mentions relevant content.
    """
    # Globally extract relevant turns.
    relevant, retrieval_meta = extract_relevant_turns(
        component,
        dialogue,
        target_turns,
        max_turns=retrieval_max_turns,
        task_type=task_type,
        constraint_spec=constraint_spec,
        retrieval_min_centers=retrieval_min_centers,
        retrieval_route_weights=retrieval_route_weights,
        return_meta=True,
    )

    # Build turn text.
    if relevant:
        # Insert separators when selected turns are not contiguous.
        lines = []
        prev_idx = -10
        for idx, text in relevant:
            if idx > prev_idx + 2:
                lines.append(f"\n  ... (skipped turns {prev_idx+1}-{idx-1}) ...")
            lines.append(text)
            prev_idx = idx
        excerpt_text = "\n".join(lines)
        location_desc = f"{len(relevant)} relevant turns extracted from {len(dialogue)}-turn dialogue"
    else:
        excerpt_text = "(No relevant turns found in the dialogue)"
        location_desc = "no matches"

    component_lines = []
    all_components = all_components or []
    for idx, comp in enumerate(all_components):
        tag = " (TARGET)" if idx == target_component_idx else ""
        component_lines.append(f"{idx}. {comp}{tag}")
    components_text = "\n".join(component_lines) if component_lines else "(not provided)"
    required_slots = required_slots or []
    candidate_evidence_turns = candidate_evidence_turns or []
    constraint_spec = constraint_spec or {}
    forbidden_hard = _dedupe_str_list(constraint_spec.get("forbidden_hard_slots", []))
    soft_slots = _dedupe_str_list(constraint_spec.get("soft_slots", []))

    strict_note = ""
    if hard_case:
        strict_note = (
            "\nHARD-CASE MODE:\n"
            "- Same rules apply, but keep answer strictly as YES or NO."
        )

    prompt = f"""You are a strict QA component verifier.

TASK TYPE:
{task_type}

QUESTION:
{query}

ALL ANSWER COMPONENTS:
{components_text}

TARGET COMPONENT:
index={target_component_idx}
text={component}

REQUIRED SLOTS (must all be true):
{required_slots}

CONSTRAINT OBJECT:
{json.dumps(constraint_spec, ensure_ascii=False)}

CANDIDATE EVIDENCE TURNS:
{candidate_evidence_turns}

RELEVANT DIALOGUE TURNS ({location_desc}):
{excerpt_text}
{strict_note}

Decision protocol (follow internally, do not output your reasoning):
1) Use QUESTION + ALL ANSWER COMPONENTS to identify what QA-critical information in TARGET COMPONENT must be checked.
   - ALL ANSWER COMPONENTS are for semantic disambiguation and role understanding only.
   - The final evidence must come from RELEVANT DIALOGUE TURNS, not from component list priors.
2) Check whether RELEVANT DIALOGUE TURNS cover that QA-critical information of TARGET COMPONENT.
   - Allow semantic paraphrase and equivalent date expressions.
   - Do not give credit to nearby non-target components unless they explicitly cover TARGET's required meaning.
   - MUST only use evidence from RELEVANT DIALOGUE TURNS. Do not use external knowledge or commonsense completion.
   - Do not reject TARGET using slots listed in forbidden_hard_slots: {forbidden_hard}.
   - Soft slots ({soft_slots}) can be semantic/fuzzy and should not be treated as strict exact unless required.
3) Output YES only if TARGET's QA-critical information has locatable support in the retrieved turns; otherwise output NO.
   - If target meaning is only indirectly implied and there is no locatable support, output NO.

Return EXACTLY one token:
- YES  (component covered)
- NO   (component missing)

Do not output JSON.
Do not output explanation.
Do not output any extra words."""

    if return_meta:
        return prompt, {
            "retrieval_centers_used": int((retrieval_meta or {}).get("centers_used", 0)),
            "relevant_turn_count": len(relevant),
            "route_slot_hits": int((retrieval_meta or {}).get("route_slot_hits", 0)),
            "route_date_hits": int((retrieval_meta or {}).get("route_date_hits", 0)),
            "route_semantic_hits": int((retrieval_meta or {}).get("route_semantic_hits", 0)),
            "overlap_ratio": float((retrieval_meta or {}).get("overlap_ratio", 0.0)),
        }
    return prompt


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "ok", "pass"}
    return False


def parse_llm_verify_response(response: str) -> Tuple[Optional[bool], str, Dict[str, Any]]:
    """
    Parse an LLM verification response.
    Returns: (verdict, reason, structured_info)
      True  = explicit YES
      False = explicit NO
      None  = unparseable response; should not count as a vote
    """
    if not response:
        return None, "LLM returned empty response", {}

    text = response.strip()
    text = re.sub(r'^\s*```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```\s*$', '', text)
    text = text.strip()
    text_upper = text.upper()

    if text_upper == "YES":
        return True, "YES", {"raw_vote": "YES"}
    if text_upper == "NO":
        return False, "NO", {"raw_vote": "NO"}
    return None, f"invalid vote output; only YES/NO accepted: {text[:50]}", {}


# ============================================================
# Main evaluation flow
# ============================================================

def adjudicate_llm_vote(
    comp_v: ComponentVerification,
    parsed_vote: Optional[bool],
    parsed_payload: Dict[str, Any],
    llm_evidence_required: bool = True,
) -> Tuple[Optional[bool], str, str]:
    """
    Convert a raw LLM vote into a final valid vote.
    Returns: (vote, reason, reject_code)
    """
    if parsed_vote is None:
        return None, "invalid_vote", ""

    reason_text = str(parsed_payload.get("reason", "")).lower()
    positive_markers = ("covered", "confirmed", "present", "found", "explicit")
    negative_markers = ("missing", "not ", "no evidence", "absent", "fail")
    has_positive_reason = any(tok in reason_text for tok in positive_markers)
    has_negative_reason = any(tok in reason_text for tok in negative_markers)
    if parsed_vote is False and has_positive_reason and not has_negative_reason:
        return None, "vote_reason_conflict_negative", "reason_conflict"
    if parsed_vote is True and has_negative_reason and not has_positive_reason:
        return None, "vote_reason_conflict_positive", "reason_conflict"

    if parsed_vote is False:
        constraint_spec = comp_v.constraint_spec or {}
        soft_slots = {
            _normalize_slot_name(s)
            for s in _dedupe_str_list(constraint_spec.get("soft_slots", []))
            if _normalize_slot_name(s)
        }
        allowed_missing = set(comp_v.required_slots) | soft_slots
        missing_slots = {
            _normalize_slot_name(s)
            for s in _dedupe_str_list(parsed_payload.get("missing_slots", []))
            if _normalize_slot_name(s)
        }
        if missing_slots and allowed_missing and not (missing_slots & allowed_missing):
            return None, "negative_vote_conflicts_with_constraint_spec", "object_conflict"
        return False, "llm_reject", ""

    slot_verdicts = parsed_payload.get("slot_verdicts", {}) or {}
    missing_required = []
    for slot in comp_v.required_slots:
        if not _to_bool(slot_verdicts.get(slot, False)):
            missing_required.append(slot)
    if missing_required:
        return False, f"missing_required_slots:{missing_required}", "missing_slot"

    evidence_turns = parsed_payload.get("evidence_turns", []) or []
    evidence_turns = [int(t) for t in evidence_turns if isinstance(t, int) or str(t).isdigit()]
    if llm_evidence_required and not evidence_turns:
        return False, "llm_yes_without_evidence", "no_evidence"

    if llm_evidence_required and "date" in comp_v.required_slots and comp_v.evidence_turns:
        if evidence_turns and not (set(evidence_turns) & set(comp_v.evidence_turns)):
            return False, "date_slot_without_anchor_evidence", "no_evidence"

    return True, "llm_accept", ""

def should_recheck_present(
    rule_confidence: str,
    present_recheck_policy: str,
) -> bool:
    """Decide whether a rule-based PRESENT result should enter LLM review."""
    conf = (rule_confidence or "").lower()
    policy = (present_recheck_policy or "low_confidence_only").lower()

    if policy == "none":
        return False
    if policy == "all":
        return True
    if policy in {"non_high", "medium_and_low", "medium_low"}:
        return conf in {"medium", "low"}

    # Default: low_confidence_only.
    return conf == "low"


def evaluate_dialogue(
    dialogue_data: Dict,
    dialogue_idx: int,
    llm: Optional[LLM] = None,
    use_llm: bool = False,
    eval_mode: str = "strict_v2",
    hard_fact_lock: bool = True,
    hard_missing_policy: str = "llm_gate",
    present_recheck_policy: str = "low_confidence_only",
    task_profile: str = "qa_task_v1",
    hard_number_mode: str = "contextual_strict",
    date_evidence_turn_cap: int = 16,
    constraint_cache: Optional[Dict[str, Any]] = None,
    component_role_votes: int = 5,
    component_role_lock_store: Optional[Dict[str, Any]] = None,
    component_role_lock_mode: str = "read_write",
    component_role_lock_mutex: Optional[Lock] = None,
) -> DialogueEvalResult:
    """Evaluate QA coverage for one dialogue."""
    character = dialogue_data.get('character', 'Unknown')
    dialogue_id = dialogue_data.get('id', '')
    dialogue = dialogue_data.get('dialogue', [])
    tasks = dialogue_data.get('tasks_covered', [])

    # Precompute lower-cased dialogue text.
    dialogue_text_lower = ' '.join(t['content'].lower() for t in dialogue)

    result = DialogueEvalResult(
        dialogue_idx=dialogue_idx,
        character=character,
        dialogue_id=dialogue_id,
    )
    hard_missing_policy = (hard_missing_policy or "llm_gate").strip().lower()
    if hard_missing_policy not in {"llm_gate", "direct_lock"}:
        hard_missing_policy = "llm_gate"
    constraint_task_stats: Dict[str, Any] = {
        "constraint_llm_success_tasks": 0,
        "constraint_fallback_tasks": 0,
        "constraint_partial_parse_rejects": 0,
        "constraint_retry_count": 0,
    }

    for task_idx, task in enumerate(tasks):
        task_type = task.get('task_type', 'Unknown')
        query = task.get('query', '')
        components = task.get('answer_components', [])
        role_decisions = classify_component_roles(
            task_type=task_type,
            query=query,
            components=list(components),
            llm=llm,
            use_llm=use_llm,
            target_valid_votes=component_role_votes,
            role_lock_store=component_role_lock_store,
            role_lock_mode=component_role_lock_mode,
            role_lock_mutex=component_role_lock_mutex,
        )

        for comp_idx, comp in enumerate(components):
            role_info = role_decisions.get(comp_idx, {
                "role": "EVIDENCE_REQUIRED",
                "should_verify": should_verify_component(task_type, comp),
                "source": "heuristic_rule",
                "confidence": "medium",
                "votes": {},
            })
            constraint_spec, constraint_source = extract_constraint_spec(
                task_type=task_type,
                query=query,
                all_components=list(components),
                target_component=comp,
                target_component_idx=comp_idx,
                llm=llm,
                use_llm=use_llm,
                cache=constraint_cache,
                task_stats=constraint_task_stats,
                task_instance_key=f"{dialogue_idx}:{task_idx}",
            )
            verify = ComponentVerification(
                task_idx=task_idx,
                comp_idx=comp_idx,
                task_type=task_type,
                component_text=comp,
                should_verify=bool(role_info.get("should_verify", True)),
                component_role=str(role_info.get("role", "")),
                component_role_source=str(role_info.get("source", "")),
                component_role_confidence=str(role_info.get("confidence", "")),
                component_role_votes=dict(role_info.get("votes", {}) or {}),
                component_role_lock_hit=bool(role_info.get("lock_hit", False)),
                component_role_lock_key=str(role_info.get("lock_key", "")),
                query=query,
                answer_components=list(components),
                target_component_idx=comp_idx,
                constraint_spec=constraint_spec,
                constraint_source=constraint_source,
                hard_object_reason=str((constraint_spec or {}).get("rationale", "")),
                match_policy=dict((constraint_spec or {}).get("match_policy", {})),
                constraint_validation_errors=list(
                    (constraint_spec or {}).get("constraint_validation_errors", [])
                ),
                constraint_anchor_hits=dict(
                    (constraint_spec or {}).get("constraint_anchor_hits", {})
                ),
                constraint_degraded_to_soft=bool(
                    (constraint_spec or {}).get("constraint_degraded_to_soft", False)
                ),
            )
            result.total_components += 1
            if role_info.get("lock_checked", False):
                if verify.component_role_lock_hit:
                    result.component_role_lock_hits += 1
                else:
                    result.component_role_lock_misses += 1

            if constraint_source == "llm_object":
                if (constraint_spec or {}).get("hard_slots"):
                    result.llm_object_hard_count += 1
                if (constraint_spec or {}).get("soft_slots"):
                    result.llm_object_soft_count += 1

            if not verify.should_verify:
                verify.final_status = "SKIPPED"
                verify.llm_status = "SKIPPED"
                verify.decision_stage = "skipped"
                verify.decision_complete = True
                verify.repair_ready = False
                verify.decision_confidence = "high"
                verify.reason = (
                    f"inference-type component ({verify.component_role_source or 'rule'}); explicit dialogue mention not required"
                )
                result.skipped_inference += 1
                result.details.append(verify)
                continue

            result.verifiable += 1

            # Stage 1: hard constraints.
            if eval_mode == "strict_v2":
                hard_status, hard_missing, hard_meta = evaluate_hard_facts(
                    comp,
                    dialogue,
                    dialogue_text_lower,
                    task_type=task_type,
                    query=query,
                    task_profile=task_profile,
                    hard_number_mode=hard_number_mode,
                    return_meta=True,
                    constraint_spec=constraint_spec,
                )
            else:
                hard_status, hard_missing, hard_meta = "PASS", [], {}
            verify.hard_status = hard_status
            verify.hard_initial_status = hard_status
            verify.hard_missing_facts = hard_missing
            verify.required_slots = list(hard_meta.get("required_slots", []))
            verify.slot_verdicts = dict(hard_meta.get("slot_verdicts", {}))
            verify.hard_number_filtered = list(hard_meta.get("hard_number_filtered", []))
            verify.date_equivalent_match_count = int(hard_meta.get("date_equivalent_match_count", 0))
            verify.constraint_validation_errors = _dedupe_str_list(
                verify.constraint_validation_errors
                + list(hard_meta.get("constraint_validation_errors", []))
            )
            verify.constraint_anchor_hits = dict(
                verify.constraint_anchor_hits or hard_meta.get("constraint_anchor_hits", {})
            )
            verify.constraint_degraded_to_soft = bool(
                verify.constraint_degraded_to_soft
                or hard_meta.get("constraint_degraded_to_soft", False)
            )
            result.number_label_filtered_count += len(verify.hard_number_filtered)
            result.hard_object_conflict_count += int(hard_meta.get("spec_conflicts", 0))
            result.constraint_validation_reject_count += len(verify.constraint_validation_errors)
            result.hard_slot_degraded_count += int(hard_meta.get("hard_slot_degraded_count", 0))
            result.date_equivalent_match_count += int(hard_meta.get("date_equivalent_match_count", 0))

            if hard_fact_lock and hard_status == "HARD_MISSING":
                result.hard_missing_initial_count += 1
                verify.hard_gate_applied = True
                verify.evidence_turns = sorted(
                    set(
                        collect_date_evidence_turns(comp, dialogue, cap=date_evidence_turn_cap)
                        + list(hard_meta.get("date_evidence_turns", []))
                    )
                )

                has_llm = use_llm and llm is not None
                if hard_missing_policy == "direct_lock" or not has_llm:
                    verify.rule_status = "MISSING"
                    verify.rule_confidence = "high"
                    verify.rule_reason_code = "hard_missing_lock"
                    verify.llm_status = "SKIPPED"
                    verify.final_status = "MISSING"
                    verify.decision_stage = "hard"
                    verify.decision_reason_code = "hard_missing_lock"
                    verify.decision_complete = True
                    verify.repair_ready = True
                    verify.decision_confidence = "high"
                    verify.hard_gate_result = "missing_confirmed"
                    verify.reason = f"hard constraints missing: {', '.join(hard_missing)}"
                    result.hard_missing_count += 1
                    result.hard_missing_llm_confirmed_missing += 1
                    result.missing += 1
                    result.details.append(verify)
                    continue

                verify.rule_status = "SUSPICIOUS"
                verify.rule_confidence = "low"
                verify.rule_reason_code = "hard_missing_needs_llm"
                verify.llm_status = "PENDING"
                verify.final_status = "PENDING_LLM"
                verify.decision_stage = "llm_pending"
                verify.decision_reason_code = "hard_missing_needs_llm"
                verify.decision_complete = False
                verify.repair_ready = False
                verify.decision_confidence = "low"
                verify.hard_gate_result = "pending"
                verify.reason = f"hard constraints initially missing; entering LLM gate: {', '.join(hard_missing)}"
                result.details.append(verify)
                continue

            # Rule-based pre-screening.
            rule_status, matched, reason, confidence, reason_code = rule_check_component(
                task_type, comp, dialogue, dialogue_text_lower, constraint_spec=constraint_spec
            )
            verify.rule_status = rule_status
            verify.rule_confidence = confidence
            verify.rule_reason_code = reason_code
            verify.matched_turns = matched
            verify.reason = reason
            verify.evidence_turns = collect_date_evidence_turns(comp, dialogue, cap=date_evidence_turn_cap)
            if verify.matched_turns:
                verify.evidence_turns = sorted(set(verify.evidence_turns + verify.matched_turns))

            if rule_status == "PRESENT":
                need_llm = (
                    use_llm
                    and llm is not None
                    and should_recheck_present(confidence, present_recheck_policy)
                )
                if need_llm:
                    verify.llm_status = "PENDING"
                    verify.final_status = "PENDING_LLM"
                    verify.decision_stage = "llm_pending"
                    verify.decision_reason_code = "rule_present_low_conf_recheck"
                    verify.decision_complete = False
                    verify.repair_ready = False
                    verify.decision_confidence = "low"
                else:
                    verify.final_status = "PRESENT"
                    verify.llm_status = "SKIPPED"
                    verify.decision_stage = "rule"
                    verify.decision_reason_code = reason_code or "rule_present"
                    verify.decision_complete = True
                    verify.repair_ready = False
                    verify.decision_confidence = confidence or "high"
                    result.present += 1
            elif rule_status == "MISSING":
                if eval_mode != "strict_v2" and use_llm and llm is not None:
                    verify.llm_status = "PENDING"
                    verify.final_status = "PENDING_LLM"
                    verify.decision_stage = "llm_pending"
                    verify.decision_reason_code = "rule_missing_recheck_legacy"
                    verify.decision_complete = False
                    verify.repair_ready = False
                    verify.decision_confidence = "low"
                else:
                    verify.final_status = "MISSING"
                    verify.llm_status = "SKIPPED"
                    verify.decision_stage = "rule"
                    verify.decision_reason_code = reason_code or "rule_missing"
                    verify.decision_complete = True
                    verify.repair_ready = True
                    verify.decision_confidence = confidence or "high"
                    result.missing += 1
            else:  # SUSPICIOUS
                if use_llm and llm is not None:
                    verify.llm_status = "PENDING"
                    verify.final_status = "PENDING_LLM"
                    verify.decision_stage = "llm_pending"
                    verify.decision_reason_code = reason_code or "rule_suspicious_llm"
                    verify.decision_complete = False
                    verify.repair_ready = False
                    verify.decision_confidence = "low"
                else:
                    # Without an LLM, treat SUSPICIOUS as MISSING.
                    verify.final_status = "MISSING"
                    verify.llm_status = "SKIPPED"
                    verify.decision_stage = "rule"
                    verify.decision_reason_code = reason_code or "rule_suspicious_missing"
                    verify.decision_complete = True
                    verify.repair_ready = True
                    verify.decision_confidence = "low"
                    result.missing += 1

            result.details.append(verify)

    result.constraint_llm_success_tasks = int(constraint_task_stats.get("constraint_llm_success_tasks", 0))
    result.constraint_fallback_tasks = int(constraint_task_stats.get("constraint_fallback_tasks", 0))
    result.constraint_partial_parse_rejects = int(constraint_task_stats.get("constraint_partial_parse_rejects", 0))
    result.constraint_retry_count = int(constraint_task_stats.get("constraint_retry_count", 0))
    return result


def evaluate_task_components(
    dialogue_data: Dict[str, Any],
    dialogue_idx: int,
    component_keys: List[Tuple[int, int]],
    llm: Optional[LLM] = None,
    llm_fallback: Optional[LLM] = None,
    use_llm: bool = False,
    eval_mode: str = "strict_v2",
    hard_fact_lock: bool = True,
    hard_missing_policy: str = "llm_gate",
    present_recheck_policy: str = "low_confidence_only",
    task_profile: str = "qa_task_v1",
    hard_number_mode: str = "contextual_strict",
    date_evidence_turn_cap: int = 16,
    num_votes: int = 3,
    focused_votes: int = 5,
    global_votes: int = 7,
    focused_strong_yes_threshold: int = 4,
    max_vote_attempts: int = 20,
    retry_until_success: bool = True,
    retry_round_sleep: float = 1.0,
    retry_backoff_cap: float = 30.0,
    retrieval_min_centers: int = 5,
    focused_turn_cap: int = 42,
    global_turn_cap: int = 72,
    retrieval_route_weights: Optional[Dict[str, float]] = None,
    vote_temperature: float = 0.0,
) -> List[ComponentVerification]:
    """
    Evaluate only a specified task/component subset by reusing the strict main flow.
    Returns ComponentVerification objects for the requested components, matching the details schema.
    """
    normalized_keys: set[Tuple[int, int]] = set()
    for task_idx, comp_idx in component_keys:
        try:
            normalized_keys.add((int(task_idx), int(comp_idx)))
        except (TypeError, ValueError):
            continue
    if not normalized_keys:
        return []

    base_result = evaluate_dialogue(
        dialogue_data=dialogue_data,
        dialogue_idx=dialogue_idx,
        llm=llm,
        use_llm=use_llm,
        eval_mode=eval_mode,
        hard_fact_lock=hard_fact_lock,
        hard_missing_policy=hard_missing_policy,
        present_recheck_policy=present_recheck_policy,
        task_profile=task_profile,
        hard_number_mode=hard_number_mode,
        date_evidence_turn_cap=date_evidence_turn_cap,
    )
    filtered_details = [
        d for d in base_result.details
        if (d.task_idx, d.comp_idx) in normalized_keys
    ]
    if not filtered_details:
        return []

    subset_result = DialogueEvalResult(
        dialogue_idx=base_result.dialogue_idx,
        character=base_result.character,
        dialogue_id=base_result.dialogue_id,
        total_components=len(filtered_details),
        skipped_inference=sum(1 for d in filtered_details if d.final_status == "SKIPPED"),
        verifiable=sum(1 for d in filtered_details if d.final_status != "SKIPPED"),
        present=sum(1 for d in filtered_details if d.final_status == "PRESENT"),
        missing=sum(1 for d in filtered_details if d.final_status == "MISSING"),
        details=filtered_details,
    )

    if use_llm and llm is not None:
        pending_exists = any(
            d.final_status == "PENDING_LLM" or d.llm_status == "PENDING"
            for d in subset_result.details
        )
        if pending_exists:
            semaphore = asyncio.Semaphore(1)
            route_weights = retrieval_route_weights or {"slot": 3.0, "date": 2.0, "semantic": 1.0}

            async def _run_subset_llm() -> None:
                await run_llm_verification(
                    result=subset_result,
                    dialogue=dialogue_data.get("dialogue", []),
                    llm=llm,
                    llm_fallback=llm_fallback,
                    semaphore=semaphore,
                    num_votes=num_votes,
                    focused_votes=focused_votes,
                    global_votes=global_votes,
                    focused_strong_yes_threshold=focused_strong_yes_threshold,
                    max_vote_attempts=max_vote_attempts,
                    retry_until_success=retry_until_success,
                    retry_round_sleep=retry_round_sleep,
                    retry_backoff_cap=retry_backoff_cap,
                    retrieval_min_centers=retrieval_min_centers,
                    focused_turn_cap=focused_turn_cap,
                    global_turn_cap=global_turn_cap,
                    retrieval_route_weights=route_weights,
                    vote_temperature=vote_temperature,
                )

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                raise RuntimeError("evaluate_task_components cannot run async LLM verification inside an active event loop")
            asyncio.run(_run_subset_llm())

    return subset_result.details


def explain_missing_reason(component_verification: Any) -> Dict[str, Any]:
    """
    Generate structured missing-reason metadata for the next repair round.
    """
    if isinstance(component_verification, ComponentVerification):
        comp = asdict(component_verification)
    elif isinstance(component_verification, dict):
        comp = dict(component_verification)
    else:
        comp = {}

    reason_code = (
        str(comp.get("decision_reason_code") or comp.get("rule_reason_code") or "missing_unknown")
    )
    reason_text = str(comp.get("reason") or "").strip()
    if not reason_text:
        if reason_code.startswith("hard_missing"):
            reason_text = "Hard constraints indicate that this component is missing key facts."
        elif "llm_missing" in reason_code or "global" in reason_code:
            reason_text = "LLM review indicates that the target fact is not clearly present in the evidence snippets."
        else:
            reason_text = "Current evidence is insufficient to support that the component is present."

    evidence_turns_raw = comp.get("evidence_turns", []) or []
    matched_turns_raw = comp.get("matched_turns", []) or []
    evidence_turns: List[int] = []
    for item in list(evidence_turns_raw) + list(matched_turns_raw):
        try:
            iv = int(item)
        except (TypeError, ValueError):
            continue
        if iv not in evidence_turns:
            evidence_turns.append(iv)

    hard_missing_facts = [str(x) for x in (comp.get("hard_missing_facts", []) or []) if str(x).strip()]
    required_slots = [str(x) for x in (comp.get("required_slots", []) or []) if str(x).strip()]
    if hard_missing_facts:
        suggested_fix_focus = "Fill in missing facts: " + "; ".join(hard_missing_facts[:3])
    elif required_slots:
        suggested_fix_focus = "Explicitly express key slots: " + ", ".join(required_slots[:4])
    else:
        suggested_fix_focus = "Write the target component's key facts into the dialogue with locatable wording."

    return {
        "reason_code": reason_code,
        "reason_text": reason_text,
        "evidence_turns": evidence_turns,
        "suggested_fix_focus": suggested_fix_focus,
    }


async def run_llm_verification(
    result: DialogueEvalResult,
    dialogue: List[Dict],
    llm: LLM,
    semaphore: asyncio.Semaphore,
    llm_fallback: Optional[LLM] = None,
    num_votes: int = 3,
    focused_votes: Optional[int] = None,
    global_votes: Optional[int] = None,
    focused_strong_yes_threshold: int = 4,
    max_vote_attempts: int = 20,
    retry_until_success: bool = True,
    retry_round_sleep: float = 1.0,
    retry_backoff_cap: float = 30.0,
    retrieval_min_centers: int = 5,
    focused_turn_cap: int = 42,
    global_turn_cap: int = 72,
    retrieval_route_weights: Optional[Dict[str, float]] = None,
    vote_temperature: float = 0.0,
) -> DialogueEvalResult:
    """
    Verify all PENDING_LLM components with hierarchical majority voting.
    - Focused stage runs first (default 5 valid votes); strong YES (default >=4) returns PRESENT.
    - Boundary cases enter Global stage (default 7 valid votes) for final adjudication.
    - Each round makes at most max_vote_attempts calls per component; insufficient valid votes carry to the next round.
    - retry_until_success=True retries until all components converge to PRESENT/MISSING.
    - retry_until_success=False leaves components with fewer than 3 valid votes as PENDING_LLM after one round.
    """
    def _validate_vote_target(name: str, value: int) -> int:
        value = int(value)
        if value < 3 or value % 2 == 0:
            raise ValueError(f"{name} must be odd and >= 3")
        return value

    if focused_votes is None:
        focused_votes = num_votes
    if global_votes is None:
        global_votes = max(num_votes, focused_votes)
    focused_votes = _validate_vote_target("focused_votes", focused_votes)
    global_votes = _validate_vote_target("global_votes", global_votes)
    focused_majority = (focused_votes // 2) + 1
    global_majority = (global_votes // 2) + 1
    focused_strong_yes_threshold = int(focused_strong_yes_threshold)
    if focused_strong_yes_threshold < focused_majority:
        focused_strong_yes_threshold = focused_majority
    if focused_strong_yes_threshold > focused_votes:
        focused_strong_yes_threshold = focused_votes

    max_vote_attempts = max(1, int(max_vote_attempts))
    retry_round_sleep = max(0.0, float(retry_round_sleep))
    retry_backoff_cap = max(0.0, float(retry_backoff_cap))
    retrieval_min_centers = max(1, int(retrieval_min_centers))
    focused_turn_cap = max(1, int(focused_turn_cap))
    global_turn_cap = max(1, int(global_turn_cap))
    route_weights = _parse_retrieval_route_weights(retrieval_route_weights)

    result.vote_target_valid_votes_focused = focused_votes
    result.vote_target_valid_votes_global = global_votes
    result.vote_majority_threshold_focused = focused_majority
    result.vote_majority_threshold_global = global_majority

    pending = [
        d for d in result.details
        if d.final_status == "PENDING_LLM" or d.llm_status == "PENDING"
    ]
    if not pending:
        return result

    def _is_content_filter_error(exc: Exception) -> bool:
        response = getattr(exc, "response", None)
        if response is None:
            return False
        status_code = getattr(response, "status_code", None)
        raw_text = str(getattr(response, "text", "") or "")
        lowered = raw_text.lower()
        if "content_filter" in lowered or "moderation block" in lowered:
            return True
        if status_code in (400, 403, 421):
            if '"code":"421"' in lowered or '"code":421' in lowered:
                return True
        return False

    async def single_llm_call(prompt: str) -> Optional[str]:
        async with semaphore:
            try:
                try:
                    completion = await llm.inference(prompt, max_tokens=200, temperature=vote_temperature)
                except TypeError:
                    # Older LLM wrappers/test stubs do not support the temperature argument.
                    completion = await llm.inference(prompt, max_tokens=200)
                return llm.decode(completion)
            except Exception as primary_err:
                if llm_fallback is not None and _is_content_filter_error(primary_err):
                    try:
                        try:
                            completion = await llm_fallback.inference(prompt, max_tokens=200, temperature=vote_temperature)
                        except TypeError:
                            completion = await llm_fallback.inference(prompt, max_tokens=200)
                        return llm_fallback.decode(completion)
                    except Exception:
                        return None
                return None

    async def _collect_votes(
        comp_v: ComponentVerification,
        vote_target: int,
        retrieval_max_turns: int,
        attempt_limit: int,
        needed_valid: int,
        hard_case: bool = False,
    ) -> Dict[str, int]:
        needed_valid = max(1, int(needed_valid))
        prompt, prompt_meta = build_llm_verify_prompt(
            component=comp_v.component_text,
            task_type=comp_v.task_type,
            dialogue=dialogue,
            target_turns=comp_v.matched_turns,
            query=comp_v.query,
            all_components=comp_v.answer_components,
            target_component_idx=comp_v.target_component_idx,
            required_slots=comp_v.required_slots,
            candidate_evidence_turns=comp_v.evidence_turns,
            constraint_spec=comp_v.constraint_spec,
            retrieval_max_turns=retrieval_max_turns,
            retrieval_min_centers=retrieval_min_centers,
            retrieval_route_weights=route_weights,
            hard_case=hard_case,
            return_meta=True,
        )
        comp_v.retrieval_centers_used = max(
            int(comp_v.retrieval_centers_used or 0),
            int((prompt_meta or {}).get("retrieval_centers_used", 0)),
        )
        comp_v.retrieval_route_slot_hits = max(
            int(comp_v.retrieval_route_slot_hits or 0),
            int((prompt_meta or {}).get("route_slot_hits", 0)),
        )
        comp_v.retrieval_route_date_hits = max(
            int(comp_v.retrieval_route_date_hits or 0),
            int((prompt_meta or {}).get("route_date_hits", 0)),
        )
        comp_v.retrieval_route_semantic_hits = max(
            int(comp_v.retrieval_route_semantic_hits or 0),
            int((prompt_meta or {}).get("route_semantic_hits", 0)),
        )
        comp_v.retrieval_overlap_ratio = max(
            float(comp_v.retrieval_overlap_ratio or 0.0),
            float((prompt_meta or {}).get("overlap_ratio", 0.0)),
        )
        yes_count = 0
        no_count = 0
        attempts_used = 0
        error_count = 0

        while (yes_count + no_count) < needed_valid and attempts_used < attempt_limit:
            attempts_used += 1
            response = await single_llm_call(prompt)
            if response is None:
                error_count += 1
                await asyncio.sleep(0.2)
                continue
            parsed_vote, _reason, _parsed_payload = parse_llm_verify_response(response)
            if parsed_vote is None:
                continue
            if parsed_vote:
                yes_count += 1
            else:
                no_count += 1

        return {
            "yes": yes_count,
            "no": no_count,
            "attempts": attempts_used,
            "errors": error_count,
            "valid": yes_count + no_count,
            "target": vote_target,
        }

    def _new_stage_state() -> Dict[str, int]:
        return {
            "yes": 0,
            "no": 0,
            "attempts": 0,
            "errors": 0,
            "rounds": 0,
        }

    def _decision_path(state: Dict[str, Any], final_decision: str) -> str:
        focused = state["focused"]
        global_ = state["global"]
        focused_done = (focused["yes"] + focused["no"]) >= focused_votes
        global_done = (global_["yes"] + global_["no"]) >= global_votes
        if final_decision == "PRESENT":
            if global_done:
                return "focused_to_global_yes"
            return "focused_strong_yes"
        if final_decision == "MISSING":
            return "focused_to_global_no"
        if final_decision == "UNFINISHED":
            return "global_incomplete" if state.get("phase") == "global" else "focused_incomplete"
        if final_decision == "IN_PROGRESS":
            if focused_done and not global_done:
                return "focused_to_global_pending"
            return "global_pending" if state.get("phase") == "global" else "focused_pending"
        return "unknown"

    def _build_vote_meta(state: Dict[str, Any], final_decision: str) -> Dict[str, Any]:
        focused = state["focused"]
        global_ = state["global"]
        phase = state.get("phase", "focused")
        if phase == "global" and (global_["yes"] + global_["no"]) > 0:
            final_yes = global_["yes"]
            final_no = global_["no"]
        else:
            final_yes = focused["yes"]
            final_no = focused["no"]
        return {
            "target_valid_votes": focused_votes,  # backward compatible field
            "focused_target_votes": focused_votes,
            "global_target_votes": global_votes,
            "focused_majority_threshold": focused_majority,
            "global_majority_threshold": global_majority,
            "focused_strong_yes_threshold": focused_strong_yes_threshold,
            "max_vote_attempts_per_round": max_vote_attempts,
            "focused_votes": {
                "yes_votes": focused["yes"],
                "no_votes": focused["no"],
                "attempts_used": focused["attempts"],
                "error_count": focused["errors"],
                "rounds_used": focused["rounds"],
                "valid_votes": focused["yes"] + focused["no"],
            },
            "global_votes": {
                "yes_votes": global_["yes"],
                "no_votes": global_["no"],
                "attempts_used": global_["attempts"],
                "error_count": global_["errors"],
                "rounds_used": global_["rounds"],
                "valid_votes": global_["yes"] + global_["no"],
            },
            "attempts_used": focused["attempts"] + global_["attempts"],
            "error_count": focused["errors"] + global_["errors"],
            "yes_votes": final_yes,
            "no_votes": final_no,
            "phase": phase,
            "final_decision": final_decision,
            "decision_path": _decision_path(state, final_decision),
        }

    def _update_round_metrics(state: Dict[str, Any]) -> None:
        total_rounds = state["focused"]["rounds"] + state["global"]["rounds"]
        if total_rounds > 1 and not state.get("retried_counted", False):
            result.components_retried_count += 1
            state["retried_counted"] = True
        if total_rounds > result.max_rounds_single_component:
            result.max_rounds_single_component = total_rounds

    def _accumulate_retrieval_stats_once(comp_v: ComponentVerification, state: Dict[str, Any]) -> None:
        if state.get("retrieval_stats_counted", False):
            return
        state["retrieval_stats_counted"] = True
        result.retrieval_route_slot_hits += int(comp_v.retrieval_route_slot_hits or 0)
        result.retrieval_route_date_hits += int(comp_v.retrieval_route_date_hits or 0)
        result.retrieval_route_semantic_hits += int(comp_v.retrieval_route_semantic_hits or 0)
        result.retrieval_overlap_ratio_sum += float(comp_v.retrieval_overlap_ratio or 0.0)
        result.retrieval_overlap_ratio_samples += 1

    def _mark_in_progress(comp_v: ComponentVerification, state: Dict[str, Any]) -> None:
        comp_v.llm_vote_meta = _build_vote_meta(state, "IN_PROGRESS")
        comp_v.decision_stage = "llm_retrying"
        comp_v.llm_status = "PENDING"
        comp_v.final_status = "PENDING_LLM"
        comp_v.decision_reason_code = "llm_retrying"
        comp_v.decision_complete = False
        comp_v.repair_ready = False
        comp_v.decision_confidence = "low"
        if comp_v.hard_gate_applied and comp_v.hard_initial_status == "HARD_MISSING":
            comp_v.hard_gate_result = "pending"

    def _mark_unfinished(
        comp_v: ComponentVerification,
        state: Dict[str, Any],
        reason_code: str,
    ) -> None:
        _accumulate_retrieval_stats_once(comp_v, state)
        comp_v.llm_vote_meta = _build_vote_meta(state, "UNFINISHED")
        comp_v.decision_stage = "llm_incomplete"
        comp_v.llm_status = "PENDING"
        comp_v.final_status = "PENDING_LLM"
        comp_v.decision_reason_code = reason_code
        comp_v.decision_complete = False
        comp_v.repair_ready = False
        comp_v.decision_confidence = "low"
        comp_v.hard_gate_result = "pending" if comp_v.hard_gate_applied else comp_v.hard_gate_result

        result.unfinished_components_count += 1
        result.failed_components_for_rerun += 1
        if comp_v.hard_gate_applied and comp_v.hard_initial_status == "HARD_MISSING":
            result.hard_missing_llm_pending += 1

    def _mark_final(
        comp_v: ComponentVerification,
        state: Dict[str, Any],
        final_present: bool,
        reason_code: str,
    ) -> None:
        _accumulate_retrieval_stats_once(comp_v, state)
        phase = state.get("phase", "focused")
        phase_votes = state.get(phase, {})
        yes_count = int(phase_votes.get("yes", 0))
        no_count = int(phase_votes.get("no", 0))
        comp_v.llm_vote_meta = _build_vote_meta(state, "PRESENT" if final_present else "MISSING")
        comp_v.decision_stage = "llm_rechecked"
        comp_v.llm_status = "PRESENT" if final_present else "MISSING"
        comp_v.final_status = "PRESENT" if final_present else "MISSING"
        comp_v.decision_reason_code = reason_code
        comp_v.decision_complete = True
        comp_v.repair_ready = not final_present
        stage_target = focused_votes if phase == "focused" else global_votes
        comp_v.decision_confidence = "high" if (yes_count == stage_target or no_count == stage_target) else "medium"

        if comp_v.hard_gate_applied and comp_v.hard_initial_status == "HARD_MISSING":
            if final_present:
                comp_v.hard_gate_result = "present_override"
                result.hard_missing_llm_overridden_present += 1
            else:
                comp_v.hard_gate_result = "missing_confirmed"
                result.hard_missing_llm_confirmed_missing += 1
                result.hard_missing_count += 1

        if comp_v.reason:
            comp_v.reason += f" | valid LLM votes ({phase}): YES={yes_count}, NO={no_count}"
        else:
            comp_v.reason = f"valid LLM votes ({phase}): YES={yes_count}, NO={no_count}"

        if final_present:
            result.present += 1
            result.llm_rechecked_present_count += 1
            if phase == "focused":
                result.focused_strong_yes_shortcuts += 1
            else:
                result.global_final_present += 1
        else:
            result.missing += 1
            if phase == "global":
                result.global_final_missing += 1

    component_by_id: Dict[int, ComponentVerification] = {id(comp_v): comp_v for comp_v in pending}
    state_by_id: Dict[int, Dict[str, Any]] = {}
    pending_ids: List[int] = []

    for comp_v in pending:
        cid = id(comp_v)
        pending_ids.append(cid)
        state_by_id[cid] = {
            "phase": "focused",
            "focused": _new_stage_state(),
            "global": _new_stage_state(),
            "error_streak": 0,
            "retried_counted": False,
            "retrieval_stats_counted": False,
        }
        _mark_in_progress(comp_v, state_by_id[cid])

    async def _collect_round(cid: int) -> Tuple[int, str, Dict[str, int]]:
        comp_v = component_by_id[cid]
        state = state_by_id[cid]
        phase = state["phase"]
        phase_state = state[phase]
        stage_target = focused_votes if phase == "focused" else global_votes
        needed_valid = stage_target - (phase_state["yes"] + phase_state["no"])
        if needed_valid <= 0:
            needed_valid = 1
        retrieval_max_turns = focused_turn_cap if phase == "focused" else global_turn_cap
        round_votes = await _collect_votes(
            comp_v=comp_v,
            vote_target=stage_target,
            retrieval_max_turns=retrieval_max_turns,
            attempt_limit=max_vote_attempts,
            needed_valid=needed_valid,
            hard_case=(phase == "global"),
        )
        return cid, phase, round_votes

    while pending_ids:
        result.retry_rounds_total += 1
        round_outputs = await asyncio.gather(*[_collect_round(cid) for cid in pending_ids])
        completed_ids: set[int] = set()

        for cid, phase, round_votes in round_outputs:
            comp_v = component_by_id[cid]
            state = state_by_id[cid]
            phase_state = state[phase]

            phase_state["yes"] += round_votes["yes"]
            phase_state["no"] += round_votes["no"]
            phase_state["attempts"] += round_votes["attempts"]
            phase_state["errors"] += round_votes["errors"]
            phase_state["rounds"] += 1

            if round_votes["errors"] > 0:
                result.network_retry_events += round_votes["errors"]
                if round_votes["valid"] == 0:
                    state["error_streak"] += 1
                else:
                    state["error_streak"] = 0
            elif round_votes["valid"] > 0:
                state["error_streak"] = 0

            _update_round_metrics(state)

            stage_valid = phase_state["yes"] + phase_state["no"]
            stage_target = focused_votes if phase == "focused" else global_votes
            stage_majority = focused_majority if phase == "focused" else global_majority
            if stage_valid >= stage_target:
                if phase == "focused":
                    if phase_state["yes"] >= focused_strong_yes_threshold:
                        _mark_final(
                            comp_v=comp_v,
                            state=state,
                            final_present=True,
                            reason_code="llm_present_focused_strong_yes",
                        )
                        completed_ids.add(cid)
                    else:
                        state["phase"] = "global"
                        result.regression_guard_triggered_count += 1
                        result.focused_to_global_escalations += 1
                        _mark_in_progress(comp_v, state)
                        comp_v.decision_reason_code = "llm_global_recheck_borderline"
                        comp_v.decision_stage = "llm_pending_global"
                else:
                    if phase_state["yes"] >= stage_majority:
                        _mark_final(
                            comp_v=comp_v,
                            state=state,
                            final_present=True,
                            reason_code="llm_present_global_majority",
                        )
                    else:
                        _mark_final(
                            comp_v=comp_v,
                            state=state,
                            final_present=False,
                            reason_code="llm_missing_global_majority",
                        )
                    completed_ids.add(cid)
            else:
                _mark_in_progress(comp_v, state)

        if completed_ids:
            pending_ids = [cid for cid in pending_ids if cid not in completed_ids]
            if not pending_ids:
                break

        if not retry_until_success:
            for cid in pending_ids:
                comp_v = component_by_id[cid]
                state = state_by_id[cid]
                _mark_unfinished(comp_v, state, "llm_not_enough_valid_votes")
            pending_ids = []
            break

        max_error_streak = max(state_by_id[cid].get("error_streak", 0) for cid in pending_ids) if pending_ids else 0
        if max_error_streak > 0:
            sleep_seconds = min(
                retry_backoff_cap,
                max(retry_round_sleep, retry_round_sleep * (2 ** (max_error_streak - 1))),
            )
        else:
            sleep_seconds = retry_round_sleep
        if sleep_seconds > 0:
            await asyncio.sleep(sleep_seconds)

    return result


def _evaluate_component_status_for_auto_check(
    task_type: str,
    query: str,
    component_text: str,
    dialogue: List[Dict[str, Any]],
) -> str:
    sample_dialogue = {
        "character": "AutoCheck",
        "id": "auto_check",
        "dialogue": dialogue,
        "tasks_covered": [
            {
                "task_type": task_type,
                "query": query,
                "answer_components": [component_text],
            }
        ],
    }
    eval_result = evaluate_dialogue(
        dialogue_data=sample_dialogue,
        dialogue_idx=0,
        llm=None,
        use_llm=False,
        eval_mode="strict_v2",
        hard_fact_lock=True,
        present_recheck_policy="none",
        task_profile="qa_task_v1",
        hard_number_mode="contextual_strict",
        date_evidence_turn_cap=16,
        constraint_cache={},
    )
    if not eval_result.details:
        return "MISSING"
    return eval_result.details[0].final_status or "MISSING"


def _build_redacted_dialogue_for_component(
    dialogue: List[Dict[str, Any]],
    component_text: str,
) -> Tuple[List[Dict[str, Any]], int]:
    keywords, phrases = extract_key_phrases(component_text)
    targets = [k.lower() for k in keywords[:8]]
    targets.extend(p.lower() for p in phrases[:4])
    for date_str in extract_dates_from_text(component_text):
        targets.extend(v.lower() for v in generate_date_variants(date_str))
    targets = [t for t in targets if t]
    if not targets:
        return list(dialogue), 0

    redacted = []
    removed = 0
    for turn in dialogue:
        content = str(turn.get("content", "")).lower()
        if any(t in content for t in targets):
            removed += 1
            continue
        redacted.append(dict(turn))
    if not redacted:
        return list(dialogue), 0
    return redacted, removed


def run_auto_quality_check(
    input_path: str,
    output_path: str,
    max_dialogues: int = 50,
    max_samples: int = 120,
):
    """Built-in automated quality proxy evaluation without human labels."""
    with open(input_path, "r", encoding="utf-8") as f:
        all_dialogues = json.load(f)

    candidates: List[Dict[str, Any]] = []
    for d_idx, dialogue_data in enumerate(all_dialogues[:max_dialogues]):
        dialogue = dialogue_data.get("dialogue", [])
        for task in dialogue_data.get("tasks_covered", []):
            task_type = task.get("task_type", "")
            query = task.get("query", "")
            for comp in task.get("answer_components", []):
                if not should_verify_component(task_type, comp):
                    continue
                candidates.append({
                    "dialogue_idx": d_idx,
                    "task_type": task_type,
                    "query": query,
                    "component_text": comp,
                    "dialogue": dialogue,
                })
                if len(candidates) >= max_samples:
                    break
            if len(candidates) >= max_samples:
                break
        if len(candidates) >= max_samples:
            break

    if not candidates:
        report = {
            "summary": {
                "max_dialogues": max_dialogues,
                "max_samples": max_samples,
                "sampled_components": 0,
            },
            "metrics": {},
            "samples": [],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return report

    false_missing = 0
    false_present = 0
    redundancy_risk = 0
    redaction_cases = 0
    repaired_present = 0
    sampled_rows = []

    for row in candidates:
        component = row["component_text"]
        dialogue = row["dialogue"]
        task_type = row["task_type"]
        query = row["query"]

        original_status = _evaluate_component_status_for_auto_check(
            task_type=task_type,
            query=query,
            component_text=component,
            dialogue=dialogue,
        )
        injected_dialogue = list(dialogue) + [{"role": "assistant", "content": component}]
        injected_status = _evaluate_component_status_for_auto_check(
            task_type=task_type,
            query=query,
            component_text=component,
            dialogue=injected_dialogue,
        )
        if injected_status != "PRESENT":
            false_missing += 1

        redacted_dialogue, removed = _build_redacted_dialogue_for_component(dialogue, component)
        if removed > 0:
            redaction_cases += 1
            redacted_status = _evaluate_component_status_for_auto_check(
                task_type=task_type,
                query=query,
                component_text=component,
                dialogue=redacted_dialogue,
            )
            if redacted_status == "PRESENT":
                false_present += 1
            repaired_status = _evaluate_component_status_for_auto_check(
                task_type=task_type,
                query=query,
                component_text=component,
                dialogue=redacted_dialogue + [{"role": "assistant", "content": component}],
            )
            if repaired_status == "PRESENT":
                repaired_present += 1
        else:
            redacted_status = "SKIPPED"
            repaired_status = "SKIPPED"

        dialogue_text_lower = " ".join(t.get("content", "") for t in dialogue).lower()
        if component.lower() in dialogue_text_lower and original_status == "MISSING":
            redundancy_risk += 1

        sampled_rows.append({
            "dialogue_idx": row["dialogue_idx"],
            "task_type": task_type,
            "component_text": component[:180],
            "original_status": original_status,
            "injected_status": injected_status,
            "redacted_status": redacted_status,
            "repaired_status": repaired_status,
            "redacted_turns_removed": removed,
        })

    total_samples = len(candidates)
    metrics = {
        "false_missing_proxy": false_missing / max(total_samples, 1),
        "false_present_proxy": false_present / max(redaction_cases, 1),
        "repair_precision_proxy": repaired_present / max(redaction_cases, 1),
        "repair_redundancy_proxy": redundancy_risk / max(total_samples, 1),
        "unfinished_components_count": 0,
    }
    report = {
        "summary": {
            "max_dialogues": max_dialogues,
            "max_samples": max_samples,
            "sampled_components": total_samples,
            "redaction_cases": redaction_cases,
        },
        "metrics": metrics,
        "samples": sampled_rows[: min(40, len(sampled_rows))],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def compare_with_reference(
    new_results: List[Dict[str, Any]],
    reference_path: str,
) -> Dict[str, Any]:
    """Compare current evaluation results with a reference report and detect status flips."""
    stability_report: Dict[str, Any] = {
        "reference_path": reference_path,
        "flip_count": 0,
        "total_compared": 0,
        "flip_rate": 0.0,
        "flipped_components": [],
    }
    try:
        with open(reference_path, "r", encoding="utf-8") as f:
            ref_data = json.load(f)
    except Exception as e:
        print(f"⚠️ Failed to load reference report {reference_path}: {e}")
        stability_report["error"] = str(e)
        return stability_report

    ref_results = ref_data.get("results", [])
    # Build reference index: (dialogue_idx, task_idx, comp_idx) -> final_status.
    ref_index: Dict[Tuple[int, int, int], str] = {}
    for r in ref_results:
        didx = r.get("dialogue_idx", -1)
        for d in r.get("details", []):
            key = (didx, d.get("task_idx", -1), d.get("comp_idx", -1))
            ref_index[key] = d.get("final_status", "")

    flip_count = 0
    total_compared = 0
    flipped: List[Dict[str, Any]] = []

    for r in new_results:
        didx = r.get("dialogue_idx", -1)
        for d in r.get("details", []):
            key = (didx, d.get("task_idx", -1), d.get("comp_idx", -1))
            new_status = d.get("final_status", "")
            if key not in ref_index:
                continue
            old_status = ref_index[key]
            total_compared += 1
            if old_status != new_status:
                flip_count += 1
                flipped.append({
                    "dialogue_idx": key[0],
                    "task_idx": key[1],
                    "comp_idx": key[2],
                    "old_status": old_status,
                    "new_status": new_status,
                })

    flip_rate = flip_count / max(total_compared, 1)
    stability_report.update({
        "flip_count": flip_count,
        "total_compared": total_compared,
        "flip_rate": round(flip_rate, 6),
        "flipped_components": flipped[:200],  # Keep at most 200 flips.
    })

    # Print summary.
    print("\n📊 Consistency comparison report:")
    print(f"  Reference: {reference_path}")
    print(f"  Compared components: {total_compared}")
    print(f"  Flips: {flip_count}")
    print(f"  Flip rate: {flip_rate:.2%}")
    if flipped:
        print("  First 5 flips:")
        for f_item in flipped[:5]:
            print(f"    dialogue={f_item['dialogue_idx']} task={f_item['task_idx']} "
                  f"component={f_item['comp_idx']}: {f_item['old_status']} -> {f_item['new_status']}")

    return stability_report


def run_evaluation(
    input_path: str,
    output_path: str,
    use_llm: bool = False,
    model_name: str = "mimo-v2-flash",
    concurrency: int = 10,
    limit: Optional[int] = None,
    checkpoint_interval: int = 10,
    num_votes: int = 3,
    focused_votes: int = 5,
    global_votes: int = 7,
    focused_strong_yes_threshold: int = 4,
    eval_mode: str = "strict_v2",
    hard_fact_lock: bool = True,
    hard_missing_policy: str = "llm_gate",
    present_recheck_policy: str = "low_confidence_only",
    task_profile: str = "qa_task_v1",
    hard_number_mode: str = "contextual_strict",
    date_evidence_turn_cap: int = 16,
    auto_check_mode: str = "off",
    auto_check_max_dialogues: int = 50,
    auto_check_max_samples: int = 120,
    max_vote_attempts: int = 20,
    fallback_model_name: str = "kimi-k2.5",
    retry_until_success: bool = True,
    retry_round_sleep: float = 1.0,
    retry_backoff_cap: float = 30.0,
    component_role_votes: int = 5,
    component_role_lock_path: str = "../output/final/component_role_lock.json",
    component_role_lock_mode: str = "read_write",
    retrieval_min_centers: int = 5,
    focused_turn_cap: int = 42,
    global_turn_cap: int = 72,
    retrieval_route_weights: str = "slot=3,date=2,semantic=1",
    vote_temperature: float = 0.0,
    constraint_cache_path: str = "../output/final/constraint_cache.json",
    constraint_cache_mode: str = "read_write",
    consistency_reference: Optional[str] = None,
):
    """Run the full batch evaluation."""
    hard_missing_policy = (hard_missing_policy or "llm_gate").strip().lower()
    if hard_missing_policy not in {"llm_gate", "direct_lock"}:
        print(f"⚠️ Invalid hard_missing_policy={hard_missing_policy}; falling back to llm_gate")
        hard_missing_policy = "llm_gate"

    def _validate_odd_vote(name: str, value: int) -> int:
        value = int(value)
        if value < 3 or value % 2 == 0:
            raise ValueError(f"{name} must be odd and >= 3")
        return value

    num_votes = max(3, int(num_votes))
    focused_votes = _validate_odd_vote("focused_votes", focused_votes)
    global_votes = _validate_odd_vote("global_votes", global_votes)
    focused_strong_yes_threshold = int(focused_strong_yes_threshold)
    focused_majority = (focused_votes // 2) + 1
    if focused_strong_yes_threshold < focused_majority:
        focused_strong_yes_threshold = focused_majority
    if focused_strong_yes_threshold > focused_votes:
        focused_strong_yes_threshold = focused_votes
    component_role_votes = max(1, int(component_role_votes))
    retrieval_min_centers = max(1, int(retrieval_min_centers))
    focused_turn_cap = max(1, int(focused_turn_cap))
    global_turn_cap = max(1, int(global_turn_cap))
    route_weights = _parse_retrieval_route_weights(retrieval_route_weights)
    component_role_lock_mode = (component_role_lock_mode or "read_write").strip().lower()
    if component_role_lock_mode not in {"read_write", "read_only", "off"}:
        print(f"⚠️ Invalid component_role_lock_mode={component_role_lock_mode}; falling back to read_write")
        component_role_lock_mode = "read_write"

    print(f"📂 Loading data: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        all_dialogues = json.load(f)

    if limit:
        all_dialogues = all_dialogues[:limit]
        print(f"🛑 Debug mode: evaluating only the first {limit} dialogues")

    print(f"📊 Total dialogues: {len(all_dialogues)}")

    component_role_lock_store: Dict[str, Any] = {}
    component_role_lock_mutex: Optional[Lock] = None
    if component_role_lock_mode != "off":
        component_role_lock_store = _load_component_role_lock(component_role_lock_path)
        component_role_lock_mutex = Lock()
        print(f"🔐 Component-role lock: mode={component_role_lock_mode}, entries={len(component_role_lock_store)}")

    # Constraint cache.
    constraint_cache_mode = (constraint_cache_mode or "read_write").strip().lower()
    if constraint_cache_mode not in {"read_write", "read_only", "off"}:
        print(f"⚠️ Invalid constraint_cache_mode={constraint_cache_mode}; falling back to read_write")
        constraint_cache_mode = "read_write"
    shared_constraint_cache: Dict[str, Any] = {}
    constraint_cache_mutex: Optional[Lock] = None
    if constraint_cache_mode != "off":
        shared_constraint_cache = _load_constraint_cache(constraint_cache_path)
        constraint_cache_mutex = Lock()
        print(f"🔐 Constraint cache: mode={constraint_cache_mode}, entries={len(shared_constraint_cache)}")

    # Load checkpoint.
    completed_results = []
    completed_indices = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
                loaded_results = checkpoint.get('results', [])
                resumed_pending = 0

                for r in loaded_results:
                    details = r.get('details', [])
                    has_pending = any(
                        d.get('final_status') == "PENDING_LLM"
                        or d.get('llm_status') == "PENDING"
                        or d.get('decision_complete') is False
                        for d in details
                    )
                    if has_pending:
                        resumed_pending += 1
                        continue
                    completed_results.append(r)

                completed_indices = {
                    r['dialogue_idx']
                    for r in completed_results
                    if isinstance(r, dict) and 'dialogue_idx' in r
                }
                print(f"🔄 Resumed from checkpoint: {len(completed_indices)} dialogues completed")
                if resumed_pending > 0:
                    print(f"  ♻️ Found {resumed_pending} dialogues with PENDING_LLM; re-evaluating them")
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")

    remaining = [(i, d) for i, d in enumerate(all_dialogues) if i not in completed_indices]
    print(f"⏳ Dialogues pending evaluation: {len(remaining)}")

    if not remaining:
        print("All dialogues have already been evaluated.")
        return

    # Initialize LLM.
    llm = None
    llm_fallback = None
    if use_llm:
        print(f"🤖 Initializing LLM: {model_name}")
        llm = get_llm(model_name, max_workers=concurrency)
        if fallback_model_name and fallback_model_name != model_name:
            try:
                print(f"🛡️ Initializing content-filter fallback LLM: {fallback_model_name}")
                llm_fallback = get_llm(fallback_model_name, max_workers=concurrency)
            except Exception as e:
                print(f"⚠️ Failed to initialize fallback LLM; fallback disabled: {e}")

    # Stage 1: rule-based pre-screening (all dialogues, parallelized by concurrency).
    print("\n📋 Stage 1: rule-based pre-screening...")
    pending_llm_results = []
    stage1_workers = max(1, concurrency)
    with ThreadPoolExecutor(max_workers=stage1_workers) as stage1_executor:
        for batch_start in range(0, len(remaining), checkpoint_interval):
            batch = remaining[batch_start:batch_start + checkpoint_interval]

            future_map = {}
            for idx, dialogue_data in batch:
                # Each dialogue gets its own constraint object to avoid cross-thread shared state.
                future = stage1_executor.submit(
                    evaluate_dialogue,
                    dialogue_data=dialogue_data,
                    dialogue_idx=idx,
                    llm=llm,
                    use_llm=use_llm,
                    eval_mode=eval_mode,
                    hard_fact_lock=hard_fact_lock,
                    hard_missing_policy=hard_missing_policy,
                    present_recheck_policy=present_recheck_policy,
                    task_profile=task_profile,
                    hard_number_mode=hard_number_mode,
                    date_evidence_turn_cap=date_evidence_turn_cap,
                    constraint_cache=shared_constraint_cache if constraint_cache_mode != "off" else {},
                    component_role_votes=component_role_votes,
                    component_role_lock_store=component_role_lock_store if component_role_lock_mode != "off" else None,
                    component_role_lock_mode=component_role_lock_mode,
                    component_role_lock_mutex=component_role_lock_mutex,
                )
                future_map[future] = (idx, dialogue_data)

            batch_outputs: List[Tuple[int, DialogueEvalResult, Dict[str, Any]]] = []
            for future in as_completed(future_map):
                idx, dialogue_data = future_map[future]
                result = future.result()
                batch_outputs.append((idx, result, dialogue_data))

            # Sort by dialogue_idx for stable output and readable checkpoints.
            batch_outputs.sort(key=lambda x: x[0])
            for idx, result, dialogue_data in batch_outputs:
                completed_results.append(asdict(result))

                pending_count = sum(1 for d in result.details if d.final_status == "PENDING_LLM")
                if pending_count > 0:
                    pending_llm_results.append((idx, result, dialogue_data))

                status = f"[{idx:4d}] {result.character:20s}: " \
                         f"verifiable={result.verifiable} present={result.present} missing={result.missing} " \
                         f"pending_llm={pending_count} skipped={result.skipped_inference}"
                print(status)

            # Save checkpoint.
            save_report(output_path, completed_results, all_dialogues, task_profile=task_profile)
            if component_role_lock_mode == "read_write":
                _save_component_role_lock(component_role_lock_path, component_role_lock_store)
            if constraint_cache_mode == "read_write":
                _save_constraint_cache(constraint_cache_path, shared_constraint_cache)

    # Stage 2: LLM verification, if enabled.
    if use_llm and llm and pending_llm_results:
        vote_str = (
            f" (vote mode: focused={focused_votes}, global={global_votes}, "
            f"focused_strong_yes>={focused_strong_yes_threshold})"
        )
        print(f"\n🤖 Stage 2: LLM semantic verification ({len(pending_llm_results)} dialogues have pending components){vote_str}...")
        semaphore = asyncio.Semaphore(concurrency)
        stage2_save_interval = max(1, int(checkpoint_interval))
        stage2_completed_dialogues = 0
        stage2_last_saved_dialogues = 0
        stage2_save_lock = asyncio.Lock()

        async def maybe_save_stage2_checkpoint(force: bool = False):
            nonlocal stage2_last_saved_dialogues
            async with stage2_save_lock:
                completed_now = stage2_completed_dialogues
                should_save = (
                    force and completed_now > stage2_last_saved_dialogues
                ) or (
                    not force
                    and completed_now > stage2_last_saved_dialogues
                    and (completed_now - stage2_last_saved_dialogues) >= stage2_save_interval
                )
                if not should_save:
                    return
                save_report(output_path, completed_results, all_dialogues, task_profile=task_profile)
                if component_role_lock_mode == "read_write":
                    _save_component_role_lock(component_role_lock_path, component_role_lock_store)
                if constraint_cache_mode == "read_write":
                    _save_constraint_cache(constraint_cache_path, shared_constraint_cache)
                stage2_last_saved_dialogues = completed_now
                print(
                    f"  💾 Stage 2 checkpoint saved: {completed_now}/{len(pending_llm_results)} dialogues"
                )

        async def verify_one_dialogue(idx, result, dialogue_data):
            nonlocal stage2_completed_dialogues
            dialogue = dialogue_data.get('dialogue', [])
            await run_llm_verification(
                result=result,
                dialogue=dialogue,
                llm=llm,
                llm_fallback=llm_fallback,
                semaphore=semaphore,
                num_votes=num_votes,
                focused_votes=focused_votes,
                global_votes=global_votes,
                focused_strong_yes_threshold=focused_strong_yes_threshold,
                max_vote_attempts=max_vote_attempts,
                retry_until_success=retry_until_success,
                retry_round_sleep=retry_round_sleep,
                retry_backoff_cap=retry_backoff_cap,
                retrieval_min_centers=retrieval_min_centers,
                focused_turn_cap=focused_turn_cap,
                global_turn_cap=global_turn_cap,
                retrieval_route_weights=route_weights,
                vote_temperature=vote_temperature,
            )
            # Update the corresponding entry in results.
            for i, r in enumerate(completed_results):
                if r['dialogue_idx'] == idx:
                    completed_results[i] = asdict(result)
                    break
            pending = sum(1 for d in result.details if d.final_status == "PENDING_LLM")
            print(f"  [{idx:4d}] {result.character}: "
                  f"present={result.present} missing={result.missing} pending={pending}")
            stage2_completed_dialogues += 1
            await maybe_save_stage2_checkpoint()

        async def run_all_llm():
            # Verify all dialogues concurrently; semaphore controls total concurrency.
            dialogue_tasks = [
                verify_one_dialogue(idx, result, dialogue_data)
                for idx, result, dialogue_data in pending_llm_results
            ]
            await asyncio.gather(*dialogue_tasks)
            await maybe_save_stage2_checkpoint(force=True)

        try:
            asyncio.run(run_all_llm())
            save_report(output_path, completed_results, all_dialogues, task_profile=task_profile)
        except KeyboardInterrupt:
            print("\n⚠️ Manual interrupt detected; saving current checkpoint...")
            save_report(output_path, completed_results, all_dialogues, task_profile=task_profile)
            if component_role_lock_mode == "read_write":
                _save_component_role_lock(component_role_lock_path, component_role_lock_store)
            if constraint_cache_mode == "read_write":
                _save_constraint_cache(constraint_cache_path, shared_constraint_cache)
            print(f"  checkpoint saved: {output_path}")
            return

    if component_role_lock_mode == "read_write":
        _save_component_role_lock(component_role_lock_path, component_role_lock_store)
    if constraint_cache_mode == "read_write":
        _save_constraint_cache(constraint_cache_path, shared_constraint_cache)

    # Final statistics.
    print_summary(completed_results)

    if auto_check_mode == "basic":
        auto_check_output = f"{output_path}.auto_check.json"
        auto_report = run_auto_quality_check(
            input_path=input_path,
            output_path=auto_check_output,
            max_dialogues=auto_check_max_dialogues,
            max_samples=auto_check_max_samples,
        )
        metrics = auto_report.get("metrics", {})
        print("\n🔍 Automated quality check:")
        print(f"  false_missing_proxy: {metrics.get('false_missing_proxy', 0):.2%}")
        print(f"  false_present_proxy: {metrics.get('false_present_proxy', 0):.2%}")
        print(f"  repair_precision_proxy: {metrics.get('repair_precision_proxy', 0):.2%}")
        print(f"  repair_redundancy_proxy: {metrics.get('repair_redundancy_proxy', 0):.2%}")
        print(f"  auto_check_report: {auto_check_output}")

    # Consistency comparison.
    if consistency_reference:
        stability_report = compare_with_reference(completed_results, consistency_reference)
        # Attach stability_report to the output report.
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                report_data = json.load(f)
            report_data["stability_report"] = stability_report
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to attach stability_report to output: {e}")


def save_report(output_path: str, results: List[Dict], all_dialogues: List[Dict], task_profile: str = "qa_task_v1"):
    """Save the evaluation report."""
    # Compute global statistics.
    total_components = sum(r.get('total_components', 0) for r in results)
    total_skipped = sum(r.get('skipped_inference', 0) for r in results)
    total_verifiable = sum(r.get('verifiable', 0) for r in results)
    total_present = sum(r.get('present', 0) for r in results)
    total_missing = sum(r.get('missing', 0) for r in results)
    total_hard_missing = sum(r.get('hard_missing_count', 0) for r in results)
    total_hard_missing_initial = sum(r.get('hard_missing_initial_count', 0) for r in results)
    total_hard_missing_override_present = sum(r.get('hard_missing_llm_overridden_present', 0) for r in results)
    total_hard_missing_confirmed_missing = sum(r.get('hard_missing_llm_confirmed_missing', 0) for r in results)
    total_hard_missing_pending = sum(r.get('hard_missing_llm_pending', 0) for r in results)
    total_regression_guard_triggered = sum(r.get('regression_guard_triggered_count', 0) for r in results)
    total_llm_rechecked_present = sum(r.get('llm_rechecked_present_count', 0) for r in results)
    total_revote_trigger = sum(r.get('revote_trigger_count', 0) for r in results)
    total_number_label_filtered = sum(r.get('number_label_filtered_count', 0) for r in results)
    total_llm_no_evidence_reject = sum(r.get('llm_no_evidence_reject_count', 0) for r in results)
    total_llm_object_hard = sum(r.get('llm_object_hard_count', 0) for r in results)
    total_llm_object_soft = sum(r.get('llm_object_soft_count', 0) for r in results)
    total_hard_object_conflict = sum(r.get('hard_object_conflict_count', 0) for r in results)
    total_constraint_validation_reject = sum(r.get('constraint_validation_reject_count', 0) for r in results)
    total_hard_slot_degraded = sum(r.get('hard_slot_degraded_count', 0) for r in results)
    total_date_equivalent_match = sum(r.get('date_equivalent_match_count', 0) for r in results)
    total_retry_rounds = sum(r.get('retry_rounds_total', 0) for r in results)
    total_components_retried = sum(r.get('components_retried_count', 0) for r in results)
    total_max_rounds_single_component = max((r.get('max_rounds_single_component', 0) for r in results), default=0)
    total_network_retry_events = sum(r.get('network_retry_events', 0) for r in results)
    total_component_role_lock_hits = sum(r.get('component_role_lock_hits', 0) for r in results)
    total_component_role_lock_misses = sum(r.get('component_role_lock_misses', 0) for r in results)
    total_constraint_llm_success_tasks = sum(r.get('constraint_llm_success_tasks', 0) for r in results)
    total_constraint_fallback_tasks = sum(r.get('constraint_fallback_tasks', 0) for r in results)
    total_constraint_partial_parse_rejects = sum(r.get('constraint_partial_parse_rejects', 0) for r in results)
    total_constraint_retry_count = sum(r.get('constraint_retry_count', 0) for r in results)
    total_retrieval_route_slot_hits = sum(r.get('retrieval_route_slot_hits', 0) for r in results)
    total_retrieval_route_date_hits = sum(r.get('retrieval_route_date_hits', 0) for r in results)
    total_retrieval_route_semantic_hits = sum(r.get('retrieval_route_semantic_hits', 0) for r in results)
    total_retrieval_overlap_ratio_sum = sum(r.get('retrieval_overlap_ratio_sum', 0.0) for r in results)
    total_retrieval_overlap_ratio_samples = sum(r.get('retrieval_overlap_ratio_samples', 0) for r in results)
    total_focused_to_global_escalations = sum(r.get('focused_to_global_escalations', 0) for r in results)
    total_focused_strong_yes_shortcuts = sum(r.get('focused_strong_yes_shortcuts', 0) for r in results)
    total_global_final_present = sum(r.get('global_final_present', 0) for r in results)
    total_global_final_missing = sum(r.get('global_final_missing', 0) for r in results)
    vote_target_focused = max((r.get('vote_target_valid_votes_focused', 0) for r in results), default=0)
    vote_target_global = max((r.get('vote_target_valid_votes_global', 0) for r in results), default=0)
    vote_majority_focused = max((r.get('vote_majority_threshold_focused', 0) for r in results), default=0)
    vote_majority_global = max((r.get('vote_majority_threshold_global', 0) for r in results), default=0)
    retrieval_overlap_ratio = (
        float(total_retrieval_overlap_ratio_sum) / max(total_retrieval_overlap_ratio_samples, 1)
    )
    repair_queue: List[Dict[str, Any]] = []
    unfinished_components: List[Dict[str, Any]] = []

    for r in results:
        didx = r.get("dialogue_idx", -1)
        character = r.get("character", "")
        for d in r.get("details", []):
            if not isinstance(d, dict):
                continue
            decision_complete = d.get("decision_complete", True)
            repair_ready = d.get("repair_ready", d.get("final_status") == "MISSING")
            if decision_complete is False:
                unfinished_components.append({
                    "dialogue_idx": didx,
                    "character": character,
                    "task_idx": d.get("task_idx", -1),
                    "comp_idx": d.get("comp_idx", -1),
                    "task_type": d.get("task_type", ""),
                    "component_text": d.get("component_text", ""),
                    "decision_reason_code": d.get("decision_reason_code", ""),
                })
                continue
            if d.get("final_status") == "MISSING" and repair_ready:
                repair_queue.append({
                    "dialogue_idx": didx,
                    "character": character,
                    "task_idx": d.get("task_idx", -1),
                    "comp_idx": d.get("comp_idx", -1),
                    "task_type": d.get("task_type", ""),
                    "component_text": d.get("component_text", ""),
                    "query": d.get("query", ""),
                    "decision_complete": True,
                    "repair_ready": True,
                    "decision_confidence": d.get("decision_confidence", ""),
                    "decision_reason_code": d.get("decision_reason_code", ""),
                    "matched_turns": d.get("matched_turns", []) or [],
                    "evidence_turns": d.get("evidence_turns", []) or [],
                    "required_slots": d.get("required_slots", []) or [],
                    "constraint_spec": d.get("constraint_spec", {}) or {},
                    "hard_missing_facts": d.get("hard_missing_facts", []) or [],
                    "hard_initial_status": d.get("hard_initial_status", d.get("hard_status", "")),
                })

    report = {
        "summary": {
            "task_profile": task_profile,
            "total_dialogues": len(results),
            "total_components": total_components,
            "skipped_inference_components": total_skipped,
            "verifiable_components": total_verifiable,
            "present": total_present,
            "missing": total_missing,
            "hard_missing_count": total_hard_missing,
            "hard_missing_initial_count": total_hard_missing_initial,
            "hard_missing_llm_overridden_present": total_hard_missing_override_present,
            "hard_missing_llm_confirmed_missing": total_hard_missing_confirmed_missing,
            "hard_missing_llm_pending": total_hard_missing_pending,
            "regression_guard_triggered_count": total_regression_guard_triggered,
            "llm_rechecked_present_count": total_llm_rechecked_present,
            "revote_trigger_count": total_revote_trigger,
            "number_label_filtered_count": total_number_label_filtered,
            "llm_no_evidence_reject_count": total_llm_no_evidence_reject,
            "llm_object_hard_count": total_llm_object_hard,
            "llm_object_soft_count": total_llm_object_soft,
            "hard_object_conflict_count": total_hard_object_conflict,
            "constraint_validation_reject_count": total_constraint_validation_reject,
            "hard_slot_degraded_count": total_hard_slot_degraded,
            "date_equivalent_match_count": total_date_equivalent_match,
            "retry_rounds_total": total_retry_rounds,
            "components_retried_count": total_components_retried,
            "max_rounds_single_component": total_max_rounds_single_component,
            "network_retry_events": total_network_retry_events,
            "component_role_lock_hits": total_component_role_lock_hits,
            "component_role_lock_misses": total_component_role_lock_misses,
            "constraint_llm_success_tasks": total_constraint_llm_success_tasks,
            "constraint_fallback_tasks": total_constraint_fallback_tasks,
            "constraint_partial_parse_rejects": total_constraint_partial_parse_rejects,
            "constraint_retry_count": total_constraint_retry_count,
            "retrieval_route_slot_hits": total_retrieval_route_slot_hits,
            "retrieval_route_date_hits": total_retrieval_route_date_hits,
            "retrieval_route_semantic_hits": total_retrieval_route_semantic_hits,
            "retrieval_overlap_ratio": retrieval_overlap_ratio,
            "focused_to_global_escalations": total_focused_to_global_escalations,
            "focused_strong_yes_shortcuts": total_focused_strong_yes_shortcuts,
            "global_final_present": total_global_final_present,
            "global_final_missing": total_global_final_missing,
            "vote_target_valid_votes": {
                "focused": vote_target_focused,
                "global": vote_target_global,
            },
            "vote_majority_threshold": {
                "focused": vote_majority_focused,
                "global": vote_majority_global,
            },
            "unfinished_components_count": len(unfinished_components),
            "failed_components_for_rerun": len(unfinished_components),
            "coverage_rate": total_present / max(total_verifiable, 1),
        },
        "results": results,
        "repair_queue": repair_queue,
        "unfinished_components": unfinished_components,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def print_summary(results: List[Dict]):
    """Print final statistics."""
    total_verifiable = sum(r.get('verifiable', 0) for r in results)
    total_present = sum(r.get('present', 0) for r in results)
    total_missing = sum(r.get('missing', 0) for r in results)
    total_failed_for_rerun = sum(r.get('failed_components_for_rerun', 0) for r in results)
    dialogues_with_rerun = sum(1 for r in results if r.get('failed_components_for_rerun', 0) > 0)
    hard_initial = sum(r.get('hard_missing_initial_count', 0) for r in results)
    hard_override = sum(r.get('hard_missing_llm_overridden_present', 0) for r in results)
    hard_confirmed = sum(r.get('hard_missing_llm_confirmed_missing', 0) for r in results)
    hard_pending = sum(r.get('hard_missing_llm_pending', 0) for r in results)
    guard_triggered = sum(r.get('regression_guard_triggered_count', 0) for r in results)
    total_retry_rounds = sum(r.get('retry_rounds_total', 0) for r in results)
    total_components_retried = sum(r.get('components_retried_count', 0) for r in results)
    max_rounds_single_component = max((r.get('max_rounds_single_component', 0) for r in results), default=0)
    total_network_retry_events = sum(r.get('network_retry_events', 0) for r in results)
    total_component_role_lock_hits = sum(r.get('component_role_lock_hits', 0) for r in results)
    total_component_role_lock_misses = sum(r.get('component_role_lock_misses', 0) for r in results)
    total_constraint_llm_success_tasks = sum(r.get('constraint_llm_success_tasks', 0) for r in results)
    total_constraint_fallback_tasks = sum(r.get('constraint_fallback_tasks', 0) for r in results)
    total_constraint_partial_parse_rejects = sum(r.get('constraint_partial_parse_rejects', 0) for r in results)
    total_constraint_retry_count = sum(r.get('constraint_retry_count', 0) for r in results)
    total_retrieval_route_slot_hits = sum(r.get('retrieval_route_slot_hits', 0) for r in results)
    total_retrieval_route_date_hits = sum(r.get('retrieval_route_date_hits', 0) for r in results)
    total_retrieval_route_semantic_hits = sum(r.get('retrieval_route_semantic_hits', 0) for r in results)
    total_retrieval_overlap_ratio_sum = sum(r.get('retrieval_overlap_ratio_sum', 0.0) for r in results)
    total_retrieval_overlap_ratio_samples = sum(r.get('retrieval_overlap_ratio_samples', 0) for r in results)
    total_focused_to_global_escalations = sum(r.get('focused_to_global_escalations', 0) for r in results)
    total_focused_strong_yes_shortcuts = sum(r.get('focused_strong_yes_shortcuts', 0) for r in results)
    total_global_final_present = sum(r.get('global_final_present', 0) for r in results)
    total_global_final_missing = sum(r.get('global_final_missing', 0) for r in results)
    vote_target_focused = max((r.get('vote_target_valid_votes_focused', 0) for r in results), default=0)
    vote_target_global = max((r.get('vote_target_valid_votes_global', 0) for r in results), default=0)
    vote_majority_focused = max((r.get('vote_majority_threshold_focused', 0) for r in results), default=0)
    vote_majority_global = max((r.get('vote_majority_threshold_global', 0) for r in results), default=0)
    retrieval_overlap_ratio = float(total_retrieval_overlap_ratio_sum) / max(total_retrieval_overlap_ratio_samples, 1)
    coverage = total_present / max(total_verifiable, 1)

    print("\n" + "=" * 60)
    print("📊 Evaluation summary")
    print("=" * 60)
    print(f"  Total verifiable: {total_verifiable}")
    print(f"  Missing: {total_missing}")
    print(f"  Coverage: {coverage:.1%}")
    print(
        f"  Hard initial missing: {hard_initial} "
        f"(overridden_to_PRESENT={hard_override}, confirmed_MISSING={hard_confirmed}, pending_rerun={hard_pending})"
    )
    print(f"  Regression guard triggers: {guard_triggered}")
    print(
        f"  LLM retries: rounds={total_retry_rounds}, retried_components={total_components_retried}, "
        f"max_rounds_single_component={max_rounds_single_component}, network_retry_events={total_network_retry_events}"
    )
    print(f"  Component-role lock hits: {total_component_role_lock_hits}, misses: {total_component_role_lock_misses}")
    print(
        "  Constraint extraction: "
        f"llm_success_tasks={total_constraint_llm_success_tasks}, "
        f"fallback_tasks={total_constraint_fallback_tasks}, "
        f"partial_parse_rejects={total_constraint_partial_parse_rejects}, "
        f"retries={total_constraint_retry_count}"
    )
    print(
        "  Retrieval route hits: "
        f"slot={total_retrieval_route_slot_hits}, "
        f"date={total_retrieval_route_date_hits}, "
        f"semantic={total_retrieval_route_semantic_hits}, "
        f"overlap_ratio={retrieval_overlap_ratio:.2%}"
    )
    print(
        "  Hierarchical voting: "
        f"focused_votes={vote_target_focused}(majority={vote_majority_focused}), "
        f"global_votes={vote_target_global}(majority={vote_majority_global}), "
        f"focused->global={total_focused_to_global_escalations}, "
        f"focused_strong_yes={total_focused_strong_yes_shortcuts}, "
        f"global_present={total_global_final_present}, global_missing={total_global_final_missing}"
    )
    print(f"  Components needing rerun: {total_failed_for_rerun}")
    print(f"  Dialogues affected: {dialogues_with_rerun}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Strict QA coverage evaluation with task-type awareness")
    parser.add_argument("--input", type=str,
                        default=str(REPO_ROOT / "dialogue_gen_api/output/final_dialogues_v8_seg150_tok3000.json"),
                        help="Input dialogue dataset path")
    parser.add_argument("--output", type=str,
                        default=str(REPO_ROOT / "dialogue_gen_api/output/final/eval_report_v8_full_budget.json"),
                        help="Output evaluation report path")
    parser.add_argument("--use_llm", action="store_true",
                        help="Enable LLM semantic verification for suspicious components and low-confidence PRESENT checks")
    parser.add_argument("--model", type=str, default="mimo-v2-flash",
                        help="LLM model name (default: mimo-v2-flash)")
    parser.add_argument("--fallback_model", type=str, default="kimi-k2.5",
                        help="Fallback model when content_filter is triggered (default: kimi-k2.5)")
    parser.add_argument("--concurrency", type=int, default=30,
                        help="LLM concurrency (default: 10)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N dialogues for debugging")
    parser.add_argument("--checkpoint_interval", type=int, default=100,
                        help="Save a checkpoint every N dialogues")
    parser.add_argument("--num_votes", type=int, default=3,
                        help="Compatibility parameter: default vote count when focused/global are not set")
    parser.add_argument("--focused_votes", type=int, default=5,
                        help="Target valid votes in the focused stage (odd and >=3; default: 5)")
    parser.add_argument("--global_votes", type=int, default=7,
                        help="Target valid votes in the global stage (odd and >=3; default: 7)")
    parser.add_argument("--focused_strong_yes_threshold", type=int, default=4,
                        help="Focused-stage strong-YES threshold for direct PRESENT (default: 4)")
    parser.add_argument("--component_role_votes", type=int, default=5,
                        help="Valid votes required for component_role classification (default: 5)")
    parser.add_argument("--component_role_lock_path", type=str, default="../output/final/component_role_lock.json",
                        help="Path to the task-level component_role lock file")
    parser.add_argument("--component_role_lock_mode", type=str, default="read_write",
                        choices=["read_write", "read_only", "off"],
                        help="component_role lock mode: read_write/read_only/off")
    parser.add_argument("--max_vote_attempts", type=int, default=20,
                        help="Max LLM attempts per component per stage per round (default: 20)")
    parser.add_argument("--retry_until_success", action=argparse.BooleanOptionalAction, default=True,
                        help="Retry rounds until convergence when valid votes are insufficient (default: True)")
    parser.add_argument("--retry_round_sleep", type=float, default=1.0,
                        help="Base sleep seconds between retry rounds (default: 1.0)")
    parser.add_argument("--retry_backoff_cap", type=float, default=30.0,
                        help="Backoff cap in seconds for consecutive network failures (default: 30.0)")
    parser.add_argument("--eval_mode", type=str, default="strict_v2",
                        help="Evaluation mode (default: strict_v2)")
    parser.add_argument("--hard_fact_lock", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable the hard-constraint signal channel (default: True)")
    parser.add_argument("--hard_missing_policy", type=str, default="llm_gate",
                        choices=["llm_gate", "direct_lock"],
                        help="Hard-missing policy: llm_gate (default; gated by LLM) or direct_lock (legacy behavior)")
    parser.add_argument("--present_recheck_policy", type=str, default="low_confidence_only",
                        choices=["low_confidence_only", "non_high", "all", "none"],
                        help="Policy for sending rule-based PRESENT cases to LLM review")
    parser.add_argument("--task_profile", type=str, default="qa_task_v1",
                        help="Task-type profile configuration (default: qa_task_v1)")
    parser.add_argument("--hard_number_mode", type=str, default="contextual_strict",
                        help="Numeric hard-constraint mode (default: contextual_strict)")
    parser.add_argument("--date_evidence_turn_cap", type=int, default=16,
                        help="Maximum date-evidence turns (default: 16)")
    parser.add_argument("--retrieval_min_centers", type=int, default=5,
                        help="Minimum centers forced during retrieval (default: 5)")
    parser.add_argument("--focused_turn_cap", type=int, default=42,
                        help="Maximum retrieved turns in the focused stage (default: 42)")
    parser.add_argument("--global_turn_cap", type=int, default=72,
                        help="Maximum retrieved turns in the global stage (default: 72)")
    parser.add_argument("--retrieval_route_weights", type=str, default="slot=3,date=2,semantic=1",
                        help="Multi-route retrieval fusion weights, e.g. slot=3,date=2,semantic=1")
    parser.add_argument("--vote_temperature", type=float, default=0.0,
                        help="LLM voting temperature (default: 0.0 greedy decoding)")
    parser.add_argument("--constraint_cache_path", type=str, default="../output/final/constraint_cache.json",
                        help="Disk cache path for constraint_spec")
    parser.add_argument("--constraint_cache_mode", type=str, default="read_write",
                        choices=["read_write", "read_only", "off"],
                        help="constraint cache mode: read_write/read_only/off")
    parser.add_argument("--consistency_reference", type=str, default=None,
                        help="Previous evaluation report path for consistency flip detection")
    parser.add_argument("--auto_check_mode", type=str, default="off", choices=["off", "basic"],
                        help="Automated quality-check mode (default: off)")
    parser.add_argument("--auto_check_max_dialogues", type=int, default=50,
                        help="Maximum dialogues used by auto-check (default: 50)")
    parser.add_argument("--auto_check_max_samples", type=int, default=120,
                        help="Maximum sampled components used by auto-check (default: 120)")
    parser.add_argument("--auto_check_only", action="store_true",
                        help="Run only the automated quality check, not the full evaluation")
    args = parser.parse_args()

    if args.auto_check_only:
        auto_output = f"{args.output}.auto_check.json"
        auto_report = run_auto_quality_check(
            input_path=args.input,
            output_path=auto_output,
            max_dialogues=args.auto_check_max_dialogues,
            max_samples=args.auto_check_max_samples,
        )
        metrics = auto_report.get("metrics", {})
        print("\n🔍 Automated quality check (only):")
        print(f"  false_missing_proxy: {metrics.get('false_missing_proxy', 0):.2%}")
        print(f"  false_present_proxy: {metrics.get('false_present_proxy', 0):.2%}")
        print(f"  repair_precision_proxy: {metrics.get('repair_precision_proxy', 0):.2%}")
        print(f"  repair_redundancy_proxy: {metrics.get('repair_redundancy_proxy', 0):.2%}")
        print(f"  auto_check_report: {auto_output}")
        return

    run_evaluation(
        input_path=args.input,
        output_path=args.output,
        use_llm=args.use_llm,
        model_name=args.model,
        concurrency=args.concurrency,
        limit=args.limit,
        checkpoint_interval=args.checkpoint_interval,
        num_votes=args.num_votes,
        focused_votes=args.focused_votes,
        global_votes=args.global_votes,
        focused_strong_yes_threshold=args.focused_strong_yes_threshold,
        eval_mode=args.eval_mode,
        hard_fact_lock=args.hard_fact_lock,
        hard_missing_policy=args.hard_missing_policy,
        present_recheck_policy=args.present_recheck_policy,
        task_profile=args.task_profile,
        hard_number_mode=args.hard_number_mode,
        date_evidence_turn_cap=args.date_evidence_turn_cap,
        auto_check_mode=args.auto_check_mode,
        auto_check_max_dialogues=args.auto_check_max_dialogues,
        auto_check_max_samples=args.auto_check_max_samples,
        max_vote_attempts=args.max_vote_attempts,
        fallback_model_name=args.fallback_model,
        retry_until_success=args.retry_until_success,
        retry_round_sleep=args.retry_round_sleep,
        retry_backoff_cap=args.retry_backoff_cap,
        component_role_votes=args.component_role_votes,
        component_role_lock_path=args.component_role_lock_path,
        component_role_lock_mode=args.component_role_lock_mode,
        retrieval_min_centers=args.retrieval_min_centers,
        focused_turn_cap=args.focused_turn_cap,
        global_turn_cap=args.global_turn_cap,
        retrieval_route_weights=args.retrieval_route_weights,
        vote_temperature=args.vote_temperature,
        constraint_cache_path=args.constraint_cache_path,
        constraint_cache_mode=args.constraint_cache_mode,
        consistency_reference=args.consistency_reference,
    )


if __name__ == "__main__":
    main()
