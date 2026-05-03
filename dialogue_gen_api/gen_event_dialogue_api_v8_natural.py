"""
Event-Grounded Dialogue Generation Pipeline V8 (Natural Dialogue Mode)

Natural dialogue generator that extends the V7 hierarchical segment
architecture with more natural scene planning, speaking style control, and
online validity checks.

Main upgrades relative to DIALOGUE_QUALITY_IMPROVEMENT_PLAN.md:
  A. Character voice system: inject speech_profile into the system prompt.
  B. Scene diversity: constrain scene planning with scene_categories.json.
  C. Expanded filler topic pool: filler_topics_v2.json.
  D. Context passing: summary plus the last three raw turns.
  E. Dialogue dynamics: randomly inject dialogue_dynamics.json events.
  F. Anchor insertion: filter reasoning-only components and keep source facts.
  G. Multi-temperature strategy by segment type.
"""

import os
import sys
import json
import re
import time
import argparse
import random
import uuid
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv
    env_path = SCRIPT_DIR / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()
except ImportError:
    pass

try:
    import httpx
except ImportError:
    print("Please install httpx: pip install httpx")
    sys.exit(1)

from natural_dialogue.speech_profile import generate_speech_profile
from natural_dialogue.director_notes_v2 import (
    generate_director_notes_v2,
    should_verify_component,
)

TRANSIENT_HTTPX_ERRORS = tuple(
    err for err in (
        getattr(httpx, "ReadTimeout", None),
        getattr(httpx, "WriteTimeout", None),
        getattr(httpx, "ConnectTimeout", None),
        getattr(httpx, "PoolTimeout", None),
        getattr(httpx, "ReadError", None),
        getattr(httpx, "WriteError", None),
        getattr(httpx, "ConnectError", None),
        getattr(httpx, "RemoteProtocolError", None),
    ) if err is not None
)

# ============================================================
# Configuration
# ============================================================
MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"
MIMO_MODEL = "mimo-v2-flash"
KIMI_BASE_URL = "https://api.moonshot.cn/v1"
KIMI_MODEL = "kimi-k2.5"
KIMI_MODEL_ALIASES = {
    "kimi-k2.5": "kimi-k2-turbo-preview",
}

DEFAULT_SEGMENT_TOKEN_TARGET = 3000
DEFAULT_TOTAL_SEGMENTS = 40
INFO_DISTRIBUTION_START = 0.05
INFO_DISTRIBUTION_END = 0.95

# Strategy G: set temperature by segment type.
TEMPERATURE_BY_SEGMENT = {
    "opening": 0.80,
    "closing": 0.80,
    "filler": 0.90,
    "info_anchor": 0.75,
    "transition": 0.85,
}


# ============================================================
# Data Classes
# ============================================================
class SegmentType(Enum):
    OPENING = "opening"
    FILLER = "filler"
    INFO_ANCHOR = "info_anchor"
    TRANSITION = "transition"
    CLOSING = "closing"


@dataclass
class InformationAnchor:
    task_id: str
    anchor_type: str  # "setup" or "fragment_N"
    content: str
    task_data: Dict
    target_segment: int = -1
    target_position: float = 0.0


@dataclass
class SegmentOutline:
    segment_id: int
    segment_type: SegmentType
    topic: str
    anchors: List[InformationAnchor] = field(default_factory=list)
    director_notes: str = ""
    dynamic_event: Optional[Dict[str, Any]] = None


@dataclass
class GeneratedSegment:
    segment_id: int
    dialogue_turns: List[Dict]
    summary: str
    token_count: int
    provider_used: str = "mimo"
    fallback_used: bool = False
    fallback_attempts: int = 0


@dataclass
class RetryDecision:
    category: str
    max_attempts: int
    status_code: Optional[int] = None
    error_excerpt: str = ""
    repeat_key: str = ""
    provider_error_code: Optional[str] = None
    is_moderation_block: bool = False


@dataclass
class LengthControlConfig:
    target_total_tokens: int
    target_total_tolerance: int
    fixed_turns_per_segment: int
    max_length_rerolls: int = 0


@dataclass
class SegmentLengthTarget:
    center: int
    low: int
    high: int
    min_words: int
    max_words: int


@dataclass
class CharacterProcessOutcome:
    result: Optional[Dict]
    failure_record: Optional[Dict] = None


# ============================================================
# API Client (reuses V7 retry/fallback logic)
# ============================================================
class DualRoleGenerator:
    def __init__(
        self,
        api_key,
        model=MIMO_MODEL,
        transient_retries: int = 8,
        bad_request_retries: int = 2,
        retry_max_wait: float = 30.0,
        kimi_api_key: Optional[str] = None,
        kimi_model: str = KIMI_MODEL,
        kimi_fallback_attempts: int = 2,
        enable_kimi_fallback: bool = True,
    ):
        self.api_key = api_key
        self.model = model
        self.transient_retries = transient_retries
        self.bad_request_retries = bad_request_retries
        self.retry_max_wait = retry_max_wait
        self.kimi_api_key = kimi_api_key or os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY")
        self.kimi_model = kimi_model
        self.kimi_request_model = KIMI_MODEL_ALIASES.get(kimi_model, kimi_model)
        self.kimi_fallback_attempts = max(1, kimi_fallback_attempts)
        self.enable_kimi_fallback = enable_kimi_fallback
        self.last_error_by_phase: Dict[str, Dict[str, Any]] = {}
        self.last_completion_by_phase: Dict[str, Dict[str, Any]] = {}
        self.client_settings = {
            "timeout": 180.0,
            "limits": httpx.Limits(max_keepalive_connections=20, max_connections=50),
        }

    def _record_success_meta(
        self,
        phase: str,
        provider: str,
        fallback_used: bool = False,
        fallback_attempts: int = 0,
    ) -> None:
        self.last_completion_by_phase[phase] = {
            "provider": provider,
            "fallback_used": fallback_used,
            "fallback_attempts": fallback_attempts,
        }
        self.last_error_by_phase.pop(phase, None)

    def _record_error_meta(
        self,
        phase: str,
        provider: str,
        decision: Optional[RetryDecision] = None,
        allow_retry_until_success: bool = True,
        **extra: Any,
    ) -> Dict[str, Any]:
        meta = {
            "provider": provider,
            "category": decision.category if decision else extra.get("category"),
            "status_code": decision.status_code if decision else extra.get("status_code"),
            "error_excerpt": decision.error_excerpt if decision else extra.get("error_excerpt", ""),
            "provider_error_code": decision.provider_error_code if decision else extra.get("provider_error_code"),
            "is_moderation_block": decision.is_moderation_block if decision else extra.get("is_moderation_block", False),
            "is_non_retryable_for_provider": extra.get(
                "is_non_retryable_for_provider",
                is_non_retryable_provider_category(decision.category) if decision else False,
            ),
            "allow_retry_until_success": allow_retry_until_success,
            "fallback_attempted": extra.get("fallback_attempted", False),
            "fallback_provider": extra.get("fallback_provider"),
            "fallback_attempts": extra.get("fallback_attempts", 0),
            "fallback_exhausted": extra.get("fallback_exhausted", False),
            "fallback_final_error": extra.get("fallback_final_error"),
            "source_provider": extra.get("source_provider"),
            "source_category": extra.get("source_category"),
            "source_error_excerpt": extra.get("source_error_excerpt"),
            "mimo_non_retryable": extra.get("mimo_non_retryable", False),
        }
        self.last_error_by_phase[phase] = meta
        self.last_completion_by_phase.pop(phase, None)
        return meta

    def _build_mimo_payload(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "stream": False,
            # Random seed changes sampling only; it is not a cache toggle and will not bypass moderation blocks.
            "seed": random.randint(0, 2**31 - 1),
        }

    def _build_kimi_payload(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        return {
            "model": self.kimi_request_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    def _normalize_content(self, content: str, json_mode: bool) -> str:
        if json_mode and isinstance(content, str):
            return content.replace("```json", "").replace("```", "").strip()
        return content

    def _extract_content(self, response: httpx.Response) -> str:
        payload = response.json()
        return payload["choices"][0]["message"]["content"]

    def _mimo_chat_completion_once(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = self._build_mimo_payload(messages, temperature, max_tokens)
        with httpx.Client(**self.client_settings) as client:
            resp = client.post(f"{MIMO_BASE_URL}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            return self._extract_content(resp)

    def _kimi_chat_completion_once(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.kimi_api_key}",
            "Content-Type": "application/json",
        }
        payload = self._build_kimi_payload(messages, temperature, max_tokens)
        with httpx.Client(**self.client_settings) as client:
            resp = client.post(f"{KIMI_BASE_URL}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            return self._extract_content(resp)

    def _attempt_kimi_fallback(
        self,
        phase: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        source_meta: Dict[str, Any],
    ) -> Optional[str]:
        if not self.enable_kimi_fallback:
            return None

        if not self.kimi_api_key:
            print(f"    [KIMI Unavailable][{phase}] Missing KIMI_API_KEY / MOONSHOT_API_KEY")
            self._record_error_meta(
                phase,
                provider="kimi",
                allow_retry_until_success=False,
                category="kimi_unavailable",
                error_excerpt="KIMI unavailable: missing KIMI_API_KEY / MOONSHOT_API_KEY",
                fallback_attempted=True,
                fallback_provider="kimi",
                fallback_attempts=0,
                fallback_exhausted=True,
                fallback_final_error="KIMI unavailable",
                source_provider=source_meta.get("provider"),
                source_category=source_meta.get("category"),
                source_error_excerpt=source_meta.get("error_excerpt"),
                mimo_non_retryable=source_meta.get("provider") == "mimo"
                and source_meta.get("is_non_retryable_for_provider", False),
                is_non_retryable_for_provider=True,
            )
            return None

        reason = source_meta.get("error_excerpt") or source_meta.get("category") or "non-retryable provider failure"
        print(f"    [Provider Switch][{phase}] MIMO -> KIMI reason={reason[:120]}")

        for attempt in range(1, self.kimi_fallback_attempts + 1):
            try:
                content = self._kimi_chat_completion_once(messages, temperature, max_tokens)
                content = self._normalize_content(content, json_mode)
                self._record_success_meta(
                    phase,
                    provider="kimi",
                    fallback_used=True,
                    fallback_attempts=attempt,
                )
                return content
            except Exception as exc:
                decision = classify_api_exception(
                    exc,
                    transient_retries=self.transient_retries,
                    bad_request_retries=self.bad_request_retries,
                )
                status_text = f" status={decision.status_code}" if decision.status_code is not None else ""
                excerpt = decision.error_excerpt[:120] if decision.error_excerpt else str(exc)[:120]
                if attempt < self.kimi_fallback_attempts:
                    print(
                        f"    [KIMI Retry {attempt}/{self.kimi_fallback_attempts}][{phase}][{decision.category}]"
                        f"{status_text} {excerpt}"
                    )
                else:
                    print(
                        f"    [KIMI Fail {attempt}/{self.kimi_fallback_attempts}][{phase}][{decision.category}]"
                        f"{status_text} {excerpt}"
                    )
                self._record_error_meta(
                    phase,
                    provider="kimi",
                    decision=decision,
                    allow_retry_until_success=False,
                    fallback_attempted=True,
                    fallback_provider="kimi",
                    fallback_attempts=attempt,
                    fallback_exhausted=attempt >= self.kimi_fallback_attempts,
                    fallback_final_error=excerpt,
                    source_provider=source_meta.get("provider"),
                    source_category=source_meta.get("category"),
                    source_error_excerpt=source_meta.get("error_excerpt"),
                    mimo_non_retryable=source_meta.get("provider") == "mimo"
                    and source_meta.get("is_non_retryable_for_provider", False),
                )
        return None

    def chat_completion(
        self,
        messages,
        temperature=0.7,
        max_tokens=4096,
        json_mode=False,
        phase: str = "generic",
        allow_kimi_fallback: bool = False,
    ):
        if not self.api_key:
            return None
        last_bad_request_key = ""
        max_possible_attempts = max(self.transient_retries, self.bad_request_retries + 1, 2)

        for attempt in range(1, max_possible_attempts + 1):
            try:
                content = self._mimo_chat_completion_once(messages, temperature, max_tokens)
                content = self._normalize_content(content, json_mode)
                self._record_success_meta(phase, provider="mimo", fallback_used=False, fallback_attempts=0)
                return content
            except Exception as e:
                decision = classify_api_exception(
                    e,
                    transient_retries=self.transient_retries,
                    bad_request_retries=self.bad_request_retries,
                )

                status_text = f" status={decision.status_code}" if decision.status_code is not None else ""
                excerpt = decision.error_excerpt[:120] if decision.error_excerpt else str(e)[:120]
                repeated_bad_request = (
                    decision.category == "bad_request"
                    and decision.repeat_key
                    and decision.repeat_key == last_bad_request_key
                )
                is_non_retryable = is_non_retryable_provider_category(decision.category)
                error_meta = self._record_error_meta(
                    phase,
                    provider="mimo",
                    decision=decision,
                    allow_retry_until_success=not (
                        phase == "segment_generation" and is_non_retryable
                    ),
                    is_non_retryable_for_provider=is_non_retryable,
                )

                if decision.is_moderation_block:
                    print(
                        f"    [Moderation Block][{phase}] provider=mimo"
                        f"{status_text} code={decision.provider_error_code or 'unknown'} {excerpt}"
                    )

                if repeated_bad_request:
                    print(
                        f"    [API Abort][{phase}][{decision.category}]"
                        f"{status_text} Repeated response: {excerpt}"
                    )
                    if phase == "segment_generation" and allow_kimi_fallback and is_non_retryable:
                        return self._attempt_kimi_fallback(
                            phase=phase,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            json_mode=json_mode,
                            source_meta=error_meta,
                        )
                    return None

                if attempt >= decision.max_attempts:
                    print(
                        f"    [API Fail {attempt}/{decision.max_attempts}][{phase}][{decision.category}]"
                        f"{status_text} {excerpt}"
                    )
                    if phase == "segment_generation" and allow_kimi_fallback and is_non_retryable:
                        return self._attempt_kimi_fallback(
                            phase=phase,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            json_mode=json_mode,
                            source_meta=error_meta,
                        )
                    return None

                wait_time = min((2 ** (attempt - 1)) + random.uniform(0, 1), self.retry_max_wait)
                print(
                    f"    [API Retry {attempt}/{decision.max_attempts}][{phase}][{decision.category}]"
                    f"{status_text} {excerpt} Waiting {wait_time:.1f}s"
                )
                last_bad_request_key = decision.repeat_key if decision.category == "bad_request" else ""
                time.sleep(wait_time)
        return None


# ============================================================
# Helpers
# ============================================================
def robust_json_parse(text: str):
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    except json.JSONDecodeError:
        pass
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except json.JSONDecodeError:
        pass
    return None


def estimate_tokens(text_or_list):
    if text_or_list is None:
        return 0
    if isinstance(text_or_list, list):
        text = json.dumps(text_or_list, ensure_ascii=False)
    else:
        text = str(text_or_list)
    return len(text) // 4


def clean_dialogue_text(text: str) -> str:
    if not text:
        return ""
    for p in [r"\([^)]*\)", r"\[[^\]]*\]", r"\{[^}]*\}", r"（[^）]*）", r"【[^】]*】", r"\*[^*]*\*"]:
        text = re.sub(p, "", text, flags=re.DOTALL)
    text = re.sub(r"\n\s*\n", "\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_keywords_from_components(components: List[str]) -> List[str]:
    return [c.strip() for c in components if len(c.strip()) >= 4]


def check_leakage(text: str, future_keywords: List[str]) -> Tuple[bool, str]:
    text_lower = text.lower()
    for kw in future_keywords:
        if kw.lower() in text_lower:
            return True, kw
    return False, ""


def detect_degeneration(turns: List[Dict], repeat_threshold: int = 4) -> bool:
    """Detect degenerate LLM output, such as repeated words or garbled text."""
    for turn in turns:
        content = turn.get("content", "")
        words = content.lower().split()
        if len(words) < 6:
            continue
        streak = 1
        for i in range(1, len(words)):
            if words[i] == words[i - 1]:
                streak += 1
                if streak >= repeat_threshold:
                    return True
            else:
                streak = 1
    return False


def _normalize_excerpt(text: str, limit: int = 160) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    return cleaned[:limit]


def extract_provider_error_details(text: str) -> Dict[str, Any]:
    parsed = robust_json_parse(text) if text else None
    error_payload = parsed.get("error", {}) if isinstance(parsed, dict) else {}
    code = error_payload.get("code") if isinstance(error_payload, dict) else None
    message = error_payload.get("message", "") if isinstance(error_payload, dict) else ""
    normalized_message = _normalize_excerpt(message)
    full_text = _normalize_excerpt(text)
    moderation_signal = f"{normalized_message} {full_text}".lower()
    is_moderation_block = str(code) == "421" or "moderation block" in moderation_signal

    return {
        "provider_error_code": str(code) if code is not None else None,
        "error_message": normalized_message or full_text,
        "is_moderation_block": is_moderation_block,
    }


def is_non_retryable_provider_category(category: str) -> bool:
    return category in {"bad_request", "fatal_http", "other_http", "unexpected", "kimi_unavailable"}


def is_mimo_non_retryable_segment_error(error_meta: Optional[Dict[str, Any]]) -> bool:
    return bool(
        error_meta
        and error_meta.get("provider") == "mimo"
        and error_meta.get("is_non_retryable_for_provider")
    )


def classify_api_exception(
    exc: Exception,
    transient_retries: int = 8,
    bad_request_retries: int = 2,
) -> RetryDecision:
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code if exc.response is not None else None
        provider_details = extract_provider_error_details(exc.response.text if exc.response is not None else "")
        body_excerpt = _normalize_excerpt(exc.response.text if exc.response is not None else "")
        error_excerpt = body_excerpt or provider_details["error_message"] or _normalize_excerpt(str(exc))
        repeat_key = f"{status_code}:{error_excerpt}" if status_code is not None else error_excerpt

        if status_code == 400:
            return RetryDecision(
                "bad_request",
                bad_request_retries + 1,
                status_code,
                error_excerpt,
                repeat_key,
                provider_details["provider_error_code"],
                provider_details["is_moderation_block"],
            )
        if status_code in {401, 403, 404, 422}:
            return RetryDecision(
                "fatal_http",
                1,
                status_code,
                error_excerpt,
                repeat_key,
                provider_details["provider_error_code"],
                provider_details["is_moderation_block"],
            )
        if status_code == 429 or (status_code is not None and 500 <= status_code < 600):
            return RetryDecision(
                "transient_http",
                transient_retries,
                status_code,
                error_excerpt,
                repeat_key,
                provider_details["provider_error_code"],
                provider_details["is_moderation_block"],
            )
        if status_code is not None and 400 <= status_code < 500:
            return RetryDecision(
                "other_http",
                2,
                status_code,
                error_excerpt,
                repeat_key,
                provider_details["provider_error_code"],
                provider_details["is_moderation_block"],
            )

    message = _normalize_excerpt(str(exc))
    lower_message = message.lower()

    if isinstance(exc, TRANSIENT_HTTPX_ERRORS):
        return RetryDecision("transient_network", transient_retries, error_excerpt=message)
    if "server disconnected without sending a response" in lower_message:
        return RetryDecision("transient_network", transient_retries, error_excerpt=message)
    if "ssl" in lower_message and "eof" in lower_message:
        return RetryDecision("transient_network", transient_retries, error_excerpt=message)
    if "timed out" in lower_message:
        return RetryDecision("transient_network", transient_retries, error_excerpt=message)

    return RetryDecision("unexpected", 2, error_excerpt=message)


def default_scene_config(scene_category: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    example_roles = (scene_category or {}).get("example_roles", []) if isinstance(scene_category, dict) else []
    return {
        "interlocutor_role": random.choice(example_roles or ["Friend"]),
        "setting": "A quiet, comfortable space",
        "interlocutor_goal": "Genuine curiosity and connection",
        "opening_style": "Casual",
    }


def compute_segment_length_target(
    config: LengthControlConfig,
    tokens_so_far: int,
    current_segment_id: int,
    total_segments: int,
    segment_type: SegmentType,
) -> SegmentLengthTarget:
    remaining_segments = max(total_segments - current_segment_id, 1)
    remaining_target = max(config.target_total_tokens - tokens_so_far, 0)
    center = round(remaining_target / remaining_segments)
    center = max(900, min(1100, center))

    if current_segment_id <= 99:
        band = 160
    elif current_segment_id <= 129:
        band = 120
    else:
        band = 80

    if segment_type in (SegmentType.OPENING, SegmentType.CLOSING):
        band += 40

    word_center = max(70, min(95, round(center * 0.08)))
    return SegmentLengthTarget(
        center=center,
        low=center - band,
        high=center + band,
        min_words=max(40, word_center - 10),
        max_words=word_center + 10,
    )


def segment_within_length_target(segment: GeneratedSegment, target: SegmentLengthTarget) -> bool:
    return target.low <= segment.token_count <= target.high


def segment_needs_length_retry(segment: GeneratedSegment, target: SegmentLengthTarget) -> bool:
    return segment.token_count < target.low


def build_length_guidance(target: SegmentLengthTarget) -> str:
    return (
        f"LENGTH TARGET:\n"
        f"- Aim for about {target.center} estimated tokens in this segment.\n"
        f"- Each turn should usually be about {target.min_words}-{target.max_words} words."
    )


def build_length_retry_instruction(segment: GeneratedSegment, target: SegmentLengthTarget) -> str:
    if segment.token_count < target.low:
        return (
            f"The previous attempt was too short at about {segment.token_count} estimated tokens. "
            f"Keep the same facts and tone, but add natural detail so the segment lands near {target.center} estimated tokens. "
            f"Each turn should usually be about {target.min_words}-{target.max_words} words."
        )
    return (
        f"The previous attempt was too long at about {segment.token_count} estimated tokens. "
        f"Keep the same facts and tone, but compress wording so the segment lands near {target.center} estimated tokens. "
        f"Each turn should usually be about {target.min_words}-{target.max_words} words."
    )


# ============================================================
# Config Loaders (strategies B, C, and E)
# ============================================================
_MODULE_DIR = Path(__file__).parent / "natural_dialogue"


def load_scene_categories(config_path: Optional[str] = None) -> List[Dict]:
    path = Path(config_path) if config_path else _MODULE_DIR / "scene_categories.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("categories", [])


def load_filler_topics(config_path: Optional[str] = None) -> Dict[str, List[str]]:
    path = Path(config_path) if config_path else _MODULE_DIR / "filler_topics_v2.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dynamics_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path else _MODULE_DIR / "dialogue_dynamics.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 1. Data Loading (reuses V7 format)
# ============================================================
def load_data_sources(stories_path: str, qa_path: str):
    print(f"Loading Events from {stories_path}...")
    with open(stories_path, "r", encoding="utf-8") as f:
        stories_data = json.load(f)

    events_map = {}
    char_map = {}
    for char_entry in stories_data:
        c_name = char_entry.get("character_name")
        char_map[c_name] = char_entry
        if "chronology" in char_entry:
            for yr in char_entry["chronology"]:
                year_val = yr.get("year")
                for ev in yr.get("events", []):
                    ev_id = ev.get("event_id")
                    ev["year_context"] = year_val
                    ev["character_name"] = c_name
                    events_map[ev_id] = ev

    print(f"Loading QA Script from {qa_path}...")
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    qa_missions = {}
    for entry in qa_data:
        c_name = entry.get("character_name")
        if c_name not in char_map:
            continue
        tasks = entry.get("validated_tasks", [])
        if tasks:
            if c_name not in qa_missions:
                qa_missions[c_name] = []
            qa_missions[c_name].extend(tasks)
    qa_missions = {k: v for k, v in qa_missions.items() if v}

    print(f"Loaded {len(events_map)} events and QA tasks for {len(qa_missions)} characters.")
    return events_map, qa_missions, char_map


# ============================================================
# 2. Segment Planner (strategy F: filter reasoning-only components)
# ============================================================
def create_information_anchors(tasks: List[Dict]) -> List[InformationAnchor]:
    """Decompose QA tasks into information anchors and filter reasoning-only components."""
    anchors = []
    for task_idx, task in enumerate(tasks):
        task_id = f"T{task_idx}_{task.get('task_type', 'unknown')[:10]}"
        task_type = task.get("task_type", "")

        anchors.append(InformationAnchor(
            task_id=task_id,
            anchor_type="setup",
            content=task.get("query", ""),
            task_data=task,
        ))

        components = task.get("answer_components", [])
        if not components:
            components = [task.get("gold_answer", "Details")]

        for comp_idx, comp in enumerate(components):
            if not should_verify_component(task_type, comp):
                continue
            anchors.append(InformationAnchor(
                task_id=task_id,
                anchor_type=f"fragment_{comp_idx}",
                content=comp,
                task_data=task,
            ))
    return anchors


def distribute_anchors_to_segments(
    anchors: List[InformationAnchor],
    total_segments: int,
    start_pct: float = INFO_DISTRIBUTION_START,
    end_pct: float = INFO_DISTRIBUTION_END,
) -> List[InformationAnchor]:
    start_segment = int(total_segments * start_pct)
    end_segment = int(total_segments * end_pct)
    available_range = end_segment - start_segment

    task_groups: Dict[str, List[InformationAnchor]] = {}
    for anchor in anchors:
        task_groups.setdefault(anchor.task_id, []).append(anchor)

    all_fragments = []
    all_setups = []
    for tid, task_anchors in task_groups.items():
        setup = next((a for a in task_anchors if a.anchor_type == "setup"), None)
        fragments = [a for a in task_anchors if a.anchor_type != "setup"]
        if setup:
            all_setups.append((tid, setup))
        all_fragments.extend([(tid, f) for f in fragments])

    random.shuffle(all_fragments)
    total_items = len(all_fragments)

    for idx, (tid, frag) in enumerate(all_fragments):
        position_ratio = idx / max(total_items - 1, 1)
        target_segment = start_segment + int(position_ratio * available_range)
        target_segment = min(target_segment, end_segment)
        frag.target_segment = target_segment
        frag.target_position = random.uniform(0.3, 0.7)

    for tid, setup in all_setups:
        task_frags = [f for t, f in all_fragments if t == tid]
        if task_frags:
            earliest = min(f.target_segment for f in task_frags)
            setup.target_segment = max(start_segment, earliest - random.randint(1, 3))
        else:
            setup.target_segment = random.randint(start_segment, end_segment)
        setup.target_position = random.uniform(0.3, 0.7)

    return anchors


# ============================================================
# 3. Filler Topic Sampler (strategy C)
# ============================================================
class FillerTopicSampler:
    """Sample diverse filler topics by type without repeating adjacent types."""

    def __init__(self, topics_by_type: Dict[str, List[str]]):
        self._pools = {t: list(items) for t, items in topics_by_type.items() if items}
        for pool in self._pools.values():
            random.shuffle(pool)
        self._type_keys = list(self._pools.keys())
        self._last_type: Optional[str] = None
        self._cursors: Dict[str, int] = {t: 0 for t in self._type_keys}

    def sample(self, char_name: str) -> Tuple[str, str]:
        """Return (topic_text, topic_type)."""
        candidates = [t for t in self._type_keys if t != self._last_type]
        if not candidates:
            candidates = self._type_keys
        chosen_type = random.choice(candidates)
        self._last_type = chosen_type

        pool = self._pools[chosen_type]
        cursor = self._cursors[chosen_type]
        if cursor >= len(pool):
            random.shuffle(pool)
            cursor = 0
        topic = pool[cursor].replace("{char_name}", char_name)
        self._cursors[chosen_type] = cursor + 1
        return topic, chosen_type


# ============================================================
# 4. Dynamics Injector (strategy E)
# ============================================================
class DynamicsInjector:
    def __init__(self, config: Dict[str, Any], total_segments: int):
        self._dynamics = config.get("dynamics", [])
        self._max_ratio = config.get("max_dynamics_per_dialogue_ratio", 0.35)
        self._budget = int(total_segments * self._max_ratio)
        self._used = 0

    def maybe_inject(self) -> Optional[Dict[str, Any]]:
        if self._used >= self._budget:
            return None
        for dyn in random.sample(self._dynamics, len(self._dynamics)):
            if random.random() < dyn.get("trigger_rate", 0.0):
                self._used += 1
                return dyn
        return None


# ============================================================
# 5. Scene Category Selector (strategy B)
# ============================================================
class SceneCategorySelector:
    def __init__(self, categories: List[Dict]):
        self._categories = categories
        self._counts: Dict[str, int] = {c["id"]: 0 for c in categories}
        self._total = 0

    def select(self) -> Dict:
        eligible = []
        for cat in self._categories:
            current_ratio = self._counts[cat["id"]] / max(self._total, 1)
            if current_ratio < cat.get("max_ratio", 1.0):
                eligible.append(cat)
        if not eligible:
            eligible = self._categories
        chosen = random.choice(eligible)
        self._counts[chosen["id"]] += 1
        self._total += 1
        return chosen


# ============================================================
# 6. Segment Outlines (strategies C and E)
# ============================================================
def create_segment_outlines(
    anchors: List[InformationAnchor],
    total_segments: int,
    char_name: str,
    filler_sampler: FillerTopicSampler,
    dynamics_injector: Optional[DynamicsInjector],
) -> List[SegmentOutline]:
    segment_anchor_map: Dict[int, List[InformationAnchor]] = {}
    for anchor in anchors:
        segment_anchor_map.setdefault(anchor.target_segment, []).append(anchor)

    outlines = []
    for seg_id in range(total_segments):
        seg_anchors = segment_anchor_map.get(seg_id, [])

        if seg_id == 0:
            seg_type = SegmentType.OPENING
            topic = f"Opening: Setting the scene, introducing {char_name}"
        elif seg_id == total_segments - 1:
            seg_type = SegmentType.CLOSING
            topic = f"Closing: Wrapping up the conversation naturally"
        elif seg_anchors:
            seg_type = SegmentType.INFO_ANCHOR
            topic = f"Information segment with {len(seg_anchors)} anchor(s)"
        else:
            seg_type = SegmentType.FILLER
            topic, _ = filler_sampler.sample(char_name)

        dynamic_event = None
        if dynamics_injector and seg_type in (SegmentType.FILLER, SegmentType.INFO_ANCHOR):
            dynamic_event = dynamics_injector.maybe_inject()

        director_notes = generate_director_notes_v2(
            seg_type=seg_type.value,
            anchors=seg_anchors,
            topic=topic,
            char_name=char_name,
            dynamic_event=dynamic_event,
        )

        outlines.append(SegmentOutline(
            segment_id=seg_id,
            segment_type=seg_type,
            topic=topic,
            anchors=seg_anchors,
            director_notes=director_notes,
            dynamic_event=dynamic_event,
        ))
    return outlines


# ============================================================
# 7. Scene Director (strategy B)
# ============================================================
def director_generate_scene(
    generator: DualRoleGenerator,
    char_profile: Dict,
    source_events: List[Dict],
    scene_category: Dict,
) -> Dict:
    char_name = char_profile.get("character_name", "The Character")
    # Strategy A bug fix: fetch demographics from the original persona.
    persona = char_profile.get("original_persona", {})
    demographics = persona.get("demographics", {})
    events_text = "\n".join([f"- {e.get('description', '')[:100]}" for e in source_events[:5]])

    prompt_hint = scene_category.get("prompt_hint", "").replace("{char_name}", char_name)
    example_roles = scene_category.get("example_roles", [])
    example_roles_text = ", ".join(example_roles[:4])

    system_prompt = f"""You are the Scene Director for a roleplay.
Character: {char_name}
Demographics: {json.dumps(demographics, ensure_ascii=False)}
Sample Events (for context only):
{events_text}

SCENE CATEGORY: {scene_category.get('label', 'General')}
{prompt_hint}

Example interlocutor roles for this category: {example_roles_text}

Design a compelling scene for a LONG, natural conversation.
The Interlocutor should have a genuine reason to talk to {char_name} for an extended time.

Output JSON:
{{
  "interlocutor_role": "A specific role fitting the category above",
  "setting": "Detailed scene description with atmosphere and sensory details",
  "interlocutor_goal": "Why they want to talk (specific and personal, not generic)",
  "opening_style": "Casual or Formal"
}}
"""
    try:
        resp = generator.chat_completion(
            [{"role": "system", "content": system_prompt}],
            temperature=0.8,
            json_mode=True,
            phase="scene_generation",
        )
        parsed = robust_json_parse(resp)
        return parsed if isinstance(parsed, dict) else default_scene_config(scene_category)
    except Exception:
        return default_scene_config(scene_category)


# ============================================================
# 8. Segment Generator (strategies A, D, and G)
# ============================================================
def generate_segment(
    generator: DualRoleGenerator,
    outline: SegmentOutline,
    char_profile: Dict,
    scene_config: Dict,
    source_events: List[Dict],
    previous_summary: str,
    previous_last_turns: List[Dict],
    speech_profile_text: str,
    segment_token_target: int = DEFAULT_SEGMENT_TOKEN_TARGET,
    forbidden_topics: Optional[List[str]] = None,
    extra_constraints: str = "",
    fixed_turns_per_segment: Optional[int] = None,
    length_target: Optional[SegmentLengthTarget] = None,
    include_summary: bool = True,
) -> GeneratedSegment:
    char_name = char_profile.get("character_name", "The Character")
    interlocutor_role = scene_config.get("interlocutor_role", "Friend")

    # Strategy D: pass summary plus the last three raw dialogue turns.
    context_section = previous_summary if previous_summary else "Start of conversation"
    if previous_last_turns:
        last_turns_text = "\n".join([
            f"{'User' if t.get('role') == 'user' else char_name}: {t.get('content', '')[:150]}"
            for t in previous_last_turns[-3:]
        ])
        context_section += f"\n\nLast few exchanges (continue in this tone):\n{last_turns_text}"

    forbidden_section = ""
    if forbidden_topics and len(forbidden_topics) > 0:
        top_forbidden = forbidden_topics[:15]
        forbidden_list = "\n".join([f"  - {t}" for t in top_forbidden])
        forbidden_section = f"""
IMPORTANT — These topics are reserved for later. Do not mention, hint at, or foreshadow them:
{forbidden_list}"""

    if extra_constraints:
        forbidden_section += f"\n\nCRITICAL CORRECTION:\n{extra_constraints}"

    length_guidance = build_length_guidance(length_target) if length_target else ""
    turn_instruction = (
        f"Generate exactly {fixed_turns_per_segment} turns as a JSON array."
        if fixed_turns_per_segment
        else "Generate 6-10 turns as a JSON array."
    )

    # Strategy A: use speech_profile instead of a generic "Be verbose" instruction.
    system_prompt = f"""You are a dialogue generator. Output ONLY a valid JSON array of dialogue turns.
Each turn has "role" (user or assistant) and "content" (the dialogue text).
user = {interlocutor_role}, assistant = {char_name}.

{speech_profile_text}

You possess memories of your past, but you are SECRETIVE.
Do NOT mention any specific event unless the User explicitly asks about it or it is required by the Director Note below.
No stage directions, no brackets, no parenthetical actions."""

    user_prompt = f"""Generate a dialogue segment for this scenario:

CHARACTER: {char_name}
SETTING: {scene_config.get('setting', 'A quiet room')[:200]}
PREVIOUS CONTEXT: {context_section}

SEGMENT INSTRUCTIONS:
{outline.director_notes[:1500]}
{forbidden_section}
{length_guidance}

OUTPUT: {turn_instruction} Example format:
[
  {{"role": "user", "content": "..."}},
  {{"role": "assistant", "content": "..."}}
]

Generate the dialogue now:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Strategy G: set temperature by segment type.
    temp = TEMPERATURE_BY_SEGMENT.get(outline.segment_type.value, 0.85)

    response = generator.chat_completion(
        messages,
        temperature=temp,
        max_tokens=min(segment_token_target + 500, 2500),
        json_mode=True,
        phase="segment_generation",
        allow_kimi_fallback=True,
    )

    if not response:
        return GeneratedSegment(outline.segment_id, [], "[Generation failed]", 0)

    turns = _parse_dialogue_response(response, outline.segment_id)

    for turn in turns:
        if isinstance(turn, dict) and "content" in turn:
            turn["content"] = clean_dialogue_text(turn["content"])

    token_count = estimate_tokens(turns)
    summary = _summarize_segment(generator, turns, char_name) if include_summary else ""
    completion_meta = getattr(generator, "last_completion_by_phase", {}).get("segment_generation", {})

    return GeneratedSegment(
        segment_id=outline.segment_id,
        dialogue_turns=turns,
        summary=summary,
        token_count=token_count,
        provider_used=completion_meta.get("provider", "mimo"),
        fallback_used=completion_meta.get("fallback_used", False),
        fallback_attempts=completion_meta.get("fallback_attempts", 0),
    )


def _parse_dialogue_response(response: str, segment_id: int) -> List[Dict]:
    cleaned = response.strip()
    if cleaned.startswith('\ufeff'):
        cleaned = cleaned[1:]

    # Try multiple parsing strategies in order.
    for parser in [_parse_direct, _parse_array_extract, _parse_truncated_fix, _parse_regex]:
        result = parser(cleaned)
        if result:
            return result

    tqdm.write(f"      [Segment {segment_id}] Parse failed. Preview: {response[:100]}...")
    return []


def _parse_direct(text: str) -> Optional[List[Dict]]:
    try:
        result = json.loads(text)
        return result if isinstance(result, list) and len(result) > 0 else None
    except json.JSONDecodeError:
        return None


def _parse_array_extract(text: str) -> Optional[List[Dict]]:
    try:
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        pass
    return None


def _parse_truncated_fix(text: str) -> Optional[List[Dict]]:
    try:
        start = text.find('[')
        if start != -1:
            truncated = text[start:]
            last = truncated.rfind('},')
            if last == -1:
                last = truncated.rfind('}')
            if last != -1:
                return json.loads(truncated[:last + 1] + ']')
    except json.JSONDecodeError:
        pass
    return None


def _parse_regex(text: str) -> Optional[List[Dict]]:
    try:
        pattern = r'\{\s*"role"\s*:\s*"(user|assistant)"\s*,\s*"content"\s*:\s*"([^"]*(?:\\.[^"]*)*)"\s*\}'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return [
                {"role": role, "content": content.replace('\\"', '"').replace('\\n', '\n')}
                for role, content in matches
            ]
    except Exception:
        pass
    return None


def _summarize_segment(generator: DualRoleGenerator, turns: List[Dict], char_name: str) -> str:
    if not turns:
        return "No conversation yet."
    dialogue_text = "\n".join([
        f"{'User' if t['role'] == 'user' else char_name}: {t['content'][:200]}..."
        for t in turns[-6:]
    ])
    prompt = f"""Summarize this conversation segment in 2-3 sentences.
Focus on: the main topics discussed, emotional tone, and any key information revealed.

Dialogue:
{dialogue_text}

Summary (2-3 sentences):"""

    response = generator.chat_completion(
        [{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=200,
        phase="segment_summary",
    )
    return response if response else "The conversation continued naturally."


# ============================================================
# 9. Full Pipeline
# ============================================================
def process_character(
    character_name: str,
    tasks: List[Dict],
    generator: DualRoleGenerator,
    events_map: Dict,
    char_map: Dict,
    total_segments: int,
    segment_token_target: int,
    scene_selector: SceneCategorySelector,
    filler_sampler: FillerTopicSampler,
    dynamics_config: Dict[str, Any],
    enable_dynamics: bool = True,
    enable_speech_profile: bool = True,
    length_control: Optional[LengthControlConfig] = None,
    retry_segments_until_success: bool = True,
) -> CharacterProcessOutcome:
    print(f"\n Processing: {character_name}")

    char_profile = char_map.get(character_name)
    if not char_profile:
        print(f"  Character profile not found")
        return CharacterProcessOutcome(result=None, failure_record=None)

    all_event_ids = set()
    for t in tasks:
        all_event_ids.update(t.get("source_event_ids", []))
    source_events = [events_map[eid] for eid in all_event_ids if eid in events_map]
    if not source_events:
        print(f"  No source events found")
        return CharacterProcessOutcome(result=None, failure_record=None)

    # --- Phase 1: Planning ---
    anchors = create_information_anchors(tasks)
    anchors = distribute_anchors_to_segments(anchors, total_segments)

    dynamics_injector = DynamicsInjector(dynamics_config, total_segments) if enable_dynamics else None
    outlines = create_segment_outlines(
        anchors, total_segments, character_name, filler_sampler, dynamics_injector,
    )

    info_segments = sum(1 for o in outlines if o.segment_type == SegmentType.INFO_ANCHOR)
    filler_segments = sum(1 for o in outlines if o.segment_type == SegmentType.FILLER)
    fragment_anchors = sum(1 for a in anchors if a.anchor_type != "setup")
    print(f"  Anchors: {fragment_anchors} fragments (after filtering), Info: {info_segments}, Filler: {filler_segments}")

    # --- Phase 2: Scene ---
    scene_category = scene_selector.select()
    scene_config = director_generate_scene(generator, char_profile, source_events, scene_category)

    # --- Phase 3: Speech Profile (strategy A) ---
    if enable_speech_profile:
        speech_profile_text = generate_speech_profile(char_profile)
    else:
        speech_profile_text = "Be verbose. Each turn should be 50-150 words. No stage directions."

    # --- Phase 4: Segment Generation ---
    segment_keywords_map: Dict[int, List[str]] = {}
    for anchor in anchors:
        seg_id = anchor.target_segment
        segment_keywords_map.setdefault(seg_id, [])
        components = anchor.task_data.get("answer_components", [])
        if anchor.anchor_type != "setup":
            segment_keywords_map[seg_id].extend(extract_keywords_from_components(components))

    all_segments = []
    previous_summary = ""
    previous_last_turns: List[Dict] = []
    failed_segments: List[int] = []
    failure_details: List[Dict[str, Any]] = []
    tokens_so_far = 0
    length_rerolls_used = 0
    provider_fallback_segments = 0
    kimi_fallback_successes = 0
    kimi_fallback_failures = 0
    mimo_non_retryable_failures = 0

    for outline in tqdm(outlines, desc=f"     Segments", leave=False):
        current_seg_id = outline.segment_id

        future_keywords = []
        for future_id in range(current_seg_id + 1, total_segments):
            if future_id in segment_keywords_map:
                future_keywords.extend(segment_keywords_map[future_id])

        segment_events = []
        seen_eids = set()
        for anchor in [a for a in anchors if a.target_segment == current_seg_id]:
            for eid in anchor.task_data.get("source_event_ids", []):
                if eid not in seen_eids and eid in events_map:
                    segment_events.append(events_map[eid])
                    seen_eids.add(eid)

        max_retries = 2
        remaining_length_rerolls = length_control.max_length_rerolls if length_control else 0
        segment = None
        extra_constraints = ""
        length_target = None
        if length_control:
            length_target = compute_segment_length_target(
                config=length_control,
                tokens_so_far=tokens_so_far,
                current_segment_id=current_seg_id,
                total_segments=total_segments,
                segment_type=outline.segment_type,
            )

        attempt = 0
        while True:
            attempt += 1
            segment = generate_segment(
                generator=generator,
                outline=outline,
                char_profile=char_profile,
                scene_config=scene_config,
                source_events=segment_events,
                previous_summary=previous_summary,
                previous_last_turns=previous_last_turns,
                speech_profile_text=speech_profile_text,
                segment_token_target=segment_token_target,
                forbidden_topics=future_keywords[:15],
                extra_constraints=extra_constraints,
                fixed_turns_per_segment=length_control.fixed_turns_per_segment if length_control else None,
                length_target=length_target,
                include_summary=False,
            )

            if not segment.dialogue_turns:
                last_error = getattr(generator, "last_error_by_phase", {}).get("segment_generation", {})
                if retry_segments_until_success and last_error.get("allow_retry_until_success", True):
                    if attempt <= 3 or attempt % 10 == 0:
                        tqdm.write(
                            f"      [Retry {attempt}] Empty/parse failure in Segment {current_seg_id}; "
                            "retrying until success"
                        )
                    time.sleep(1)
                    continue
                if retry_segments_until_success and not last_error.get("allow_retry_until_success", True):
                    tqdm.write(
                        f"      [Segment Stop Retry] Segment {current_seg_id} due to non-retryable provider failure"
                    )
                    break
                if attempt <= max_retries:
                    tqdm.write(f"      [Retry {attempt}] Format error in Segment {current_seg_id}")
                    time.sleep(1)
                    continue
                break

            if detect_degeneration(segment.dialogue_turns):
                if attempt <= max_retries:
                    tqdm.write(f"      [Retry {attempt}] Degeneration detected in Segment {current_seg_id}")
                    time.sleep(1)
                    continue
                tqdm.write(f"      Warning: Degeneration unresolved in Segment {current_seg_id}")
                break

            dialogue_text = " ".join([t["content"] for t in segment.dialogue_turns])
            has_leak, leaked_word = check_leakage(dialogue_text, future_keywords)

            if has_leak and attempt <= max_retries:
                tqdm.write(f"      [Retry {attempt}] Leak detected: '{leaked_word}' in Segment {current_seg_id}")
                extra_constraints = (
                    f"You prematurely mentioned '{leaked_word}'. "
                    f"This information belongs to a future time. "
                    f"REWRITE the dialogue to REMOVE any mention of '{leaked_word}'."
                )
                time.sleep(1)
                continue

            if has_leak:
                tqdm.write(f"      Warning: Unresolved leak '{leaked_word}' in Segment {current_seg_id}")
            if length_target and segment_needs_length_retry(segment, length_target):
                if remaining_length_rerolls > 0:
                    remaining_length_rerolls -= 1
                    length_rerolls_used += 1
                    tqdm.write(
                        f"      [Length Retry] Segment {current_seg_id} length "
                        f"{segment.token_count} outside {length_target.low}-{length_target.high}"
                    )
                    extra_constraints = build_length_retry_instruction(segment, length_target)
                    continue
            break

        if not segment.dialogue_turns:
            failed_segments.append(current_seg_id)
            last_error = getattr(generator, "last_error_by_phase", {}).get("segment_generation")
            if last_error and last_error.get("fallback_attempted") and last_error.get("fallback_exhausted"):
                kimi_fallback_failures += 1
            if last_error and (
                last_error.get("mimo_non_retryable")
                or (
                    last_error.get("provider") == "mimo"
                    and last_error.get("is_non_retryable_for_provider")
                )
            ):
                mimo_non_retryable_failures += 1
            failure_details.append({
                "segment_id": current_seg_id,
                "failure_stage": "segment_generation",
                "last_error_summary": last_error,
                "failed_provider": last_error.get("provider") if last_error else None,
                "failed_category": last_error.get("category") if last_error else None,
                "provider_error_code": last_error.get("provider_error_code") if last_error else None,
                "fallback_attempted": last_error.get("fallback_attempted") if last_error else False,
                "fallback_attempts": last_error.get("fallback_attempts") if last_error else 0,
                "fallback_final_error": last_error.get("fallback_final_error") if last_error else None,
            })

        if segment.dialogue_turns and not segment.summary:
            segment.summary = _summarize_segment(generator, segment.dialogue_turns, character_name)

        if segment.dialogue_turns and segment.fallback_used:
            provider_fallback_segments += 1
            kimi_fallback_successes += 1

        all_segments.append(segment)
        previous_summary = f"Previously: {segment.summary}"
        previous_last_turns = segment.dialogue_turns[-3:] if segment.dialogue_turns else []
        tokens_so_far += segment.token_count

    if failed_segments:
        tqdm.write(f"  {len(failed_segments)} segments failed after retries")
        last_error = failure_details[-1]["last_error_summary"] if failure_details else None
        return CharacterProcessOutcome(
            result=None,
            failure_record={
                "character": character_name,
                "failed_segments": failed_segments,
                "failure_stage": "segment_generation",
                "last_error_summary": last_error,
                "failed_provider": last_error.get("provider") if last_error else None,
                "failed_category": last_error.get("category") if last_error else None,
                "provider_error_code": last_error.get("provider_error_code") if last_error else None,
                "fallback_attempted": last_error.get("fallback_attempted") if last_error else False,
                "fallback_attempts": last_error.get("fallback_attempts") if last_error else 0,
                "fallback_final_error": last_error.get("fallback_final_error") if last_error else None,
                "failures": failure_details,
                "statistics": {
                    "total_segments": total_segments,
                    "completed_segments": len([seg for seg in all_segments if seg.dialogue_turns]),
                    "estimated_tokens": tokens_so_far,
                    "length_rerolls": length_rerolls_used,
                    "provider_fallback_segments": provider_fallback_segments,
                    "kimi_fallback_successes": kimi_fallback_successes,
                    "kimi_fallback_failures": kimi_fallback_failures,
                    "mimo_non_retryable_failures": mimo_non_retryable_failures,
                },
            },
        )

    # --- Phase 5: Stitching & Validation ---
    full_dialogue = []
    for seg in all_segments:
        full_dialogue.extend(seg.dialogue_turns)

    total_tokens = sum(seg.token_count for seg in all_segments)
    validation_results = validate_information_coverage(full_dialogue, anchors)

    print(f"  Result: {len(full_dialogue)} turns, ~{total_tokens} tokens, "
          f"coverage: {validation_results['coverage_rate']:.1%}")

    return CharacterProcessOutcome(
        result={
            "id": str(uuid.uuid4()),
            "character": character_name,
            "generation_mode": "natural_v8",
            "tasks_covered": tasks,
            "segment_outlines": [
                {
                    "segment_id": o.segment_id,
                    "type": o.segment_type.value,
                    "topic": o.topic,
                    "anchor_count": len(o.anchors),
                    "has_dynamic": o.dynamic_event is not None,
                }
                for o in outlines
            ],
            "dialogue": full_dialogue,
            "scene_config": scene_config,
            "scene_category": scene_category.get("id", "unknown") if isinstance(scene_category, dict) else "unknown",
            "statistics": {
                "total_segments": total_segments,
                "total_turns": len(full_dialogue),
                "estimated_tokens": total_tokens,
                "info_anchors": len(anchors),
                "info_segments": info_segments,
                "filler_segments": filler_segments,
                "length_rerolls": length_rerolls_used,
                "length_target_tokens": length_control.target_total_tokens if length_control else None,
                "length_target_band": [
                    length_control.target_total_tokens - length_control.target_total_tolerance,
                    length_control.target_total_tokens + length_control.target_total_tolerance,
                ] if length_control else None,
                "failed_segments": 0,
                "provider_fallback_segments": provider_fallback_segments,
                "kimi_fallback_successes": kimi_fallback_successes,
                "kimi_fallback_failures": kimi_fallback_failures,
                "mimo_non_retryable_failures": mimo_non_retryable_failures,
            },
            "validation": validation_results,
        },
        failure_record=None,
    )


def validate_information_coverage(dialogue: List[Dict], anchors: List[InformationAnchor]) -> Dict:
    """
    Validate information coverage with a loose keyword-based check.

    A component is considered covered when enough content-bearing keywords
    appear in the dialogue. The return value also includes diagnostic details.
    """
    dialogue_text = " ".join([m.get("content", "") for m in dialogue]).lower()

    # Common synonym map used to improve coverage-check robustness.
    synonyms = {
        "larger": ["bigger", "spacious", "roomy", "expanded"],
        "smaller": ["tinier", "compact", "cozy"],
        "apartment": ["place", "flat", "unit", "home", "residence"],
        "house": ["home", "residence", "property"],
        "moved": ["relocated", "transferred", "shifted"],
        "bought": ["purchased", "acquired", "got"],
        "sold": ["disposed", "liquidated"],
        "started": ["began", "commenced", "initiated", "launched"],
        "ended": ["finished", "concluded", "completed", "terminated"],
        "hired": ["employed", "recruited", "brought"],
        "fired": ["dismissed", "terminated", "let go"],
        "la": ["los angeles", "l.a."],
        "nyc": ["new york", "new york city", "n.y.c."],
        "sf": ["san francisco", "san fran"],
    }
    
    def check_keyword_with_synonyms(keyword: str, text: str) -> bool:
        """Check whether a keyword or one of its synonyms appears in text."""
        if keyword in text:
            return True
        for syn in synonyms.get(keyword, []):
            if syn in text:
                return True
        return False


    stop_words = {
        "the", "a", "an", "is", "was", "were", "are", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "must", "can", "could", "and", "but", "or",
        "nor", "not", "so", "yet", "both", "either", "neither", "each",
        "every", "all", "any", "few", "more", "most", "other", "some",
        "such", "than", "too", "very", "just", "also", "then", "that",
        "this", "these", "those", "with", "from", "into", "over", "after",
        "before", "between", "under", "above", "about", "for", "during",
        "through", "upon", "onto", "within", "without", "along",
        "start", "end", "date", "period", "duration", "action", "subject",
        "benefit", "status", "vehicle", "model", "condition", "asset",
        "funding", "source", "type", "result", "initial", "final",
    }

    results = {
        "total_anchors": 0, 
        "found_anchors": 0, 
        "missing_anchors": [], 
        "coverage_rate": 0.0,
        "filtered_anchors": 0,  # Number of filtered reasoning-only components.
        "setup_anchors": 0,     # Number of setup anchors.
    }

    for anchor in anchors:
        if anchor.anchor_type == "setup":
            results["setup_anchors"] += 1
            continue
        
        task_type = anchor.task_data.get("task_type", "")
        if not should_verify_component(task_type, anchor.content):
            results["filtered_anchors"] += 1
            continue

        results["total_anchors"] += 1
        content_lower = anchor.content.lower()
        keywords = [w for w in re.findall(r'\w+', content_lower) if len(w) > 2 and w not in stop_words]

        if not keywords:
            results["found_anchors"] += 1
            continue

        hits = sum(1 for kw in keywords if check_keyword_with_synonyms(kw, dialogue_text))
        ratio = hits / len(keywords) if keywords else 0

        # Require at least half of the content keywords to match.
        if ratio >= 0.5:
            results["found_anchors"] += 1
        else:
            results["missing_anchors"].append({
                "task_id": anchor.task_id,
                "task_type": task_type,
                "type": anchor.anchor_type,
                "content": anchor.content[:80],
                "keyword_hit_ratio": f"{hits}/{len(keywords)} ({ratio:.0%})",
                "missing_keywords": [kw for kw in keywords if kw not in dialogue_text][:5],
            })

    if results["total_anchors"] > 0:
        results["coverage_rate"] = results["found_anchors"] / results["total_anchors"]
    return results


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Natural Dialogue Generator V8")
    parser.add_argument("--stories_file",
                        default=str(REPO_ROOT / "data_gen/output/stories_v4.json"))
    parser.add_argument("--qa_file",
                        default=str(REPO_ROOT / "data_gen/output/stories_v4_characters_qa.json"))
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--total_segments", type=int, default=DEFAULT_TOTAL_SEGMENTS)
    parser.add_argument("--segment_tokens", type=int, default=DEFAULT_SEGMENT_TOKEN_TARGET)
    parser.add_argument("--max_workers", type=int, default=20)
    # V8-specific options.
    parser.add_argument("--scene_config", default=None, help="Path to scene_categories.json")
    parser.add_argument("--filler_config", default=None, help="Path to filler_topics_v2.json")
    parser.add_argument("--dynamics_config", default=None, help="Path to dialogue_dynamics.json")
    parser.add_argument("--disable_dynamics", action="store_true", help="Disable dynamic events")
    parser.add_argument("--disable_speech_profile", action="store_true", help="Fall back to V7 style prompts")
    parser.add_argument("--target_total_tokens", type=int, default=None, help="Target estimated tokens per dialogue")
    parser.add_argument("--target_total_tolerance", type=int, default=None, help="Tolerance around target_total_tokens")
    parser.add_argument("--fixed_turns_per_segment", type=int, default=None, help="Exact turns to request for each segment")
    parser.add_argument("--max_length_rerolls", type=int, default=0, help="Max rerolls for segments outside the target length band")
    parser.add_argument("--api_transient_retries", type=int, default=8, help="Max attempts for transient API failures")
    parser.add_argument("--api_bad_request_retries", type=int, default=2, help="Retry count for HTTP 400 failures")
    parser.add_argument("--api_retry_max_wait", type=float, default=30.0, help="Maximum backoff wait between API retries")
    parser.add_argument(
        "--disable_kimi_fallback",
        action="store_true",
        help="Disable KIMI fallback for non-retryable MIMO segment_generation failures",
    )
    parser.add_argument(
        "--kimi_fallback_model",
        default=KIMI_MODEL,
        help="KIMI model alias to use for segment_generation fallback",
    )
    parser.add_argument(
        "--kimi_fallback_attempts",
        type=int,
        default=2,
        help="Total number of KIMI fallback attempts for a non-retryable segment_generation request",
    )
    parser.add_argument(
        "--disable_retry_segments_until_success",
        action="store_true",
        help="Use bounded segment retries instead of retrying empty segments until success",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    api_key = os.getenv("MIMO_API_KEY")
    if not api_key:
        print("Error: MIMO_API_KEY required")
        return

    generator = DualRoleGenerator(
        api_key,
        transient_retries=args.api_transient_retries,
        bad_request_retries=args.api_bad_request_retries,
        retry_max_wait=args.api_retry_max_wait,
        kimi_api_key=os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY"),
        kimi_model=args.kimi_fallback_model,
        kimi_fallback_attempts=args.kimi_fallback_attempts,
        enable_kimi_fallback=not args.disable_kimi_fallback,
    )

    length_control = None
    if args.target_total_tokens is not None:
        target_tolerance = args.target_total_tolerance if args.target_total_tolerance is not None else 10000
        fixed_turns = args.fixed_turns_per_segment if args.fixed_turns_per_segment is not None else 10
        length_control = LengthControlConfig(
            target_total_tokens=args.target_total_tokens,
            target_total_tolerance=target_tolerance,
            fixed_turns_per_segment=fixed_turns,
            max_length_rerolls=args.max_length_rerolls,
        )

    # Load data
    events_map, qa_missions, char_map = load_data_sources(args.stories_file, args.qa_file)

    if args.limit:
        limited_keys = list(qa_missions.keys())[:args.limit]
        qa_missions = {k: qa_missions[k] for k in limited_keys}
        print(f"Testing Mode: Limiting to {args.limit} characters.")

    # Load configs for strategies B, C, and E.
    scene_categories = load_scene_categories(args.scene_config)
    filler_topics = load_filler_topics(args.filler_config)
    dynamics_config = load_dynamics_config(args.dynamics_config)

    scene_selector = SceneCategorySelector(scene_categories)
    filler_sampler = FillerTopicSampler(filler_topics)

    print(f"\nStarting V8 Natural Dialogue Generation")
    print(f"   Total Segments: {args.total_segments}")
    print(f"   Target Tokens/Segment: {args.segment_tokens}")
    print(f"   Characters: {len(qa_missions)}")
    print(f"   Dynamics: {'ON' if not args.disable_dynamics else 'OFF'}")
    print(f"   Speech Profile: {'ON' if not args.disable_speech_profile else 'OFF'}")
    print(f"   Retry Segments Until Success: {'OFF' if args.disable_retry_segments_until_success else 'ON'}")
    print(f"   KIMI Fallback: {'OFF' if args.disable_kimi_fallback else 'ON'}")
    print(f"   KIMI Model: {args.kimi_fallback_model}")
    print(f"   KIMI Attempts: {args.kimi_fallback_attempts}")
    print(f"   Scene Categories: {len(scene_categories)}")
    print(f"   Filler Topic Types: {len(filler_topics)} ({sum(len(v) for v in filler_topics.values())} topics)")

    results = []
    failed_characters: Dict[str, Dict[str, Any]] = {}

    # Checkpoint resume
    tag = f"v8_seg{args.total_segments}_tok{args.segment_tokens}"
    checkpoint_file = os.path.join(args.output_dir, f"partial_results_{tag}.json")
    failed_characters_file = os.path.join(args.output_dir, f"failed_characters_{tag}.json")
    completed_chars = set()

    if os.path.exists(checkpoint_file):
        print(f"Found checkpoint: {checkpoint_file}")
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            for res in results:
                char_name = res.get("character")
                if char_name:
                    completed_chars.add(char_name)
            print(f"Loaded {len(completed_chars)} completed characters. Resuming...")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            results = []

    if os.path.exists(failed_characters_file):
        try:
            with open(failed_characters_file, "r", encoding="utf-8") as f:
                for record in json.load(f):
                    char_name = record.get("character")
                    if char_name:
                        failed_characters[char_name] = record
        except Exception as e:
            print(f"Failed to load failed character quarantine: {e}")
            failed_characters = {}
    for char_name in completed_chars:
        failed_characters.pop(char_name, None)

    original_count = len(qa_missions)
    qa_missions = {c: tasks for c, tasks in qa_missions.items() if c not in completed_chars}
    if len(qa_missions) < original_count:
        print(f"Skipping {original_count - len(qa_missions)} characters. Remaining: {len(qa_missions)}")

    if not qa_missions:
        print("All characters completed!")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def process_one(item):
        c_name, tasks = item
        per_char_filler = FillerTopicSampler(filler_topics)
        return process_character(
            character_name=c_name,
            tasks=tasks,
            generator=generator,
            events_map=events_map,
            char_map=char_map,
            total_segments=args.total_segments,
            segment_token_target=args.segment_tokens,
            scene_selector=scene_selector,
            filler_sampler=per_char_filler,
            dynamics_config=dynamics_config,
            enable_dynamics=not args.disable_dynamics,
            enable_speech_profile=not args.disable_speech_profile,
            length_control=length_control,
            retry_segments_until_success=not args.disable_retry_segments_until_success,
        )

    print(f"   Max Workers: {args.max_workers}")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_char = {
            executor.submit(process_one, (c_name, tasks)): c_name
            for c_name, tasks in qa_missions.items()
        }
        for future in tqdm(as_completed(future_to_char), total=len(future_to_char), desc="Characters"):
            c_name = future_to_char[future]
            try:
                outcome = future.result()
                if outcome.result:
                    results.append(outcome.result)
                    failed_characters.pop(outcome.result["character"], None)
                    with open(checkpoint_file, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                if outcome.failure_record:
                    failed_characters[outcome.failure_record["character"]] = outcome.failure_record
                if outcome.result or outcome.failure_record:
                    with open(failed_characters_file, "w", encoding="utf-8") as f:
                        json.dump(list(failed_characters.values()), f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"  Error processing {c_name}: {e}")

    out_path = os.path.join(args.output_dir, f"final_dialogues_{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("V8 GENERATION COMPLETE")
    print("=" * 60)
    if results:
        total_turns = sum(r["statistics"]["total_turns"] for r in results)
        total_tokens = sum(r["statistics"]["estimated_tokens"] for r in results)
        avg_tokens = total_tokens / len(results)
        avg_coverage = sum(r["validation"]["coverage_rate"] for r in results) / len(results)
        print(f"   Dialogues: {len(results)}")
        print(f"   Total Turns: {total_turns}")
        print(f"   Total Tokens: {total_tokens:,}")
        print(f"   Avg Tokens/Dialogue: {avg_tokens:,.0f}")
        print(f"   Avg Info Coverage: {avg_coverage:.1%}")
    if failed_characters:
        print(f"   Quarantined Characters: {len(failed_characters)}")
        print(f"   Quarantine File: {failed_characters_file}")
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
