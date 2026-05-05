from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib import error, request


MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
KIMI_BASE_URL = os.getenv("MOONSHOT_API_URL") or os.getenv("MOONSHOT_API_BASE") or "https://api.moonshot.cn/v1"


@dataclass(frozen=True)
class ProviderConfig:
    base_url: str
    default_model: str
    key_envs: tuple[str, ...]
    auth_style: str
    token_field: str


PROVIDERS: Dict[str, ProviderConfig] = {
    "mimo": ProviderConfig(
        base_url=MIMO_BASE_URL,
        default_model="mimo-v2-flash",
        key_envs=("MIMO_API_KEY", "XIAOMI_MIMO_API_KEY"),
        auth_style="api-key",
        token_field="max_completion_tokens",
    ),
    "siliconflow": ProviderConfig(
        base_url=SILICONFLOW_BASE_URL,
        default_model="Qwen/Qwen3-8B",
        key_envs=("SILICONFLOW_API_KEY",),
        auth_style="bearer",
        token_field="max_tokens",
    ),
    "kimi": ProviderConfig(
        base_url=KIMI_BASE_URL,
        default_model="moonshot-v1-128k",
        key_envs=("MOONSHOT_API_KEY", "KIMI_API_KEY"),
        auth_style="bearer",
        token_field="max_tokens",
    ),
}


class ChatClient:
    """Small OpenAI-compatible chat client for the providers used in this project."""

    def __init__(
        self,
        provider: str = "mimo",
        *,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_s: int = 120,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ) -> None:
        if provider not in PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}")
        self.provider = provider
        self.config = PROVIDERS[provider]
        self.model = model or self.config.default_model
        self.base_url = (base_url or self.config.base_url).rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    def _api_key(self) -> str:
        for env in self.config.key_envs:
            value = os.getenv(env)
            if value:
                return value
        names = " or ".join(self.config.key_envs)
        raise RuntimeError(f"Missing API key. Set {names}.")

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = self._api_key()
        if self.config.auth_style == "api-key":
            headers["api-key"] = key
        else:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        max_completion_tokens: int = 4096,
        temperature: float = 0.2,
        top_p: float = 0.95,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            self.config.token_field: max_completion_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        url = f"{self.base_url}/chat/completions"
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            req = request.Request(url, data=data, headers=self._headers(), method="POST")
            try:
                with request.urlopen(req, timeout=self.timeout_s) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                obj = json.loads(raw)
                return (obj.get("choices", [{}])[0].get("message", {}) or {}).get("content", "").strip()
            except error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
                last_error = RuntimeError(f"{self.provider} HTTP {exc.code}: {body}")
                if exc.code in (400, 401, 403):
                    raise last_error from exc
            except Exception as exc:
                last_error = RuntimeError(f"{self.provider} request failed: {exc}")

            if attempt < self.max_retries:
                time.sleep(self.retry_backoff * attempt)

        raise last_error or RuntimeError(f"{self.provider} request failed.")


def extract_json_value(text: str) -> Any:
    if not isinstance(text, str) or not text.strip():
        return None
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except Exception:
        pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(stripped):
        if ch not in "[{":
            continue
        try:
            obj, _ = decoder.raw_decode(stripped[idx:])
            return obj
        except Exception:
            continue
    return None
