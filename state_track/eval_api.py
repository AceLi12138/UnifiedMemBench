#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error, request

from fact_track.evaluation import (
    IncrementalWriter,
    collect_prompt_files,
    evaluate_file,
    load_jobs,
    print_summary,
    safe_name,
    write_run_config,
)


@dataclass(frozen=True)
class ProviderConfig:
    base_url: str
    default_model: str
    key_envs: Tuple[str, ...]
    auth_style: str
    token_field: str
    max_retries: int
    default_temperature: float


PROVIDERS: Dict[str, ProviderConfig] = {
    "mimo": ProviderConfig(
        base_url="https://api.xiaomimimo.com/v1",
        default_model="mimo-v2-flash",
        key_envs=("MIMO_API_KEY", "XIAOMI_MIMO_API_KEY"),
        auth_style="api-key",
        token_field="max_completion_tokens",
        max_retries=3,
        default_temperature=0.0,
    ),
    "kimi": ProviderConfig(
        base_url=os.getenv("MOONSHOT_API_URL") or os.getenv("MOONSHOT_API_BASE") or "https://api.moonshot.cn/v1",
        default_model="kimi-k2.5",
        key_envs=("MOONSHOT_API_KEY", "KIMI_API_KEY", "MOONSHOTAI_API_KEY"),
        auth_style="bearer",
        token_field="max_tokens",
        max_retries=10,
        default_temperature=0.6,
    ),
}


class ApiChatClient:
    def __init__(
        self,
        provider: str,
        *,
        model: str,
        base_url: Optional[str],
        timeout_s: int,
        max_retries: Optional[int],
    ) -> None:
        if provider not in PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}")
        self.provider = provider
        self.config = PROVIDERS[provider]
        self.model = model
        self.base_url = (base_url or self.config.base_url).rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries or self.config.max_retries
        self.api_key = self._resolve_api_key()

    def _resolve_api_key(self) -> str:
        for env_name in self.config.key_envs:
            value = os.getenv(env_name)
            if value:
                return value
        names = " or ".join(self.config.key_envs)
        raise RuntimeError(f"Missing API key for {self.provider}. Set {names}.")

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.auth_style == "api-key":
            headers["api-key"] = self.api_key
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _extract_content(obj: Dict[str, Any]) -> str:
        message = (obj.get("choices", [{}])[0].get("message", {}) or {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "".join(parts).strip()
        return str(content).strip()

    def chat(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            self.config.token_field: max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        if self.provider == "kimi" and self.model.lower().startswith("kimi-k2.5"):
            payload["thinking"] = {"type": "disabled"}

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        url = f"{self.base_url}/chat/completions"
        last_error: Optional[BaseException] = None

        for attempt in range(self.max_retries):
            req = request.Request(url, data=data, headers=self._headers(), method="POST")
            try:
                with request.urlopen(req, timeout=self.timeout_s) as response:
                    raw = response.read().decode("utf-8", errors="replace")
                return self._extract_content(json.loads(raw))
            except error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
                last_error = RuntimeError(f"{self.provider} HTTP {exc.code}: {body[:1000]}")
                if exc.code in (400, 401, 403):
                    raise last_error from exc
                if attempt < self.max_retries - 1:
                    wait_s = self._retry_wait(attempt, exc)
                    print(
                        f"[warn] {self.provider} HTTP {exc.code}; retry {attempt + 1}/{self.max_retries} "
                        f"after {wait_s:.1f}s",
                        file=sys.stderr,
                    )
                    time.sleep(wait_s)
                    continue
                raise last_error from exc
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    wait_s = min(60.0, 2.0 * (2 ** attempt)) + random.uniform(0.0, 1.5)
                    print(
                        f"[warn] {self.provider} request error; retry {attempt + 1}/{self.max_retries} "
                        f"after {wait_s:.1f}s: {exc}",
                        file=sys.stderr,
                    )
                    time.sleep(wait_s)
                    continue
                raise RuntimeError(f"{self.provider} request failed: {exc}") from exc

        raise RuntimeError(f"{self.provider} request failed: {last_error}")

    @staticmethod
    def _retry_wait(attempt: int, exc: error.HTTPError) -> float:
        retry_after = exc.headers.get("Retry-After") if exc.headers else None
        if retry_after:
            try:
                return min(60.0, float(retry_after)) + random.uniform(0.0, 1.5)
            except ValueError:
                pass
        base = 3.0 if exc.code == 429 else 2.0
        return min(60.0, base * (2 ** attempt)) + random.uniform(0.0, 1.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate schema_longest prompt files with mimo or kimi APIs.")
    parser.add_argument("-i", "--input", default="fact_track_schema_longest", help="Prompt JSON file or directory.")
    parser.add_argument("-o", "--output-dir", default="eval_results/api", help="Directory for CSV/JSONL outputs.")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default="mimo")
    parser.add_argument("--model", default="", help="Model name. Defaults to provider-specific model.")
    parser.add_argument("--base-url", default="", help="Override provider base URL.")
    parser.add_argument("--num-samples", type=int, default=0, help="Max samples per file. 0 means all.")
    parser.add_argument("--level-start", type=int, default=None)
    parser.add_argument("--level-end", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=16384, help="Maximum completion tokens.")
    parser.add_argument("--temperature", type=float, default=None, help="Default: provider-specific.")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=0, help="0 means provider default.")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds after each request per worker.")
    parser.add_argument("--system", default="", help="Optional system prompt.")
    parser.add_argument("--include-prompt", action="store_true", help="Store full prompt text in JSONL details.")
    parser.add_argument("--dry-run", action="store_true", help="Only list selected prompt files and sample counts.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing result files.")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip files already present in the CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provider_config = PROVIDERS[args.provider]
    model = args.model.strip() or provider_config.default_model
    temperature = provider_config.default_temperature if args.temperature is None else args.temperature
    max_samples = args.num_samples if args.num_samples > 0 else None
    max_files = args.max_files if args.max_files > 0 else None
    system_prompt = args.system.strip() or None

    prompt_files = collect_prompt_files(
        args.input,
        level_start=args.level_start,
        level_end=args.level_end,
        max_files=max_files,
    )

    if args.dry_run:
        total = 0
        print(f"Selected {len(prompt_files)} prompt files:")
        for file_path in prompt_files:
            samples = len(load_jobs(file_path, max_samples))
            total += samples
            print(f"  {file_path.name}: {samples} samples")
        print(f"Total selected samples: {total}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"eval_api_{args.provider}_{safe_name(model)}"
    csv_path = output_dir / f"{prefix}.csv"
    jsonl_path = output_dir / f"{prefix}.jsonl"
    config_path = output_dir / f"{prefix}.config.json"

    writer = IncrementalWriter(
        csv_path=csv_path,
        jsonl_path=jsonl_path,
        runner="api",
        provider=args.provider,
        model=model,
        resume=not args.no_resume,
        overwrite=args.overwrite,
    )

    write_run_config(
        config_path,
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "runner": "api",
            "provider": args.provider,
            "model": model,
            "base_url": args.base_url.strip() or provider_config.base_url,
            "input": str(args.input),
            "files": [file.name for file in prompt_files],
            "num_samples": max_samples,
            "max_tokens": args.max_tokens,
            "temperature": temperature,
            "top_p": args.top_p,
            "concurrency": args.concurrency,
            "sleep": args.sleep,
            "resume": not args.no_resume,
        },
    )

    client = ApiChatClient(
        args.provider,
        model=model,
        base_url=args.base_url.strip() or None,
        timeout_s=args.timeout,
        max_retries=args.max_retries if args.max_retries > 0 else None,
    )

    def call_model(prompt: str) -> str:
        return client.chat(
            prompt,
            system_prompt=system_prompt,
            max_tokens=args.max_tokens,
            temperature=temperature,
            top_p=args.top_p,
        )

    print(
        f"provider={args.provider} model={model} files={len(prompt_files)} "
        f"concurrency={args.concurrency} output={output_dir}"
    )
    evaluated = []
    for idx, file_path in enumerate(prompt_files, start=1):
        if writer.is_done(file_path.name):
            print(f"[{idx}/{len(prompt_files)}] {file_path.name} -> skip")
            continue

        started = time.time()
        print(f"[{idx}/{len(prompt_files)}] evaluating {file_path.name}", flush=True)
        result, records = evaluate_file(
            file_path,
            model_call=call_model,
            max_samples=max_samples,
            concurrency=args.concurrency,
            sleep_s=args.sleep,
            include_prompt=args.include_prompt,
        )
        writer.save(result, records)
        elapsed = time.time() - started
        evaluated.append(result)
        print(
            f"  {result.level}: prompts={result.total_prompts} "
            f"prompt_em={result.prompt_exact_match}/{result.total_prompts} "
            f"schema_acc={result.schema_accuracy:.4f} parse={result.parse_rate:.4f} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    print_summary(evaluated, provider=args.provider, model=model)
    print(f"Saved:\n  {csv_path}\n  {jsonl_path}\n  {config_path}")


if __name__ == "__main__":
    main()
