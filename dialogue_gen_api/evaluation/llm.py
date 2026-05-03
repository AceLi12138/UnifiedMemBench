"""
LLM wrapper classes for model inference.
Supports SiliconFlow (Qwen, etc.), OpenAI, MiMo and other models.
Uses requests library to avoid httpx version conflicts.
"""

import os
import asyncio
import random
import requests
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from concurrent.futures import ThreadPoolExecutor


class LLM(ABC):
    """Abstract base class for language models."""

    def __init__(self, model_name: str, max_workers: int = 10):
        self.model_name = model_name
        self.max_workers = max_workers
        self.usage = self._initialize_usage()

    @abstractmethod
    def _initialize_usage(self) -> Dict:
        """Initializes the usage statistics dictionary."""
        pass

    @abstractmethod
    async def inference(self, prompt, max_tokens: int = 16384) -> Any:
        """
        Performs inference using the language model.
        
        Args:
            prompt: Either a string or List[Dict] messages in OpenAI format
            max_tokens: Maximum tokens for response
        """
        pass

    @abstractmethod
    def decode(self, completion: Any) -> str:
        """Decodes the completion response from the model."""
        pass

    def save_usage(self, save_dir: str):
        """Saves the usage statistics to a file."""
        import json
        if self.usage["api_calls"] > 0:
            avg_usage = {}
            for key in self.usage:
                if key != "api_calls":
                    avg_usage[f"avg_{key}"] = self.usage[key] / self.usage["api_calls"]
            self.usage.update(avg_usage)

        with open(os.path.join(save_dir, "usage.json"), "w") as f:
            json.dump(self.usage, f, indent=2)


class SiliconFlow_LLM(LLM):
    """LLM wrapper for SiliconFlow API (supports Qwen, GLM, etc.)."""
    
    BASE_URL = "https://api.siliconflow.cn/v1"

    def _initialize_usage(self) -> Dict:
        return {"input": 0, "output": 0, "total": 0, "api_calls": 0}

    def __init__(self, model_name: str, api_key: Optional[str] = None, max_workers: int = 10):
        super().__init__(model_name, max_workers)
        self.api_key = api_key or os.environ.get("SILICONFLOW_API_KEY")
        if not self.api_key:
            raise ValueError("SILICONFLOW_API_KEY not found. Set it in .env or pass via --api_key")
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def _call_api(self, prompt, max_tokens: int = 16384, max_retries: int = 20) -> dict:
        """
        Synchronous API call with aggressive retry mechanism for high reliability.
        
        Args:
            prompt: Either a string or List[Dict] messages in OpenAI format
            max_tokens: Maximum tokens for response
            max_retries: Maximum retry attempts
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Support both string and messages list format
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "enable_thinking": False
        }
        
        last_error = None
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=300
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as e:
                last_error = e
                status_code = e.response.status_code if e.response else 0
                
                # 400 Bad Request - don't retry
                if status_code == 400:
                    print(f"\n  ❌ Bad Request (400), not retrying: {e}")
                    raise
                
                # 429 Rate Limit
                if status_code == 429 and attempt < max_retries - 1:
                    retry_after = e.response.headers.get("Retry-After")
                    if retry_after:
                        wait_time = min(60, float(retry_after)) + random.uniform(0, 2)
                    else:
                        wait_time = min(60, (2 ** attempt) * 3) + random.uniform(0, 3)
                    print(f"\n  ⏳ Rate limited (429), retry {attempt + 1}/{max_retries}, waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                
                # 5xx Server errors
                if status_code >= 500 and attempt < max_retries - 1:
                    wait_time = min(60, (2 ** attempt) * 2) + random.uniform(0, 2)
                    print(f"\n  Retry {attempt + 1}/{max_retries} after {status_code} error, waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                raise
            except requests.exceptions.Timeout as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = min(60, (2 ** attempt) * 2) + random.uniform(0, 2)
                    print(f"\n  Retry {attempt + 1}/{max_retries} after timeout, waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                raise
            except requests.exceptions.ConnectionError as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = min(60, (2 ** attempt) * 2) + random.uniform(0, 2)
                    print(f"\n  Retry {attempt + 1}/{max_retries} after connection error, waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                raise
        
        raise last_error

    async def inference(self, prompt: str, max_tokens: int = 16384) -> Any:
        """Async inference using thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            lambda: self._call_api(prompt, max_tokens)
        )

    def decode(self, completion: Any) -> str:
        """Decode the API response."""
        if isinstance(completion, dict):
            if "usage" in completion:
                usage = completion["usage"]
                self.usage["input"] += usage.get("prompt_tokens", 0)
                self.usage["output"] += usage.get("completion_tokens", 0)
                self.usage["total"] += usage.get("total_tokens", 0)
            self.usage["api_calls"] += 1
            
            return completion["choices"][0]["message"]["content"]
        return str(completion)


class OpenAI_LLM(LLM):
    """LLM wrapper for OpenAI API using requests."""
    
    BASE_URL = "https://api.openai.com/v1"

    def _initialize_usage(self) -> Dict:
        return {"input": 0, "output": 0, "total": 0, "api_calls": 0}

    def __init__(self, model_name: str, api_key: Optional[str] = None, max_workers: int = 10):
        super().__init__(model_name, max_workers)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found. Set it in .env or pass via --api_key")
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def _call_api(self, prompt, max_tokens: int = 16384, max_retries: int = 5) -> dict:
        """
        Synchronous API call with retry mechanism.
        
        Args:
            prompt: Either a string or List[Dict] messages in OpenAI format
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Support both string and messages list format
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        
        last_error = None
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=300
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as e:
                last_error = e
                status_code = e.response.status_code if e.response else 0
                
                if status_code == 429 and attempt < max_retries - 1:
                    retry_after = e.response.headers.get("Retry-After")
                    if retry_after:
                        wait_time = float(retry_after) + random.uniform(0, 2)
                    else:
                        wait_time = (2 ** attempt) * 3 + random.uniform(0, 3)
                    print(f"\n  ⏳ Rate limited (429), retry {attempt + 1}/{max_retries}, waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                
                if status_code >= 500 and attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2 + random.uniform(0, 2)
                    print(f"\n  Retry {attempt + 1}/{max_retries} after {status_code} error, waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                raise
            except requests.exceptions.Timeout as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2 + random.uniform(0, 2)
                    print(f"\n  Retry {attempt + 1}/{max_retries} after timeout, waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                raise
        
        raise last_error

    async def inference(self, prompt: str, max_tokens: int = 16384) -> Any:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            lambda: self._call_api(prompt, max_tokens)
        )

    def decode(self, completion: Any) -> str:
        if isinstance(completion, dict):
            if "usage" in completion:
                usage = completion["usage"]
                self.usage["input"] += usage.get("prompt_tokens", 0)
                self.usage["output"] += usage.get("completion_tokens", 0)
                self.usage["total"] += usage.get("total_tokens", 0)
            self.usage["api_calls"] += 1
            return completion["choices"][0]["message"]["content"]
        return str(completion)


class Kimi_LLM(LLM):
    """LLM wrapper for Moonshot Kimi API using OpenAI-compatible schema."""

    BASE_URL = "https://api.moonshot.cn/v1"
    MODEL_ALIASES = {
        "kimi-k2.5": "kimi-k2-turbo-preview",
    }

    def _initialize_usage(self) -> Dict:
        return {"input": 0, "output": 0, "total": 0, "api_calls": 0}

    def __init__(self, model_name: str, api_key: Optional[str] = None, max_workers: int = 10):
        super().__init__(model_name, max_workers)
        self.api_key = (
            api_key
            or os.environ.get("KIMI_API_KEY")
            or os.environ.get("MOONSHOT_API_KEY")
        )
        if not self.api_key:
            raise ValueError("KIMI_API_KEY (or MOONSHOT_API_KEY) not found. Set it in .env or pass via --api_key")
        self.request_model = self.MODEL_ALIASES.get(model_name, model_name)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def _call_api(self, prompt, max_tokens: int = 16384, max_retries: int = 20) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": prompt}]

        payload = {
            "model": self.request_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }

        last_error = None
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=300,
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as e:
                last_error = e
                status_code = e.response.status_code if e.response else 0

                if status_code == 400:
                    raise

                if status_code == 429 and attempt < max_retries - 1:
                    retry_after = e.response.headers.get("Retry-After")
                    if retry_after:
                        wait_time = min(60, float(retry_after)) + random.uniform(0, 3)
                    else:
                        wait_time = min(60, (2 ** attempt) * 3) + random.uniform(0, 5)
                    print(f"\n  ⏳ Rate limited (429), retry {attempt + 1}/{max_retries}, waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue

                if status_code >= 500 and attempt < max_retries - 1:
                    wait_time = min(60, (2 ** attempt) * 2) + random.uniform(0, 3)
                    print(f"\n  Retry {attempt + 1}/{max_retries} after {status_code} error, waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                raise
            except requests.exceptions.Timeout as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = min(60, (2 ** attempt) * 2) + random.uniform(0, 3)
                    print(f"\n  Retry {attempt + 1}/{max_retries} after timeout, waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                raise
            except requests.exceptions.ConnectionError as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = min(60, (2 ** attempt) * 2) + random.uniform(0, 3)
                    print(f"\n  Retry {attempt + 1}/{max_retries} after connection error, waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                raise

        if last_error:
            raise last_error
        raise RuntimeError("Max retries reached without success")

    async def inference(self, prompt: str, max_tokens: int = 16384) -> Any:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: self._call_api(prompt, max_tokens),
        )

    def decode(self, completion: Any) -> str:
        if isinstance(completion, dict):
            if "usage" in completion:
                usage = completion["usage"]
                self.usage["input"] += usage.get("prompt_tokens", 0)
                self.usage["output"] += usage.get("completion_tokens", 0)
                self.usage["total"] += usage.get("total_tokens", 0)
            self.usage["api_calls"] += 1
            return completion["choices"][0]["message"]["content"]
        return str(completion)


class MiMo_LLM(LLM):
    """LLM wrapper for Xiaomi MiMo API."""
    
    BASE_URL = "https://api.xiaomimimo.com/v1"

    def _initialize_usage(self) -> Dict:
        return {"input": 0, "output": 0, "total": 0, "api_calls": 0}

    def __init__(self, model_name: str, api_key: Optional[str] = None, max_workers: int = 10):
        super().__init__(model_name, max_workers)
        self.api_key = api_key or os.environ.get("MIMO_API_KEY")
        if not self.api_key:
            raise ValueError("MIMO_API_KEY not found. Set it in .env or pass via --api_key")
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def _call_api(self, prompt, max_tokens: int = 16384, max_retries: int = 20) -> dict:
        """
        Synchronous API call with aggressive retry mechanism for high reliability.
        
        Args:
            prompt: Either a string or List[Dict] messages in OpenAI format
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Support both string and messages list format
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "top_p": 0.95,
            "stream": False,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
        
        last_error = None
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=300
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as e:
                last_error = e
                status_code = e.response.status_code if e.response else 0
                
                # 400 Bad Request - don't retry, raise immediately
                if status_code == 400:
                    print(f"\n  ❌ Bad Request (400), not retrying: {e}")
                    raise
                
                # 429 Rate Limit - aggressive wait with Retry-After support
                if status_code == 429 and attempt < max_retries - 1:
                    retry_after = e.response.headers.get("Retry-After") if e.response else None
                    if retry_after:
                        try:
                            wait_time = min(60, float(retry_after)) + random.uniform(0, 3)
                        except ValueError:
                            wait_time = min(60, (2 ** attempt) * 3) + random.uniform(0, 5)
                    else:
                        # Exponential backoff capped at 60s
                        wait_time = min(60, (2 ** attempt) * 3) + random.uniform(0, 5)
                    print(f"\n  ⏳ Rate limited (429), retry {attempt + 1}/{max_retries}, waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                
                # 5xx Server errors - retry with backoff
                if status_code >= 500 and attempt < max_retries - 1:
                    wait_time = min(60, (2 ** attempt) * 2) + random.uniform(0, 3)
                    print(f"\n  Retry {attempt + 1}/{max_retries} after {status_code} error, waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                raise
            except requests.exceptions.Timeout as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = min(60, (2 ** attempt) * 2) + random.uniform(0, 3)
                    print(f"\n  Retry {attempt + 1}/{max_retries} after timeout, waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                raise
            except requests.exceptions.ConnectionError as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = min(60, (2 ** attempt) * 2) + random.uniform(0, 3)
                    print(f"\n  Retry {attempt + 1}/{max_retries} after connection error, waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                raise
        
        raise last_error

    async def inference(self, prompt: str, max_tokens: int = 16384) -> Any:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            lambda: self._call_api(prompt, max_tokens)
        )

    def decode(self, completion: Any) -> str:
        if isinstance(completion, dict):
            if "usage" in completion:
                usage = completion["usage"]
                self.usage["input"] += usage.get("prompt_tokens", 0)
                self.usage["output"] += usage.get("completion_tokens", 0)
                self.usage["total"] += usage.get("total_tokens", 0)
            self.usage["api_calls"] += 1
            return completion["choices"][0]["message"]["content"]
        return str(completion)


class Gemini_LLM(LLM):
    """LLM wrapper for Google Gemini API using requests."""
    
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def _initialize_usage(self) -> Dict:
        return {"input": 0, "output": 0, "total": 0, "api_calls": 0}

    def __init__(self, model_name: str, api_key: Optional[str] = None, max_workers: int = 10):
        super().__init__(model_name, max_workers)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found. Set it in .env or pass via --api_key")
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        # Map friendly names to actual model IDs
        self.model_id = self._get_model_id(model_name)

    def _get_model_id(self, model_name: str) -> str:
        """Map model name to Gemini model ID."""
        model_map = {
            # Gemini 2.0
            "gemini-2.0-flash": "gemini-2.0-flash",
            "gemini-2.0-flash-exp": "gemini-2.0-flash-exp",
            "gemini-2.0-flash-lite": "gemini-2.0-flash-lite",
            # Gemini 2.5
            "gemini-2.5-flash": "gemini-2.5-flash",
            "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
            "gemini-2.5-pro": "gemini-2.5-pro",
            # Gemini 3.x
            "gemini-3-flash-preview": "gemini-3-flash-preview",
            "gemini-3-pro-preview": "gemini-3-pro-preview",
            # Legacy
            "gemini-1.5-pro": "gemini-1.5-pro",
            "gemini-1.5-flash": "gemini-1.5-flash",
            "gemini-1.5-flash-8b": "gemini-1.5-flash-8b",
            "gemini-pro": "gemini-pro",
        }
        return model_map.get(model_name, model_name)

    def _convert_messages_to_gemini_format(self, prompt) -> list:
        """
        Convert OpenAI-style messages to Gemini format.
        Gemini uses 'user' and 'model' roles instead of 'user' and 'assistant'.
        """
        if isinstance(prompt, str):
            return [{"role": "user", "parts": [{"text": prompt}]}]
        
        contents = []
        system_instruction = None
        
        for msg in prompt:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                # Gemini handles system prompts differently
                system_instruction = content
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})
            else:  # user
                contents.append({"role": "user", "parts": [{"text": content}]})
        
        # Prepend system instruction to first user message if exists
        if system_instruction and contents:
            first_user_idx = next((i for i, c in enumerate(contents) if c["role"] == "user"), None)
            if first_user_idx is not None:
                original_text = contents[first_user_idx]["parts"][0]["text"]
                contents[first_user_idx]["parts"][0]["text"] = f"{system_instruction}\n\n{original_text}"
        
        return contents

    def _call_api(self, prompt, max_tokens: int = 16384, max_retries: int = 20) -> dict:
        """
        Synchronous API call with aggressive retry mechanism.
        
        Args:
            prompt: Either a string or List[Dict] messages in OpenAI format
        """
        contents = self._convert_messages_to_gemini_format(prompt)
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.0,
            }
        }
        
        url = f"{self.BASE_URL}/{self.model_id}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        
        last_error = None
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=300
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as e:
                last_error = e
                status_code = e.response.status_code if e.response else 0
                
                # 400 Bad Request - don't retry
                if status_code == 400:
                    print(f"\n  ❌ Bad Request (400), not retrying: {e}")
                    raise
                
                if status_code == 429 and attempt < max_retries - 1:
                    retry_after = e.response.headers.get("Retry-After")
                    if retry_after:
                        wait_time = min(60, float(retry_after)) + random.uniform(0, 3)
                    else:
                        wait_time = min(60, (2 ** attempt) * 3) + random.uniform(0, 5)
                    print(f"\n  ⏳ Rate limited (429), retry {attempt + 1}/{max_retries}, waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                
                if status_code >= 500 and attempt < max_retries - 1:
                    wait_time = min(60, (2 ** attempt) * 2) + random.uniform(0, 3)
                    print(f"\n  Retry {attempt + 1}/{max_retries} after {status_code} error, waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                
                raise
            except requests.exceptions.Timeout as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = min(60, (2 ** attempt) * 2) + random.uniform(0, 3)
                    print(f"\n  Retry {attempt + 1}/{max_retries} after timeout, waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                raise
            except requests.exceptions.ConnectionError as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = min(60, (2 ** attempt) * 2) + random.uniform(0, 3)
                    print(f"\n  Retry {attempt + 1}/{max_retries} after connection error, waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                raise
        
        if last_error:
            raise last_error
        raise RuntimeError("Max retries reached without success")

    async def inference(self, prompt, max_tokens: int = 16384) -> Any:
        """Async wrapper around synchronous API call."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            lambda: self._call_api(prompt, max_tokens)
        )

    def decode(self, completion: Any) -> str:
        """Decode the Gemini API response."""
        if isinstance(completion, dict):
            # Track usage if available
            if "usageMetadata" in completion:
                usage = completion["usageMetadata"]
                self.usage["input"] += usage.get("promptTokenCount", 0)
                self.usage["output"] += usage.get("candidatesTokenCount", 0)
                self.usage["total"] += usage.get("totalTokenCount", 0)
            self.usage["api_calls"] += 1
            
            # Extract text from response
            try:
                candidates = completion.get("candidates", [])
                if candidates:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    if parts:
                        return parts[0].get("text", "")
            except (KeyError, IndexError) as e:
                print(f"Warning: Failed to parse Gemini response: {e}")
                return ""
        return str(completion)


# Model name to LLM class mapping
LLM_PROVIDERS = {
    # SiliconFlow models (Qwen, GLM, etc.)
    "Qwen/Qwen3-8B": SiliconFlow_LLM,
    "Qwen/Qwen3-14B": SiliconFlow_LLM,
    "Qwen/Qwen3-32B": SiliconFlow_LLM,
    "Qwen/Qwen3-30B-A3B": SiliconFlow_LLM,
    "Qwen/Qwen3-235B-A22B": SiliconFlow_LLM,
    "Pro/zai-org/GLM-4.7": SiliconFlow_LLM,
    "zai-org/GLM-4.6": SiliconFlow_LLM,
    "deepseek-ai/DeepSeek-V3.1-Terminus": SiliconFlow_LLM,
    # MiMo models
    "mimo-v2-flash": MiMo_LLM,
    # OpenAI models
    "gpt-4o": OpenAI_LLM,
    "gpt-4o-mini": OpenAI_LLM,
    "gpt-4-turbo": OpenAI_LLM,
    "gpt-4": OpenAI_LLM,
    "o1": OpenAI_LLM,
    "o1-mini": OpenAI_LLM,
    # Kimi models
    "kimi-k2.5": Kimi_LLM,
    "kimi-k2-turbo-preview": Kimi_LLM,
    # Gemini models
    
    "gemini-2.0-flash": Gemini_LLM,
    "gemini-2.0-flash-exp": Gemini_LLM,
    "gemini-1.5-pro": Gemini_LLM,
    "gemini-1.5-flash": Gemini_LLM,
    "gemini-1.5-flash-8b": Gemini_LLM,
    "gemini-pro": Gemini_LLM,
}


def get_llm(model_name: str, api_key: Optional[str] = None, max_workers: int = 10) -> LLM:
    """
    Factory function to get the appropriate LLM class for a model.
    
    Args:
        model_name: Name of the model
        api_key: Optional API key
        max_workers: Max concurrent workers for the thread pool
    
    For SiliconFlow models, you can use any model name - if not in the predefined list,
    it will default to SiliconFlow_LLM.
    """
    if model_name in LLM_PROVIDERS:
        llm_class = LLM_PROVIDERS[model_name]
    elif model_name.startswith(("Qwen/", "Pro/", "zai-org/", "deepseek-ai/", "tencent/")):
        llm_class = SiliconFlow_LLM
    elif model_name.startswith(("gpt-", "o1", "o3")):
        llm_class = OpenAI_LLM
    elif model_name.startswith("kimi"):
        llm_class = Kimi_LLM
    elif model_name.startswith("mimo"):
        llm_class = MiMo_LLM
    elif model_name.startswith("gemini"):
        llm_class = Gemini_LLM
    else:
        print(f"Warning: Unknown model '{model_name}', assuming SiliconFlow API")
        llm_class = SiliconFlow_LLM
    
    return llm_class(model_name, api_key=api_key, max_workers=max_workers)
