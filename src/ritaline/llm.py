"""Minimal OpenAI-compatible Chat Completions client with retries."""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
from typing import Any

import httpx

from .config import EndpointConfig
from .exceptions import LLMError

_RETRIABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _extract_content(payload: dict[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError("Endpoint response did not contain choices[0].message.content") from exc

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "".join(parts)
    raise LLMError("Endpoint returned an unsupported message content format")


def parse_json_object(content: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating markdown code fences around it."""
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            raise LLMError("Model output was not a JSON object") from None
        try:
            value = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"Model output contained invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise LLMError("Model output JSON must be an object")
    return value


class OpenAICompatibleClient:
    """Async client for OpenAI-compatible `/chat/completions` endpoints."""

    def __init__(self, config: EndpointConfig):
        self.config = config
        headers = {"Content-Type": "application/json", **config.headers}
        if config.api_key_env:
            api_key = os.getenv(config.api_key_env)
            if not api_key:
                raise LLMError(
                    f"Environment variable {config.api_key_env!r} is not set for the API key"
                )
            headers.setdefault("Authorization", f"Bearer {api_key}")
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            headers=headers,
            timeout=config.timeout_seconds,
        )

    async def __aenter__(self) -> OpenAICompatibleClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def generate_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        if self.config.response_format == "json_object":
            body["response_format"] = {"type": "json_object"}
        body.update(self.config.extra_body)

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = await self._client.post(self.config.chat_completions_path, json=body)
                if response.status_code in _RETRIABLE_STATUS:
                    raise httpx.HTTPStatusError(
                        f"Retriable HTTP {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise LLMError("Endpoint response body must be a JSON object")
                return parse_json_object(_extract_content(payload))
            except (httpx.HTTPError, json.JSONDecodeError, LLMError) as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
                delay = self.config.retry_backoff_seconds * (2**attempt)
                delay *= random.uniform(0.8, 1.2)
                await asyncio.sleep(delay)
        raise LLMError(f"LLM request failed after retries: {last_error}") from last_error
