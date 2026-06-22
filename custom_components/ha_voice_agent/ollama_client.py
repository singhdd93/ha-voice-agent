"""Ollama native API client for HA Voice Agent."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

_LOGGER = logging.getLogger(__name__)


class OllamaError(Exception):
    """Raised when Ollama returns an error."""


async def chat(
    ollama_url: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    num_ctx: int = 4096,
    temperature: float = 0.1,
    num_predict: int = 512,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """
    Call Ollama /api/chat and return the parsed response dict.

    Response shape:
    {
      "model": "...",
      "message": {
        "role": "assistant",
        "content": "...",
        "tool_calls": [          # present only when model calls a tool
          {
            "function": {
              "name": "execute_services",
              "arguments": { ... }
            }
          }
        ]
      },
      "done": true,
      "done_reason": "stop" | "tool_calls",
      "prompt_eval_count": N,
      "eval_count": N,
    }
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "num_ctx": num_ctx,
            "temperature": temperature,
            "num_predict": num_predict,
            "repeat_penalty": 1.1,
        },
    }
    if tools:
        payload["tools"] = tools

    url = f"{ollama_url.rstrip('/')}/api/chat"
    _LOGGER.debug("Ollama request → %s  model=%s  msgs=%d", url, model, len(messages))

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
    except httpx.TimeoutException as err:
        raise OllamaError(f"Ollama request timed out after {timeout}s") from err
    except httpx.HTTPStatusError as err:
        raise OllamaError(
            f"Ollama returned HTTP {err.response.status_code}: {err.response.text[:200]}"
        ) from err
    except httpx.RequestError as err:
        raise OllamaError(f"Could not reach Ollama at {url}: {err}") from err

    data = resp.json()
    _LOGGER.debug(
        "Ollama response: done_reason=%s prompt_tokens=%s gen_tokens=%s",
        data.get("done_reason"),
        data.get("prompt_eval_count"),
        data.get("eval_count"),
    )
    return data


async def test_connection(ollama_url: str, model: str) -> bool:
    """Return True if Ollama is reachable and the model exists."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{ollama_url.rstrip('/')}/api/tags", timeout=5.0
            )
        resp.raise_for_status()
        tags = resp.json()
        model_names = [m["name"] for m in tags.get("models", [])]
        return any(m == model or m.startswith(model.split(":")[0]) for m in model_names)
    except Exception:  # noqa: BLE001
        return False
