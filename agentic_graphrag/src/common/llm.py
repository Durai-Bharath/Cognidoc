"""Provider-agnostic, JSON-schema-constrained LLM access.

``LLMClient.complete_json`` returns a validated Pydantic object. Retries with
exponential backoff cover transient provider errors *and* malformed/invalid JSON
(the validation error is fed back to the model on the next attempt).
"""

from __future__ import annotations

import base64
import json
import os
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from common.base_agent import TokenUsage
from common.config import LLMConfig
from common.logging import get_logger

T = TypeVar("T", bound=BaseModel)
_log = get_logger(__name__)


class LLMError(RuntimeError):
    """Permanent failure (bad request, auth, refusal, retries exhausted)."""


class TransientLLMError(LLMError):
    """Retryable failure (rate limit, timeout, 5xx, connection reset)."""


class InvalidLLMOutput(LLMError):
    """The response was not valid JSON for the requested schema (retryable)."""


class LLMResult(BaseModel, Generic[T]):
    value: T
    usage: TokenUsage
    model: str
    attempts: int = 1


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic schema → strict structured-output schema.

    Inlines ``$ref``/``$defs``, marks every property required, forbids additional
    properties and strips keywords structured-output validators reject.
    """
    raw = model.model_json_schema()
    defs: dict[str, Any] = raw.pop("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(n) for n in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            ref = node["$ref"].split("/")[-1]
            return walk(deepcopy(defs[ref]))
        out = {k: walk(v) for k, v in node.items() if k not in {"default", "title", "examples"}}
        if out.get("type") == "object" and "properties" in out:
            out["required"] = list(out["properties"].keys())
            out["additionalProperties"] = False
        return out

    result: dict[str, Any] = walk(raw)
    return result


def _image_media_type(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] in (b"GIF8",):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


class LLMClient(ABC):
    """Base class: subclasses only implement :meth:`_complete_raw`."""

    def __init__(
        self,
        *,
        max_retries: int = 3,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self._sleep = sleep

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    def _complete_raw(
        self,
        *,
        task: str,
        system: str,
        prompt: str,
        schema: dict[str, Any],
        schema_name: str,
        images: Sequence[bytes],
    ) -> tuple[str, TokenUsage]:
        """Return the raw JSON text and token usage for one attempt."""

    def complete_json(
        self,
        *,
        task: str,
        system: str,
        prompt: str,
        schema: type[T],
        images: Sequence[bytes] = (),
    ) -> LLMResult[T]:
        strict = strict_json_schema(schema)
        usage = TokenUsage()
        attempt_prompt = prompt
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                raw, step_usage = self._complete_raw(
                    task=task,
                    system=system,
                    prompt=attempt_prompt,
                    schema=strict,
                    schema_name=schema.__name__,
                    images=images,
                )
                usage.add(step_usage)
                try:
                    value = schema.model_validate_json(_strip_fences(raw))
                except ValidationError as exc:
                    raise InvalidLLMOutput(str(exc)) from exc
                return LLMResult[T](
                    value=value, usage=usage, model=self.model_name, attempts=attempt + 1
                )
            except InvalidLLMOutput as exc:
                last_error = exc
                attempt_prompt = (
                    f"{prompt}\n\nYour previous answer was not valid for the required JSON "
                    f"schema ({str(exc)[:500]}). Return only JSON matching the schema."
                )
            except TransientLLMError as exc:
                last_error = exc
            if attempt < self.max_retries:
                delay = min(self.backoff_base_s * (2**attempt), self.backoff_max_s)
                delay += random.uniform(0, self.backoff_base_s / 2) if delay else 0.0
                _log.warning(
                    "llm_retry",
                    extra={
                        "task": task,
                        "attempt": attempt + 1,
                        "delay_s": round(delay, 2),
                        "error": str(last_error)[:200],
                    },
                )
                self._sleep(delay)
        raise LLMError(
            f"{task}: giving up after {self.max_retries + 1} attempts: {last_error}"
        ) from last_error


def _strip_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0]
    return text.strip()


# --------------------------------------------------------------------------- Anthropic


class AnthropicLLM(LLMClient):
    """Claude via the official SDK, using ``output_config.format`` structured outputs."""

    def __init__(self, config: LLMConfig, *, api_key: str | None = None) -> None:
        super().__init__(
            max_retries=config.max_retries,
            backoff_base_s=config.backoff_base_s,
            backoff_max_s=config.backoff_max_s,
        )
        import anthropic

        self._anthropic = anthropic
        self.config = config
        # Retries are owned by LLMClient so they also cover invalid JSON.
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            max_retries=0,
            timeout=config.timeout_s,
        )

    @property
    def model_name(self) -> str:
        return self.config.model

    def _complete_raw(
        self,
        *,
        task: str,
        system: str,
        prompt: str,
        schema: dict[str, Any],
        schema_name: str,
        images: Sequence[bytes],
    ) -> tuple[str, TokenUsage]:
        anthropic = self._anthropic
        content: list[dict[str, Any]] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": _image_media_type(img),
                    "data": base64.standard_b64encode(img).decode("ascii"),
                },
            }
            for img in images
        ]
        content.append({"type": "text", "text": prompt})
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": content}],
            "output_config": {
                "effort": self.config.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
        }
        try:
            if self.config.anthropic_fallbacks:
                response = self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
                )
            else:
                response = self.client.messages.create(**kwargs)
        except (anthropic.RateLimitError, anthropic.InternalServerError) as exc:
            raise TransientLLMError(str(exc)) from exc
        except anthropic.APIConnectionError as exc:  # includes timeouts
            raise TransientLLMError(str(exc)) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500 or exc.status_code in (408, 409, 429):
                raise TransientLLMError(str(exc)) from exc
            raise LLMError(f"{exc.status_code}: {exc.message}") from exc

        usage = TokenUsage(
            input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens
        )
        if response.stop_reason == "refusal":
            raise LLMError(f"{task}: request refused by the model")
        if response.stop_reason == "max_tokens":
            raise InvalidLLMOutput("response truncated at max_tokens")
        text = next((b.text for b in response.content if b.type == "text"), "")
        return text, usage


# --------------------------------------------------------------------------- OpenAI-compatible


class OpenAICompatibleLLM(LLMClient):
    """Azure OpenAI or OpenRouter through the ``openai`` SDK with json_schema output."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(
            max_retries=config.max_retries,
            backoff_base_s=config.backoff_base_s,
            backoff_max_s=config.backoff_max_s,
        )
        import openai

        self._openai = openai
        self.config = config
        self._json_schema_supported = True
        if config.provider == "azure_openai":
            self.client: Any = openai.AzureOpenAI(
                azure_endpoint=_require_env("AZURE_OPENAI_ENDPOINT"),
                api_key=_require_env("AZURE_OPENAI_API_KEY"),
                api_version=_require_env("OPENAI_API_VERSION"),
                max_retries=0,
                timeout=config.timeout_s,
            )
            self._model = config.model or _require_env("AZURE_OPENAI_DEPLOYMENT")
        else:
            self.client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=_require_env("OPENROUTER_API_KEY"),
                max_retries=0,
                timeout=config.timeout_s,
            )
            self._model = config.model

    @property
    def model_name(self) -> str:
        return self._model

    def _complete_raw(
        self,
        *,
        task: str,
        system: str,
        prompt: str,
        schema: dict[str, Any],
        schema_name: str,
        images: Sequence[bytes],
    ) -> tuple[str, TokenUsage]:
        openai = self._openai
        user_text = prompt
        if not self._json_schema_supported:
            user_text += "\n\nRespond with JSON matching this schema:\n" + json.dumps(schema)
        parts: list[dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{_image_media_type(img)};base64,"
                    + base64.standard_b64encode(img).decode("ascii")
                },
            }
            for img in images
        ]
        parts.append({"type": "text", "text": user_text})
        response_format: dict[str, Any] = (
            {
                "type": "json_schema",
                "json_schema": {"name": schema_name[:64], "schema": schema, "strict": True},
            }
            if self._json_schema_supported
            else {"type": "json_object"}
        )
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": parts},
            ],
            "response_format": response_format,
            "max_completion_tokens": self.config.max_tokens,
        }
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        try:
            response = self.client.chat.completions.create(**kwargs)
        except openai.BadRequestError as exc:
            message = str(exc)
            if self._json_schema_supported and "response_format" in message:
                self._json_schema_supported = False
                raise TransientLLMError(
                    "json_schema unsupported; retrying with json_object"
                ) from exc
            if "temperature" in message and self.config.temperature is not None:
                self.config = self.config.model_copy(update={"temperature": None})
                raise TransientLLMError("temperature unsupported; retrying without") from exc
            raise LLMError(message) from exc
        except (
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.InternalServerError,
        ) as exc:
            raise TransientLLMError(str(exc)) from exc
        except openai.APIStatusError as exc:
            if exc.status_code >= 500 or exc.status_code in (408, 409, 429):
                raise TransientLLMError(str(exc)) from exc
            raise LLMError(str(exc)) from exc
        usage = TokenUsage()
        if response.usage is not None:
            usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens or 0,
                output_tokens=response.usage.completion_tokens or 0,
            )
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise InvalidLLMOutput("response truncated at max tokens")
        return choice.message.content or "", usage


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise LLMError(f"environment variable {name} is not set")
    return value


# --------------------------------------------------------------------------- Mock

Handler = Callable[[str, str, Sequence[bytes]], Mapping[str, Any] | BaseModel | str]


class MockLLM(LLMClient):
    """Deterministic offline LLM for tests.

    ``responses[task]`` may be a dict / model / raw string, a callable
    ``(prompt, system, images) -> ...``, or a list consumed one item per call. Tasks
    without a response return the schema's defaults (every LLM output schema in this
    project has defaults for all fields). ``calls`` records every request.
    """

    def __init__(
        self,
        responses: Mapping[str, Any] | None = None,
        *,
        max_retries: int = 2,
    ) -> None:
        super().__init__(max_retries=max_retries, backoff_base_s=0.0, sleep=lambda _s: None)
        self.responses: dict[str, Any] = dict(responses or {})
        self.calls: list[dict[str, Any]] = []

    @property
    def model_name(self) -> str:
        return "mock-llm"

    def _complete_raw(
        self,
        *,
        task: str,
        system: str,
        prompt: str,
        schema: dict[str, Any],
        schema_name: str,
        images: Sequence[bytes],
    ) -> tuple[str, TokenUsage]:
        self.calls.append(
            {
                "task": task,
                "system": system,
                "prompt": prompt,
                "images": len(images),
                "schema": schema_name,
            }
        )
        spec = self.responses.get(task)
        if isinstance(spec, list):
            spec = spec.pop(0) if spec else None
        if isinstance(spec, Exception):
            raise spec
        if callable(spec):
            spec = spec(prompt, system, images)
        usage = TokenUsage(input_tokens=len(prompt) // 4, output_tokens=16)
        if spec is None:
            return "{}", usage
        if isinstance(spec, BaseModel):
            return spec.model_dump_json(), usage
        if isinstance(spec, str):
            return spec, usage
        return json.dumps(dict(spec)), usage


def build_llm(config: LLMConfig) -> LLMClient:
    if config.provider == "anthropic":
        return AnthropicLLM(config)
    if config.provider in ("azure_openai", "openrouter"):
        return OpenAICompatibleLLM(config)
    return MockLLM()
