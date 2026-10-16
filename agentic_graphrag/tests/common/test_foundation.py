"""Shared foundation: BaseAgent, LLMClient, Embedder, config, ReviewQueue, text utils."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from common.base_agent import AgentContext, AgentFailure, BaseAgent
from common.config import ExtractionConfig, load_config
from common.embedder import FakeEmbedder, TransientEmbeddingError, cosine
from common.llm import LLMError, MockLLM, TransientLLMError, build_llm, strict_json_schema
from common.review_queue import ReviewQueue
from common.text import count_tokens, detect_language, keywords, normalize_text, split_sentences
from common.validation import CheckResult, Severity, ValidationOutcome

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------- BaseAgent


class _In(BaseModel):
    value: int


class _Out(BaseModel):
    doubled: int
    validation: ValidationOutcome = ValidationOutcome()


class _LLMAnswer(BaseModel):
    answer: str = ""


class _Doubler(BaseAgent[_In, _Out]):
    name = "Doubler"
    algorithm = 0

    def _execute(self, data: _In) -> _Out:
        if data.value < 0:
            raise AgentFailure("negative input", error_type="BadInput")
        if data.value == 13:
            raise RuntimeError("unlucky")
        self.llm_json(task="t", system="s", prompt="p", schema=_LLMAnswer)
        return _Out(
            doubled=data.value * 2,
            validation=ValidationOutcome(checks=[CheckResult(name="a", passed=True)]),
        )


def test_base_agent_success_records_trace(ctx: AgentContext) -> None:
    result = _Doubler(ctx).run(_In(value=2), trace_context={"section": "s1"})
    assert result.ok and result.unwrap().doubled == 4
    trace = ctx.traces.records[-1]
    assert trace.agent == "Doubler" and trace.status == "ok" and trace.latency_ms >= 0
    assert trace.confidence == 1.0 and trace.context == {"section": "s1"}
    assert trace.token_usage.total > 0


def test_base_agent_typed_failure(ctx: AgentContext) -> None:
    result = _Doubler(ctx).run(_In(value=-1))
    assert not result.ok and result.error is not None
    assert result.error.error_type == "BadInput"
    assert ctx.traces.records[-1].status == "error"
    with pytest.raises(AgentFailure):
        result.unwrap()


def test_base_agent_unexpected_exception_becomes_error(ctx: AgentContext) -> None:
    result = _Doubler(ctx).run(_In(value=13))
    assert result.error is not None and result.error.error_type == "RuntimeError"


def test_flag_if_failed_writes_review_queue(ctx: AgentContext) -> None:
    agent = _Doubler(ctx)
    bad = ValidationOutcome(checks=[CheckResult(name="x", passed=False, reasons=["boom"])])
    assert agent.flag_if_failed(bad, "bad thing", item_ref="i1") is False
    assert agent.flag_if_failed(ValidationOutcome(), "fine") is True
    item = ctx.review_queue.items[-1]
    assert item.agent == "Doubler" and item.details["failures"] == ["x: boom"]


def test_validation_outcome_warnings_do_not_fail() -> None:
    outcome = ValidationOutcome(
        checks=[
            CheckResult(name="w", passed=False, severity=Severity.WARNING),
            CheckResult(name="e", passed=True),
        ]
    )
    assert outcome.passed and outcome.warnings and outcome.pass_ratio == 0.5


# ----------------------------------------------------------------- LLMClient


class _Nested(BaseModel):
    x: int = 0


class _Schema(BaseModel):
    name: str = ""
    items: list[_Nested] = []


def test_strict_json_schema_inlines_refs_and_requires_all() -> None:
    schema = strict_json_schema(_Schema)
    assert schema["required"] == ["name", "items"]
    assert schema["additionalProperties"] is False
    item = schema["properties"]["items"]["items"]
    assert item["required"] == ["x"] and "$ref" not in json.dumps(schema)


def test_mock_llm_default_and_callable() -> None:
    llm = MockLLM({"named": lambda prompt, _s, _i: {"name": prompt.upper()}})
    assert llm.complete_json(task="other", system="", prompt="", schema=_Schema).value.name == ""
    out = llm.complete_json(task="named", system="", prompt="hi", schema=_Schema)
    assert out.value.name == "HI" and llm.calls[-1]["task"] == "named"


def test_llm_retries_invalid_json_with_feedback() -> None:
    llm = MockLLM({"t": ["not json", {"name": "ok"}]})
    result = llm.complete_json(task="t", system="", prompt="base", schema=_Schema)
    assert result.value.name == "ok" and result.attempts == 2
    assert "not valid" in llm.calls[1]["prompt"]


def test_llm_retries_transient_errors_with_backoff() -> None:
    delays: list[float] = []
    llm = MockLLM({"t": [TransientLLMError("429"), TransientLLMError("503"), {"name": "x"}]})
    llm.max_retries, llm.backoff_base_s, llm._sleep = 3, 1.0, delays.append
    assert llm.complete_json(task="t", system="", prompt="", schema=_Schema).value.name == "x"
    assert len(delays) == 2 and delays[1] > delays[0]


def test_llm_gives_up_after_max_retries() -> None:
    llm = MockLLM({"t": [TransientLLMError("x")] * 5}, max_retries=2)
    with pytest.raises(LLMError, match="giving up after 3 attempts"):
        llm.complete_json(task="t", system="", prompt="", schema=_Schema)


def test_llm_permanent_error_is_not_retried() -> None:
    llm = MockLLM({"t": [LLMError("401"), {"name": "never"}]})
    with pytest.raises(LLMError, match="401"):
        llm.complete_json(task="t", system="", prompt="", schema=_Schema)
    assert len(llm.calls) == 1


def test_build_llm_mock() -> None:
    from common.config import LLMConfig

    assert isinstance(build_llm(LLMConfig(provider="mock")), MockLLM)


# ----------------------------------------------------------------- Embedder


def test_fake_embedder_is_deterministic_and_normalised() -> None:
    emb = FakeEmbedder(64)
    a, b, c = emb.embed(["pump pressure valve", "pump pressure valve", "guitar music concert"])
    assert np.allclose(a, b) and abs(float(np.linalg.norm(a)) - 1) < 1e-6
    assert cosine(a, b) > 0.99 > cosine(a, c)
    assert emb.embed([]).shape == (0, 64)
    assert cosine(np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32)) == 0.0


def test_embedder_retries_transient_errors_with_backoff() -> None:
    delays: list[float] = []

    class Flaky(FakeEmbedder):
        failures = 2

        def _embed_batch(self, texts):  # type: ignore[no-untyped-def]
            if Flaky.failures:
                Flaky.failures -= 1
                raise TransientEmbeddingError("429")
            return super()._embed_batch(texts)

    emb = Flaky(16)
    emb.max_retries, emb._sleep = 3, delays.append
    assert emb.embed(["pump"]).shape == (1, 16)
    assert delays == [2.0, 4.0]
    emb.max_retries, Flaky.failures = 0, 1
    with pytest.raises(TransientEmbeddingError):
        emb.embed(["new text"])


# ----------------------------------------------------------------- config


def test_project_config_yaml_is_valid() -> None:
    cfg = load_config(PROJECT_ROOT / "config.yaml")
    assert cfg.llm.provider == "azure_openai" and cfg.embedder.provider == "openrouter"
    assert Path(cfg.paths.schemas_dir).is_absolute()
    assert load_config(None) == ExtractionConfig()


@pytest.mark.parametrize(
    "data",
    [
        {"ocr_quality": {"weights": {"contrast": -1.0}}},
        {"ocr_quality": {"weights": {"made_up_metric": 1.0}}},
        {"confidence": {"medium_threshold": 0.9, "high_threshold": 0.8}},
        {"unknown_section": {}},
        {"classification": {"tau": 2.0}},
    ],
)
def test_config_rejects_invalid_values(tmp_path: Path, data: dict[str, object]) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(json.dumps(data))
    with pytest.raises(ValidationError):
        load_config(path)


# ----------------------------------------------------------------- ReviewQueue


def test_review_queue_appends_jsonl(tmp_path: Path) -> None:
    queue = ReviewQueue(tmp_path / "q" / "review.jsonl")
    queue.flag(agent="A", reason="r1", document_id="d", item_ref="x")
    queue.flag(agent="B", reason="r2")
    lines = (tmp_path / "q" / "review.jsonl").read_text().splitlines()
    assert [json.loads(line)["reason"] for line in lines] == ["r1", "r2"]
    assert [i.agent for i in ReviewQueue(tmp_path / "q" / "review.jsonl").read_all()] == ["A", "B"]


def test_review_queue_never_blocks_on_io_error(tmp_path: Path) -> None:
    queue = ReviewQueue(tmp_path)  # a directory: writing fails
    item = queue.flag(agent="A", reason="still recorded")
    assert queue.items == [item]


# ----------------------------------------------------------------- text


def test_text_utilities() -> None:
    assert normalize_text("ﬁle\x07  name\r\n") == "file name"
    assert count_tokens("Hello, world") == 3
    assert split_sentences("One here. Two there! three") == ["One here.", "Two there! three"]
    assert keywords("pump pump valve the and pump valve seal", 2) == ["pump", "valve"]
    assert detect_language("The pump and the valve of the system are in the plant.") == "en"
    assert detect_language("Die Pumpe und das Ventil ist nicht mit den Teilen von zu.") == "de"
    assert detect_language("x y") == "und"
