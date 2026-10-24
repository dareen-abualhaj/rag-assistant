import json

import numpy as np
import pytest

from src.llm import LLMConfigError, LLMUnavailableError
from src.prompts import NO_ANSWER
from src.rag import RAGPipeline
import src.rag as rag_module

GOOD = "Wings create lift for the airplane. [1]"
NO_CITATION = "Wings create lift for the airplane."
HALLUCINATION = "Seagulls flap their wings to hover in air. [1]"
BAD_CITATION = "Wings create lift for the airplane. [9]"


class FakeEmbedder:
    """Texts mentioning 'lift' -> [1,0]; everything else -> [0,1] (orthogonal)."""
    def encode(self, texts, normalize_embeddings=True):
        return np.array([[1.0, 0.0] if "lift" in t.lower() else [0.0, 1.0] for t in texts])


class FakeLLM:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def _hit(distance=0.3):
    return {"text": "Wings create lift that keeps the airplane in the air.",
            "metadata": {"title": "Aircraft", "url": "https://x/Aircraft"},
            "distance": distance}


@pytest.fixture
def make(monkeypatch, tmp_path):
    def _make(llm, distance=0.3):
        monkeypatch.setattr(rag_module, "search", lambda coll, emb, k: [_hit(distance)])
        return RAGPipeline(embedder=FakeEmbedder(), collection=object(), llm=llm,
                           log_path=str(tmp_path / "logs" / "latency.jsonl"))
    return _make


def test_empty_question_raises(make):
    with pytest.raises(ValueError):
        make(FakeLLM()).answer("   ")


def test_out_of_domain_does_not_call_llm(make):
    llm = FakeLLM()
    r = make(llm, distance=0.9).answer("Write a python sort function")
    assert r["status"] == "out_of_domain"
    assert r["answer"] == NO_ANSWER
    assert llm.calls == []
    assert r["latency"]["generation_ms"] == 0.0


def test_good_answer_is_accepted(make):
    llm = FakeLLM(GOOD)
    r = make(llm).answer("How do airplanes fly?")
    assert r["status"] == "answered"
    assert r["answer"] == GOOD
    assert len(llm.calls) == 1
    assert r["sources"][0]["title"] == "Aircraft"


def test_missing_citation_is_retried_and_fixed(make):
    llm = FakeLLM(NO_CITATION, GOOD)
    r = make(llm).answer("How do airplanes fly?")
    assert r["status"] == "answered"
    assert len(llm.calls) == 2
    assert "rejected" in llm.calls[1][0]["content"]  # strict prompt used on retry


def test_missing_citation_twice_is_refused(make):
    llm = FakeLLM(NO_CITATION, NO_CITATION)
    r = make(llm).answer("How do airplanes fly?")
    assert r["status"] == "unverified"
    assert "can't answer this reliably" in r["answer"]
    assert r["rejected_answer"] == NO_CITATION
    assert any("citation" in w for w in r["warnings"])


def test_invalid_citation_number_is_refused(make):
    r = make(FakeLLM(BAD_CITATION, BAD_CITATION)).answer("How do airplanes fly?")
    assert r["status"] == "unverified"


def test_hallucinated_sentence_is_refused(make):
    r = make(FakeLLM(HALLUCINATION, HALLUCINATION)).answer("How do airplanes fly?")
    assert r["status"] == "unverified"
    assert any("not supported" in w for w in r["warnings"])
    assert HALLUCINATION not in r["answer"]  # user never sees the hallucination


def test_model_no_answer_is_passed_through_without_retry(make):
    llm = FakeLLM(NO_ANSWER)
    r = make(llm).answer("Something obscure")
    assert r["status"] == "no_answer"
    assert len(llm.calls) == 1


def test_llm_unavailable_returns_error_status(make):
    r = make(FakeLLM(LLMUnavailableError("down"))).answer("How do airplanes fly?")
    assert r["status"] == "error"
    assert "temporarily unavailable" in r["answer"]
    assert len(r["sources"]) == 1  # sources still shown


def test_llm_config_error_has_distinct_message(make):
    r = make(FakeLLM(LLMConfigError("bad key"))).answer("How do airplanes fly?")
    assert r["status"] == "error"
    assert "misconfigured" in r["answer"]


def test_retry_failure_keeps_refusal(make):
    r = make(FakeLLM(NO_CITATION, LLMUnavailableError("down"))).answer("How do airplanes fly?")
    assert r["status"] == "unverified"


def test_latency_is_logged_to_jsonl(make, tmp_path):
    p = make(FakeLLM(GOOD))
    p.answer("How do airplanes fly?")
    lines = (tmp_path / "logs" / "latency.jsonl").read_text().strip().splitlines()
    rec = json.loads(lines[0])
    assert rec["status"] == "answered"
    for key in ("retrieval_ms", "generation_ms", "verification_ms", "total_ms"):
        assert key in rec