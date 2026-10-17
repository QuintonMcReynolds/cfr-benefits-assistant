import numpy as np
import pandas as pd

from app.answer import CITE_RE, Assistant
from app.monitor import QueryDrift, Rolling


class FakeLLM:
    def __init__(self, text):
        self.text = text

    def chat(self, messages, max_tokens=0, temperature=0.0):
        return {"text": self.text, "prompt_tokens": 100, "completion_tokens": 20, "latency_s": 0.01}


class FakeIndex:
    chunks = pd.DataFrame({"heading": ["§ 3.304 Direct service connection", "§ 3.102 Reasonable doubt"],
                           "text": ["Service connection means ...", "When there is an approximate balance ..."]})

    def candidates(self, q):
        c = pd.DataFrame({"chunk_idx": [0, 1], "section": ["3.304", "3.102"], "rrf_rank": [0, 1]})
        c.attrs["latency_s"] = 0.001
        return c


def _assistant(text, refuse_below=None, top=None):
    a = Assistant.__new__(Assistant)
    a.ix, a.k, a.refuse_below, a.reranker, a.ltr, a.model_id = FakeIndex(), 4, refuse_below, "hybrid", None, "fake"
    a._llm = FakeLLM(text)
    return a


def test_citation_regex():
    assert CITE_RE.findall("see [§3.304] and [§ 4.71a]") == ["3.304", "4.71a"]


def test_grounded_answer_passes():
    a = _assistant("Service connection requires ... [§3.304]")("what is service connection?")
    assert a.citations == ["3.304"] and not a.ungrounded_citations and not a.needs_review and not a.refused


def test_invented_citation_is_flagged():
    a = _assistant("You get 100% under [§4.130] and [§3.304].")("q")
    assert a.ungrounded_citations == ["4.130"] and a.needs_review


def test_uncited_answer_needs_review_and_refusal_detected():
    assert _assistant("Yes, you qualify.")("q").needs_review
    r = _assistant("I couldn't find this in 38 CFR Parts 3-4. Check VA.gov.")("q")
    assert r.refused and r.refused_by == "model" and not r.needs_review


def test_gate_refuses_without_calling_the_model():
    a = _assistant("should not be used", refuse_below=10.0)
    r = a("q")
    assert r.refused and r.refused_by == "gate" and r.tokens["prompt"] == 0


def test_drift_detects_shift_not_noise():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, (150, 8))
    d = QueryDrift(ref, n_perm=100)
    assert not d.test(rng.normal(0, 1, (80, 8)))["alarm"]
    assert d.test(rng.normal(0.8, 1, (80, 8)))["alarm"]


def test_rolling_metrics():
    r = Rolling(10)
    r.add(True, "gate", False, 100)
    r.add(False, None, True, 900)
    s = r.snapshot()
    assert s["n"] == 2 and s["refusal_rate"] == 0.5 and s["gate_refusal_rate"] == 0.5 and s["needs_review_rate"] == 0.5
