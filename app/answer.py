"""Answering: retrieve -> rerank -> (refuse early | generate with citations) -> citation checks."""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np

from .retrieval import FEATURES, Index

ROOT = Path(__file__).resolve().parents[1]

SYSTEM = """You answer questions about U.S. veterans' benefits regulations (38 CFR Parts 3 and 4) for veterans and their families.
Use ONLY the regulation excerpts provided. After every statement, cite the section it came from in square brackets, like [§3.304].
If the excerpts do not answer the question, say: "I couldn't find this in 38 CFR Parts 3-4." and suggest checking VA.gov or an accredited representative. Do not guess.
Write in plain language, at most 120 words. This is general information, not legal advice."""

REFUSAL = ("I couldn't find this in 38 CFR Parts 3-4, the regulations this assistant covers. VA.gov or an accredited "
           "Veterans Service Organization representative can help with it.")
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CITE_RE = re.compile(r"§\s*(\d+\.\d+[a-z]?)")


@dataclass
class Answer:
    question: str
    text: str
    refused: bool
    refused_by: str | None                 # "gate" (retrieval confidence) | "model" | None
    citations: list[str]
    retrieved_sections: list[str]
    ungrounded_citations: list[str]        # cited but not among the excerpts the model was shown
    needs_review: bool
    top_score: float
    latency_ms: dict = field(default_factory=dict)
    tokens: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class Assistant:
    """reranker: 'hybrid' (RRF order), 'ltr' (LambdaRank), 'ce' (cross-encoder on the hybrid top 30), or 'ltr_ce'
    (LambdaRank with the cross-encoder score as a feature). refuse_below gates on the top reranker score."""

    def __init__(self, index: Index | None = None, model_id: str = "mlx-community/Qwen2.5-7B-Instruct-4bit",
                 reranker: str = "ce", k_context: int = 4, refuse_below: float | None = None):
        self.ix = index or Index()
        self.model_id, self.k, self.refuse_below, self.reranker = model_id, k_context, refuse_below, reranker
        self.ltr = lgb.Booster(model_file=str(ROOT / "runs" / f"{reranker}.txt")) if reranker in ("ltr", "ltr_ce") else None
        self._ce = None
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            from .llm import LLM
            self._llm = LLM(self.model_id)
        return self._llm

    @property
    def ce(self):
        if self._ce is None:
            from sentence_transformers import CrossEncoder
            self._ce = CrossEncoder(CE_MODEL)
        return self._ce

    def _ce_scores(self, q: str, c):
        top = c.nsmallest(30, "rrf_rank")
        sc = self.ce.predict([(q, self.ix.chunks.heading.iloc[i] + "\n" + self.ix.chunks.text.iloc[i]) for i in top.chunk_idx],
                             batch_size=32, show_progress_bar=False)
        c = c.copy()
        c["ce"] = -10.0
        c.loc[top.index, "ce"] = sc
        return c

    def retrieve(self, q: str):
        t0 = time.perf_counter()
        c = self.ix.candidates(q)
        if self.reranker in ("ce", "ltr_ce"):
            c = self._ce_scores(q, c)
        if self.reranker in ("ltr", "ltr_ce"):
            assert self.ltr is not None
            cols = FEATURES + (["ce"] if self.reranker == "ltr_ce" else [])
            c["score"] = self.ltr.predict(c[cols].to_numpy())
        elif self.reranker == "ce":
            c["score"] = c["ce"]
        else:
            c["score"] = -c["rrf_rank"].astype(float)
        c = c.sort_values("score", ascending=False)
        top = c.drop_duplicates("section").head(self.k)   # one chunk per section: more distinct sources in context
        return top, float(c.score.iloc[0]), (time.perf_counter() - t0) * 1000

    def __call__(self, q: str) -> Answer:
        top, top_score, t_ret = self.retrieve(q)
        secs = top.section.tolist()
        if self.refuse_below is not None and top_score < self.refuse_below:
            return Answer(q, REFUSAL, True, "gate", [], secs, [], False, top_score, {"retrieve": t_ret, "generate": 0.0},
                          {"prompt": 0, "completion": 0})
        ctx = "\n\n".join(f"[§{r.section}] {self.ix.chunks.heading.iloc[r.chunk_idx]}\n{self.ix.chunks.text.iloc[r.chunk_idx]}"
                          for r in top.itertuples())
        out = self.llm.chat([{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": f"Regulation excerpts:\n{ctx}\n\nQuestion: {q}"}], max_tokens=260)
        text = out["text"]
        cites = list(dict.fromkeys(CITE_RE.findall(text)))
        refused = "couldn't find this in 38 cfr" in text.lower()
        ungrounded = [c for c in cites if c not in secs]
        needs_review = bool(ungrounded) or (not refused and not cites)
        return Answer(q, text, refused, "model" if refused else None, cites, secs, ungrounded, needs_review, top_score,
                      {"retrieve": t_ret, "generate": out["latency_s"] * 1000},
                      {"prompt": out["prompt_tokens"], "completion": out["completion_tokens"]})


def refusal_threshold(reranker: str = "ltr_ce") -> float | None:  # only LambdaRank scores are calibrated
    import json
    p = ROOT / "reports" / "retrieval.json"
    return json.loads(p.read_text())["refusal"].get(f"threshold_{reranker}") if p.exists() else None


__all__ = ["Assistant", "Answer", "REFUSAL", "np"]
