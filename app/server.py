"""FastAPI service.

    uvicorn app.server:app --port 8000
POST /ask {"question": "..."} -> answer with citations, refusal flag, review flag, latency breakdown
GET  /health, GET /metrics (rolling quality + latency), POST /drift (recent questions vs reference)
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .answer import Assistant, refusal_threshold
from .monitor import QueryDrift, Rolling

app = FastAPI(title="cfr-benefits-assistant", version="0.1.0",
              description="Grounded Q&A over 38 CFR Parts 3-4. General information, not legal advice. Not affiliated with VA.")
ROLLING = Rolling()


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class DriftReq(BaseModel):
    questions: list[str] = Field(min_length=20, max_length=2000)


@lru_cache(maxsize=1)
def assistant() -> Assistant:
    # default = best end-to-end configuration (reports/answers.json): cross-encoder reranking, no retrieval gate
    # (the model refuses unanswerable questions itself; the gate cost more answerable questions than it saved tokens).
    # CFR_RERANKER=ltr_ce CFR_GATE=1 enables the calibrated gate where token cost matters more.
    reranker = os.environ.get("CFR_RERANKER", "ce")
    gate = os.environ.get("CFR_GATE", "0") == "1" and reranker in ("ltr", "ltr_ce")
    return Assistant(model_id=os.environ.get("CFR_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit"), reranker=reranker,
                     refuse_below=refusal_threshold(reranker) if gate else None)


@lru_cache(maxsize=1)
def drift_ref() -> QueryDrift:
    import pandas as pd

    from .answer import ROOT
    q = pd.read_json(ROOT / "data" / "eval" / "questions.jsonl", lines=True)
    ref = q[(q.split == "train") & q.answerable].question.sample(300, random_state=0).tolist()
    return QueryDrift(assistant().ix.encode_queries(ref))


READY = {"ok": False}


@app.on_event("startup")
def warm_up():
    """Load the embedding model, cross-encoder and reranker before taking traffic: the first live test spent 36 s of
    its first request loading them lazily. The LLM is warmed only for the local backend (a remote endpoint is not ours to load)."""
    a = assistant()
    a.retrieve("warm-up query about service connection")
    if os.environ.get("CFR_LLM_BACKEND", "mlx") == "mlx" and os.environ.get("CFR_WARM_LLM", "1") == "1":
        _ = a.llm
    READY["ok"] = True


@app.get("/health")
def health():
    return {"status": "ok" if READY["ok"] else "warming"}


@app.post("/ask")
def ask(req: Ask):
    a = assistant()(req.question)
    ROLLING.add(a.refused, a.refused_by, a.needs_review, sum(a.latency_ms.values()))
    return {**a.to_dict(), "disclaimer": "General information from 38 CFR, not legal advice. Not affiliated with VA."}


@app.get("/metrics")
def metrics():
    return ROLLING.snapshot()


@app.post("/drift")
def drift(req: DriftReq):
    if len(req.questions) < 20:
        raise HTTPException(400, "need at least 20 questions")
    e = assistant().ix.encode_queries(req.questions)
    return drift_ref().test(np.asarray(e))
