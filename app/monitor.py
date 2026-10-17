"""Production monitors for the assistant.

QueryDrift   kernel MMD between recent query embeddings and a reference window (permutation p-value), so a shift in
             what people ask (new topic, new phrasing, a policy change in the news) shows up before quality metrics do.
Rolling      rates over the last N answers: refusal, needs-review (ungrounded / uncited), gate refusals, latency p95.
"""
from __future__ import annotations

from collections import deque

import numpy as np


def _mmd2(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    def k(a, b):
        d = (a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * a @ b.T
        return np.exp(-gamma * np.maximum(d, 0))
    kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
    np.fill_diagonal(kxx, 0)
    np.fill_diagonal(kyy, 0)
    n, m = len(x), len(y)
    return float(kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2 * kxy.mean())


class QueryDrift:
    def __init__(self, reference: np.ndarray, n_perm: int = 200, seed: int = 0):
        self.ref = reference
        d = (reference * reference).sum(1)[:, None] + (reference * reference).sum(1)[None] - 2 * reference @ reference.T
        self.gamma = 1.0 / float(np.median(d[np.triu_indices(len(reference), 1)]))
        self.n_perm, self.rng = n_perm, np.random.default_rng(seed)

    def test(self, recent: np.ndarray) -> dict:
        stat = _mmd2(self.ref, recent, self.gamma)
        pool = np.concatenate([self.ref, recent])
        n = len(self.ref)
        null = []
        for _ in range(self.n_perm):
            p = self.rng.permutation(len(pool))
            null.append(_mmd2(pool[p[:n]], pool[p[n:]], self.gamma))
        pv = (1 + np.sum(np.asarray(null) >= stat)) / (self.n_perm + 1)
        return {"mmd2": stat, "p_value": float(pv), "alarm": bool(pv < 0.01)}


class Rolling:
    def __init__(self, n: int = 500):
        self.buf: deque = deque(maxlen=n)

    def add(self, refused: bool, refused_by: str | None, needs_review: bool, latency_ms: float):
        self.buf.append((refused, refused_by == "gate", needs_review, latency_ms))

    def snapshot(self) -> dict:
        if not self.buf:
            return {"n": 0}
        a = np.array(self.buf, dtype=float)
        return {"n": len(a), "refusal_rate": a[:, 0].mean(), "gate_refusal_rate": a[:, 1].mean(),
                "needs_review_rate": a[:, 2].mean(), "latency_p95_ms": float(np.percentile(a[:, 3], 95))}
