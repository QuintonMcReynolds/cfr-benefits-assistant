"""Retrieval over 38 CFR chunks: BM25, dense (bge-small), hybrid (reciprocal-rank fusion), plus candidate features
for the learning-to-rank reranker. Relevance is judged at the section level: any chunk of the gold section counts,
because the section is what gets cited."""
from __future__ import annotations

import re
import time
from functools import cached_property
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data" / "index"
EMB_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
STOP = set("the a an of to and or in for on by with is are be as at that this from any which who whom what when how does do "
           "can i my me if it its not no may shall will under than such".split())


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?|°|%", text.lower())
    return [t for t in toks if t not in STOP]


class Index:
    def __init__(self, db: Path = ROOT / "data" / "cfr.duckdb"):
        con = duckdb.connect(str(db), read_only=True)
        self.chunks = con.execute("SELECT chunk_id, section, part, heading, text, n_words FROM chunks ORDER BY chunk_id").df()
        con.close()
        self.toks = [tokenize(h + " " + t) for h, t in zip(self.chunks.heading, self.chunks.text, strict=True)]
        self.bm25 = BM25Okapi(self.toks)
        self.heading_toks = [set(tokenize(h)) for h in self.chunks.heading]
        self.table_frac = np.array([np.mean([("|" in ln) for ln in t.split("\n")]) for t in self.chunks.text])
        self.numbers = [set(re.findall(r"\d+", t)) for t in self.chunks.text]

    @cached_property
    def encoder(self):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(EMB_MODEL)

    @cached_property
    def emb(self) -> np.ndarray:
        INDEX.mkdir(parents=True, exist_ok=True)
        f = INDEX / "chunk_emb.npy"
        ids = INDEX / "chunk_ids.txt"
        if f.exists() and ids.exists() and ids.read_text().split("\n") == list(self.chunks.chunk_id):
            return np.load(f)
        docs = [f"{h}\n{t}" for h, t in zip(self.chunks.heading, self.chunks.text, strict=True)]
        e = self.encoder.encode(docs, batch_size=32, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        np.save(f, e)
        ids.write_text("\n".join(self.chunks.chunk_id))
        return e

    def encode_queries(self, qs: list[str]) -> np.ndarray:
        return self.encoder.encode([QUERY_PREFIX + q for q in qs], batch_size=64, normalize_embeddings=True,
                                   show_progress_bar=False).astype(np.float32)

    # ---- first-stage retrievers: return (indices, scores) best first
    def bm25_scores(self, q: str) -> np.ndarray:
        return self.bm25.get_scores(tokenize(q))

    def search(self, q: str, method: str = "hybrid", k: int = 10, qemb: np.ndarray | None = None):
        b = self.bm25_scores(q)
        d = self.emb @ (qemb if qemb is not None else self.encode_queries([q])[0])
        if method == "bm25":
            s = b
        elif method == "dense":
            s = d
        else:
            s = rrf([b, d])
        idx = np.argsort(-s)[:k]
        return idx, s[idx]

    def candidates(self, q: str, n: int = 50, qemb: np.ndarray | None = None) -> pd.DataFrame:
        """Union of BM25 / dense / hybrid top-n with features for the reranker."""
        t0 = time.perf_counter()
        b = self.bm25_scores(q)
        d = self.emb @ (qemb if qemb is not None else self.encode_queries([q])[0])
        h = rrf([b, d])
        rb, rd, rh = _ranks(b), _ranks(d), _ranks(h)
        pool = np.unique(np.r_[np.argsort(-b)[:n], np.argsort(-d)[:n], np.argsort(-h)[:n]])
        qt = set(tokenize(q))
        qnum = set(re.findall(r"\d+", q))
        sec = self.chunks.section.to_numpy()
        # section-level aggregates over the pool: best dense / bm25 of any chunk in the same section
        sec_best_d = pd.Series(d[pool]).groupby(sec[pool]).transform("max").to_numpy()
        sec_best_b = pd.Series(b[pool]).groupby(sec[pool]).transform("max").to_numpy()
        f = pd.DataFrame({
            "chunk_idx": pool, "chunk_id": self.chunks.chunk_id.to_numpy()[pool], "section": sec[pool],
            "bm25": b[pool], "bm25_rank": rb[pool], "bm25_norm": b[pool] / (b.max() + 1e-9),
            "dense": d[pool], "dense_rank": rd[pool], "dense_gap": d.max() - d[pool],
            "rrf": h[pool], "rrf_rank": rh[pool],
            "sec_best_dense": sec_best_d, "sec_best_bm25": sec_best_b / (b.max() + 1e-9),
            "heading_overlap": [len(qt & self.heading_toks[i]) / (len(qt) + 1e-9) for i in pool],
            "query_overlap": [len(qt & set(self.toks[i])) / (len(qt) + 1e-9) for i in pool],
            "number_match": [len(qnum & self.numbers[i]) for i in pool],
            "table_frac": self.table_frac[pool], "n_words": self.chunks.n_words.to_numpy()[pool],
            "part4": (self.chunks.part.to_numpy()[pool] == "4").astype(int), "q_len": len(qt),
        })
        f.attrs["latency_s"] = time.perf_counter() - t0
        return f


FEATURES = ["bm25", "bm25_rank", "bm25_norm", "dense", "dense_rank", "dense_gap", "rrf", "rrf_rank", "sec_best_dense",
            "sec_best_bm25", "heading_overlap", "query_overlap", "number_match", "table_frac", "n_words", "part4", "q_len"]


def _ranks(s: np.ndarray) -> np.ndarray:
    r = np.empty(len(s), int)
    r[np.argsort(-s)] = np.arange(len(s))
    return r


def rrf(score_lists: list[np.ndarray], k: int = 60) -> np.ndarray:
    out = np.zeros(len(score_lists[0]))
    for s in score_lists:
        out += 1.0 / (k + _ranks(s) + 1)
    return out


def sections_ranked(chunk_sections: list[str]) -> list[str]:
    """Collapse a ranked chunk list to a ranked, de-duplicated section list."""
    return list(dict.fromkeys(chunk_sections))
