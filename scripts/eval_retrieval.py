"""Retrieval + reranking evaluation, section-level relevance.

Methods: bm25, dense, hybrid (RRF), hybrid -> cross-encoder (top 30), hybrid pool -> LightGBM LambdaRank (LTR),
LTR with the cross-encoder score as an extra feature. LTR trains on 'train' questions (disjoint sections from test),
early-stopped on 20 % of the train sections. Refusal threshold on the top reranker score is set from *answerable*
train questions only (5th percentile), then tested on the hand-written unanswerable questions.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.retrieval import FEATURES, Index, sections_ranked  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KS = (1, 3, 5, 10)


def load_questions() -> pd.DataFrame:
    return pd.read_json(ROOT / "data" / "eval" / "questions.jsonl", lines=True)


def section_metrics(ranked_secs: list[str], gold: str) -> dict:
    r = {f"recall@{k}": float(gold in ranked_secs[:k]) for k in KS}
    r["mrr@10"] = next((1.0 / (i + 1) for i, s in enumerate(ranked_secs[:10]) if s == gold), 0.0)
    return r


def build_candidates(ix: Index, qs: pd.DataFrame, ce=None) -> pd.DataFrame:
    qemb = ix.encode_queries(qs.question.tolist())
    rows = []
    for (_, q), e in zip(qs.iterrows(), qemb, strict=True):
        c = ix.candidates(q.question, qemb=e)
        c["qid"], c["label"] = q.qid, (c.section == q.gold_section).astype(int) if q.answerable else 0
        c["cand_ms"] = c.attrs["latency_s"] * 1000
        if ce is not None:
            top = c.nsmallest(30, "rrf_rank")
            t0 = time.perf_counter()
            sc = ce.predict([(q.question, ix.chunks.heading.iloc[i] + "\n" + ix.chunks.text.iloc[i]) for i in top.chunk_idx],
                            batch_size=32, show_progress_bar=False)
            c["ce"] = -10.0
            c.loc[top.index, "ce"] = sc
            c["ce_ms"] = (time.perf_counter() - t0) * 1000
        rows.append(c)
    return pd.concat(rows, ignore_index=True)


def fit_ltr(cand: pd.DataFrame, feats: list[str], seed: int = 0):
    qids = cand.qid.unique()
    secs = pd.Series(qids).str.split(":").str[0]
    rng = np.random.default_rng(seed)
    val_secs = set(rng.choice(secs.unique(), int(0.2 * secs.nunique()), replace=False))
    is_val = cand.qid.str.split(":").str[0].isin(val_secs)
    tr, va = cand[~is_val].sort_values("qid"), cand[is_val].sort_values("qid")
    m = lgb.LGBMRanker(objective="lambdarank", n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=seed)
    m.fit(tr[feats], tr.label, group=tr.groupby("qid", sort=True).size().to_numpy(),
          eval_set=[(va[feats], va.label)], eval_group=[va.groupby("qid", sort=True).size().to_numpy()],
          eval_at=[5], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


def oof_scores(cand: pd.DataFrame, feats: list[str], k: int = 5, seed: int = 0) -> np.ndarray:
    """Score every train candidate with a ranker fit on the other folds (folds by section, so no question's section is
    seen by the model that scores it)."""
    sec = cand.qid.str.split(":").str[0]
    secs = sec.unique()
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(secs, rng.integers(0, k, len(secs)), strict=True))
    fold = sec.map(fold_of).to_numpy()
    out = np.zeros(len(cand))
    for f in range(k):
        m = fit_ltr(cand[fold != f], feats, seed)
        out[fold == f] = m.predict(cand.loc[fold == f, feats])
    return out


def rank_by(c: pd.DataFrame, col: str, ascending: bool = False) -> list[str]:
    return sections_ranked(c.sort_values(col, ascending=ascending).section.tolist())


def main():
    ix = Index()
    qs = load_questions()
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    tr_q, te_q = qs[(qs.split == "train") & qs.answerable], qs[qs.split == "test"]
    t0 = time.time()
    cand_tr, cand_te = build_candidates(ix, tr_q, ce), build_candidates(ix, te_q, ce)
    print(f"candidates built in {time.time() - t0:.0f}s", flush=True)
    ltr = fit_ltr(cand_tr, FEATURES)
    ltr_ce = fit_ltr(cand_tr, FEATURES + ["ce"])
    for name, m, f in (("ltr", ltr, FEATURES), ("ltr_ce", ltr_ce, FEATURES + ["ce"])):
        cand_te[name] = m.predict(cand_te[f])
        cand_tr[name] = m.predict(cand_tr[f])
        m.booster_.save_model(str(ROOT / "runs" / f"{name}.txt"))
    rows = []
    for qid, c in cand_te.groupby("qid"):
        q = te_q.set_index("qid").loc[qid]
        if not q.answerable:
            continue
        rankings = {"bm25": rank_by(c, "bm25_rank", True), "dense": rank_by(c, "dense_rank", True),
                    "hybrid": rank_by(c, "rrf_rank", True), "hybrid+cross-encoder": rank_by(c, "ce"),
                    "hybrid+LTR": rank_by(c, "ltr"), "hybrid+LTR(+ce feature)": rank_by(c, "ltr_ce")}
        for meth, r in rankings.items():
            rows.append({"qid": qid, "style": q["style"], "method": meth, **section_metrics(r, q.gold_section)})
    res = pd.DataFrame(rows)
    summ = res.groupby(["method", "style"]).mean(numeric_only=True).round(3)
    overall = res.groupby("method").mean(numeric_only=True).round(3)
    # refusal: top reranker score, threshold from answerable TRAIN questions only
    # Calibrate refusal on OUT-OF-FOLD scores: a model scoring the questions it was trained on is overconfident, which
    # put the first threshold too high (29 % of answerable test questions refused instead of the intended ~5 %).
    for name, f in (("ltr", FEATURES), ("ltr_ce", FEATURES + ["ce"])):
        cand_tr[f"{name}_oof"] = oof_scores(cand_tr, f)
    refusal = {"calibration": "5th percentile of out-of-fold top scores on answerable train questions (5-fold by section)"}
    for name in ("ltr", "ltr_ce"):
        top_tr = cand_tr.groupby("qid")[f"{name}_oof"].max()
        thr = float(np.quantile(top_tr, 0.05))
        top_te = cand_te.groupby("qid")[name].max()
        ans = te_q.set_index("qid").answerable.reindex(top_te.index).astype(bool)
        refusal[f"threshold_{name}"] = thr
        refusal[f"{name}"] = {"auroc_answerable_vs_not": float(roc_auc_score(ans, top_te)),
                              "refused_unanswerable": float((top_te[~ans] < thr).mean()),
                              "refused_answerable": float((top_te[ans] < thr).mean()), "n_unanswerable": int((~ans).sum())}
    lat = {"candidates_ms_p50": float(cand_te.groupby("qid").cand_ms.first().median()),
           "cross_encoder_ms_p50": float(cand_te.groupby("qid").ce_ms.first().median())}
    imp = dict(sorted(zip(FEATURES + ["ce"], ltr_ce.booster_.feature_importance("gain").round(1).tolist(), strict=True),
                      key=lambda kv: -kv[1]))
    out = {"n_test_answerable": int(te_q.answerable.sum()), "n_train": int(len(tr_q)), "overall": overall.to_dict(orient="index"),
           "by_style": {f"{m}|{s}": v for (m, s), v in summ.to_dict(orient="index").items()}, "refusal": refusal,
           "latency": lat, "ltr_feature_gain": imp, "refusal_threshold_source": "5th pct of top score, answerable train questions"}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "retrieval.json").write_text(json.dumps(out, indent=1))
    res.to_parquet(ROOT / "reports" / "retrieval_per_query.parquet", index=False)
    cand_te.to_parquet(ROOT / "runs" / "candidates_test.parquet", index=False)
    print(overall.to_string())
    print(summ[["recall@1", "recall@5", "mrr@10"]].unstack("style").to_string())
    print(json.dumps(refusal, indent=1), json.dumps(lat))


if __name__ == "__main__":
    main()
