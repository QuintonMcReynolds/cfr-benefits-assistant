"""End-to-end answer evaluation across configurations (accuracy vs latency vs tokens).

Sample: 50 test sections x {formal, plain} + all 30 unanswerable questions (fixed seed).
Metrics per answer:
  gold_cited        cites the gold section (answerable questions)
  grounded          every citation is one of the excerpts shown (no invented citations)
  refused           said it couldn't find it (gate or model)
  judged_correct    LLM judge vs the reference answer (answerable, not refused)
The judge is sanity-checked first: reference answers must be judged correct, and a reference answer for a *different*
question must be judged incorrect.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.answer import Assistant  # noqa: E402
from app.llm import LLM  # noqa: E402
from app.retrieval import Index  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
JUDGE = """Question: {q}
Reference answer: {ref}
Candidate answer: {cand}

Does the candidate answer state the same key facts as the reference answer, without contradicting it? Extra correct
detail is fine. Reply with one word: CORRECT or INCORRECT."""

CONFIGS = {
    "7B + hybrid (no reranker)": {"model_id": "mlx-community/Qwen2.5-7B-Instruct-4bit", "reranker": "hybrid", "gate": False},
    "7B + cross-encoder": {"model_id": "mlx-community/Qwen2.5-7B-Instruct-4bit", "reranker": "ce", "gate": False},
    "7B + LTR(+CE) + refusal gate": {"model_id": "mlx-community/Qwen2.5-7B-Instruct-4bit", "reranker": "ltr_ce", "gate": True},
    "3B + cross-encoder": {"model_id": "mlx-community/Qwen2.5-3B-Instruct-4bit", "reranker": "ce", "gate": False},
}


def sample(seed: int = 0) -> pd.DataFrame:
    q = pd.read_json(ROOT / "data" / "eval" / "questions.jsonl", lines=True)
    te = q[q.split == "test"]
    secs = te[te.answerable].gold_section.drop_duplicates().sample(50, random_state=seed)
    return pd.concat([te[te.gold_section.isin(secs)], te[~te.answerable]], ignore_index=True)


def judge(llm: LLM, q: str, ref: str, cand: str) -> bool:
    r = llm.chat([{"role": "user", "content": JUDGE.format(q=q, ref=ref, cand=cand)}], max_tokens=4)
    return r["text"].strip().upper().startswith("CORRECT")


def main():
    qs = sample()
    thr = json.loads((ROOT / "reports" / "retrieval.json").read_text())["refusal"]["threshold_ltr_ce"]
    ix = Index()
    jl = LLM("mlx-community/Qwen2.5-7B-Instruct-4bit")
    # ---- judge sanity check
    ans = qs[qs.answerable].drop_duplicates("gold_section")
    same = [judge(jl, r.question, r.reference_answer, r.reference_answer) for r in ans.itertuples()]
    shuffled = ans.reference_answer.sample(frac=1, random_state=1).to_numpy()
    diff = [judge(jl, r.question, r.reference_answer, s) for r, s in zip(ans.itertuples(), shuffled, strict=True)
            if s != r.reference_answer]
    judge_check = {"reference_judged_correct": float(np.mean(same)), "other_reference_judged_incorrect": float(1 - np.mean(diff)),
                   "n": len(same)}
    print("judge check", judge_check, flush=True)
    rows = []
    for name, cfg in CONFIGS.items():
        a = Assistant(ix, cfg["model_id"], cfg["reranker"], refuse_below=thr if cfg["gate"] else None)
        t0 = time.time()
        for r in qs.itertuples():
            out = a(r.question)
            row = {"config": name, "qid": r.qid, "style": r.style, "answerable": r.answerable, "refused": out.refused,
                   "refused_by": out.refused_by, "grounded": not out.ungrounded_citations, "n_citations": len(out.citations),
                   "gold_cited": (r.gold_section in out.citations) if r.answerable else None,
                   "gold_retrieved": (r.gold_section in out.retrieved_sections) if r.answerable else None,
                   "needs_review": out.needs_review, "latency_ms": sum(out.latency_ms.values()),
                   "prompt_tokens": out.tokens["prompt"], "completion_tokens": out.tokens["completion"], "answer": out.text}
            row["judged_correct"] = (judge(jl, r.question, r.reference_answer, out.text)
                                     if r.answerable and not out.refused else None)
            rows.append(row)
        print(name, f"{time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(ROOT / "reports" / "answers.parquet", index=False)

    def summarize(g):
        a, u = g[g.answerable], g[~g.answerable]
        answered = a[~a.refused]
        return pd.Series({
            "answer_correct": float(answered.judged_correct.fillna(False).sum() / len(a)),   # refused answerable = wrong
            "gold_cited": float(a.gold_cited.mean()), "false_refusal": float(a.refused.mean()),
            "refused_unanswerable": float(u.refused.mean()), "grounded_citations": float(g[~g.refused].grounded.mean()),
            "needs_review": float(g.needs_review.mean()),
            "p50_latency_s": float(g.latency_ms.median() / 1000), "p95_latency_s": float(g.latency_ms.quantile(0.95) / 1000),
            "tokens_per_question": float((g.prompt_tokens + g.completion_tokens).mean())})

    summ = df.groupby("config").apply(summarize, include_groups=False)
    by_style = df[df.answerable].groupby(["config", "style"]).apply(
        lambda g: pd.Series({"answer_correct": float(g[~g.refused].judged_correct.fillna(False).sum() / len(g)),
                             "gold_cited": float(g.gold_cited.mean())}), include_groups=False)
    out = {"judge_check": judge_check, "summary": summ.round(3).to_dict(orient="index"),
           "by_style": {f"{c}|{s}": v for (c, s), v in by_style.round(3).to_dict(orient="index").items()},
           "n_answerable": int(qs.answerable.sum()), "n_unanswerable": int((~qs.answerable).sum()), "refusal_threshold": thr}
    (ROOT / "reports" / "answers.json").write_text(json.dumps(out, indent=1))
    print(summ.round(3).to_string())
    print(by_style.round(3).to_string())


if __name__ == "__main__":
    main()
