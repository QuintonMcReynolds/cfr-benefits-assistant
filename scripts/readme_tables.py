"""Regenerate README tables from reports/*.json (between <!-- gen:NAME --> markers)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
METHODS = ["bm25", "dense", "hybrid", "hybrid+cross-encoder", "hybrid+LTR", "hybrid+LTR(+ce feature)"]
NAMES = {"bm25": "BM25", "dense": "dense embeddings (bge-small)", "hybrid": "hybrid (reciprocal-rank fusion)",
         "hybrid+cross-encoder": "hybrid → cross-encoder (MiniLM)", "hybrid+LTR": "hybrid → LambdaRank (LightGBM)",
         "hybrid+LTR(+ce feature)": "hybrid → LambdaRank + cross-encoder feature"}


def pct(x):
    return f"{100 * x:.0f} %"


def retrieval():
    r = json.loads((REP / "retrieval.json").read_text())
    bs = r["by_style"]
    rows = ["| retriever | formal: top-1 | formal: top-5 | **plain: top-1** | **plain: top-5** | plain MRR@10 |", "|---|---|---|---|---|---|"]
    for m in METHODS:
        f, p = bs[f"{m}|formal"], bs[f"{m}|plain"]
        rows.append(f"| {NAMES[m]} | {pct(f['recall@1'])} | {pct(f['recall@5'])} | {pct(p['recall@1'])} | {pct(p['recall@5'])} | {p['mrr@10']:.2f} |")
    n = r["n_test_answerable"] // 2
    rows.append("")
    rows.append(f"{n} held-out sections, each asked two ways ({r['n_test_answerable']} questions). LambdaRank trained on "
                f"{r['n_train']} questions from different sections. Cross-encoder adds {r['latency']['cross_encoder_ms_p50']:.0f} ms "
                f"per question (median, M4 CPU/GPU); first-stage retrieval takes {r['latency']['candidates_ms_p50']:.0f} ms.")
    return "\n".join(rows)


def refusal():
    r = json.loads((REP / "retrieval.json").read_text())["refusal"]
    rows = ["| refusal gate on the top reranker score | AUROC (answerable vs not) | unanswerable refused | answerable wrongly refused |",
            "|---|---|---|---|"]
    for k, n in (("ltr", "LambdaRank score"), ("ltr_ce", "LambdaRank + cross-encoder score")):
        v = r[k]
        rows.append(f"| {n} | {v['auroc_answerable_vs_not']:.2f} | {pct(v['refused_unanswerable'])} | {pct(v['refused_answerable'])} |")
    return "\n".join(rows)


def answers():
    p = REP / "answers.json"
    if not p.exists():
        return "_not run yet_"
    a = json.loads(p.read_text())
    s = a["summary"]
    rows = ["| configuration | answered correctly | cites the right section | citations grounded | unanswerable refused | "
            "answerable wrongly refused | needs review | p50 / p95 latency | tokens / question |", "|---|---|---|---|---|---|---|---|---|"]
    for name, v in s.items():
        rows.append(f"| {name} | **{pct(v['answer_correct'])}** | {pct(v['gold_cited'])} | {pct(v['grounded_citations'])} | "
                    f"{pct(v['refused_unanswerable'])} | {pct(v['false_refusal'])} | {pct(v['needs_review'])} | "
                    f"{v['p50_latency_s']:.1f} s / {v['p95_latency_s']:.1f} s | {v['tokens_per_question']:.0f} |")
    rows.append("")
    rows.append(f"{a['n_answerable']} answerable questions (50 held-out sections × 2 wordings) + {a['n_unanswerable']} unanswerable. "
                "\"Answered correctly\" counts a refused answerable question as wrong.")
    return "\n".join(rows)


def answers_by_style():
    p = REP / "answers.json"
    if not p.exists():
        return "_not run yet_"
    b = json.loads(p.read_text())["by_style"]
    cfgs = list(dict.fromkeys(k.split("|")[0] for k in b))
    rows = ["| configuration | formal: correct | plain: correct | gap |", "|---|---|---|---|"]
    for c in cfgs:
        f, pl = b[f"{c}|formal"]["answer_correct"], b[f"{c}|plain"]["answer_correct"]
        rows.append(f"| {c} | {pct(f)} | {pct(pl)} | {100 * (f - pl):+.0f} pt |")
    return "\n".join(rows)


def judge():
    p = REP / "answers.json"
    if not p.exists():
        return "_not run yet_"
    j = json.loads(p.read_text())["judge_check"]
    return (f"Judge sanity check on {j['n']} questions: reference answers judged correct {pct(j['reference_judged_correct'])}; "
            f"another question's reference answer judged incorrect {pct(j['other_reference_judged_incorrect'])}.")


GEN = {"retrieval": retrieval, "refusal": refusal, "answers": answers, "answers_by_style": answers_by_style, "judge": judge}


def main():
    p = ROOT / "README.md"
    s = p.read_text()
    for name, fn in GEN.items():
        s = re.sub(rf"(<!-- gen:{name} -->\n).*?(<!-- /gen:{name} -->)", lambda m, fn=fn: m.group(1) + fn() + "\n" + m.group(2), s, flags=re.S)
    p.write_text(s)
    print("README tables regenerated")


if __name__ == "__main__":
    print(GEN[sys.argv[1]]()) if len(sys.argv) > 1 else main()
