"""Build the evaluation set.

For each section: pick one chunk, ask the local LLM for (a) a question in regulation language, (b) the same need in
plain language as a veteran would ask it, (c) a one-sentence reference answer, all answerable from that chunk alone.
A second call verifies answerability and rejects questions that leak the section number. Sections are split
train/test by section (the reranker never trains on a test section). Hand-written unanswerable questions are added
to the test split.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.llm import LLM  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "eval"

GEN = """You write test questions for a search system over U.S. veterans-benefits regulations (38 CFR).
Read the passage and write ONE question that the passage alone fully answers.
Return JSON only:
{"formal": "<the question in regulatory language>",
 "plain": "<the same need, typed by a veteran about their own situation into a search box: first person, under 25 words, everyday words; do NOT reuse the passage's distinctive wording or legal terms>",
 "answer": "<one-sentence answer taken from the passage>"}
Rules: never mention section or paragraph numbers; the question must be specific enough that this passage answers it."""

CHECK = """Passage:
{passage}

Question: {q}
Proposed answer: {a}

Does the passage alone answer the question, and does the proposed answer match the passage? Reply YES or NO."""

UNANSWERABLE = [
    "How do I apply for a VA home loan guaranty?",
    "What is the 2026 monthly housing allowance under the Post-9/11 GI Bill?",
    "What is the phone number for my VA regional office?",
    "How much will I get paid each month at a 70 percent rating in 2026?",
    "Does the VA cover IVF treatments for veterans?",
    "How do I schedule a VA medical appointment online?",
    "Can I get a VA-backed loan for a manufactured home?",
    "What are the eligibility rules for VA burial in Arlington National Cemetery?",
    "How do I update my direct deposit information with VA?",
    "What vocational rehabilitation subsistence allowance rates apply this year?",
    "Is there a VA program to pay for my child's college tuition at a state school, and how much is it in 2026?",
    "How long does a disability claim take to process right now?",
    "What is the cost-of-living adjustment for VA compensation in 2026?",
    "Can a veteran get a free cell phone from the VA?",
    "How do I file an appeal with the Board of Veterans' Appeals online?",
    "What does the PACT Act toxic exposure screening at a VA clinic involve?",
    "Which VA hospitals have the shortest wait times?",
    "How do I get a Veteran ID card?",
    "What life insurance does the VA offer, and what are the premiums?",
    "Can VA pay for adaptive equipment for my car, and how much is the grant this year?",
    "How do I transfer my GI Bill benefits to my spouse?",
    "What are the income limits for VA health care copays in 2026?",
    "Does the VA offer small business loans to veterans?",
    "How do I contact the Veterans Crisis Line by text?",
    "What is the VA's policy on telework for its employees?",
    "Can I use VA benefits to buy a boat?",
    "What dental care does VA provide to veterans without a service-connected condition?",
    "How do I change my address on VA.gov?",
    "What is the maximum VA home loan amount in my county?",
    "How does the VA caregiver stipend get calculated in 2026?",
]


def main(limit: int = 0, seed: int = 0):
    rng = random.Random(seed)
    con = duckdb.connect(str(ROOT / "data" / "cfr.duckdb"), read_only=True)
    chunks = con.execute("SELECT chunk_id, section, part, heading, text, n_words FROM chunks WHERE n_words >= 60").df()
    con.close()
    sections = sorted(chunks.section.unique())
    rng.shuffle(sections)
    if limit:
        sections = sections[:limit]
    test_secs = set(sections[: len(sections) // 2])
    llm = LLM()
    OUT.mkdir(parents=True, exist_ok=True)
    kept, rejected = [], []
    for i, sec in enumerate(sections):
        cand = chunks[chunks.section == sec]
        ch = cand.iloc[rng.randrange(len(cand))]
        r = llm.chat([{"role": "system", "content": GEN}, {"role": "user", "content": f"Passage ({ch.heading}):\n{ch.text}"}],
                     max_tokens=260)
        m = re.search(r"\{.*\}", r["text"], re.S)
        try:
            q = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            q = None
        if not q or not all(isinstance(q.get(k), str) and q[k].strip() for k in ("formal", "plain", "answer")):
            rejected.append({"section": sec, "why": "unparseable"})
            continue
        leak = re.search(r"§|\b\d+\.\d+[a-z]?\b|section \d|paragraph \(", q["formal"] + " " + q["plain"], re.I)
        if leak:
            rejected.append({"section": sec, "why": "leaks section number"})
            continue
        ok = llm.chat([{"role": "user", "content": CHECK.format(passage=ch.text, q=q["formal"], a=q["answer"])}], max_tokens=4)
        if not ok["text"].upper().startswith("YES"):
            rejected.append({"section": sec, "why": "verifier said no"})
            continue
        split = "test" if sec in test_secs else "train"
        for style in ("formal", "plain"):
            kept.append({"qid": f"{sec}:{style}", "split": split, "style": style, "question": q[style].strip(),
                         "reference_answer": q["answer"].strip(), "gold_section": sec, "gold_chunk": ch.chunk_id,
                         "answerable": True})
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(sections)} kept {len(kept) // 2} rejected {len(rejected)}", flush=True)
    for j, q in enumerate(UNANSWERABLE):
        kept.append({"qid": f"unans:{j}", "split": "test", "style": "unanswerable", "question": q, "reference_answer": "",
                     "gold_section": None, "gold_chunk": None, "answerable": False})
    (OUT / "questions.jsonl").write_text("".join(json.dumps(k) + "\n" for k in kept))
    (OUT / "rejected.json").write_text(json.dumps(rejected, indent=1))
    by = {}
    for k in kept:
        by[(k["split"], k["style"])] = by.get((k["split"], k["style"]), 0) + 1
    print("kept", {f"{a}/{b}": v for (a, b), v in sorted(by.items())}, "rejected", len(rejected))


if __name__ == "__main__":
    main(*map(int, sys.argv[1:]))
