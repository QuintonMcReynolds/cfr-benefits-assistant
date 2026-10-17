"""CI gate on the committed retrieval report: fails if the learned reranker stops beating the hybrid baseline, if
plain-language recall@5 falls below the floor, or if the refusal gate stops separating unanswerable questions."""
import json
import sys
from pathlib import Path

r = json.loads((Path(__file__).resolve().parents[1] / "reports" / "retrieval.json").read_text())
# Checks the reranker that ships (cross-encoder; best end-to-end in reports/answers.json) and the optional calibrated
# gate (LambdaRank + cross-encoder feature). The first version of this gate checked plain LambdaRank and failed
# (MRR 0.834 < hybrid 0.839; plain-language recall@5 0.82 < hybrid 0.87): that's how its over-reliance on keyword
# overlap was found.
o, bs, ref = r["overall"], r["by_style"], r["refusal"]["ltr_ce"]
ship = "hybrid+cross-encoder"
checks = {
    "shipped reranker MRR@10 > hybrid": o[ship]["mrr@10"] > o["hybrid"]["mrr@10"],
    "plain-language recall@5 >= 0.60": bs[f"{ship}|plain"]["recall@5"] >= 0.60,
    "plain-language recall@5 not below hybrid": bs[f"{ship}|plain"]["recall@5"] >= bs["hybrid|plain"]["recall@5"],
    "optional gate: refusal AUROC >= 0.80": ref["auroc_answerable_vs_not"] >= 0.80,
}
for k, v in checks.items():
    print(("PASS " if v else "FAIL ") + k)
sys.exit(0 if all(checks.values()) else 1)
