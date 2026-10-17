"""Figures for the README (dataviz reference palette slots 1-2 for the two question styles: validated pair)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
STYLE_C = {"formal": "#2a78d6", "plain": "#eb6834"}
METHODS = ["bm25", "dense", "hybrid", "hybrid+cross-encoder", "hybrid+LTR", "hybrid+LTR(+ce feature)"]
LABEL = {"bm25": "BM25", "dense": "dense (bge-small)", "hybrid": "hybrid (RRF)", "hybrid+cross-encoder": "+ cross-encoder",
         "hybrid+LTR": "+ LambdaRank", "hybrid+LTR(+ce feature)": "+ LambdaRank w/ CE feature"}


def _style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlabel(ylabel, color=INK2, fontsize=9)


def retrieval_bars():
    r = json.loads((ROOT / "reports" / "retrieval.json").read_text())["by_style"]
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    y = np.arange(len(METHODS))
    h = 0.36
    for j, st in enumerate(("formal", "plain")):
        vals = [100 * r[f"{m}|{st}"]["recall@5"] for m in METHODS]
        ax.barh(y + (j - 0.5) * (h + 0.04), vals, height=h, color=STYLE_C[st], label=f"{st} wording")
        for yi, v in zip(y + (j - 0.5) * (h + 0.04), vals, strict=True):
            ax.text(v + 1, yi, f"{v:.0f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[m] for m in METHODS])
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    _style(ax, "Right section in the top 5 (held-out sections)\n", "recall@5 (%)")
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "retrieval_recall5.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def tradeoff():
    p = ROOT / "reports" / "answers.json"
    if not p.exists():
        return
    s = json.loads(p.read_text())["summary"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    for name, v in s.items():
        ax.scatter(v["p50_latency_s"], 100 * v["answer_correct"], s=80, color="#2a78d6", edgecolor=SURFACE, lw=1.5, zorder=3)
        ax.annotate(name, (v["p50_latency_s"], 100 * v["answer_correct"]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK2)
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_xlabel("median latency per question (s, M4 laptop)", color=INK2, fontsize=9)
    ax.set_ylabel("answerable questions answered correctly (%)", color=INK2, fontsize=9)
    ax.set_title("Accuracy vs latency by configuration", loc="left", fontsize=11, color=INK, pad=10)
    xs = [v["p50_latency_s"] for v in s.values()]
    ax.set_xlim(0, max(xs) * 1.9)
    fig.tight_layout()
    fig.savefig(FIG / "tradeoff.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    retrieval_bars()
    tradeoff()
    print(sorted(x.name for x in FIG.glob("*.png")))
