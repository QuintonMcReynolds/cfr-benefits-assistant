"""eCFR XML -> sections -> chunks, stored in DuckDB with a manifest (source hashes, counts).

Sections are the citable unit (e.g. 38 CFR 3.304). Chunks never cross a section boundary, so every retrieved chunk
maps to exactly one citation. Rating tables are flattened row by row ('col | col | col') so a row stays one unit.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import duckdb
import pandas as pd
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DB = ROOT / "data" / "cfr.duckdb"
ISSUE_DATE = "2026-08-27"


def _text(el) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


SKIP = {"HEAD", "CITA", "SECAUTH", "PARAUTH", "img", "AUTH", "SOURCE"}


def _table_lines(tbl) -> list[str]:
    """Both table dialects: GPOTABLE (ROW/ENT) and HTML (TR/TD/TH). One line per row, caption first."""
    lines = []
    cap = tbl.find(".//CAPTION")
    if cap is not None and _text(cap):
        lines.append(_text(cap))
    for row in tbl.iter("ROW", "TR"):
        cells = [_text(c) for c in row if c.tag in ("ENT", "TD", "TH")]
        if any(cells):
            lines.append(" | ".join(c for c in cells if c))
    return lines


def _blocks(el) -> list[str]:
    """Flatten a section's children into paragraphs; tables become one line per row."""
    out = []
    for child in el:
        if child.tag in SKIP or not isinstance(child.tag, str):
            continue
        if child.tag in ("GPOTABLE", "TABLE"):
            out.extend(_table_lines(child))
        elif child.find(".//TABLE") is not None or child.find(".//GPOTABLE") is not None or child.tag in ("DIV", "DIV9", "EXTRACT"):
            out.extend(_blocks(child))
        else:
            t = _text(child)
            if t:
                out.append(t)
    return out


def parse_part(path: Path) -> list[dict]:
    root = etree.parse(str(path)).getroot()
    out = []
    for sec in root.iter("DIV8"):
        sid = sec.get("N", "").replace("§", "").strip()
        head = _text(sec.find("HEAD")) if sec.find("HEAD") is not None else sid
        subpart = ""
        p = sec.getparent()
        while p is not None:
            if p.tag == "DIV6":
                subpart = _text(p.find("HEAD")) if p.find("HEAD") is not None else ""
                break
            p = p.getparent()
        paras = _blocks(sec)
        if paras and "[Reserved]" not in head:
            out.append({"section": sid, "part": sid.split(".")[0], "heading": head, "subpart": subpart, "paragraphs": paras})
    return out


def chunk_section(sec: dict, max_words: int = 250) -> list[dict]:
    chunks: list[list[str]] = []
    cur: list[str] = []
    n = 0
    for para in sec["paragraphs"]:
        w = len(para.split())
        if cur and n + w > max_words:
            chunks.append(cur)
            cur, n = cur[-1:], len(cur[-1].split())   # one-paragraph overlap keeps context across the cut
        cur.append(para)
        n += w
    if cur:
        chunks.append(cur)
    return [{"chunk_id": f"{sec['section']}#{i}", "section": sec["section"], "part": sec["part"], "heading": sec["heading"],
             "subpart": sec["subpart"], "text": "\n".join(c)} for i, c in enumerate(chunks)]


def main():
    secs, src = [], {}
    for f in sorted(RAW.glob("38cfr_part*.xml")):
        src[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
        secs += parse_part(f)
    sections = pd.DataFrame([{**{k: v for k, v in s.items() if k != "paragraphs"}, "text": "\n".join(s["paragraphs"]),
                              "n_words": sum(len(x.split()) for x in s["paragraphs"])} for s in secs])
    chunks = pd.DataFrame([c for s in secs for c in chunk_section(s)])
    chunks["n_words"] = chunks.text.str.split().str.len()
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB))
    # register under distinct names: inside CREATE OR REPLACE TABLE chunks ... FROM chunks, DuckDB would read the
    # *existing table*, not the DataFrame, and silently keep stale data
    con.register("sections_df", sections)
    con.register("chunks_df", chunks)
    con.execute("CREATE OR REPLACE TABLE sections AS SELECT * FROM sections_df")
    con.execute("CREATE OR REPLACE TABLE chunks AS SELECT * FROM chunks_df")
    stats = con.execute("""SELECT part, count(DISTINCT section) n_sections, count(*) n_chunks, round(avg(n_words)) avg_words,
                                  max(n_words) max_words FROM chunks GROUP BY part ORDER BY part""").df()
    con.close()
    manifest = {"source": "eCFR versioner API, title 38 parts 3-4", "issue_date": ISSUE_DATE, "source_sha256": src,
                "n_sections": int(len(sections)), "n_chunks": int(len(chunks)), "per_part": stats.to_dict(orient="records")}
    (ROOT / "data" / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
