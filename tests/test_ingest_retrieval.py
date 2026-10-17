import numpy as np
from lxml import etree

from app.ingest import _table_lines, chunk_section
from app.retrieval import rrf, sections_ranked, tokenize


def test_both_table_dialects_flatten_row_by_row():
    gpo = etree.fromstring("<GPOTABLE><ROW><ENT>Flexion limited to 45°</ENT><ENT>10</ENT></ROW></GPOTABLE>")
    html = etree.fromstring("<TABLE><CAPTION>Knee</CAPTION><TR><TH>Code</TH><TH>Rating</TH></TR><TR><TD>5260</TD><TD>10</TD></TR></TABLE>")
    assert _table_lines(gpo) == ["Flexion limited to 45° | 10"]
    assert _table_lines(html) == ["Knee", "Code | Rating", "5260 | 10"]


def test_chunks_stay_inside_their_section_and_overlap():
    sec = {"section": "3.1", "part": "3", "heading": "h", "subpart": "s", "paragraphs": ["word " * 200, "next " * 100, "last " * 100]}
    ch = chunk_section(sec, max_words=250)
    assert all(c["section"] == "3.1" for c in ch) and len(ch) >= 2
    assert ch[1]["text"].startswith(ch[0]["text"].split("\n")[-1][:20])   # one-paragraph overlap


def test_tokenizer_keeps_numbers_and_units():
    t = tokenize("Flexion limited to 45° and 10% of the knee")
    assert "45" in t and "°" in t and "%" in t and "the" not in t


def test_rrf_and_section_dedup():
    a, b = np.array([3.0, 2.0, 1.0]), np.array([1.0, 3.0, 2.0])
    s = rrf([a, b])
    assert s.argmax() in (0, 1) and s.argmin() == 2
    assert sections_ranked(["3.1", "3.1", "4.2", "3.1", "4.3"]) == ["3.1", "4.2", "4.3"]
