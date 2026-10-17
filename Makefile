PY ?= .venv/bin/python
ingest:      ## eCFR XML -> DuckDB sections/chunks + manifest
	$(PY) -m app.ingest
evalset:     ## LLM-generated + verified questions (formal & plain), hand-written unanswerables (~1.5 h on an M4)
	$(PY) scripts/make_eval_set.py
retrieval:   ## BM25 / dense / hybrid / cross-encoder / LambdaRank; refusal calibration
	$(PY) scripts/eval_retrieval.py
answers:     ## end-to-end answer eval across model sizes / rerankers / gating
	$(PY) scripts/eval_answers.py
report:
	$(PY) scripts/make_report.py && $(PY) scripts/readme_tables.py
serve:
	$(PY) -m uvicorn app.server:app --port 8000
check:
	$(PY) -m ruff check app scripts tests && $(PY) -m mypy && $(PY) -m pytest -q
.PHONY: ingest evalset retrieval answers report serve check
