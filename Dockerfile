# Service image (Linux, CPU). Retrieval + reranking run in-process; the LLM is any OpenAI-compatible endpoint
# (vLLM / Ollama / hosted) set with CFR_LLM_BACKEND=openai CFR_LLM_BASE_URL=... CFR_LLM_MODEL=...
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/app/.hf CFR_LLM_BACKEND=openai
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY app/ app/
COPY data/raw/ data/raw/
COPY data/eval/ data/eval/
COPY runs/ltr.txt runs/ltr_ce.txt runs/
COPY reports/retrieval.json reports/retrieval.json
# build the DuckDB store and the embedding index at image build time (models baked in; no network at runtime)
RUN python -m app.ingest >/dev/null && python -c "from app.retrieval import Index; Index().emb" \
 && python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
RUN useradd -r -u 10001 app && chown -R app /app
USER app
EXPOSE 8000
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
