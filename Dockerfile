# One image, three roles: the all-in-one demo (default), or `api` / `ui` in compose.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/app/corpus PORT=7860

# Hugging Face Spaces run containers as uid 1000; the index must be writable by it.
RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=user app ./app
COPY --chown=user ui ./ui
COPY --chown=user corpus ./corpus
COPY --chown=user eval/golden.jsonl ./eval/golden.jsonl
COPY --chown=user start.sh .
RUN mkdir -p data && chown user data

USER user
EXPOSE 7860
CMD ["./start.sh"]
