#!/bin/sh
# All-in-one container: index the corpus, run the API privately, serve the UI on $PORT.
set -e

# Idempotent: unchanged files are skipped, so restarts with a persistent volume are free.
python -m app.cli ingest || echo "ingest failed - is GOOGLE_API_KEY set? The UI will say so."

uvicorn app.api:app --host 127.0.0.1 --port 8000 &

exec streamlit run ui/streamlit_app.py \
  --server.port "${PORT:-7860}" --server.address 0.0.0.0 \
  --server.headless true --browser.gatherUsageStats false
