.PHONY: setup check ingest demo rebuild stats ask debug chat serve ui smoke test lint eval eval-cheap eval-matrix up deploy clean

PY := .venv/bin/python
PIP := .venv/bin/pip

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@test -f .env || (cp .env.example .env && echo ">> Created .env - add your GOOGLE_API_KEY")

check:
	$(PY) -m app.cli check

ingest:
	$(PY) -m app.cli ingest --verbose

# Index the bundled Halcyon Robotics demo corpus (the one the golden set targets).
demo:
	DATA_DIR=corpus $(PY) -m app.cli ingest --verbose

rebuild:
	$(PY) -m app.cli ingest --rebuild --verbose

stats:
	$(PY) -m app.cli stats

# usage: make ask Q="what is the leave policy?"
ask:
	$(PY) -m app.cli ask "$(Q)" --verbose

# usage: make debug Q="what is the leave policy?"  - shows every retrieval stage
debug:
	$(PY) -m app.cli debug "$(Q)"

chat:
	$(PY) -m app.cli chat

# API on :8000 - docs at http://localhost:8000/docs
serve:
	$(PY) -m app.cli serve --reload

# Streamlit UI on :8501. Needs `make serve` running in another terminal.
ui:
	.venv/bin/streamlit run ui/streamlit_app.py

# Full offline end-to-end check: no API key, no network, no tokens.
smoke:
	$(PY) scripts/smoke_test.py

test:
	$(PY) -m pytest tests/ -v

lint:
	.venv/bin/ruff check app ui scripts tests

# Golden-set evaluation against the demo corpus. Fails below the success criteria.
eval: demo
	$(PY) -m app.cli eval --gate

# Retrieval metrics + abstention only: no generation, no judge.
eval-cheap: demo
	$(PY) -m app.cli eval --retrieval-only --gate

# Retrieval-only A/B: every mode x reranker on the same questions.
eval-matrix: demo
	$(PY) -m app.cli eval --matrix

# api + ui + postgres/pgvector
up:
	docker compose up --build

# Push to a Hugging Face Space. usage: make deploy SPACE=user/citewise (needs HF_TOKEN)
deploy:
	$(PY) scripts/deploy_hf.py $(SPACE)

clean:
	rm -rf data/chroma data/manifest.json data/lexical_index.json eval/reports
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
