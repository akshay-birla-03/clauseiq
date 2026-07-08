.PHONY: install dev data test lint run ui eval docker

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

data:
	python scripts/generate_contracts.py

test:
	pytest -q --cov=clauseiq --cov-report=term-missing

lint:
	ruff check src tests

run:
	uvicorn clauseiq.api:app --reload --port 8000

ui:
	streamlit run src/clauseiq/ui_streamlit.py

eval:
	python -m clauseiq.eval.harness --cases data/eval_cases.json

docker:
	docker compose up --build
