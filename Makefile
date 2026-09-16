.PHONY: setup install run-api run-prod ingest test evaluate heal drift-check clean docker-build docker-run

setup:
	python -m venv .venv
	.venv/bin/pip install -r requirements.txt || .venv\Scripts\pip install -r requirements.txt
	@echo "Setup complete! Activate with: source .venv/bin/activate (Linux/Mac) or .venv\Scripts\activate (Windows)"

install:
	pip install -r requirements.txt

run-api:
	python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-prod:
	python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 2

ingest:
	python -m src.ingestion.embedder

test:
	pytest tests/ -v

evaluate:
	python -m src.evaluation.eval_runner

heal:
	python scripts/test_heal_pipeline.py

drift-check:
	python -c "from src.observability.drift_monitor import drift_monitor; import json; print(json.dumps(drift_monitor.get_drift_assessment(), indent=2))"

docker-build:
	docker build -t self-healing-rag:latest .

docker-run:
	docker run -p 8000:8000 --env-file .env self-healing-rag:latest

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	@echo "Cache cleaned!"
