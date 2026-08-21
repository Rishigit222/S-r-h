.PHONY: setup install run-api run-ui ingest test evaluate clean

setup:
	python -m venv .venv
	.venv\Scripts\pip install -r requirements.txt
	@echo "Setup complete! Activate with: .venv\Scripts\activate"

install:
	pip install -r requirements.txt

run-api:
	python -m uvicorn src.api.main:app --reload --port 8000

run-ui:
	python -m streamlit run src/ui/app.py --server.port 8501

ingest:
	python -m src.ingestion.embedder

test:
	python -m pytest tests/ -v

evaluate:
	python -m src.evaluation.eval_runner

clean:
	rm -rf data/chroma_db
	rm -rf __pycache__
	@echo "Cleaned!"
