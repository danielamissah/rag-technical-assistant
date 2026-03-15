PYTHON = python3

.PHONY: setup ingest chunk embed serve evaluate mlflow clean

setup:
	pip install -r requirements.txt

ingest:
	$(PYTHON) src/ingestion/fetch_arxiv.py

chunk:
	$(PYTHON) src/ingestion/chunk_documents.py

embed:
	$(PYTHON) src/embeddings/build_vectorstore.py

pipeline: ingest chunk embed
	@echo "Pipeline complete — vector store ready"

serve:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

evaluate:
	PYTHONPATH=. $(PYTHON) src/evaluation/evaluate_rag.py

mlflow:
	mlflow ui --port 5001

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
