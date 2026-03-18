.PHONY: install train api test clean

install:
    pip install -r requirements.txt

train-cf:
    python scripts/train_collaborative_filtering.py

train-hybrid:
    python scripts/hybrid_reranker.py

api:
    uvicorn app:app --reload

test:
    pytest tests/

clean:
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
