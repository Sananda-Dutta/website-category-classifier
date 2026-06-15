# Website Category Classifier

A DistilBERT-powered Website Category Classification API built using FastAPI and HuggingFace Transformers.

## Features

- Website category prediction
- URL classification
- Text classification
- Batch prediction
- Safe-check endpoint
- Explainability endpoint
- HuggingFace model integration
- FastAPI backend

## Tech Stack

- Python
- FastAPI
- DistilBERT
- HuggingFace Transformers
- PyTorch
- BeautifulSoup
- Render Deployment

## Run Locally

```bash
pip install -r requirements.txt
uvicorn api:app --reload
```

## API Docs

After running:

```text
http://127.0.0.1:8000/docs
```

## Model

Hosted on HuggingFace Hub.

## Accuracy

85.4%