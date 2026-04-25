# Relevant Priors API

This project implements an HTTP API for the relevant-priors challenge.

## Endpoint

POST `/predict`

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

The API will run at:

```text
http://localhost:5000/predict
```

## Test with curl

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

## Deployment

For Render/Railway/Fly.io, use:

```bash
gunicorn app:app
```

## Approach

The model is a deterministic rule-based classifier. It compares each current study against each prior study using:

- imaging modality
- anatomy/body-region keyword matching
- normalized text similarity
- contrast matching
- study-date recency

It always returns exactly one prediction per prior study.
