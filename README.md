# Technical Engineering Document: Building an Adversarial-Resistant Spam Detection Pipeline

## System Architecture & Evolution Report

**Core Stack:** Python, FastAPI, Scikit-Learn, Docker

## 1. Executive Summary

This project evolved from a basic academic text classification example into a more robust anti-spam microservice designed to handle adversarial input patterns and production-style inference challenges. The pipeline was strengthened by improving preprocessing, switching to sub-word feature representations, and adding a backend decision layer that makes the system behave more conservatively when suspicious signals appear.

## 2. System Evolution

### Phase 1: Baseline Academic Model

The initial version used a basic bag-of-words approach with a standard Multinomial Naive Bayes classifier. Although effective on ordinary examples, it was vulnerable to simple obfuscation such as misspellings, punctuation variations, and lookalike substitutions.

### Phase 2: Stronger Preprocessing and Feature Engineering

The preprocessing pipeline was improved to normalize common adversarial patterns before classification. The system now:

- maps URLs to a generic token such as urltoken
- maps numbers to a generic token such as numbertoken
- removes punctuation in a controlled way
- standardizes whitespace and casing

In addition, the project moved from word-level counting to a sub-word TF-IDF representation using character n-grams. This makes the model more resilient to small spelling variations and character-level obfuscation.

### Phase 3: Adversarial-Resistance Strategy

A major issue discovered during testing was the tendency of the model to produce a high confidence ham prediction for messages that contained many benign words but also a suspicious link. This is a classic failure mode in Naive Bayes classification because a long stream of ordinary tokens can overpower a single suspicious signal.

To address this, the backend was adjusted to evaluate inference in the log-probability space (predict_log_proba) rather than standard raw probabilities. By applying a custom log-offset threshold cushion (spam_log > ham_log - 4.5), the API successfully overrides dataset bias to catch highly diluted adversarial inputs.

## 3. Engineering Approach

### Preprocessing

The shared cleaning logic is implemented in both training and inference paths so the model and the live API follow the same normalization rules.

### Feature Representation

The model uses a TF-IDF vectorizer configured with character n-grams in the range 3 to 5. This allows it to capture subtle text patterns and structural similarities rather than relying only on exact words.

### Classification Strategy

The model uses Multinomial Naive Bayes with:

- alpha = 0.1
- fit_prior = False

This helps reduce the effect of class imbalance and makes the classifier less biased toward the majority class during inference.

## 4. Why This Is a Strong Portfolio Project

This project is not just a simple classifier demo. It demonstrates:

- practical preprocessing for real-world text noise
- adversarial-aware model design
- production-oriented API development with FastAPI
- deployment readiness using Docker
- a thoughtful engineering response to model weaknesses rather than simply retraining with more data

## 5. Deployment and Usage

### Prerequisites

- Python 3.11 or later
- pip
- Docker (optional, for containerized deployment)

### Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Train the model

```bash
python train.py
```

### Run the API

```bash
python api.py
```

Then open:

- http://127.0.0.1:8000/
- http://127.0.0.1:8000/docs

### Docker Usage

```bash
docker build -t text-spam-detection .
docker run -p 8000:8000 text-spam-detection
```

## 6. API Example

### Health check

```bash
curl http://127.0.0.1:8000/health
```

### Prediction request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"Congratulations! You won a free iPhone."}'
```

Example response:

```json
{
  "text": "Congratulations! You won a free iPhone.",
  "prediction": "SPAM",
  "confidence": 99.61,
  "algorithm": "Optimized Multinomial Naive Bayes"
}
```

## 7. Future Directions

Potential next steps include:

- migrating to transformer-based embeddings for deeper semantic understanding
- adding more robust adversarial evaluation datasets
- improving observability and logging for production monitoring
- expanding deployment to cloud platforms
