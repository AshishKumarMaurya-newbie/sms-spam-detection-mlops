# Text Analytics for Spam Detection using Naive Bayes

## Project Overview

This project implements a complete spam detection system for SMS messages using a Multinomial Naive Bayes classifier. It includes dataset loading, preprocessing, model training, evaluation, model serialization, a FastAPI backend, and a lightweight browser-based frontend for real-time prediction.

## Features

- Automatically downloads the SMS spam dataset if it is not already present locally
- Stores the dataset in the dataset folder as spam.csv
- Cleans and prepares the data using pandas
- Trains an optimized Multinomial Naive Bayes model with CountVectorizer
- Improves text normalization with cleaner preprocessing for SMS-style messages
- Evaluates the model using accuracy, precision, recall, F1 score, confusion matrix, and classification report
- Saves the trained model and vectorizer for reuse
- Exposes a REST API for predictions using FastAPI
- Includes a polished frontend UI with example prompts, confidence feedback, and recent-check history
- Supports Docker-based deployment

## Project Structure

```text
text-spam-detection/
├── dataset/
│   └── spam.csv
├── model/
│   ├── spam_model.pkl
│   └── vectorizer.pkl
├── train.py
├── api.py
├── predict.py
├── index.html
├── requirements.txt
├── Dockerfile
├── deploy.sh
├── README.md
└── .gitignore
```

## Dataset

The project uses a SMS Spam Collection dataset sourced from a Kaggle-based GitHub repository.

Dataset source:
https://raw.githubusercontent.com/mdrsyed/SMS-Spam-Collection-Dataset-Kaggle/main/spam.csv

The training script automatically downloads the CSV file if it is not already available locally and saves it inside the dataset directory as spam.csv.

## Model

- Algorithm: Optimized Multinomial Naive Bayes
- Feature extraction: CountVectorizer with English stop words removed
- Expected accuracy: approximately 98%

## Verified Performance

The upgraded model was trained successfully and achieved:

- Accuracy: 0.9834
- Precision: 0.9580
- Recall: 0.9048
- F1 Score: 0.9306

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running Locally

### 1. Train the model

```bash
python train.py
```

### 2. Run the API and frontend

```bash
python api.py
```

Then open:

- http://127.0.0.1:8000/
- http://127.0.0.1:8000/docs

## Docker Usage

```bash
docker build -t text-spam-detection .
docker run -p 8000:8000 text-spam-detection
```

## API Examples

### Health check

```bash
curl http://127.0.0.1:8000/health
```

Example response:

```json
{
  "status": "running"
}
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
  "confidence": 99.37,
  "algorithm": "Multinomial Naive Bayes"
}
```

## Future Improvements

- Add a more advanced analytics dashboard with charts and trends
- Compare the current Naive Bayes model with Logistic Regression or SVM
- Add further preprocessing enhancements such as stemming and lemmatization
- Deploy the project to a cloud platform such as Azure or AWS
