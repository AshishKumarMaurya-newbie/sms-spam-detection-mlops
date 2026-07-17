# Text Spam Detection

## Project Overview

This project is a simple and effective spam detection system for SMS messages. It uses a Multinomial Naive Bayes classifier combined with text preprocessing to classify incoming messages as either spam or ham.

The application includes:

- dataset loading and preprocessing
- model training and evaluation
- model saving for reuse
- a FastAPI backend for predictions
- a lightweight frontend for real-time testing
- Docker support for containerized deployment

## Features

- Automatically downloads the SMS spam dataset if it is not already available locally
- Stores the dataset in the dataset folder as spam.csv
- Cleans and prepares text data using pandas and scikit-learn
- Trains an optimized Multinomial Naive Bayes model with CountVectorizer
- Evaluates the model using accuracy, precision, recall, F1 score, and a confusion matrix
- Saves the trained model and vectorizer for reuse
- Exposes a REST API for predictions using FastAPI
- Provides a simple browser-based frontend for testing messages
- Supports Docker-based deployment

## Tech Stack

- Python 3.11
- FastAPI
- scikit-learn
- pandas
- joblib
- Docker

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

The project uses the SMS Spam Collection dataset, sourced from a Kaggle-based GitHub repository.

Dataset source:
https://raw.githubusercontent.com/mdrsyed/SMS-Spam-Collection-Dataset-Kaggle/main/spam.csv

If the dataset is not already present locally, the training script will download it automatically and save it in the dataset folder.

## Model

- Algorithm: Multinomial Naive Bayes
- Feature extraction: CountVectorizer with English stop words removed
- Expected performance: approximately 98% accuracy

## Verified Performance

The trained model achieved the following results:

- Accuracy: 0.9834
- Precision: 0.9580
- Recall: 0.9048
- F1 Score: 0.9306

## Prerequisites

Make sure you have the following installed:

- Python 3.11 or later
- pip
- Docker (optional, for containerized deployment)

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Project Locally

### 1. Train the model

```bash
python train.py
```

This step prepares the model files in the model folder.

### 2. Start the API and frontend

```bash
python api.py
```

Then open:

- http://127.0.0.1:8000/
- http://127.0.0.1:8000/docs

## Docker Usage

Build and run the application in a container:

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
- Improve preprocessing with stemming or lemmatization
- Deploy the project to a cloud platform such as Azure or AWS
