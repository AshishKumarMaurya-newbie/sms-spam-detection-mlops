"""Train a spam detection model using the UCI SMS Spam Collection dataset."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
MODEL_DIR = PROJECT_ROOT / "model"
DATASET_FILE = DATASET_DIR / "spam.csv"
LEGACY_DATASET_FILE = DATASET_DIR / "SMSSpamCollection"


def ensure_directories() -> None:
    """Create the dataset and model directories if they do not exist."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset() -> None:
    """Download the requested SMS Spam Collection CSV dataset if it is not already present."""
    if DATASET_FILE.exists():
        logger.info("Dataset already exists. Skipping download.")
        return

    url = "https://raw.githubusercontent.com/mdrsyed/SMS-Spam-Collection-Dataset-Kaggle/main/spam.csv"
    logger.info("Downloading dataset from %s", url)

    try:
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=30) as response:
            content = response.read()
            if not content:
                raise ValueError("Downloaded dataset is empty.")
            DATASET_FILE.write_bytes(content)
            logger.info("Dataset downloaded successfully.")
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        logger.warning("Primary download failed: %s", exc)
        if LEGACY_DATASET_FILE.exists():
            logger.info("Creating local spam.csv from the existing legacy dataset file.")
            legacy_df = pd.read_csv(LEGACY_DATASET_FILE, sep="\t", header=None, names=["label", "message"])
            legacy_df[["label", "message"]].to_csv(DATASET_FILE, index=False)
            logger.info("Fallback dataset created at %s", DATASET_FILE)
        else:
            raise RuntimeError(
                "Unable to download the requested dataset and no local fallback dataset is available."
            ) from exc


def clean_text(text: str) -> str:
    """Normalize SMS text using regex tokenization and structural maps for robust Naive Bayes performance."""
    if not isinstance(text, str):
        return ""
    
    # 1. Standardize case early
    text = text.lower()
    
    # 2. Map URLs and numerical indicators to generic variables before character stripping
    text = re.sub(r'http\S+|www\S+', 'urltoken', text)
    text = re.sub(r'\d+', 'numbertoken', text)
    
    # 3. Collapse aggressive token separation and punctuation padding
    text = re.sub(r'[^\w\s]', '', text)
    
    # 4. Standardize space blocks
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset() -> pd.DataFrame:
    """Load and prepare the SMS dataset for training."""
    if not DATASET_FILE.exists():
        raise FileNotFoundError("The dataset file was not found.")

    logger.info("Loading dataset.")
    df = pd.read_csv(DATASET_FILE)

    if "v1" in df.columns and "v2" in df.columns:
        df = df.rename(columns={"v1": "label", "v2": "message"})
    elif "label" in df.columns and "message" in df.columns:
        df = df[["label", "message"]].copy()
    else:
        raise ValueError("Unexpected dataset columns. Expected 'label'/'message' or 'v1'/'v2'.")

    df = df[["label", "message"]].copy()
    df["label"] = df["label"].map({"spam": 1, "ham": 0})
    df["message"] = df["message"].astype(str).apply(clean_text)

    df = df.drop_duplicates().reset_index(drop=True)
    df = df.dropna().reset_index(drop=True)

    if df.empty:
        raise ValueError("The dataset is empty after cleaning.")

    logger.info("Dataset shape after cleaning: %s", df.shape)
    return df


def split_dataset(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split the dataset into training and testing subsets."""
    X = df["message"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.Series, y_train: pd.Series) -> Tuple[TfidfVectorizer, MultinomialNB]:
    """Train a Naive Bayes classifier with improved text features."""
    vectorizer = TfidfVectorizer(
    analyzer='char_wb', 
    ngram_range=(3, 5), 
    min_df=2
)
    X_train_vec = vectorizer.fit_transform(X_train)

    model = MultinomialNB(alpha=0.1, fit_prior=False)
    model.fit(X_train_vec, y_train)

    return vectorizer, model


def evaluate_model(model: MultinomialNB, vectorizer: TfidfVectorizer, X_test: pd.Series, y_test: pd.Series) -> None:
    """Evaluate the trained model and print the performance metrics."""
    X_test_vec = vectorizer.transform(X_test)
    predictions = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=["ham", "spam"])

    logger.info("Accuracy: %.4f", accuracy)
    logger.info("Precision: %.4f", precision)
    logger.info("Recall: %.4f", recall)
    logger.info("F1 Score: %.4f", f1)
    logger.info("Confusion Matrix:\n%s", cm)
    logger.info("Classification Report:\n%s", report)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)


def save_artifacts(vectorizer: TfidfVectorizer, model: MultinomialNB) -> None:
    """Persist the trained vectorizer and model to disk."""
    joblib.dump(model, MODEL_DIR / "spam_model.pkl")
    joblib.dump(vectorizer, MODEL_DIR / "vectorizer.pkl")
    logger.info("Model artifacts saved to %s", MODEL_DIR)


def main() -> None:
    """Run the full training pipeline."""
    ensure_directories()
    download_dataset()
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(df)
    vectorizer, model = train_model(X_train, y_train)
    evaluate_model(model, vectorizer, X_test, y_test)
    save_artifacts(vectorizer, model)


if __name__ == "__main__":
    main()
