
import pandas as pd
import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)
from sklearn.pipeline import Pipeline

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "product_reviews_cleaned.csv"
RESULTS_PATH = ROOT / "data" / "processed" / "model_results.json"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Use the actual lowercase column names
df["review"] = df["review"].fillna("").astype(str)
df["summary"] = df["summary"].fillna("").astype(str)
df["sentiment"] = df["sentiment"].fillna("").astype(str).str.strip().str.lower()

# Combine summary and review text
df["text"] = (
    df["summary"].str.strip() + " " + df["review"].str.strip()
).str.strip()

# Remove empty texts and missing sentiment labels
df = df[
    (df["text"].str.len() > 0) &
    (df["sentiment"] != "")
].copy()

# Check class counts
class_counts = df["sentiment"].value_counts()
valid_classes = class_counts[class_counts >= 2].index
df = df[df["sentiment"].isin(valid_classes)].copy()

X = df["text"]
y = df["sentiment"]

print("Dataset shape:", df.shape)
print("\nSentiment counts:")
print(y.value_counts())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create the ML pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    )
])

# Train the model
print("\nTraining Logistic Regression model...")
model.fit(X_train, y_train)

# Evaluate on unseen test data
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(
    y_test, predictions, average="macro", zero_division=0
)
weighted_f1 = f1_score(
    y_test, predictions, average="weighted", zero_division=0
)

print("\n===== MODEL PERFORMANCE =====")
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Macro F1-score: {macro_f1:.4f}")
print(f"Weighted F1-score: {weighted_f1:.4f}")

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, predictions, zero_division=0))

# Save actual metrics
results = {
    "algorithm": "TF-IDF + Logistic Regression",
    "input_features": ["summary", "review"],
    "target": "sentiment",
    "test_size": 0.20,
    "random_state": 42,
    "training_records": int(len(X_train)),
    "test_records": int(len(X_test)),
    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1)
}

RESULTS_PATH.write_text(
    json.dumps(results, indent=4),
    encoding="utf-8"
)

print(f"\nResults saved to: {RESULTS_PATH}")
