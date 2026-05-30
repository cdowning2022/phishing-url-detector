"""
Train a baseline Logistic Regression model on the PhiUSIIL phishing URL dataset.

Usage:
    python -m src.train

Outputs:
    models/logreg_baseline.pkl        — trained model + scaler + feature list
    models/training_metrics.json      — accuracy, precision, recall, F1 on test set
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------- Config ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "PhiUSIIL_Phishing_URL_Dataset.csv"
FEATURE_LIST_PATH = PROJECT_ROOT / "data" / "feature_list.txt"
MODEL_PATH = PROJECT_ROOT / "models" / "logreg_baseline.pkl"
METRICS_PATH = PROJECT_ROOT / "models" / "training_metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20  # 20% held out for testing

# Features to exclude because they are suspected of being leakage.
# Day 2 EDA flagged URLSimilarityIndex as having near-perfect correlation with
# the label, which suggests it was computed using label information.
# We exclude it from the baseline to get an honest accuracy number, and document
# this choice in the writeup.
LEAKAGE_FEATURES = {"URLSimilarityIndex"}


def load_data() -> pd.DataFrame:
    """Load the raw dataset CSV."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}.\n"
            "Download it from the UCI link in the README and place it in data/."
        )
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns from {DATA_PATH.name}")
    return df


def load_feature_list() -> list[str]:
    """Load the feature list saved by the Day 2 EDA notebook."""
    if not FEATURE_LIST_PATH.exists():
        raise FileNotFoundError(
            f"Feature list not found at {FEATURE_LIST_PATH}.\n"
            "Run the Day 2 EDA notebook first — it saves the feature list."
        )
    features = FEATURE_LIST_PATH.read_text().strip().splitlines()
    print(f"Loaded {len(features)} features from feature_list.txt")
    return features


def prepare_features(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Build the X feature matrix and y label vector.

    Drops leakage-suspect features so we get an honest baseline.
    """
    # Drop leakage features if present
    used = [f for f in features if f in df.columns and f not in LEAKAGE_FEATURES]
    excluded = [f for f in features if f in LEAKAGE_FEATURES]
    if excluded:
        print(f"Excluding {len(excluded)} leakage-suspect feature(s): {excluded}")

    X = df[used].copy()
    y = df["label"].copy()

    # Any remaining NaN values would crash training — fill with 0 as a safe default.
    # (We confirmed on Day 1 there shouldn't be any, but defensive coding is cheap.)
    if X.isna().any().any():
        print("  Note: filling NaN values with 0 before training.")
        X = X.fillna(0)

    print(f"Final feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
    return X, y, used


def train_baseline(X: pd.DataFrame, y: pd.Series) -> dict:
    """Train a Logistic Regression model and report metrics on a held-out test set."""

    # Stratified split keeps the class balance the same in train and test.
    # Important when one class might be slightly larger than the other.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"Train set: {len(X_train):,}   Test set: {len(X_test):,}")

    # Logistic Regression benefits from feature scaling — features on very
    # different scales (e.g. URL length in the hundreds vs. binary flags)
    # would otherwise let the model put too much weight on the larger ones.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,           # default 100 is sometimes not enough to converge
        random_state=RANDOM_STATE,
        n_jobs=-1,               # use all CPU cores
    )

    print("Training Logistic Regression…")
    model.fit(X_train_scaled, y_train)
    print("Done.")

    # Evaluate
    y_pred = model.predict(X_test_scaled)

    metrics = {
        "accuracy":  round(accuracy_score(y_test,  y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test,    y_pred), 4),
        "f1":        round(f1_score(y_test,        y_pred), 4),
        "test_size": len(y_test),
        "train_size": len(y_train),
    }

    print("\n----- Test set performance -----")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<10} {v:.4f}")
        else:
            print(f"  {k:<10} {v:,}")

    print("\nConfusion matrix (rows = actual, cols = predicted):")
    print("              pred=phish  pred=legit")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  actual=phish  {cm[0,0]:>10,}  {cm[0,1]:>10,}")
    print(f"  actual=legit  {cm[1,0]:>10,}  {cm[1,1]:>10,}")

    print("\nDetailed report:")
    print(classification_report(y_test, y_pred, target_names=["phishing", "legitimate"]))

    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
    }


def save_artifacts(result: dict, feature_names: list[str]) -> None:
    """Persist the model + scaler + feature list so predict.py can load them later."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save everything the predict step needs as a single bundle.
    bundle = {
        "model": result["model"],
        "scaler": result["scaler"],
        "feature_names": feature_names,
        "model_type": "LogisticRegression",
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nSaved model bundle → {MODEL_PATH.relative_to(PROJECT_ROOT)}")

    # Save metrics as JSON for the README to reference later.
    METRICS_PATH.write_text(json.dumps({
        "model_type": "LogisticRegression",
        "metrics": result["metrics"],
        "confusion_matrix": result["confusion_matrix"],
    }, indent=2))
    print(f"Saved metrics    → {METRICS_PATH.relative_to(PROJECT_ROOT)}")


def main() -> None:
    df = load_data()
    features = load_feature_list()
    X, y, used = prepare_features(df, features)
    result = train_baseline(X, y)
    save_artifacts(result, used)
    print("\nDay 3 complete. Tomorrow: add Random Forest and compare.")


if __name__ == "__main__":
    main()
