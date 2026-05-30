"""
Train a Random Forest model on the PhiUSIIL phishing URL dataset.

Critical design choice: we ONLY train on features that the CLI can extract
from a raw URL string at prediction time. The dataset includes ~54 features
total, but many of them (page content, WHOIS data, etc.) aren't available
when a user types a URL into our CLI. Training on those would mean the
production model never sees them — they'd always be 0 — and predictions
would be garbage.

Equally important: we re-extract features from the raw URL strings using the
*same code* the CLI uses at prediction time, rather than using the dataset's
pre-computed columns. The dataset's definitions for several features (e.g.
NoOfOtherSpecialCharsInURL, TLDLegitimateProb) differ from our implementation,
so training on the dataset values would cause systematic misclassification of
real-world URLs.

Usage:
    python -m src.train

Outputs:
    models/phishing_model.pkl       — trained model bundle
    models/training_metrics.json    — metrics on the held-out test set
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.features import extract_features

# ---------- Config ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "PhiUSIIL_Phishing_URL_Dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "phishing_model.pkl"
METRICS_PATH = PROJECT_ROOT / "models" / "training_metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def get_cli_extractable_features() -> list[str]:
    """
    Return the list of feature names that src.features.extract_features() produces.

    We get this by running it on a sample URL and listing the keys. This way
    the training script is always in sync with the CLI — if you add a new
    feature to features.py, it automatically gets used here too.
    """
    sample = extract_features("https://example.com")
    return list(sample.keys())


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}.\n"
            "Download from UCI and place CSV in data/."
        )
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns from dataset.")
    return df


def compute_features_from_urls(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Re-extract features from raw URLs using the same code the CLI uses at
    prediction time.

    The dataset ships with pre-computed feature columns, but their definitions
    don't always match ours (e.g. 'NoOfOtherSpecialCharsInURL' uses a different
    character set; 'TLDLegitimateProb' is a continuous probability in the dataset
    but we emit 0/1). Training on the dataset's values produces a model that sees
    different numbers at prediction time than it saw during training, causing
    systematic misclassification of simple legitimate URLs.

    By re-extracting here, training and prediction are guaranteed to use identical
    feature values.
    """
    feature_names = get_cli_extractable_features()
    print(f"\nRe-extracting {len(feature_names)} features from {len(df):,} URLs "
          f"(ensures training/prediction consistency)...")

    rows = []
    for i, url in enumerate(df["URL"]):
        try:
            feats = extract_features(str(url))
        except Exception:
            feats = {}
        rows.append([float(feats.get(f, 0.0)) for f in feature_names])
        if (i + 1) % 50_000 == 0:
            print(f"  {i + 1:,} / {len(df):,}")

    X = pd.DataFrame(rows, columns=feature_names)
    print(f"Done. Feature matrix: {X.shape}")

    print(f"\nTraining feature set ({len(feature_names)} features):")
    for f in feature_names:
        print(f"  - {f}")
    return X, feature_names


def train(X: pd.DataFrame, y: pd.Series) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTrain: {len(X_train):,}   Test: {len(X_test):,}")

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    print("Training Random Forest...")
    rf.fit(X_train, y_train)
    print("Done.")

    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test,  y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test,    y_pred), 4),
        "f1":        round(f1_score(y_test,        y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test,   y_proba), 4),
        "train_size": len(X_train),
        "test_size":  len(X_test),
    }

    print("\n----- Test set performance -----")
    for k, v in metrics.items():
        print(f"  {k:<10} {v}")

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion matrix:")
    print(f"  Phishing missed (FN):  {cm[0, 1]:,}")
    print(f"  Legit flagged   (FP):  {cm[1, 0]:,}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["phishing", "legitimate"]))

    return {"model": rf, "metrics": metrics, "confusion_matrix": cm.tolist()}


def save(result: dict, features: list[str]) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": result["model"],
        "scaler": None,
        "feature_names": features,
        "model_type": "RandomForestClassifier",
        "metrics": result["metrics"],
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nSaved model    -> {MODEL_PATH.relative_to(PROJECT_ROOT)}")

    METRICS_PATH.write_text(json.dumps({
        "model_type": "RandomForestClassifier",
        "metrics": result["metrics"],
        "confusion_matrix": result["confusion_matrix"],
        "feature_count": len(features),
    }, indent=2))
    print(f"Saved metrics  -> {METRICS_PATH.relative_to(PROJECT_ROOT)}")


def main() -> None:
    df = load_data()
    X, features = compute_features_from_urls(df)
    result = train(X, df["label"])
    save(result, features)
    print("\nDone. Model trained on features computed identically to how the CLI computes them.")


if __name__ == "__main__":
    main()
