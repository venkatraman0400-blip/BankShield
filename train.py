"""
BankShield — Model Training Pipeline
Random Forest classifier with feature engineering for mobile financial fraud detection.
Run: python train.py
Output: models/model.pkl  +  models/feature_names.pkl  +  models/label_encoder.pkl
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH  = Path("data/transactions.csv")
MODEL_DIR  = Path("models")
MODEL_PATH = MODEL_DIR / "model.pkl"
FEAT_PATH  = MODEL_DIR / "feature_names.pkl"
LE_PATH    = MODEL_DIR / "label_encoder.pkl"

RANDOM_SEED = 42
TEST_SIZE   = 0.20

# Features fed to the model (all engineered from raw inputs)
FEATURE_COLS = [
    "step", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "type_encoded",
    "balanceOrigDiff", "balanceDestDiff",
    "errorBalanceOrig", "errorBalanceDest",
    "isHighAmount", "zeroBalanceOrig", "zeroBalanceDest",
    "amountToBalance", "amountToDestBalance", "accountDrained",
]
TARGET_COL = "isFraud"


# ── Feature engineering (mirrors notebook engineer_features) ───────────────────
def engineer_features(df: pd.DataFrame, le: LabelEncoder = None):
    df = df.copy()

    df["balanceOrigDiff"]    = df["newbalanceOrig"] - df["oldbalanceOrg"]
    df["balanceDestDiff"]    = df["newbalanceDest"] - df["oldbalanceDest"]
    df["errorBalanceOrig"]   = df["amount"] - (df["oldbalanceOrg"] - df["newbalanceOrig"])
    df["errorBalanceDest"]   = df["amount"] - (df["newbalanceDest"] - df["oldbalanceDest"])
    df["amountToBalance"]    = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["amountToDestBalance"]= df["amount"] / (df["oldbalanceDest"] + 1)
    df["isHighAmount"]       = (df["amount"] > df["amount"].quantile(0.75)).astype(int)
    df["zeroBalanceOrig"]    = (df["oldbalanceOrg"] == 0).astype(int)
    df["zeroBalanceDest"]    = (df["oldbalanceDest"] == 0).astype(int)
    df["accountDrained"]     = (df["newbalanceOrig"] == 0).astype(int)

    if le is not None:
        df["type_encoded"] = le.transform(df["type"])
    else:
        le_new = LabelEncoder()
        df["type_encoded"] = le_new.fit_transform(df["type"])

    return df


# ── 1. Load data ───────────────────────────────────────────────────────────────
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python generate_data.py` first."
        )
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows | Fraud rate: {df[TARGET_COL].mean():.2%}")
    return df


# ── 2. Split ───────────────────────────────────────────────────────────────────
def split(X, y):
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )


# ── 3. Train Random Forest ────────────────────────────────────────────────────
def train_model(X_train, y_train) -> RandomForestClassifier:
    print("\nTraining Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight="balanced",   # handles class imbalance without SMOTE
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


# ── 4. Evaluate ───────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc       = roc_auc_score(y_test, y_prob)
    f1        = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print("  BankShield -- Evaluation Results")
    print("=" * 50)
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print("=" * 50)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")

    return {"auc": auc, "f1": f1, "precision": precision, "recall": recall}


# ── 5. Save artefacts ─────────────────────────────────────────────────────────
def save(model, feature_names, le):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model,         MODEL_PATH)
    joblib.dump(feature_names, FEAT_PATH)
    joblib.dump(le,            LE_PATH)
    print(f"\nModel saved         -> {MODEL_PATH}")
    print(f"Features saved      -> {FEAT_PATH}")
    print(f"Label encoder saved -> {LE_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  BankShield -- Training Pipeline")
    print("=" * 50)

    df = load_data()

    # Fit label encoder on full dataset before split
    le = LabelEncoder()
    le.fit(df["type"])

    df_feat = engineer_features(df, le=le)

    X = df_feat[FEATURE_COLS]
    y = df_feat[TARGET_COL]

    X_train, X_test, y_train, y_test = split(X, y)
    print(f"\nTrain: {len(X_train):,} samples | Test: {len(X_test):,} samples")

    model   = train_model(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    save(model, FEATURE_COLS, le)

    print("\nTraining complete.")
    return model, metrics


if __name__ == "__main__":
    main()
