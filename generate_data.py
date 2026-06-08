"""
BankShield — Synthetic Transaction Data Generator
Generates PaySim-style mobile financial transaction data.
Fraud occurs only in CASH_OUT and TRANSFER types, matching real-world patterns.
Run: python generate_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED  = 42
N_SAMPLES    = 50_000
FRAUD_RATE   = 0.10   # ~10% fraud — matches notebook dataset

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# Type mix (approximate PaySim distribution)
TYPE_PROBS_LEGIT  = [0.22, 0.30, 0.05, 0.33, 0.10]
TYPE_PROBS_FRAUD  = [0.00, 0.50, 0.00, 0.00, 0.50]  # fraud only in CASH_OUT & TRANSFER


def _names(prefix: str, n: int, rng) -> list[str]:
    ids = rng.integers(100_000, 999_999, size=n)
    return [f"{prefix}{i}" for i in ids]


def generate_legitimate(n: int, rng: np.random.Generator) -> pd.DataFrame:
    types  = rng.choice(TRANSACTION_TYPES, size=n, p=TYPE_PROBS_LEGIT)
    amount = rng.lognormal(mean=5.5, sigma=1.8, size=n).clip(1, 500_000)

    old_orig = rng.lognormal(mean=7.0, sigma=2.0, size=n).clip(0, 5_000_000)
    new_orig = np.maximum(0, old_orig - amount * rng.uniform(0.8, 1.0, size=n))

    old_dest = rng.lognormal(mean=6.5, sigma=2.2, size=n).clip(0, 5_000_000)
    new_dest = old_dest + amount * rng.uniform(0.9, 1.0, size=n)

    return pd.DataFrame({
        "step":           rng.integers(1, 744, size=n),   # up to 744 hours (31 days)
        "type":           types,
        "amount":         amount.round(2),
        "nameOrig":       _names("C", n, rng),
        "oldbalanceOrg":  old_orig.round(2),
        "newbalanceOrig": new_orig.round(2),
        "nameDest":       _names("M", n, rng),
        "oldbalanceDest": old_dest.round(2),
        "newbalanceDest": new_dest.round(2),
        "isFraud":        0,
    })


def generate_fraudulent(n: int, rng: np.random.Generator) -> pd.DataFrame:
    types  = rng.choice(TRANSACTION_TYPES, size=n, p=TYPE_PROBS_FRAUD)

    # Fraud: large amounts that drain the origin account
    old_orig = rng.lognormal(mean=7.5, sigma=1.8, size=n).clip(100, 10_000_000)
    amount   = old_orig * rng.uniform(0.85, 1.0, size=n)   # drain the account
    new_orig = np.zeros(n)                                  # account drained to 0

    # Destination: zero balance (mule account)
    old_dest = rng.choice([0.0], size=n) * np.ones(n)
    new_dest = np.zeros(n)                                  # money disappears

    return pd.DataFrame({
        "step":           rng.integers(1, 744, size=n),
        "type":           types,
        "amount":         amount.round(2),
        "nameOrig":       _names("C", n, rng),
        "oldbalanceOrg":  old_orig.round(2),
        "newbalanceOrig": new_orig.round(2),
        "nameDest":       _names("C", n, rng),   # fraud: C->C, not C->M
        "oldbalanceDest": old_dest.round(2),
        "newbalanceDest": new_dest.round(2),
        "isFraud":        1,
    })


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    n_fraud = int(N_SAMPLES * FRAUD_RATE)
    n_legit = N_SAMPLES - n_fraud

    print(f"Generating {n_legit:,} legitimate + {n_fraud:,} fraudulent transactions...")

    df = pd.concat(
        [generate_legitimate(n_legit, rng), generate_fraudulent(n_fraud, rng)],
        ignore_index=True,
    ).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    out_path = Path("data/transactions.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df):,} rows -> {out_path}")
    print(f"Fraud rate: {df['isFraud'].mean():.2%}")
    print(f"\nTransaction type breakdown:")
    print(df.groupby("type")["isFraud"].agg(["count", "sum", "mean"])
           .rename(columns={"count": "total", "sum": "fraud_n", "mean": "fraud_rate"})
           .assign(fraud_rate=lambda d: d["fraud_rate"].map("{:.2%}".format))
           .to_string())


if __name__ == "__main__":
    main()
