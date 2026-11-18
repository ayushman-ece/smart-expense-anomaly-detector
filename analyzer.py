
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest


NUMERIC_FEATURES = ["amount", "hour", "day_of_week"]
CATEGORICAL_FEATURES = ["category"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

VALID_CATEGORIES = {
    "Groceries", "Food", "Transport", "Shopping", "Electronics"
}


def load_expenses(file_path):
    df = pd.read_csv(file_path)

    required = ["transaction_id"] + FEATURES

    if not all(col in df.columns for col in required):
        raise ValueError("Required columns are missing.")

    if df.empty or df[required].isnull().any().any():
        raise ValueError("Dataset is empty or contains missing values.")

    if df["transaction_id"].duplicated().any():
        raise ValueError("Duplicate transaction IDs found.")

    for col in ["transaction_id"] + NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="raise")

    if not np.isfinite(
        df[["transaction_id"] + NUMERIC_FEATURES].to_numpy()
    ).all():
        raise ValueError("Invalid numeric values found.")

    if (df[NUMERIC_FEATURES] < 0).any().any():
        raise ValueError("Negative values are not allowed.")

    if not df["hour"].between(0, 23).all():
        raise ValueError("Hour must be between 0 and 23.")

    if not df["day_of_week"].between(1, 7).all():
        raise ValueError("Day must be between 1 and 7.")

    if not df["category"].isin(VALID_CATEGORIES).all():
        raise ValueError("Unknown expense category.")

    print(f"Loaded {len(df)} transactions successfully.")
    return df


def analyze_expenses(df):
    print("\n========== EXPENSE ANALYSIS ==========")

    print("Total transactions:", len(df))
    print(f"Total spending: ₹{df['amount'].sum():,.2f}")
    print(f"Average expense: ₹{df['amount'].mean():,.2f}")

    print("\nCategory-wise spending:")
    print(df.groupby("category")["amount"].sum().sort_values(
        ascending=False
    ))


def train_anomaly_model(df):
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("category", OneHotEncoder(handle_unknown="ignore"),
             CATEGORICAL_FEATURES)
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("detector", IsolationForest(
            n_estimators=100,
            contamination=0.15,
            random_state=42
        ))
    ])

    model.fit(df[FEATURES])

    print("\nIsolation Forest model trained successfully.")
    return model


def detect_anomalies(model, df):
    results = df.copy()

    predictions = model.predict(df[FEATURES])

    results["status"] = np.where(
        predictions == -1, "Anomaly", "Normal"
    )

    # Higher score means more unusual.
    results["anomaly_score"] = -model.decision_function(
        df[FEATURES]
    )

    print("\n========== ANOMALY DETECTION ==========")
    print(results["status"].value_counts())

    print("\nFlagged transactions:")
    print(
        results[results["status"] == "Anomaly"][
            ["transaction_id", "amount", "category", "status",
             "anomaly_score"]
        ].to_string(index=False)
    )

    return results


def predict_new_expense(model, amount, hour, day, category):
    if not np.isfinite([amount, hour, day]).all():
        raise ValueError("Inputs must be finite numbers.")

    if amount <= 0:
        raise ValueError("Amount must be greater than zero.")

    if not 0 <= hour <= 23:
        raise ValueError("Hour must be between 0 and 23.")

    if not 1 <= day <= 7:
        raise ValueError("Day must be between 1 and 7.")

    if category not in VALID_CATEGORIES:
        raise ValueError("Invalid expense category.")

    transaction = pd.DataFrame([{
        "amount": amount,
        "hour": hour,
        "day_of_week": day,
        "category": category
    }])

    prediction = model.predict(transaction)[0]
    score = -model.decision_function(transaction)[0]

    print("\n========== NEW EXPENSE CHECK ==========")
    print(f"Amount: ₹{amount:,.2f}")
    print(f"Category: {category}")
    print("Result:", "ANOMALY" if prediction == -1 else "NORMAL")
    print(f"Anomaly score: {score:.4f}")

    return prediction, score


def visualize_anomalies(results):
    normal = results[results["status"] == "Normal"]
    anomalies = results[results["status"] == "Anomaly"]

    plt.figure(figsize=(9, 5))

    plt.scatter(
        normal["transaction_id"],
        normal["amount"],
        label="Normal"
    )

    plt.scatter(
        anomalies["transaction_id"],
        anomalies["amount"],
        label="Anomaly",
        marker="x"
    )

    plt.xlabel("Transaction ID")
    plt.ylabel("Expense Amount (₹)")
    plt.title("Expense Anomaly Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig("expense_anomalies.png")
    plt.close()

    print("\nChart saved to expense_anomalies.png")


def export_results(results):
    results.to_csv("expense_anomaly_report.csv", index=False)
    print("Report saved to expense_anomaly_report.csv")
