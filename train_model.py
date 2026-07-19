import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Load Dataset
# -----------------------------
DATASET_PATH = "data/dataset.csv"

print("Loading dataset...")

df = pd.read_csv(DATASET_PATH)

# Remove unwanted unnamed columns
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

print(f"Dataset Shape: {df.shape}")

# -----------------------------
# Check Required Column
# -----------------------------
if "prognosis" not in df.columns:
    raise ValueError(
        "Dataset must contain a 'prognosis' column.\n"
        f"Columns found:\n{list(df.columns)}"
    )

# -----------------------------
# Features and Labels
# -----------------------------
X = df.drop(columns=["prognosis"])
y = df["prognosis"]

print(f"Number of Symptoms: {X.shape[1]}")
print(f"Number of Diseases: {len(y.unique())}")

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Train Model
# -----------------------------
print("Training Random Forest Model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print(f"Model Accuracy : {accuracy * 100:.2f}%")
print("==============================\n")

print(classification_report(y_test, y_pred))

# -----------------------------
# Save Model
# -----------------------------
os.makedirs("model", exist_ok=True)

MODEL_PATH = "model/model.pkl"

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"\n✅ Model saved successfully!")
print(f"📁 Location: {MODEL_PATH}")

# Save feature names
with open("model/features.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

print("✅ Features saved!")
