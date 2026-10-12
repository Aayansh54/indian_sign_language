
from pathlib import Path
import csv
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = Path(r"D:\ISL\data\landmarks.csv")
MODEL_DIR = Path(r"D:\ISL\model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Loading landmark data...")

with DATA_PATH.open("r", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

X = np.asarray([row[2:] for row in rows], dtype=np.float32)
y = np.asarray([row[0] for row in rows])

print("Samples:", len(y))
print("Features per sample:", X.shape[1])
print("Classes:", len(np.unique(y)))

# Hold out data the model won't train on.
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=150,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

predictions = model.predict(X_val)
accuracy = accuracy_score(y_val, predictions)

print(f"\nValidation accuracy: {accuracy:.2%}")
print("\nClassification report:")
print(classification_report(y_val, predictions, zero_division=0))

model_path = MODEL_DIR / "isl_alphabet_model.joblib"
joblib.dump(model, model_path)

print("\nModel saved to:", model_path)
print("Training complete.")
