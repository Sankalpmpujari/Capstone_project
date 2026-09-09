import os
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier

df = pd.read_csv("data/respiratory/extracted_features.csv")

# Binary framing: what the patient counts can actually support
df["diagnosis"] = df["diagnosis"].apply(lambda d: "COPD" if d == "COPD" else "Not_COPD")

patient_counts = df.drop_duplicates("patient_id")["diagnosis"].value_counts()
print("Unique patients per class:")
print(patient_counts)

X = df.drop(columns=["patient_id", "diagnosis"])
y = df["diagnosis"]
groups = df["patient_id"]

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
clf = LGBMClassifier(
    n_estimators=400, learning_rate=0.03, num_leaves=31,
    class_weight="balanced", random_state=42, verbosity=-1
)

y_pred_oof = cross_val_predict(clf, X, y, cv=sgkf, groups=groups)
print("\nOut-of-fold report:")
print(classification_report(y, y_pred_oof, zero_division=0))

# ↓↓↓ NEW BLOCK — ADD THIS ↓↓↓
results_df = pd.DataFrame({
    "patient_id": df["patient_id"].values,
    "y_true": y.values,
    "y_pred": y_pred_oof
})

patient_level = results_df.groupby("patient_id").agg(
    y_true=("y_true", lambda x: x.mode()[0]),
    y_pred=("y_pred", lambda x: x.mode()[0])
).reset_index()

print("\nPatient-level report (majority vote across clips):")
print(classification_report(patient_level["y_true"], patient_level["y_pred"], zero_division=0))
# ↑↑↑ NEW BLOCK — END ↑↑↑

clf.fit(X, y)
os.makedirs("models/respiratory", exist_ok=True)
joblib.dump(clf, "models/respiratory/lgbm_model.pkl")
joblib.dump(list(X.columns), "models/respiratory/feature_order.pkl")
print("Saved models to models/respiratory/")