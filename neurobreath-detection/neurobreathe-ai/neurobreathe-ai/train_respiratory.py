import os
import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier

df = pd.read_csv("data/respiratory/extracted_features.csv")
print("Shape:", df.shape)
print(df["diagnosis"].value_counts())

X = df.drop(columns=["patient_id", "diagnosis"])
y = df["diagnosis"]
groups = df["patient_id"]

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# sanity check: no patient overlap between train and test
assert set(groups.iloc[train_idx]).isdisjoint(set(groups.iloc[test_idx]))
print("Patient split verified: no leakage")

clf = LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

os.makedirs("models/respiratory", exist_ok=True)
joblib.dump(clf, "models/respiratory/lgbm_model.pkl")
joblib.dump(list(X.columns), "models/respiratory/feature_order.pkl")
print("Saved models to models/respiratory/")