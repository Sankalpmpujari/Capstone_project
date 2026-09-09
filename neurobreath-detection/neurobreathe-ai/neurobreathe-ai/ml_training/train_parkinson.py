import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

# 1. Load
df = pd.read_csv("data/parkinson/parkinsons.data")
print("Shape:", df.shape)
print(df["status"].value_counts())

X = df.drop(columns=["name", "status"])
y = df["status"]

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Feature selection
selector_model = XGBClassifier(n_estimators=200, random_state=42)
selector_model.fit(X_train_scaled, y_train)
selector = SelectFromModel(selector_model, threshold="median", prefit=True)
X_train_sel = selector.transform(X_train_scaled)
X_test_sel = selector.transform(X_test_scaled)
print("Kept features:", list(X.columns[selector.get_support()]))

# 5. Train
clf = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X_train_sel, y_train, cv=cv, scoring="accuracy")
print(f"5-fold CV accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

clf.fit(X_train_sel, y_train)

# 6. Evaluate
y_pred = clf.predict(X_test_sel)
y_proba = clf.predict_proba(X_test_sel)[:, 1]
print(classification_report(y_test, y_pred))
print("Test ROC-AUC:", roc_auc_score(y_test, y_proba))

# 7. Save
os.makedirs("models/parkinson", exist_ok=True)
joblib.dump(clf, "models/parkinson/xgb_model.pkl")
joblib.dump(scaler, "models/parkinson/scaler.pkl")
joblib.dump(selector, "models/parkinson/selector.pkl")
joblib.dump(list(X.columns), "models/parkinson/feature_order.pkl")
print("Saved models to models/parkinson/")