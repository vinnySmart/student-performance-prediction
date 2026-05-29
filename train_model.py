"""
train_model.py
Run this script once to generate the serialized model artifacts:
    models/random_forest_model.pkl
    models/label_encoder.pkl
    models/scaler.pkl
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Synthetic Dataset Generation (2000 records) ──────────────────────────────
N = 2000

attendance = np.clip(np.random.normal(70, 20, N), 10, 100)
internal_marks = np.clip(np.random.normal(60, 20, N), 10, 100)
assignment_score = np.clip(np.random.normal(65, 18, N), 10, 100)
prev_gpa = np.clip(np.random.normal(6.5, 2.0, N), 2.0, 10.0)
study_hours = np.clip(np.random.normal(3.5, 2.0, N), 0, 12)
participation = np.clip(np.random.normal(5, 2.5, N), 0, 10)
gender = np.random.choice([0, 1], N)          # 0=Male, 1=Female
income = np.random.choice([0, 1, 2], N)       # 0=Low, 1=Med, 2=High
extra = np.random.choice([0, 1], N)

score = (
    0.214 * internal_marks
    + 0.198 * prev_gpa * 10
    + 0.156 * attendance
    + 0.112 * assignment_score
    + 0.098 * study_hours * 8.33
    + 0.087 * participation * 10
    + 0.062 * income * 33.3
    + 0.041 * extra * 50
    + 0.032 * gender * 20
    + np.random.normal(0, 5, N)
)

labels = np.where(score >= np.percentile(score, 70), 'High Performer',
         np.where(score >= np.percentile(score, 35), 'Average', 'At-Risk'))

df = pd.DataFrame({
    'attendance_pct': attendance,
    'internal_marks': internal_marks,
    'assignment_score': assignment_score,
    'prev_gpa': prev_gpa,
    'study_hours': study_hours,
    'participation': participation,
    'gender': gender,
    'income_level': income,
    'extra_curricular': extra,
    'label': labels,
})

# ── Preprocessing ─────────────────────────────────────────────────────────────
le = LabelEncoder()
y = le.fit_transform(df['label'])
X = df.drop('label', axis=1).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=SEED, stratify=y)

# ── Model Comparison ──────────────────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=SEED),
    'Decision Tree': DecisionTreeClassifier(random_state=SEED),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=SEED),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=SEED),
}

print("\n=== Algorithm Comparison ===")
for name, clf in models.items():
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"  {name:25s}: {acc*100:.1f}%")

# ── Hyperparameter Tuning: Random Forest ─────────────────────────────────────
print("\n=== Tuning Random Forest with GridSearchCV ===")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}
gs = GridSearchCV(
    RandomForestClassifier(random_state=SEED),
    param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=0,
)
gs.fit(X_train, y_train)
best_rf = gs.best_estimator_
print(f"  Best params: {gs.best_params_}")

# ── Final Evaluation ──────────────────────────────────────────────────────────
y_pred = best_rf.predict(X_test)
print(f"\n=== Final Random Forest Evaluation (n_test={len(y_test)}) ===")
print(f"  Accuracy : {accuracy_score(y_test, y_pred)*100:.1f}%")
print(classification_report(y_test, y_pred,
                              target_names=le.classes_))
print("  Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
feature_names = ['attendance_pct', 'internal_marks', 'assignment_score',
                 'prev_gpa', 'study_hours', 'participation',
                 'gender', 'income_level', 'extra_curricular']
importances = best_rf.feature_importances_
ranked = sorted(zip(feature_names, importances), key=lambda x: -x[1])
print("\n  Feature Importances:")
for feat, imp in ranked:
    print(f"    {feat:25s}: {imp:.3f}")

# ── Serialize Artifacts ───────────────────────────────────────────────────────
os.makedirs('models', exist_ok=True)
joblib.dump(best_rf, 'models/random_forest_model.pkl')
joblib.dump(le,      'models/label_encoder.pkl')
joblib.dump(scaler,  'models/scaler.pkl')
print("\n✓ Model artifacts saved to models/")
