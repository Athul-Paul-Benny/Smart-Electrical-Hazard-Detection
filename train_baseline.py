"""
Baseline classifier for the single-phase hazard dataset.
Run: python3 train_baseline.py
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

df = pd.read_csv("features_single_phase.csv")
X = df.drop(columns=["sample_id", "label"])
y = df["label"]

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
clf.fit(Xtr, ytr)
pred = clf.predict(Xte)

print(classification_report(yte, pred))
print("Confusion matrix (rows=true, cols=pred):")
labels = sorted(y.unique())
cm = confusion_matrix(yte, pred, labels=labels)
print(pd.DataFrame(cm, index=labels, columns=labels))

imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature importance:\n", imp)
