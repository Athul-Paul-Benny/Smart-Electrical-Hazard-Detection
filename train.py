"""
AI-Based Electrical Hazard & Arc-Fault Detection Pipeline
Primary Model Training and Evaluation Script (train.py)

Performs:
1. Data loading & verification (features_single_phase.csv)
2. Stratified 70% Train / 15% Val / 15% Test split
3. Preprocessing (StandardScaler fit strictly on training set)
4. Evaluates Candidate Models:
   - Random Forest
   - XGBoost
   - Extra Trees
   - HistGradientBoosting
   - Logistic Regression
5. Model selection based on Validation Macro F1 & Recall
6. Unbiased single final evaluation on untouched Test set
7. Model artifact persistence (joblib + JSON metadata)
8. Visualizations (Confusion Matrix, Feature Importances, Model Comparison)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def set_seed(seed=42):
    np.random.seed(seed)


def load_dataset(csv_path="data/features_single_phase.csv"):
    if not os.path.exists(csv_path):
        # Fallback to root directory if data folder copy is not present
        if os.path.exists("features_single_phase.csv"):
            csv_path = "features_single_phase.csv"
        else:
            raise FileNotFoundError(f"Dataset file not found at {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded dataset from '{csv_path}' with shape: {df.shape}")

    # Validate nulls
    assert df.isna().sum().sum() == 0, "Dataset contains unexpected missing values!"

    # Drop ID column if present
    if "sample_id" in df.columns:
        df = df.drop(columns=["sample_id"])

    X = df.drop(columns=["label"])
    y = df["label"]

    feature_names = list(X.columns)
    classes = list(np.unique(y))
    print(f"[INFO] Features count: {len(feature_names)}")
    print(f"[INFO] Target classes ({len(classes)}): {classes}")
    print("[INFO] Class breakdown:")
    print(y.value_counts())

    return X, y, feature_names, classes


def split_data(X, y, seed=42):
    """
    Split data into 70% Train, 15% Validation, 15% Test.
    Uses stratification to ensure equal class proportions across all splits.
    """
    # First split off 15% for final test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=seed
    )

    # Next split remaining 85% into train (70% total) and val (15% total)
    # 0.15 / 0.85 = ~0.17647
    val_ratio = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=seed
    )

    print("\n[INFO] Data split summary:")
    print(f"  - Training Set:   {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"  - Validation Set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"  - Test Set:       {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_candidate_models(classes, seed=42):
    label_encoder = LabelEncoder()
    encoded_classes = label_encoder.fit_transform(classes)

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=seed, class_weight="balanced", max_depth=12
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=seed,
            eval_metric="mlogloss",
            verbosity=0,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300, random_state=seed, class_weight="balanced"
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=seed, max_iter=200),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=seed, class_weight="balanced"
        ),
    }

    return models, label_encoder


def evaluate_model_performance(model, X_scaled, y_true, label_encoder=None, is_xgb=False):
    """
    Computes accuracy, macro precision, recall, macro F1, weighted F1, and multi-class ROC-AUC.
    """
    if is_xgb and label_encoder is not None:
        y_eval = label_encoder.transform(y_true)
        preds_num = model.predict(X_scaled)
        preds = label_encoder.inverse_transform(preds_num)
        probs = model.predict_proba(X_scaled)
    else:
        preds = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled) if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_true, preds)
    prec_macro = precision_score(y_true, preds, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, preds, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, preds, average="weighted", zero_division=0)

    roc_auc = None
    if probs is not None:
        try:
            # One-vs-Rest multi-class ROC AUC
            if is_xgb and label_encoder is not None:
                roc_auc = roc_auc_score(y_eval, probs, multi_class="ovr", average="macro")
            else:
                roc_auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
        except Exception:
            roc_auc = None

    return {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "roc_auc_macro": float(roc_auc) if roc_auc is not None else None,
        "predictions": preds,
        "probabilities": probs,
    }


def main():
    set_seed(42)

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # 1. Load Data
    X, y, feature_names, classes = load_dataset()

    # 2. Train/Val/Test Split
    X_tr, X_val, X_te, y_tr, y_val, y_te = split_data(X, y)

    # 3. Data Preprocessing & Scaling (Fit scaler strictly on Train data)
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)
    X_te_scaled = scaler.transform(X_te)

    # 4. Train & Compare Candidate Models on Validation Set
    candidate_models, label_encoder = build_candidate_models(classes)
    val_results = {}
    trained_model_objs = {}

    y_tr_xgb = label_encoder.transform(y_tr)

    print("\n=======================================================")
    print(" CANDIDATE MODEL EVALUATION (VALIDATION SET)")
    print("=======================================================")

    best_name = None
    best_val_f1 = -1.0

    for name, model in candidate_models.items():
        is_xgb = name == "XGBoost"
        if is_xgb:
            model.fit(X_tr_scaled, y_tr_xgb)
        else:
            model.fit(X_tr_scaled, y_tr)

        metrics = evaluate_model_performance(
            model, X_val_scaled, y_val, label_encoder=label_encoder, is_xgb=is_xgb
        )
        val_results[name] = metrics
        trained_model_objs[name] = model

        f1_m = metrics["f1_macro"]
        acc = metrics["accuracy"]
        rec = metrics["recall_macro"]
        print(f" -> {name:22s} | Val Acc: {acc:.4f} | Val Macro F1: {f1_m:.4f} | Val Recall: {rec:.4f}")

        if f1_m > best_val_f1:
            best_val_f1 = f1_m
            best_name = name

    print(f"\n[WINNER] Best Model selected based on Validation Macro F1: {best_name} ({best_val_f1:.4f})")

    best_model = trained_model_objs[best_name]
    is_best_xgb = best_name == "XGBoost"

    # 5. Untouched Final Test Set Evaluation
    print("\n=======================================================")
    print(f" UNTOUCHED TEST SET EVALUATION ({best_name})")
    print("=======================================================")

    test_metrics = evaluate_model_performance(
        best_model, X_te_scaled, y_te, label_encoder=label_encoder, is_xgb=is_best_xgb
    )

    print(f"Final Test Accuracy:     {test_metrics['accuracy']:.4f}")
    print(f"Final Test Macro F1:     {test_metrics['f1_macro']:.4f}")
    print(f"Final Test Weighted F1:  {test_metrics['f1_weighted']:.4f}")
    if test_metrics["roc_auc_macro"]:
        print(f"Final Test Macro ROC-AUC:{test_metrics['roc_auc_macro']:.4f}")

    y_test_preds = test_metrics["predictions"]
    report_str = classification_report(y_te, y_test_preds, digits=4)
    print("\nClassification Report (Test Set):")
    print(report_str)

    # Save text classification report
    with open("results/classification_report.txt", "w") as f:
        f.write(f"Best Selected Model: {best_name}\n\n")
        f.write(report_str)

    # 6. Save Plots
    # A. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_te, y_test_preds, labels=classes)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
    )
    plt.title(f"Confusion Matrix - {best_name} (Test Set)", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("plots/confusion_matrix.png", dpi=300)
    plt.close()

    # B. Feature Importance Plot
    feature_importances = {}
    if hasattr(best_model, "feature_importances_"):
        imps = best_model.feature_importances_
        feature_importances = dict(zip(feature_names, imps.tolist()))
    elif hasattr(best_model, "coef_"):
        imps = np.mean(np.abs(best_model.coef_), axis=0)
        feature_importances = dict(zip(feature_names, imps.tolist()))

    if feature_importances:
        imp_series = pd.Series(feature_importances).sort_values(ascending=True)
        plt.figure(figsize=(9, 6))
        imp_series.plot(kind="barh", color="#1f77b4")
        plt.title(f"Feature Importances - {best_name}", fontsize=14, fontweight="bold")
        plt.xlabel("Relative Importance Score", fontsize=12)
        plt.ylabel("Feature Name", fontsize=12)
        plt.grid(axis="x", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig("plots/feature_importance.png", dpi=300)
        plt.close()

    # C. Model Comparison Chart
    model_names = list(val_results.keys())
    val_f1s = [val_results[m]["f1_macro"] for m in model_names]
    val_accs = [val_results[m]["accuracy"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, val_accs, width, label="Val Accuracy", color="#4C72B0")
    plt.bar(x + width / 2, val_f1s, width, label="Val Macro F1", color="#55A868")
    plt.ylabel("Score", fontsize=12)
    plt.title("Candidate Model Validation Performance Comparison", fontsize=14, fontweight="bold")
    plt.xticks(x, model_names, rotation=20, ha="right")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/model_comparison.png", dpi=300)
    plt.close()

    # 7. Save Model & Metadata Artifacts
    model_filepath = "models/best_model.joblib"
    scaler_filepath = "models/preprocessor.joblib"
    metadata_filepath = "models/metadata.json"
    results_filepath = "results/metrics.json"

    joblib.dump(best_model, model_filepath)
    joblib.dump(scaler, scaler_filepath)
    print(f"\n[INFO] Saved best model to '{model_filepath}'")
    print(f"[INFO] Saved preprocessor scaler to '{scaler_filepath}'")

    metadata = {
        "selected_model": best_name,
        "is_xgb": is_best_xgb,
        "feature_names": feature_names,
        "classes": classes,
        "label_mapping": {c: int(i) for i, c in enumerate(label_encoder.classes_)},
        "metrics": {
            "validation_macro_f1": float(best_val_f1),
            "test_accuracy": float(test_metrics["accuracy"]),
            "test_macro_f1": float(test_metrics["f1_macro"]),
            "test_weighted_f1": float(test_metrics["f1_weighted"]),
            "test_roc_auc_macro": float(test_metrics["roc_auc_macro"])
            if test_metrics["roc_auc_macro"]
            else None,
        },
        "feature_importances": feature_importances,
    }

    with open(metadata_filepath, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"[INFO] Saved metadata to '{metadata_filepath}'")

    # Clean JSON serializable metrics for all candidates
    summary_results = {
        "best_model": best_name,
        "validation_candidate_comparison": {
            k: {
                "accuracy": v["accuracy"],
                "precision_macro": v["precision_macro"],
                "recall_macro": v["recall_macro"],
                "f1_macro": v["f1_macro"],
                "f1_weighted": v["f1_weighted"],
            }
            for k, v in val_results.items()
        },
        "final_test_results": {
            "accuracy": test_metrics["accuracy"],
            "precision_macro": test_metrics["precision_macro"],
            "recall_macro": test_metrics["recall_macro"],
            "f1_macro": test_metrics["f1_macro"],
            "f1_weighted": test_metrics["f1_weighted"],
        },
    }

    with open(results_filepath, "w") as f:
        json.dump(summary_results, f, indent=4)
    print(f"[INFO] Saved comprehensive metrics log to '{results_filepath}'")
    print("\n[SUCCESS] Pipeline training and evaluation complete!")


if __name__ == "__main__":
    main()
