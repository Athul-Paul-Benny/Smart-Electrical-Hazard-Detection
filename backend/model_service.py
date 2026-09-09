"""
AMPIX FastAPI Backend — Centralized Model Service (backend/model_service.py)
Single Source of Truth for Model Loading and Inference
"""

import os
import json
import joblib
import numpy as np
import pandas as pd


class ModelService:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.model = None
        self.scaler = None
        self.metadata = None
        self.feature_names = []
        self.classes = []
        self.is_xgb = False

        self._load_artifacts()

    def _load_artifacts(self):
        model_path = os.path.join(self.models_dir, "best_model.joblib")
        scaler_path = os.path.join(self.models_dir, "preprocessor.joblib")
        meta_path = os.path.join(self.models_dir, "metadata.json")

        if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(meta_path)):
            raise FileNotFoundError(
                f"Model artifacts missing in '{self.models_dir}'! Please run 'python train.py' first."
            )

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        with open(meta_path, "r") as f:
            self.metadata = json.load(f)

        self.feature_names = self.metadata["feature_names"]
        self.classes = self.metadata["classes"]
        self.is_xgb = self.metadata.get("is_xgb", False)

    def predict(self, feature_dict: dict):
        """
        Accepts a dictionary of feature values, aligns them with exact 17-feature column order,
        transforms via StandardScaler, and predicts class & probabilities.
        """
        # Align with exact trained feature order
        input_df = pd.DataFrame([feature_dict])[self.feature_names]
        scaled_feats = self.scaler.transform(input_df)

        if self.is_xgb and "label_mapping" in self.metadata:
            inv_map = {v: k for k, v in self.metadata["label_mapping"].items()}
            pred_idx = self.model.predict(scaled_feats)[0]
            pred_class = inv_map.get(pred_idx, str(pred_idx))
            probs_arr = self.model.predict_proba(scaled_feats)[0]
        else:
            pred_class = self.model.predict(scaled_feats)[0]
            probs_arr = self.model.predict_proba(scaled_feats)[0]

        prob_dict = {self.classes[i]: float(probs_arr[i]) for i in range(len(self.classes))}
        confidence = float(np.max(probs_arr)) * 100.0

        # Top 3 indicator features
        top_features = []
        importances = self.metadata.get("feature_importances", {})
        if importances:
            scores = {}
            for f_name in self.feature_names:
                val = input_df[f_name].iloc[0]
                imp = importances.get(f_name, 0.01)
                scores[f_name] = abs(val) * imp

            sorted_feats = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
            for f_name, _ in sorted_feats:
                val = input_df[f_name].iloc[0]
                top_features.append(f"{f_name} = {val:.4f}")

        return {
            "prediction": pred_class,
            "confidence": round(confidence, 2),
            "probabilities": prob_dict,
            "top_features": top_features,
            "feature_df": input_df,
        }


# Singleton instance
model_service = ModelService()
