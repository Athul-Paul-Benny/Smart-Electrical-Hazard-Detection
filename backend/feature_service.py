"""
AMPIX FastAPI Backend — Centralized Feature Service (backend/feature_service.py)
Single Source of Truth for Feature Alignment and Signal Processing
"""

import numpy as np
import pandas as pd


class FeatureService:
    @staticmethod
    def process_and_align_features(input_data: dict, feature_names: list) -> dict:
        """
        Validates input dictionary, computes derived ratios (over_rated_ratio),
        and aligns all 17 feature names in exact required order.
        """
        feats = {}
        for col in feature_names:
            feats[col] = float(input_data.get(col, 0.0))

        # Compute over_rated_ratio if missing or zero
        rated = feats.get("rated_current_A", 16.0)
        if rated <= 0:
            rated = 16.0
            feats["rated_current_A"] = rated

        if "over_rated_ratio" not in input_data or input_data.get("over_rated_ratio") is None:
            feats["over_rated_ratio"] = feats["rms_line_A"] / rated

        # Ensure non-negative bounds
        feats["rms_line_A"] = max(0.0, feats["rms_line_A"])
        feats["rms_neutral_A"] = max(0.0, feats["rms_neutral_A"])
        feats["peak_A"] = max(feats["rms_line_A"], feats["peak_A"])
        feats["thd_pct"] = max(0.0, feats["thd_pct"])

        return feats


feature_service = FeatureService()
