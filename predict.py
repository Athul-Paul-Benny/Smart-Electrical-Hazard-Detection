"""
Inference / Prediction Engine for AI-Based Electrical Hazard & Arc-Fault Detection
Run:
  python predict.py --sample_index 0
  python predict.py --csv sample_input.csv
  python predict.py --simulate --fault_type arc_fault_series --rated 16
"""

import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd

# Import physics simulator and feature extractor from existing simulate.py
import simulate


FEATURE_EXPLANATIONS = {
    "h5_ratio": "5th Harmonic Ratio (High 5th harmonic indicates non-linear load degradation or distortion)",
    "h3_ratio": "3rd Harmonic Ratio (Elevated 3rd harmonic reflects core saturation or arcing distortion)",
    "h7_ratio": "7th Harmonic Ratio (Indicates higher-order non-linear harmonic noise)",
    "h9_ratio": "9th Harmonic Ratio (Indicates upper odd harmonic distortion)",
    "over_rated_ratio": "Over-Rated Current Ratio (Line RMS / Rated MCB Current; >1.0 indicates overload)",
    "crest_factor": "Crest Factor (Peak / RMS; abnormal values signal sharp impulse arcing or spikes)",
    "thd_pct": "Total Harmonic Distortion % (Sustained elevation models degrading appliance/motor)",
    "leakage_rms_A": "Leakage Current RMS (Line - Neutral CT divergence; >30mA signals ground fault)",
    "hf_band_energy_1p5_2p5kHz": "High-Frequency Band Energy (1.5-2.5 kHz broadband noise burst during arc reignition)",
    "di_dt_max_A_per_s": "Max Rate of Current Change di/dt (Catches rapid short-circuit fault onset)",
    "kurtosis": "Waveform Kurtosis (Catches extreme impulsive arcing spikes)",
    "skew": "Waveform Skewness (Asymmetry in positive vs negative half-cycles)",
    "zero_crossing_noise_std": "Zero-Crossing Noise Std Dev (Catches series-arc shouldering/flattening)",
    "peak_A": "Peak Current (Maximum instantaneous amplitude)",
    "rms_line_A": "Line Conductor RMS Current (Main load current magnitude)",
    "rms_neutral_A": "Neutral Conductor RMS Current (Return path magnitude)",
    "rated_current_A": "MCB Rated Current (Nameplate circuit rating in Amperes)",
}


def load_pipeline_artifacts(models_dir="models"):
    model_path = os.path.join(models_dir, "best_model.joblib")
    scaler_path = os.path.join(models_dir, "preprocessor.joblib")
    meta_path = os.path.join(models_dir, "metadata.json")

    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(meta_path)):
        raise FileNotFoundError(
            "Model artifacts missing! Please run 'python train.py' first to build and save the model."
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    return model, scaler, metadata


def predict_single_sample(features_dict, model, scaler, metadata):
    feature_names = metadata["feature_names"]
    classes = metadata["classes"]
    is_xgb = metadata.get("is_xgb", False)
    importances = metadata.get("feature_importances", {})

    # Ensure feature vector matches exact trained column order
    input_df = pd.DataFrame([features_dict])[feature_names]
    scaled_feats = scaler.transform(input_df)

    if is_xgb and "label_mapping" in metadata:
        inv_map = {v: k for k, v in metadata["label_mapping"].items()}
        pred_idx = model.predict(scaled_feats)[0]
        pred_class = inv_map.get(pred_idx, str(pred_idx))
        probs = model.predict_proba(scaled_feats)[0]
    else:
        pred_class = model.predict(scaled_feats)[0]
        probs = model.predict_proba(scaled_feats)[0]

    prob_dict = {classes[i]: float(probs[i]) for i in range(len(classes))}
    confidence = float(np.max(probs)) * 100.0

    # Determine key contributing features using feature importance & feature values
    top_features = []
    if importances:
        # Rank by (feature importance * normalized input deviation)
        scores = {}
        for f_name in feature_names:
            val = input_df[f_name].iloc[0]
            imp = importances.get(f_name, 0.01)
            scores[f_name] = abs(val) * imp

        sorted_feats = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        for f_name, _ in sorted_feats:
            desc = FEATURE_EXPLANATIONS.get(f_name, f_name)
            val = input_df[f_name].iloc[0]
            top_features.append(f"{f_name} = {val:.4f} ({desc})")

    return {
        "predicted_class": pred_class,
        "confidence_pct": round(confidence, 2),
        "probabilities": prob_dict,
        "top_contributing_features": top_features,
    }


def format_prediction_report(res):
    lines = []
    lines.append("\n=======================================================")
    lines.append("        AI ELECTRICAL HAZARD DETECTION RESULT")
    lines.append("=======================================================")
    lines.append(f" Predicted Hazard : {res['predicted_class'].upper()}")
    lines.append(f" Confidence       : {res['confidence_pct']:.1f}%")
    lines.append("-------------------------------------------------------")
    lines.append(" Class Probability Distribution:")
    for cls, prob in sorted(res["probabilities"].items(), key=lambda x: x[1], reverse=True):
        bar = "#" * int(prob * 20)
        lines.append(f"  - {cls:22s} : {prob*100:5.1f}%  {bar}")
    lines.append("-------------------------------------------------------")
    lines.append(" Top Physical Indicator Features:")
    for feat in res["top_contributing_features"]:
        lines.append(f"  * {feat}")
    lines.append("=======================================================\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Predict Electrical Hazard Class from Features or Waveform")
    parser.add_argument("--csv", type=str, help="Path to input CSV row")
    parser.add_argument("--sample_index", type=int, help="Index of sample in dataset features_single_phase.csv")
    parser.add_argument("--simulate", action="store_true", help="Generate simulated physics waveform reading")
    parser.add_argument(
        "--fault_type",
        type=str,
        default="arc_fault_series",
        choices=[
            "normal",
            "overcurrent_overload",
            "short_circuit",
            "arc_fault_series",
            "arc_fault_parallel",
            "ground_fault_leakage",
            "harmonic_distortion",
        ],
        help="Fault type to simulate",
    )
    parser.add_argument("--rated", type=float, default=16.0, help="Rated MCB current in Amperes")
    args = parser.parse_args()

    model, scaler, metadata = load_pipeline_artifacts()

    if args.simulate:
        print(f"[INFO] Generating simulated waveform for '{args.fault_type}' (MCB Rated: {args.rated}A)...")
        gen_fn = simulate.CLASS_GENERATORS[args.fault_type]
        line, neutral = gen_fn(args.rated)
        feats = simulate.extract_features(line, neutral, args.rated)
    elif args.csv:
        print(f"[INFO] Reading sample from CSV: {args.csv}")
        df_in = pd.read_csv(args.csv)
        feats = df_in.iloc[0].to_dict()
    elif args.sample_index is not None:
        dataset_path = "data/features_single_phase.csv"
        if not os.path.exists(dataset_path):
            dataset_path = "features_single_phase.csv"
        df_all = pd.read_csv(dataset_path)
        row = df_all.iloc[args.sample_index]
        actual_label = row.get("label", "Unknown")
        print(f"[INFO] Using dataset sample #{args.sample_index} (Actual Ground Truth: '{actual_label}')")
        feats = row.to_dict()
    else:
        print("[INFO] No input specified. Defaulting to dataset sample #0.")
        df_all = pd.read_csv("data/features_single_phase.csv")
        row = df_all.iloc[0]
        feats = row.to_dict()

    res = predict_single_sample(feats, model, scaler, metadata)
    print(format_prediction_report(res))


if __name__ == "__main__":
    main()
