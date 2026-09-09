# ⚡ AMPIX — AI Electrical Safety & Hazard Monitoring System
### *(Ampere + Intelligence)*

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue?logo=python)
![Framework](https://img.shields.io/badge/framework-Streamlit%20%7C%20Scikit--Learn-orange?logo=streamlit)
![ML Accuracy](https://img.shields.io/badge/Test%20Accuracy-99.37%25-success)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Production%20Ready-brightgreen)

---

## 📌 Project Overview

**AMPIX** (**Amp**ere + **Intel**ligence) is an end-to-end Machine Learning system and industrial software dashboard for predictive electrical hazard classification, arc-fault detection, earth leakage monitoring, and overcurrent prevention based on physics-simulated 5 kHz CT sensor signatures.

Traditional circuit protection devices like **Miniature Circuit Breakers (MCBs)** and **Residual Current Devices (RCDs)** only trip when sustained thermal overcurrents occur or when earth leakage current exceeds ~30 mA. They are **blind to arc faults** (series and parallel arcing), which generate localized temperatures exceeding 5,000°C without exceeding the rated RMS current limit, causing over 40% of electrical fires. 

AMPIX analyzes 5 kHz dual-CT (Line + Neutral) current signatures to classify **7 distinct electrical hazard conditions** before catastrophic fire ignition occurs.

---

## ✨ Key Features

- **🧠 Machine Learning Engine**: Pre-trained Random Forest Classifier achieving **99.37% test accuracy** and **99.35% macro F1** across 7 target hazard classes.
- **⚡ Real-Time Physics Waveform Simulator**: Generates physical dual-CT 5 kHz current waveforms (`simulate.py`) adhering to IEEE C37 asymmetrical fault decay and UL 1699 zero-crossing shouldering standards.
- **💡 Common-Man Hazard Explanations**: Translates complex spectral features into simple, non-technical explanations for householders while preserving exact technical parameters (`rms_line_A`, `thd_pct`, `crest_factor`, `hf_band_energy`) for engineers.
- **📢 Highlighted Safety Action Notes**: Provides prominent, color-coded householder action guidance based on risk level (*"🚨 HIGH FIRE RISK: Move away from MCB panel! Call a certified electrician."*).
- **📊 Constant Hazard Risk Score Mapping**: Assigns fixed, deterministic risk scores (5 to 100) per hazard class to prevent confusing number fluctuations.
- **📋 Formatted Human Timeline History Log**: Tracks events in clear human terms (`03:05 PM — Overload Warning... 03:06 PM — Trip Activated... 03:10 PM — Returned to Normal`).
- **🚨 Simulated MCB Safety Trip Interlock**: Automatically trips the virtual breaker when critical hazards occur, dropping load current to 0.0A until reset.
- **🎬 Automatic Viva Demo Mode**: Scripted 45-second sequence through 5 presentation steps for evaluators.
- **🔬 Technical ML Diagnostics Tab**: Houses ML confidence scores (`99.7%`), class probability distribution tables, and feature rationale rankings.

---

## 📊 Model Performance Comparison Table

| Model Algorithm | Validation Macro F1 | Untouched Test Accuracy | Untouched Test Macro F1 | Test ROC-AUC | Status |
|---|---|---|---|---|---|
| **Random Forest (AMPIX Winner)** | **1.0000** | **99.37%** | **99.35%** | **1.0000** | **Winning Model** |
| Extra Trees | 1.0000 | 99.37% | 99.35% | 1.0000 | Candidate |
| XGBoost | 0.9938 | 99.37% | 99.35% | 1.0000 | Candidate |
| HistGradientBoosting | 0.9938 | 99.37% | 99.35% | 1.0000 | Candidate |
| Logistic Regression | 0.9938 | 98.73% | 0.9870 | 0.9995 | Baseline |
| 1D-CNN (Raw Waveforms) | N/A | 99.37% | 99.35% | N/A | DL Benchmark |

---

## 🚀 Quick Start Guide

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/your-username/ampix-hazard-detector.git
cd ampix-hazard-detector
pip install -r requirements.txt
```

### 2. Launch AMPIX Web Application

```powershell
python -m streamlit run app.py
```

Open your browser at: **[http://localhost:8501](http://localhost:8501)**

---

### 3. Re-train Machine Learning Models (Optional)

To execute the dataset split (70/15/15), train candidate models, and save artifacts:

```bash
python train.py
```

### 4. Run CLI Inference Engine

Predict hazards on dataset samples or live simulated waveforms:

```bash
# Predict on dataset sample #0
python predict.py --sample_index 0

# Predict on live simulated Series Arc Fault
python predict.py --simulate --fault_type arc_fault_series --rated 16
```

---

## 📁 Repository Directory Structure

```
ampix-hazard-detector/
├── data/
│   ├── features_single_phase.csv      # Primary engineered tabular dataset (1,050 samples)
│   ├── features_load_pattern.csv      # Long-duration duty-cycle profile dataset
│   └── features_three_phase_imbalance.csv # 3-phase imbalance dataset
├── models/
│   ├── best_model.joblib              # Winning Random Forest classifier
│   ├── preprocessor.joblib            # Fitted StandardScaler
│   └── metadata.json                  # Model metadata, feature list, & test metrics
├── results/
│   ├── metrics.json                   # Comprehensive metrics log
│   ├── classification_report.txt      # Text classification report
│   └── dl_metrics.json                # 1D-CNN benchmark results
├── plots/
│   ├── confusion_matrix.png           # Confusion matrix heatmap
│   ├── feature_importance.png         # Feature importance bar chart
│   └── model_comparison.png           # Candidate model comparison chart
├── app.py                             # Main Streamlit web app (AMPIX Dashboard)
├── train.py                           # Primary tabular ML training script
├── predict.py                         # CLI prediction script
├── simulate.py                        # Scipy/Numpy physics waveform & feature generator
├── train_dl_waveform.py               # 1D-CNN waveform training benchmark script
├── requirements.txt                   # Dependency specifications
├── .gitignore                         # Git repository ignore rules
├── LICENSE                            # MIT open-source license
├── README.md                          # Main GitHub repository documentation
├── PROJECT_SUMMARY.md                 # Capstone viva Q&A document
├── SMART_MCB_DASHBOARD.md             # Dashboard manual & guide
└── AI_SYSTEM_SPECIFICATION.md         # Full system AI specification manual
```

---

## 📘 Comprehensive AI & Developer Manuals

For AI platforms (ChatGPT, Claude, Gemini, custom agents) or developers seeking in-depth architectural details:
- **[AI_SYSTEM_SPECIFICATION.md](file:///c:/Users/ATHUL/Music/hazard-dataset/AI_SYSTEM_SPECIFICATION.md)**: Full system AI specification detailing data schemas, risk math, simulation equations, and state transitions.
- **[PROJECT_SUMMARY.md](file:///c:/Users/ATHUL/Music/hazard-dataset/PROJECT_SUMMARY.md)**: Capstone presentation talking points and viva Q&A reference.
- **[SMART_MCB_DASHBOARD.md](file:///c:/Users/ATHUL/Music/hazard-dataset/SMART_MCB_DASHBOARD.md)**: Dashboard layout and layman hazard translation manual.

---

## 📜 License & Disclaimer

This project is licensed under the **MIT License** — see the [LICENSE](file:///c:/Users/ATHUL/Music/hazard-dataset/LICENSE) file for details.

> ⚠️ **DISCLAIMER:** Software Simulation Only — AMPIX is a software simulation of a smart household electrical monitoring device created for college capstone presentation and viva demonstration. It is not connected to live physical high-voltage electrical hardware.
