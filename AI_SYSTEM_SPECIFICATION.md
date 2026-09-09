# AMPIX — Complete AI System Specification & Architecture Manual
## (Ampere + Intelligence)
### Technical Reference for AI Agents, LLMs, and Software Developers

---

## 1. System Identity & Core Purpose

- **System Name**: **AMPIX** (**Amp**ere + **Intel**ligence)
- **Model / System Version**: AMPIX-S1 PRO (v2.4-PRO)
- **Domain**: AI-Based Electrical Hazard & Arc-Fault Detection System for Household Distribution Boards.
- **Problem Statement**: Traditional Miniature Circuit Breakers (MCBs) and Residual Current Devices (RCDs) trip only under sustained thermal overcurrent or earth leakage >30mA. They fail to detect **arc faults** (series and parallel arcing), which generate localized temperatures exceeding 5,000°C without exceeding the rated RMS current limit, causing over 40% of electrical fires.
- **Solution**: AMPIX uses machine learning trained on dual-CT (Line + Neutral) 5 kHz current signatures to classify 7 distinct electrical hazard conditions before fire ignition occurs.

---

## 2. Directory Structure & Key Files

```
hazard-dataset/
├── data/
│   ├── features_single_phase.csv      # Primary engineered tabular dataset (1,050 samples)
│   ├── features_load_pattern.csv      # Long-duration duty-cycle profile dataset
│   └── features_three_phase_imbalance.csv # 3-phase imbalance dataset
├── models/
│   ├── best_model.joblib              # Serialized winning Random Forest classifier
│   ├── preprocessor.joblib            # Serialized fitted StandardScaler
│   ├── metadata.json                  # Feature names, label mappings, and test metrics
│   └── cnn_waveform_model.keras       # Optional 1D-CNN deep learning waveform model
├── results/
│   ├── metrics.json                   # Detailed candidate comparison metrics
│   ├── classification_report.txt      # Text classification report
│   └── dl_metrics.json                # 1D-CNN baseline benchmark metrics
├── plots/
│   ├── confusion_matrix.png           # Test set confusion matrix heatmap
│   ├── feature_importance.png         # Feature importance bar chart
│   └── model_comparison.png           # Candidate model comparison chart
├── app.py                             # Main Streamlit web application (AMPIX Dashboard)
├── train.py                           # Primary tabular ML training & evaluation script
├── predict.py                         # CLI inference script with feature explanations
├── simulate.py                        # Physics-based waveform & feature generator engine
├── train_dl_waveform.py               # 1D-CNN waveform training benchmark script
├── requirements.txt                   # Project dependencies
├── .gitignore                         # Git repository ignore rules
├── LICENSE                            # MIT open-source license
├── README.md                          # Main GitHub repository documentation
├── PROJECT_SUMMARY.md                 # Viva Q&A document
├── SMART_MCB_DASHBOARD.md             # Dashboard architecture manual
└── AI_SYSTEM_SPECIFICATION.md         # Full system AI specification manual (THIS FILE)
```

---

## 3. Data Pipeline & 17 Feature Schemas

The ML model (`models/best_model.joblib`) expects an exact 17-feature numerical vector in the following precise order:

| Index | Feature Column Name | Data Type | Physical Definition & Unit |
|---|---|---|---|
| 0 | `rated_current_A` | float64 | MCB Nameplate Rated Current (e.g. 6A, 10A, 16A, 20A, 32A, 40A, 63A) |
| 1 | `rms_line_A` | float64 | Line Conductor RMS Current magnitude ($A$) |
| 2 | `rms_neutral_A` | float64 | Neutral Conductor RMS Current magnitude ($A$) |
| 3 | `over_rated_ratio` | float64 | Ratio of Line RMS Current to Rated MCB Current ($\frac{I_{\text{line}}}{I_{\text{rated}}}$) |
| 4 | `peak_A` | float64 | Instantaneous Peak Current Amplitude ($A$) |
| 5 | `crest_factor` | float64 | Crest Factor ratio ($\frac{I_{\text{peak}}}{I_{\text{rms}}}$) |
| 6 | `thd_pct` | float64 | Total Harmonic Distortion percentage ($\text{THD} \% = 100 \times \frac{\sqrt{\sum_{h=2}^9 I_h^2}}{I_1}$) |
| 7 | `h3_ratio` | float64 | 3rd Harmonic Ratio relative to fundamental ($\frac{I_3}{I_1}$) |
| 8 | `h5_ratio` | float64 | 5th Harmonic Ratio relative to fundamental ($\frac{I_5}{I_1}$) |
| 9 | `h7_ratio` | float64 | 7th Harmonic Ratio relative to fundamental ($\frac{I_7}{I_1}$) |
| 10 | `h9_ratio` | float64 | 9th Harmonic Ratio relative to fundamental ($\frac{I_9}{I_1}$) |
| 11 | `hf_band_energy_1p5_2p5kHz` | float64 | Power Spectral Density energy in 1.5–2.5 kHz frequency band |
| 12 | `di_dt_max_A_per_s` | float64 | Maximum rate of current change over time ($\max |\frac{di}{dt}|$) |
| 13 | `zero_crossing_noise_std` | float64 | Noise standard deviation near fundamental zero crossings (UL 1699 shouldering) |
| 14 | `leakage_rms_A` | float64 | Differential CT leakage current RMS ($I_{\text{line}} - I_{\text{neutral}}$) |
| 15 | `kurtosis` | float64 | Waveform statistical kurtosis (measures impulsive arcing spikes) |
| 16 | `skew` | float64 | Waveform statistical skewness (measures positive vs negative cycle asymmetry) |

---

## 4. Target Hazard Classes & Constant Risk Score Mapping

The model classifies inputs into 7 mutually exclusive target classes. Each class is mapped to a **Constant Risk Score** (0 to 100) to ensure deterministic presentation:

| Target Class | Constant Risk Score | Risk Tier | UI LED Badge Color | Common-Man Layman Reason |
|---|---|---|---|---|
| `normal` | **5 / 100** | **LOW** | Green (`#34d399`) | Healthy circuit, smooth 50 Hz power flow |
| `harmonic_distortion` | **35 / 100** | **MEDIUM** | Amber (`#fbbf24`) | Degrading motor/compressor distortion |
| `overcurrent_overload` | **65 / 100** | **HIGH** | Amber (`#fbbf24`) | Appliance overload causing wire heating |
| `ground_fault_leakage` | **80 / 100** | **CRITICAL** | Red (`#f87171`) | Power escaping to earth / shock hazard |
| `arc_fault_series` | **90 / 100** | **CRITICAL** | Red (`#f87171`) | Loose terminal / wire spark (5,000°C) |
| `arc_fault_parallel` | **95 / 100** | **CRITICAL** | Red (`#f87171`) | Wire insulation breakdown spark |
| `short_circuit` | **100 / 100** | **CRITICAL** | Red (`#f87171`) | Bare live wire direct contact surge |

---

## 5. Machine Learning Pipeline Architecture

1. **Dataset Split**:
   - Total dataset: 1,050 samples (150 per class).
   - **70% Training Set**: 734 samples.
   - **15% Validation Set**: 158 samples.
   - **15% Test Set**: 158 samples.
   - Stratified split based on target label (`stratify=y`, `random_state=42`).

2. **Preprocessing**:
   - `StandardScaler` fitted **strictly on the Training set**.
   - No data leakage to validation or test sets.

3. **Candidate Models Evaluated**:
   - Random Forest (`RandomForestClassifier`, 300 estimators, balanced class weights).
   - XGBoost (`XGBClassifier`, 200 estimators, learning_rate=0.05).
   - Extra Trees (`ExtraTreesClassifier`, 300 estimators).
   - HistGradientBoosting (`HistGradientBoostingClassifier`).
   - Logistic Regression baseline (`LogisticRegression`).

4. **Winning Model Results**:
   - **Winning Model**: Random Forest Classifier.
   - **Validation Macro F1**: 1.0000 (100.0%).
   - **Untouched Test Set Accuracy**: 99.37%.
   - **Untouched Test Set Macro F1**: 99.35%.
   - **Untouched Test Set ROC-AUC**: 1.0000.

---

## 6. Web Application UI & Navigation (`app.py`)

### Brand Identity
- **Name**: **AMPIX** (Ampere + Intelligence).
- **Typography**: Google Fonts (`Outfit` for headings/brand, `JetBrains Mono` for numbers).
- **Brand Title Style**: Large 2.85rem bold gradient text (`#38bdf8` to `#c084fc`).

### Tabbed Architecture
1. **Tab 1: ⚡ Real-Time Safety Dashboard**:
   - Header panel bar with device info and glowing status LED (`🟢 NORMAL`, `🟡 WARNING`, `🔴 TRIPPED`).
   - Alert status banner.
   - 6 Metric Cards: Voltage, Line Current, Active Power, Power Factor, THD %, Constant Risk Score.
   - Common-Man Hazard Explanation card.
   - Householder Safety & Action Instruction card.
   - 3 Real-Time Live Rolling Analytics Line Charts (Voltage/Current, Power/THD, Risk Score Trend).
2. **Tab 2: 📋 Event & Timeline History Log**:
   - Formatted timeline history table (`03:05 PM — Overload Warning... 03:06 PM — Trip Activated... 03:10 PM — Returned to Normal`).
3. **Tab 3: 🔌 Appliance Load & Controls**:
   - Interactive appliance toggles (Lights, Refrigerator, TV, Washing Machine, AC, Water Heater) and active power demand breakdown.
4. **Tab 4: 🔬 Technical ML Diagnostics**:
   - **ML Model Confidence Score** (e.g. `99.7%`) kept strictly here.
   - Full class probability distribution table and top feature rationale rankings.

### Sidebar Controls
- Household Load Appliance checkboxes.
- Manual Hazard Simulator buttons (`Normal`, `Series Arc`, `Parallel Arc`, `Overload`, `Short Circuit`, `Ground Fault`, `Harmonics`, `Reset MCB`).
- Automatic Viva Demo Mode controller (`▶ START AUTOMATIC DEMO`).

---

## 7. Simulated MCB Trip Interlock & State Machine

```
   [ NORMAL OPERATION ]
            │
            ▼ (Hazard Event / Risk Score >= 80)
   [ CRITICAL HAZARD DETECTED ]
            │
            ▼ (Safety Interlock Activated)
   [ SIMULATED MCB TRIPPED ]
    • Current drops to 0.0 A
    • Active Power drops to 0 W
    • Red alert banner displayed
    • Hazard cause retained in prediction & log
            │
            ▼ (User clicks 'Reset MCB' / 'Normal')
   [ RETURNED TO NORMAL CONDITIONS ]
```

---

## 8. Automatic Viva Presentation Demo Mode Sequence

When `[▶ START AUTOMATIC DEMO]` is activated, a 45-second sequence runs automatically:
- **Step 1/5**: Normal Household Operation (Low Load)
- **Step 2/5**: Increasing Load (High Household Consumption)
- **Step 3/5**: Early Warning Signal (Distorted Harmonic Waveform)
- **Step 4/5**: Critical Series Arc Fault Detected (UL 1699 Shouldering)
- **Step 5/5**: Short Circuit Fault & Automatic MCB Safety Trip

---

## 9. Hardware & Simulation Limitations

- **Simulated Component**: Waveform signal generation is driven by scipy/numpy physics equations (`simulate.py`) reproducing IEEE/UL standards rather than physical CT coils.
- **Deployed Component**: Feature extraction (`extract_features`), preprocessor scaler (`joblib`), Random Forest classifier (`joblib`), confidence calculations, risk scoring, trip interlock logic, and visualization UI are 100% production code ready for physical microcontroller integration (e.g. STM32 / ESP32 + CT sensors).
