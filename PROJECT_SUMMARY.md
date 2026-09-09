# 🎓 Capstone Viva & Project Summary Guide
## AI-Based Electrical Hazard & Arc-Fault Detection System

---

### 1. What problem does this system solve?
Traditional circuit protection devices like **Miniature Circuit Breakers (MCBs)** and **Residual Current Devices (RCDs)** only trip when sustained thermal overcurrents occur or when earth leakage current exceeds ~30 mA. They are **blind to arc faults** (series and parallel arcing), which generate localized temperatures exceeding 5,000°C without exceeding the rated RMS current limit, causing over 40% of electrical fires. This system uses machine learning on high-frequency dual-CT current signatures to classify 7 distinct healthy and hazard conditions before catastrophic ignition or equipment failure occurs.

---

### 2. What dataset was used?
- **Dataset**: `features_single_phase.csv` (along with `raw_waveforms_single_phase.npz`).
- **Total Samples**: 1,050 windows (0.5-second duration @ 5 kHz sampling rate = 2,500 samples per window).
- **Target Classes (7 Balanced Classes, 150 samples per class)**:
  1. `normal`: Fundamental 50 Hz sine + low-level background harmonics + sensor noise.
  2. `overcurrent_overload`: Sustained RMS current 1.1×–1.55× above rated MCB capacity with thermal drift.
  3. `short_circuit`: Rapid high fault current onset with exponentially decaying DC offset (IEEE C37 fault theory) followed by breaker collapse.
  4. `arc_fault_series`: Current shouldering/flattening near zero-crossing (arc restrike voltage) + 1.5–2.5 kHz broadband noise bursts (UL 1699 standard).
  5. `arc_fault_parallel`: Elevated erratic current with random high-frequency noise spikes not locked to zero crossing.
  6. `ground_fault_leakage`: Divergence between Line CT and Neutral CT currents (`leakage_rms_A`), modeling differential RCD operation.
  7. `harmonic_distortion`: Progressive growth of 3rd, 5th, 7th, and 9th harmonics across the window, modeling degrading motor/compressor loads.
- **Engineered Features**: 17 numerical spectral, temporal, and differential features extracted per window.

---

### 3. What preprocessing was performed?
- **Feature Scaling**: `StandardScaler` (Z-score normalization) fit **strictly on the Training set only** to prevent data leakage.
- **Verification**: Validated that no missing/null values, infinity, or duplicate sample IDs exist.

---

### 4. Which models were tested?
1. **Random Forest Classifier**: Ensemble of 300 decision trees.
2. **XGBoost Classifier**: Gradient boosted decision trees (`XGBClassifier`).
3. **Extra Trees Classifier**: Extremely randomized trees ensemble.
4. **HistGradientBoosting Classifier**: Histogram-based gradient boosting.
5. **Logistic Regression**: Scaled linear baseline model.
6. **1D-CNN (Deep Learning)**: 3-layer 1D Convolutional Neural Network trained directly on raw 5 kHz dual-channel waveforms (`raw_waveforms_single_phase.npz`).

---

### 5. Why was the final model selected?
- **Winning Model**: **Random Forest Classifier** (and **XGBoost**).
- **Validation Macro F1**: **1.0000 (100%)**
- **Test Accuracy**: **99.37%**
- **Test Macro F1**: **99.35%**
- **Selection Rationale**:
  - Outperformed simple linear baselines.
  - Achieved near-perfect classification across all 7 hazard conditions.
  - Highly interpretable via feature importance scores (essential for safety-critical electrical applications during viva and compliance review).
  - Extremely fast inference (<1 ms per sample) suitable for low-cost embedded edge microcontrollers.

---

### 6. How does the Train/Validation/Test splitting work?
- **Dataset Split**:
  - **70% Training Set**: 734 samples (used to fit scaler and model parameters).
  - **15% Validation Set**: 158 samples (used to evaluate candidate models and select hyper-parameters without touching the test set).
  - **15% Test Set**: 158 samples (kept completely untouched until final evaluation).
- **Stratification**: Class distribution is strictly preserved across all 3 subsets using `stratify=y` with `random_state=42` for exact scientific reproducibility.

---

### 7. How was class imbalance handled?
- The dataset is naturally balanced (150 samples per class across 7 classes).
- For robustness against any operational imbalance, models were configured with `class_weight="balanced"`. No synthetic oversampling (like SMOTE) was required or applied, avoiding synthetic data distortion on test boundaries.

---

### 8. What do the evaluation metrics mean?
- **Accuracy (99.37%)**: Proportion of total predictions that were correct.
- **Precision (99.38% Macro)**: Out of all alarms raised for a class, how many were genuine (low false alarm rate).
- **Recall (99.35% Macro)**: Out of all actual hazard events, how many were correctly caught by the model (critical for life safety to prevent missed arc faults).
- **Macro F1-Score (99.35%)**: Harmonic mean of Precision and Recall calculated per class equally, ensuring minority/critical classes are not ignored.
- **ROC-AUC (1.0000)**: Area Under Receiver Operating Characteristic curve measuring class separation capability across all thresholds.

---

### 9. How is the model saved?
- The winning classifier is serialized to `models/best_model.joblib`.
- The preprocessing scaler is saved to `models/preprocessor.joblib`.
- Complete pipeline metadata (feature names in exact order, class label mappings, test metrics) is stored in `models/metadata.json`.

---

### 10. How does the demonstration work?
- **CLI Prediction Script (`predict.py`)**: Accepts sample indices, CSV input files, or live simulated waveforms, running feature extraction, scaling, inference, confidence calculation, and top physical feature contribution explanation.
- **Interactive Streamlit Web Dashboard (`app.py`)**:
  - Allows evaluators to select MCB ratings (6A–63A) and choose fault conditions.
  - Calls physics generator in `simulate.py` to generate live 5 kHz dual-CT waveforms.
  - Displays real-time dual-channel current waveforms, hazard alert cards, confidence gauges, per-class probabilities, feature contribution rankings, and batch CSV testing.

---

### 11. Why is simulation used instead of hardware?
- **Safety & Regulations**: Live arcing (5,000°C) and short-circuit faults (hundreds of Amperes) pose severe fire and high-voltage hazards that cannot be safely created in a classroom/viva setting.
- **Physics Basis**: The dataset is generated using a validated physics simulator (`simulate.py`) reproducing exact IEEE C37 asymmetrical fault current decay, UL 1699 zero-crossing arc shouldering, and differential CT leakage theory.

---

### 12. What are the most important features?
1. `h5_ratio` / `h3_ratio`: 5th and 3rd harmonic ratios (detect non-linear load degradation & arc distortion).
2. `over_rated_ratio`: Ratio of Line RMS current to MCB rated capacity (detects overcurrent/overload).
3. `crest_factor`: Peak-to-RMS ratio (detects sharp impulse spikes during arcing).
4. `hf_band_energy_1p5_2p5kHz`: Broadband 1.5–2.5 kHz energy (detects high-frequency noise bursts during arc reignition).
5. `zero_crossing_noise_std`: Noise standard deviation around zero-crossing (detects series arc shouldering).
6. `leakage_rms_A`: Difference between Line and Neutral CT RMS currents (detects ground faults/leakage).

---

### 13. System Limitations
- Synthetic data, while physically grounded, lacks real-world CT saturation non-linearities, severe grid voltage sags, and complex site-specific electrical noise.
- Current models evaluate 0.5-second windows independently without multi-minute temporal memory.

---

### 14. Future Improvements
- Fine-tuning the trained models with real physical CT sensor data collected from test benches.
- Quantizing the Random Forest/XGBoost model to C code using `emlearn` or `microML` to deploy directly onto ARM Cortex-M microcontrollers (e.g., STM32 / ESP32).
