# AMPIX — AI Electrical Safety & Hazard Monitor
## (Ampere + Intelligence)
### Capstone Software Architecture & Layman Viva Guide

---

### 1. Brand Identity & Overview
**AMPIX** stands for **Ampere + Intelligence**. It represents a next-generation AI-driven smart circuit breaker and electrical hazard monitoring system installed in household distribution panels.

#### Key Features:
- **Constant Hazard Risk Score**: Fixed, deterministic risk scores assigned according to the inherent danger of each hazard class (eliminating confusing fluctuations).
- **Steady Non-Flickering UI**: Clean, steady component rendering without jarring full-page refresh flashes.
- **Layman Hazard Translations**: Simple, non-technical explanations for householders on the main dashboard, with actual physical feature readings displayed right below.
- **Technical ML Diagnostics Tab**: ML confidence percentage (`99.7%`), class probability distributions, and tree importances are housed strictly in the **Technical ML Diagnostics** tab for evaluators.
- **Human Timeline History Log**: Formatted events (`03:05 PM — Overload Warning... 03:10 PM — Returned to Normal`).

---

### 2. Constant Hazard Risk Score Mapping Table

| Target Hazard Class | Constant Risk Score | Risk Tier | CSS Color Indicator | Description |
|---|---|---|---|---|
| `normal` | **5 / 100** | **LOW** | Green (`#34d399`) | Healthy circuit, smooth 50 Hz power |
| `harmonic_distortion` | **35 / 100** | **MEDIUM** | Amber (`#fbbf24`) | Degrading motor/compressor harmonics |
| `overcurrent_overload` | **65 / 100** | **HIGH** | Amber (`#fbbf24`) | Heavy appliance thermal overload |
| `ground_fault_leakage` | **80 / 100** | **CRITICAL** | Red (`#f87171`) | Earth leakage / electric shock hazard |
| `arc_fault_series` | **90 / 100** | **CRITICAL** | Red (`#f87171`) | Series loose wire arcing (UL 1699) |
| `arc_fault_parallel` | **95 / 100** | **CRITICAL** | Red (`#f87171`) | Parallel wire insulation breakdown spark |
| `short_circuit` | **100 / 100** | **CRITICAL** | Red (`#f87171`) | Massive direct short-circuit surge |

---

### 3. Layman Explanations & Action Notes

1. **Healthy Circuit (`normal`)**:
   - *Layman Reason*: "Electricity is flowing cleanly and safely to your appliances without any sparking, wire overheating, or earth power leakage."
   - *Action Note*: "🟢 SAFE TO OPERATE: No action needed."

2. **Ground Fault / Earth Leakage (`ground_fault_leakage`)**:
   - *Constant Risk Score*: **80 / 100 (CRITICAL)**
   - *Layman Reason*: "Electric current is escaping out of the wire into appliance metal frames or damp walls instead of returning safely through neutral wire."
   - *Action Note*: "⚡ SHOCK HAZARD: Do NOT touch wet appliances or metal sinks! Unplug suspect device and call an electrician."

3. **Series Arc Fault (`arc_fault_series`)**:
   - *Constant Risk Score*: **90 / 100 (CRITICAL)**
   - *Layman Reason*: "A loose screw terminal in a wall socket or broken wire strand is creating invisible 5,000°C electric sparks inside the wall."
   - *Action Note*: "🚨 HIGH FIRE RISK: Stand back from MCB panel! High-temperature arcing ignites wall insulation quickly. Call an electrician immediately."

4. **Severe Short Circuit (`short_circuit`)**:
   - *Constant Risk Score*: **100 / 100 (CRITICAL)**
   - *Layman Reason*: "A bare live wire is directly touching a neutral or ground wire, drawing a massive surge of current that can cause an instant electrical flash or explosion."
   - *Action Note*: "🚨 DO NOT TOUCH BREAKER: Move away from panel! Automatic MCB tripped open to prevent explosion."

---

### 4. Technical ML Diagnostics Placement
To keep the main page clean and beginner-friendly for householders:
- Main dashboard displays layman explanations, constant risk scores, color-coded metric cards, user safety action notes, and real-time graphs.
- **ML Confidence %** (e.g. `99.7%`), full probability distribution tables, and top feature rankings are located in **Tab 4: 🔬 Technical ML Diagnostics**.

---

### 5. Quick Run Instructions
Launch AMPIX in your terminal:
```powershell
python -m streamlit run app.py
```
Dashboard URL: [http://localhost:8501](http://localhost:8501)
