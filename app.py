"""
AMPIX — AI Electrical Safety & Hazard Monitor (app.py)
AMPIX = Ampere + Intelligence

Supports Dual Acquisition Modes:
1. SIMULATION MODE (Powered by simulate.py physics generator)
2. HARDWARE MODE (Ready for ESP32 / MCU Wi-Fi HTTP POST Ingestion via FastAPI backend)
"""

import os
import json
import time
import requests
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import simulate
from predict import predict_single_sample

# Backend API Configuration
API_BASE_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Page Configuration & AMPIX Industrial Theme Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AMPIX - AI Electrical Safety Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for AMPIX Dark Theme with Outfit & JetBrains Mono Fonts
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800;900&family=JetBrains+Mono:wght@500;700&display=swap');

    .main {
        background-color: #0b0f19;
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }
    .stAppHeader {
        background-color: rgba(11, 15, 25, 0.9);
    }
    
    /* AMPIX Large Glowing Gradient Brand Title */
    .ampix-brand-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.85rem;
        font-weight: 900;
        letter-spacing: 2px;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.1;
        text-shadow: 0 0 30px rgba(56, 189, 248, 0.3);
    }
    .ampix-sub-tag {
        font-family: 'Outfit', sans-serif;
        font-size: 0.92rem;
        color: #94a3b8;
        margin-top: 4px;
        letter-spacing: 0.5px;
    }
    
    /* AMPIX Header Bar Container */
    .ampix-header-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-top: 5px solid #38bdf8;
        border-radius: 12px;
        padding: 18px 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    
    /* Status LED Badges */
    .led-green {
        background-color: #064e3b;
        color: #34d399;
        border: 1px solid #059669;
        padding: 8px 18px;
        border-radius: 20px;
        font-weight: 800;
        font-size: 0.9rem;
        box-shadow: 0 0 12px rgba(52, 211, 153, 0.3);
    }
    .led-amber {
        background-color: #78350f;
        color: #fbbf24;
        border: 1px solid #d97706;
        padding: 8px 18px;
        border-radius: 20px;
        font-weight: 800;
        font-size: 0.9rem;
        box-shadow: 0 0 12px rgba(251, 191, 36, 0.3);
    }
    .led-red {
        background-color: #7f1d1d;
        color: #f87171;
        border: 1px solid #dc2626;
        padding: 8px 18px;
        border-radius: 20px;
        font-weight: 800;
        font-size: 0.9rem;
        box-shadow: 0 0 14px rgba(248, 113, 113, 0.4);
    }

    /* Metric Display Box */
    .ind-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
    }
    .ind-val-normal { color: #34d399; font-size: 1.65rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; }
    .ind-val-warning { color: #fbbf24; font-size: 1.65rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; }
    .ind-val-hazard { color: #f87171; font-size: 1.65rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; }
    .ind-lbl { color: #94a3b8; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }

    /* Alert Banners */
    .banner-normal {
        background: linear-gradient(135deg, #064e3b 0%, #047857 100%);
        color: #ecfdf5;
        padding: 16px 20px;
        border-radius: 8px;
        border-left: 6px solid #10b981;
        margin-bottom: 16px;
    }
    .banner-warning {
        background: linear-gradient(135deg, #78350f 0%, #b45309 100%);
        color: #fffbeb;
        padding: 16px 20px;
        border-radius: 8px;
        border-left: 6px solid #f59e0b;
        margin-bottom: 16px;
    }
    .banner-critical {
        background: linear-gradient(135deg, #7f1d1d 0%, #b91c1c 100%);
        color: #fef2f2;
        padding: 16px 20px;
        border-radius: 8px;
        border-left: 6px solid #ef4444;
        margin-bottom: 16px;
    }
    .banner-tripped {
        background: linear-gradient(135deg, #450a0a 0%, #7f1d1d 100%);
        color: #fef2f2;
        padding: 18px 22px;
        border-radius: 8px;
        border: 2px dashed #f87171;
        margin-bottom: 16px;
    }

    /* Safety Action Cards */
    .safety-box-critical {
        background: #450a0a;
        border: 2px solid #ef4444;
        color: #fef2f2;
        border-radius: 10px;
        padding: 18px;
    }
    .safety-box-warning {
        background: #451a03;
        border: 2px solid #f59e0b;
        color: #fffbeb;
        border-radius: 10px;
        padding: 18px;
    }
    .safety-box-safe {
        background: #022c22;
        border: 2px solid #10b981;
        color: #ecfdf5;
        border-radius: 10px;
        padding: 18px;
    }

    .footer-text {
        color: #64748b;
        font-size: 0.8rem;
        text-align: center;
        margin-top: 30px;
        padding-top: 15px;
        border-top: 1px solid #1e293b;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Load ML Model Artifacts
# ---------------------------------------------------------------------------
@st.cache_resource
def load_ml_pipeline():
    model_path = "models/best_model.joblib"
    scaler_path = "models/preprocessor.joblib"
    meta_path = "models/metadata.json"

    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(meta_path)):
        st.error("Model artifacts missing! Please run 'python train.py' to train and save the model.")
        st.stop()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    return model, scaler, metadata


model, scaler, metadata = load_ml_pipeline()


# ---------------------------------------------------------------------------
# Initialize Persistent Session State
# ---------------------------------------------------------------------------
if "mcb_tripped" not in st.session_state:
    st.session_state.mcb_tripped = False

if "sim_condition" not in st.session_state:
    st.session_state.sim_condition = "normal"

if "prev_condition" not in st.session_state:
    st.session_state.prev_condition = "normal"

if "demo_active" not in st.session_state:
    st.session_state.demo_active = False

if "demo_step" not in st.session_state:
    st.session_state.demo_step = 0

if "cumulative_energy_kwh" not in st.session_state:
    st.session_state.cumulative_energy_kwh = 142.50

if "history_buffer" not in st.session_state:
    st.session_state.history_buffer = []

if "timeline_logs" not in st.session_state:
    st.session_state.timeline_logs = [
        {
            "Time": time.strftime("%I:%M %p"),
            "System Event & Hazard State": "System Normal Operation — All Parameters Safe",
            "Severity": "NORMAL",
            "Action Taken": "Continuous Monitoring Active",
        }
    ]


# ---------------------------------------------------------------------------
# CONSTANT RISK SCORE PER HAZARD CLASS
# Format: (Risk Score / 100, Tier Name, CSS Color Class)
# ---------------------------------------------------------------------------
HAZARD_CONSTANT_RISK = {
    "normal": (5.0, "LOW", "ind-val-normal"),
    "harmonic_distortion": (35.0, "MEDIUM", "ind-val-warning"),
    "overcurrent_overload": (65.0, "HIGH", "ind-val-warning"),
    "ground_fault_leakage": (80.0, "CRITICAL", "ind-val-hazard"),
    "arc_fault_series": (90.0, "CRITICAL", "ind-val-hazard"),
    "arc_fault_parallel": (95.0, "CRITICAL", "ind-val-hazard"),
    "short_circuit": (100.0, "CRITICAL", "ind-val-hazard"),
}


# ---------------------------------------------------------------------------
# Layman Explanations & User Safety Instructions Mapping
# ---------------------------------------------------------------------------
HAZARD_LAYMAN_MAP = {
    "normal": {
        "title": "Healthy Circuit — Normal Electricity Flow",
        "layman_reason": "Electricity is flowing cleanly and safely to your appliances without any sparking, wire overheating, or earth power leakage.",
        "layman_symptom": "Voltage stable ~230V, Current normal within breaker rating.",
        "safety_type": "safe",
        "safety_badge": "🟢 SAFE TO OPERATE",
        "user_action": "✅ SYSTEM SAFE: No action needed. All electrical parameters are operating within standard safety limits.",
    },
    "overcurrent_overload": {
        "title": "Current Usage Very High (Appliance Overload)",
        "layman_reason": "Too many high-power appliances (e.g. AC, Water Heater, Washing Machine) are running at the same time, causing the wires to heat up.",
        "layman_symptom": "Current flow has exceeded the breaker rated limit (>16 Amps).",
        "safety_type": "warning",
        "safety_badge": "⚠️ UNPLUG HEAVY APPLIANCES",
        "user_action": "⚡ ACTION REQUIRED: Turn off or unplug heavy appliances (Air Conditioner, Water Heater, Microwave) immediately. If MCB trips, wait 2 minutes for wires to cool before flipping switch back ON.",
    },
    "short_circuit": {
        "title": "Severe Short Circuit (Direct Wire Contact)",
        "layman_reason": "A bare live wire is directly touching a neutral or ground wire, drawing a massive surge of current that can cause an instant electrical flash or explosion.",
        "layman_symptom": "Extreme current surge detected in milliseconds (di/dt > 20,000 A/s).",
        "safety_type": "critical",
        "safety_badge": "🚨 DO NOT TOUCH BREAKER — CALL ELECTRICIAN",
        "user_action": "🚨 CRITICAL SAFETY DANGER: Move away from the electrical panel immediately! The automatic MCB breaker has tripped open to prevent an explosion. Do NOT force the switch back ON until a certified electrician inspects the fault.",
    },
    "arc_fault_series": {
        "title": "Series Arc Fault (Loose Connection / Broken Wire Sparking)",
        "layman_reason": "A loose screw terminal in a wall socket or broken wire strand is creating invisible high-temperature electrical sparks (over 5,000°C) inside the wall.",
        "layman_symptom": "Current flattening near zero-voltage restrike + high-frequency noise bursts (1.5-2.5 kHz).",
        "safety_type": "critical",
        "safety_badge": "🚨 HIGH FIRE RISK — CALL ELECTRICIAN IMMEDIATELY",
        "user_action": "🚨 CRITICAL FIRE RISK: Stand back from the MCB panel! High-temperature arcing can ignite wall insulation within seconds. Unplug all devices on this branch circuit and call a licensed electrician immediately.",
    },
    "arc_fault_parallel": {
        "title": "Parallel Arc Fault (Damaged Wire Insulation Sparking)",
        "layman_reason": "Insulation between Live and Neutral wires has cracked or degraded, causing high-energy electrical sparks jumping directly between conductors.",
        "layman_symptom": "Random high-frequency noise spikes occurring anywhere in the AC cycle.",
        "safety_type": "critical",
        "safety_badge": "🚨 HIGH FIRE RISK — DO NOT RESET MCB",
        "user_action": "🚨 CRITICAL FIRE RISK: Do NOT attempt to flip the breaker back ON! Parallel arcing generates extreme heat. Keep distance and call an electrician to test wiring insulation.",
    },
    "ground_fault_leakage": {
        "title": "Ground Fault / Earth Leakage (Electric Shock Danger)",
        "layman_reason": "Electric current is escaping out of the wire into appliance metal frames, damp walls, or water pipes instead of returning safely through the neutral wire.",
        "layman_symptom": "Line CT current does not match Neutral CT current (Line Current ≠ Neutral Current).",
        "safety_type": "warning",
        "safety_badge": "⚡ SHOCK HAZARD — DO NOT TOUCH WET APPLIANCES",
        "user_action": "⚡ ELECTRIC SHOCK HAZARD: Do NOT touch wet appliances, metal sinks, or bare pipes! Unplug the suspect appliance carefully and call an electrician to verify your home earth grounding.",
    },
    "harmonic_distortion": {
        "title": "Abnormal Harmonic Distortion (Failing Appliance Motor)",
        "layman_reason": "An appliance motor (e.g. fridge compressor, AC pump) or faulty adapter is failing and feeding dirty, distorted electrical waves back into your house.",
        "layman_symptom": "Total Harmonic Distortion (THD) is high (>15%) with elevated 3rd/5th harmonics.",
        "safety_type": "warning",
        "safety_badge": "⚠️ SERVICE FAULTY APPLIANCE",
        "user_action": "⚠️ MAINTENANCE ADVISORY: Listen for unusual buzzing or hums from your refrigerator, AC, or washing machine motor. Have degrading appliances serviced to prevent motor burnout.",
    },
}


# ---------------------------------------------------------------------------
# Sidebar Controls & Data Source Selector (Simulation Mode vs Hardware Mode)
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    """
<div style="text-align:center; padding:12px 0; border-bottom:1px solid #334155; margin-bottom:14px;">
    <h1 style="font-family:'Outfit', sans-serif; font-size:2.5rem; font-weight:900; letter-spacing:2px; background:linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0;">⚡ AMPIX</h1>
    <p style="margin:2px 0 0 0; font-size:0.85rem; color:#94a3b8; font-weight:600;">(Ampere + Intelligence)</p>
</div>
""",
    unsafe_allow_html=True,
)

# DATA SOURCE SELECTOR
st.sidebar.subheader("📡 Data Source Selector")
data_source_mode = st.sidebar.radio(
    "Select Mode:",
    ["Simulation Mode", "Hardware Mode (ESP32)"],
    index=0,
)

# Breaker Switch State Badge
if st.session_state.mcb_tripped:
    st.sidebar.error("🔴 **MCB BREAKER STATE: TRIPPED**\n\nSafety interlock active. Click 'Reset MCB' below after inspecting circuit.")
else:
    st.sidebar.success("🟢 **MCB BREAKER STATE: CLOSED (ON)**\n\nProtected by AMPIX AI Detector.")

# Load Appliance Selection
st.sidebar.subheader("🔌 Household Load Simulator")
load_lights = st.sidebar.checkbox("Lights & Fans (150W)", value=True)
load_fridge = st.sidebar.checkbox("Refrigerator (350W)", value=True)
load_tv = st.sidebar.checkbox("TV & Electronics (250W)", value=True)
load_washer = st.sidebar.checkbox("Washing Machine (1200W)", value=False)
load_ac = st.sidebar.checkbox("Air Conditioner (2200W)", value=False)
load_heater = st.sidebar.checkbox("Water Heater (3000W)", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("🕹️ Simulated Hazard Triggers")

col_h1, col_h2 = st.sidebar.columns(2)
with col_h1:
    if st.button("🟢 Normal", use_container_width=True):
        st.session_state.sim_condition = "normal"
        st.session_state.demo_active = False
        if st.session_state.mcb_tripped:
            st.session_state.mcb_tripped = False
            st.session_state.timeline_logs.append({
                "Time": time.strftime("%I:%M %p"),
                "System Event & Hazard State": "Returned to Normal Conditions — Circuit Clear",
                "Severity": "NORMAL",
                "Action Taken": "MCB Switch Closed by User",
            })

    if st.button("⚡ Series Arc", use_container_width=True):
        st.session_state.sim_condition = "arc_fault_series"
        st.session_state.mcb_tripped = False
        st.session_state.demo_active = False

    if st.button("⚠️ Overload", use_container_width=True):
        st.session_state.sim_condition = "overcurrent_overload"
        st.session_state.mcb_tripped = False
        st.session_state.demo_active = False

    if st.button("🌊 Ground Fault", use_container_width=True):
        st.session_state.sim_condition = "ground_fault_leakage"
        st.session_state.mcb_tripped = False
        st.session_state.demo_active = False

with col_h2:
    if st.button("💥 Parallel Arc", use_container_width=True):
        st.session_state.sim_condition = "arc_fault_parallel"
        st.session_state.mcb_tripped = False
        st.session_state.demo_active = False

    if st.button("🔥 Short Circuit", use_container_width=True):
        st.session_state.sim_condition = "short_circuit"
        st.session_state.mcb_tripped = False
        st.session_state.demo_active = False

    if st.button("📈 Harmonics", use_container_width=True):
        st.session_state.sim_condition = "harmonic_distortion"
        st.session_state.mcb_tripped = False
        st.session_state.demo_active = False

    if st.button("🔄 Reset MCB", type="primary", use_container_width=True):
        st.session_state.mcb_tripped = False
        st.session_state.sim_condition = "normal"
        st.session_state.demo_active = False
        st.session_state.timeline_logs.append({
            "Time": time.strftime("%I:%M %p"),
            "System Event & Hazard State": "Returned to Normal Conditions — MCB Switch Reset",
            "Severity": "NORMAL",
            "Action Taken": "Manual Reset Verified Safe",
        })

st.sidebar.markdown("---")
st.sidebar.subheader("🎬 Automatic Viva Demo Mode")
if not st.session_state.demo_active:
    if st.sidebar.button("▶ START AUTOMATIC DEMO", type="primary", use_container_width=True):
        st.session_state.demo_active = True
        st.session_state.demo_step = 0
        st.session_state.mcb_tripped = False
else:
    if st.sidebar.button("⏹ STOP DEMO", use_container_width=True):
        st.session_state.demo_active = False


# ---------------------------------------------------------------------------
# Automatic Viva Demo Sequence Controller
# ---------------------------------------------------------------------------
if st.session_state.demo_active:
    step = st.session_state.demo_step
    demo_script = [
        ("normal", False, "Step 1/5: Normal Household Operation (Low Load)"),
        ("normal", False, "Step 2/5: Increasing Household Consumption (High Load)"),
        ("harmonic_distortion", False, "Step 3/5: Early Warning Signal (Distorted Harmonic Waveform)"),
        ("arc_fault_series", False, "Step 4/5: Critical Series Arc Fault Detected (UL 1699 Sparks)"),
        ("short_circuit", True, "Step 5/5: Short Circuit Fault & Automatic MCB Safety Trip"),
    ]
    curr_cond, force_trip, demo_msg = demo_script[step % len(demo_script)]
    st.session_state.sim_condition = curr_cond
    if force_trip:
        st.session_state.mcb_tripped = True
    st.sidebar.warning(f"🎬 **DEMO RUNNING:**\n{demo_msg}")
    st.session_state.demo_step = (step + 1) % len(demo_script)


# ---------------------------------------------------------------------------
# DATA ACQUISITION PIPELINE (SIMULATION vs HARDWARE MODE)
# ---------------------------------------------------------------------------
rated_mcb_amp = 16.0
hardware_is_online = False
hardware_error_msg = ""

if data_source_mode == "Hardware Mode (ESP32)":
    # Attempt to fetch latest reading and status from FastAPI backend
    try:
        resp_status = requests.get(f"{API_BASE_URL}/api/v1/device/status", timeout=1.5)
        resp_latest = requests.get(f"{API_BASE_URL}/api/v1/readings/latest", timeout=1.5)

        if resp_status.status_code == 200 and resp_latest.status_code == 200:
            status_data = resp_status.json()
            latest_data = resp_latest.json()

            if status_data.get("status") == "ONLINE":
                hardware_is_online = True
                pred_class = latest_data["prediction"]
                confidence_pct = latest_data["confidence"]
                probs = latest_data["probabilities"]
                elec = latest_data["electrical_readings"]

                # Use readings from hardware API
                voltage_v = elec.get("voltage_V", 230.0)
                rms_current_a = elec.get("rms_line_A", 0.0) if not st.session_state.mcb_tripped else 0.0
                active_power_w = elec.get("active_power_W", 0.0) if not st.session_state.mcb_tripped else 0.0
                apparent_power_va = round(voltage_v * rms_current_a, 1)
                power_factor = 0.95
                panel_temp = 34.0

                extracted_feats = {
                    "rms_line_A": rms_current_a,
                    "rms_neutral_A": elec.get("rms_neutral_A", rms_current_a),
                    "peak_A": elec.get("peak_A", rms_current_a * 1.414),
                    "crest_factor": 1.44,
                    "thd_pct": elec.get("thd_pct", 2.5),
                    "h5_ratio": 0.01,
                    "hf_band_energy_1p5_2p5kHz": 0.001,
                    "leakage_rms_A": elec.get("leakage_rms_A", 0.02),
                }

                pred_res = {
                    "prediction": pred_class,
                    "confidence_pct": confidence_pct,
                    "probabilities": probs,
                    "top_contributing_features": latest_data.get("top_features", []),
                }
            else:
                hardware_error_msg = "Device SMCB-001 is OFFLINE. No Wi-Fi reading received in last 10 seconds."
        else:
            hardware_error_msg = "FastAPI backend server is not responding."
    except Exception as e:
        hardware_error_msg = f"Cannot connect to AMPIX API ({e}). Run 'python hardware_test.py' or start backend server."

# Fallback or Simulation Mode Execution
if data_source_mode == "Simulation Mode" or not hardware_is_online:
    active_condition = st.session_state.sim_condition

    # Generate fresh physical waveform window corresponding to active condition
    gen_fn = simulate.CLASS_GENERATORS[active_condition]
    line_wave, neutral_wave = gen_fn(rated_mcb_amp)
    extracted_feats = simulate.extract_features(line_wave, neutral_wave, rated_mcb_amp)

    # Predict via ML Model
    pred_res = predict_single_sample(extracted_feats, model, scaler, metadata)
    pred_class = pred_res["predicted_class"]
    confidence_pct = pred_res["confidence_pct"]
    probs = pred_res["probabilities"]

    # Fluctuate displayed parameters realistically around generator
    voltage_v = round(230.0 + np.random.uniform(-1.8, 1.8), 1)
    frequency_hz = round(50.0 + np.random.uniform(-0.09, 0.09), 2)

    base_temp = 31.0 + (extracted_feats["rms_line_A"] / rated_mcb_amp) * 12.0
    if active_condition in ["arc_fault_series", "arc_fault_parallel"]:
        base_temp += 18.0
    elif active_condition == "short_circuit":
        base_temp += 28.0
    panel_temp = round(base_temp + np.random.uniform(-0.6, 0.6), 1)

    if st.session_state.mcb_tripped:
        rms_current_a = 0.0
        power_factor = 1.0
        active_power_w = 0.0
        apparent_power_va = 0.0
    else:
        rms_current_a = round(extracted_feats["rms_line_A"], 2)
        power_factor = round(max(0.70, min(0.99, 1.0 - (extracted_feats["thd_pct"] / 150.0))), 2)
        active_power_w = round(voltage_v * rms_current_a * power_factor, 1)
        apparent_power_va = round(voltage_v * rms_current_a, 1)

# Constant Risk Score lookup
risk_score, risk_tier, risk_color_class = HAZARD_CONSTANT_RISK.get(
    pred_class, HAZARD_CONSTANT_RISK["normal"]
)

# Number Color Highlighting
if st.session_state.mcb_tripped or pred_class in ["short_circuit", "arc_fault_series", "arc_fault_parallel"]:
    num_color_class = "ind-val-hazard"
elif pred_class in ["overcurrent_overload", "harmonic_distortion", "ground_fault_leakage"] or risk_tier == "MEDIUM":
    num_color_class = "ind-val-warning"
else:
    num_color_class = "ind-val-normal"

# Track Event Timeline Log History on Condition Changes
if data_source_mode == "Simulation Mode" and active_condition != st.session_state.prev_condition:
    curr_time_str = time.strftime("%I:%M %p")
    if active_condition == "normal":
        st.session_state.timeline_logs.append({
            "Time": curr_time_str,
            "System Event & Hazard State": "Returned to Normal Conditions — All Parameters Safe",
            "Severity": "NORMAL",
            "Action Taken": "Monitoring Healthy Circuit",
        })
    else:
        st.session_state.timeline_logs.append({
            "Time": curr_time_str,
            "System Event & Hazard State": f"Electrical {active_condition.replace('_', ' ').title()} Detected",
            "Severity": "CRITICAL" if active_condition in ["arc_fault_series", "arc_fault_parallel", "short_circuit"] else "WARNING",
            "Action Taken": f"Constant Risk Score {int(risk_score)}/100 ({risk_tier})",
        })
    st.session_state.prev_condition = active_condition

# MCB Safety Interlock Trip Trigger
if (pred_class in ["short_circuit", "arc_fault_series", "arc_fault_parallel"] or risk_score >= 80.0) and not st.session_state.mcb_tripped:
    st.session_state.mcb_tripped = True
    st.session_state.timeline_logs.append({
        "Time": time.strftime("%I:%M %p"),
        "System Event & Hazard State": f"Trip Activated at {time.strftime('%I:%M %p')} — {pred_class.replace('_', ' ').title()} Safety Interlock",
        "Severity": "CRITICAL",
        "Action Taken": "Power Cut Automatically to Prevent Fire",
    })

# Increment energy
st.session_state.cumulative_energy_kwh += (active_power_w / 1000.0) * (0.5 / 3600.0)

# Append to History Buffer (Limit to 60 data points)
st.session_state.history_buffer.append({
    "time": time.strftime("%I:%M:%S %p"),
    "voltage": voltage_v,
    "current": rms_current_a,
    "power_kw": active_power_w / 1000.0,
    "thd": extracted_feats["thd_pct"],
    "risk": risk_score,
    "confidence": confidence_pct,
})
if len(st.session_state.history_buffer) > 60:
    st.session_state.history_buffer.pop(0)


# ---------------------------------------------------------------------------
# MAIN DASHBOARD UI - AMPIX TOP HEADER BAR
# ---------------------------------------------------------------------------
source_badge = (
    "📡 SIMULATION MODE (simulate.py)" if data_source_mode == "Simulation Mode"
    else ("🟢 HARDWARE ONLINE (SMCB-001)" if hardware_is_online else "🔴 HARDWARE OFFLINE")
)

st.markdown(
    f"""
<div class="ampix-header-box">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <div class="ampix-brand-title">⚡ AMPIX</div>
            <div class="ampix-sub-tag">
                <strong>(Ampere + Intelligence)</strong> — Smart Household Electrical Safety Monitor | 
                <strong>Data Source:</strong> {source_badge} | <strong>Panel ID:</strong> DB-MAIN-01
            </div>
        </div>
        <div>
            <span class="{'led-red' if st.session_state.mcb_tripped else ('led-amber' if risk_tier in ['MEDIUM', 'HIGH'] else 'led-green')}">
                {'🔴 MCB TRIPPED (OPEN)' if st.session_state.mcb_tripped else ('🟡 SYSTEM WARNING' if risk_tier in ['MEDIUM', 'HIGH'] else '🟢 NORMAL OPERATION')}
            </span>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# Hardware Offline Warning Banner
if data_source_mode == "Hardware Mode (ESP32)" and not hardware_is_online:
    st.warning(
        f"🔴 **HARDWARE OFFLINE**: {hardware_error_msg}\n\n"
        "💡 **To Test Hardware Mode:**\n"
        "1. Start FastAPI Backend: `python -m uvicorn backend.main:app --port 8000`\n"
        "2. Run ESP32 Simulator Client: `python hardware_test.py --fault arc_fault_series`\n"
        "*(Or switch back to 'Simulation Mode' in the sidebar to run without backend API)*"
    )


# ---------------------------------------------------------------------------
# MAIN TABBED INTERFACE
# ---------------------------------------------------------------------------
tab_main, tab_timeline, tab_loads, tab_tech = st.tabs([
    "⚡ Real-Time Safety Dashboard",
    "📋 Event & Timeline History Log",
    "🔌 Appliance Load & Controls",
    "🔬 Technical ML Diagnostics",
])


# ---------------------------------------------------------------------------
# TAB 1: REAL-TIME SAFETY DASHBOARD
# ---------------------------------------------------------------------------
with tab_main:
    # Alert Banners (NO ML CONFIDENCE ON MAIN PAGE)
    if st.session_state.mcb_tripped:
        st.markdown(
            f"""
        <div class="banner-tripped">
            <h2 style="margin:0;">🚨 SIMULATED MCB BREAKER TRIPPED — CURRENT FLOW HALTED</h2>
            <p style="margin:6px 0 0 0; font-size:1.02rem;">
                <strong>Reason for Trip:</strong> {pred_class.replace('_', ' ').upper()} DETECTED (Constant Hazard Risk: {int(risk_score)}/100).
                Power was cut automatically to prevent electrical fire or wire damage.
            </p>
            <p style="margin:4px 0 0 0; font-size:0.88rem; color:#fca5a5;">
                Click <strong>"Reset MCB"</strong> in the sidebar after inspecting circuit.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    elif pred_class == "normal" and risk_score < 30:
        st.markdown(
            f"""
        <div class="banner-normal">
            <h3 style="margin:0;">✅ SYSTEM NORMAL — CIRCUIT HEALTHY & STABLE</h3>
            <p style="margin:4px 0 0 0;">All electrical parameters within safe operating limits | <strong>Constant Hazard Risk:</strong> {int(risk_score)}/100 (LOW RISK)</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    elif risk_tier in ["MEDIUM", "WARNING"] or pred_class in ["harmonic_distortion", "overcurrent_overload"]:
        st.markdown(
            f"""
        <div class="banner-warning">
            <h3 style="margin:0;">⚠️ EARLY WARNING: ABNORMAL ELECTRICAL CONDITION</h3>
            <p style="margin:4px 0 0 0;"><strong>Detected Condition:</strong> {pred_class.replace('_', ' ').upper()} | <strong>Constant Hazard Risk:</strong> {int(risk_score)}/100 — Elevated monitoring active.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
        <div class="banner-critical">
            <h3 style="margin:0;">🚨 CRITICAL ELECTRICAL HAZARD DETECTED</h3>
            <p style="margin:4px 0 0 0;"><strong>Hazard Type:</strong> {pred_class.replace('_', ' ').upper()} | <strong>Constant Hazard Risk:</strong> {int(risk_score)}/100 — Immediate inspection required!</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Real-Time Color-Coded Electrical Readings (Continuously Changing)
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.markdown(
            f"""
        <div class="ind-card">
            <div class="ind-lbl">Mains Voltage</div>
            <div class="ind-val-normal">{voltage_v} V</div>
            <div style="font-size:0.75rem; color:#64748b;">Frequency: {frequency_hz} Hz</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
        <div class="ind-card">
            <div class="ind-lbl">Line Current</div>
            <div class="{num_color_class}">{rms_current_a} A</div>
            <div style="font-size:0.75rem; color:#64748b;">Peak: {extracted_feats['peak_A']:.1f} A</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
        <div class="ind-card">
            <div class="ind-lbl">Active Power</div>
            <div class="{num_color_class}">{active_power_w:.0f} W</div>
            <div style="font-size:0.75rem; color:#64748b;">Apparent: {apparent_power_va:.0f} VA</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
        <div class="ind-card">
            <div class="ind-lbl">Power Factor</div>
            <div class="ind-val-normal">{power_factor:.2f}</div>
            <div style="font-size:0.75rem; color:#64748b;">Cos(φ) Efficiency</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with c5:
        st.markdown(
            f"""
        <div class="ind-card">
            <div class="ind-lbl">THD Distortion</div>
            <div class="{'ind-val-hazard' if extracted_feats['thd_pct'] > 15 else ('ind-val-warning' if extracted_feats['thd_pct'] > 8 else 'ind-val-normal')}">{extracted_feats['thd_pct']:.1f}%</div>
            <div style="font-size:0.75rem; color:#64748b;">H5 Ratio: {extracted_feats['h5_ratio']*100:.1f}%</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with c6:
        st.markdown(
            f"""
        <div class="ind-card">
            <div class="ind-lbl">Constant Risk Score</div>
            <div class="{num_color_class}">{int(risk_score)} / 100</div>
            <div style="font-size:0.75rem; color:#64748b;">Panel Temp: {panel_temp} °C</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Layman Explanation & Highlighted Safety Action Note
    layman_info = HAZARD_LAYMAN_MAP.get(pred_class, HAZARD_LAYMAN_MAP["normal"])

    col_lay1, col_lay2 = st.columns([1.3, 1])

    with col_lay1:
        st.subheader("💡 Common-Man Hazard Reason")
        st.markdown(
            f"""
        <div style="background:#1e293b; border:1px solid #334155; border-radius:10px; padding:18px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h3 style="margin:0; color:#f8fafc; font-size:1.15rem;">{layman_info['title']}</h3>
                <span style="font-size:0.85rem; font-weight:bold; padding:4px 10px; border-radius:12px; background:#0f172a; color:#38bdf8;">
                    Constant Risk: {int(risk_score)}/100 ({risk_tier})
                </span>
            </div>
            <hr style="border-color:#334155; margin:10px 0;">
            <p style="font-size:1.02rem; color:#e2e8f0; margin-bottom:8px;">
                <strong>🔍 What is happening (Simple Terms):</strong><br>{layman_info['layman_reason']}
            </p>
            <p style="font-size:0.92rem; color:#94a3b8; margin-bottom:0;">
                <strong>📈 Simple Symptom Observed:</strong> {layman_info['layman_symptom']}
            </p>
            <hr style="border-color:#334155; margin:10px 0;">
            <p style="font-size:0.8rem; color:#64748b; margin:0;">
                <strong>🔬 Dynamic Sensor Parameter Readings (Changing Live):</strong><br>
                • Line Current: <code>{extracted_feats['rms_line_A']:.2f} A</code> &nbsp;|&nbsp; 
                • Peak Amplitude: <code>{extracted_feats['peak_A']:.2f} A</code> &nbsp;|&nbsp; 
                • Crest Factor: <code>{extracted_feats['crest_factor']:.2f}</code><br>
                • THD: <code>{extracted_feats['thd_pct']:.2f}%</code> &nbsp;|&nbsp; 
                • HF Noise Energy (1.5-2.5kHz): <code>{extracted_feats['hf_band_energy_1p5_2p5kHz']:.4f}</code> &nbsp;|&nbsp; 
                • Leakage Current: <code>{extracted_feats['leakage_rms_A']:.3f} A</code>
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_lay2:
        st.subheader("📢 User Safety & Action Instructions")
        safety_box_class = (
            "safety-box-critical" if layman_info["safety_type"] == "critical"
            else ("safety-box-warning" if layman_info["safety_type"] == "warning" else "safety-box-safe")
        )
        st.markdown(
            f"""
        <div class="{safety_box_class}">
            <div style="font-size:0.85rem; font-weight:bold; letter-spacing:0.5px; text-transform:uppercase; margin-bottom:6px;">
                {layman_info['safety_badge']}
            </div>
            <h4 style="margin:0 0 8px 0; color:#ffffff;">Instruction for Householder:</h4>
            <p style="font-size:1.02rem; font-weight:600; line-height:1.45; margin:0;">
                {layman_info['user_action']}
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Real-Time Live Rolling Analytics Charts
    st.subheader("📈 Real-Time Live Graph Analytics")

    if len(st.session_state.history_buffer) > 1:
        df_hist = pd.DataFrame(st.session_state.history_buffer)

        fig_g, axes = plt.subplots(1, 3, figsize=(12, 3.2))
        fig_g.patch.set_facecolor("#0b0f19")

        # Chart 1: Voltage & Current Trends
        ax1 = axes[0]
        ax1.set_facecolor("#1e293b")
        ax1.plot(df_hist["time"], df_hist["voltage"], color="#38bdf8", label="Voltage (V)", linewidth=1.5)
        ax1_i = ax1.twinx()
        ax1_i.plot(df_hist["time"], df_hist["current"], color="#f87171" if num_color_class == "ind-val-hazard" else ("#fbbf24" if num_color_class == "ind-val-warning" else "#34d399"), label="Current (A)", linewidth=1.5, linestyle="--")
        ax1.set_title("Mains Voltage & Current", color="#f8fafc", fontsize=9.5, fontweight="bold")
        ax1.tick_params(colors="#94a3b8", labelsize=6.5)
        ax1_i.tick_params(colors="#94a3b8", labelsize=6.5)
        ax1.set_xticks(ax1.get_xticks()[:: max(1, len(df_hist) // 5)])

        # Chart 2: Active Power & THD %
        ax2 = axes[1]
        ax2.set_facecolor("#1e293b")
        ax2.plot(df_hist["time"], df_hist["power_kw"], color="#34d399", label="Power (kW)", linewidth=1.5)
        ax2_thd = ax2.twinx()
        ax2_thd.plot(df_hist["time"], df_hist["thd"], color="#fbbf24", label="THD (%)", linewidth=1.2, linestyle=":")
        ax2.set_title("Active Power (kW) & THD %", color="#f8fafc", fontsize=9.5, fontweight="bold")
        ax2.tick_params(colors="#94a3b8", labelsize=6.5)
        ax2_thd.tick_params(colors="#94a3b8", labelsize=6.5)
        ax2.set_xticks(ax2.get_xticks()[:: max(1, len(df_hist) // 5)])

        # Chart 3: Constant Hazard Risk Score
        ax3 = axes[2]
        ax3.set_facecolor("#1e293b")
        ax3.plot(df_hist["time"], df_hist["risk"], color="#f87171", label="Constant Risk Score", linewidth=1.8)
        ax3.set_title("Constant Hazard Risk Score Trend", color="#f8fafc", fontsize=9.5, fontweight="bold")
        ax3.set_ylim(0, 105)
        ax3.tick_params(colors="#94a3b8", labelsize=6.5)
        ax3.set_xticks(ax3.get_xticks()[:: max(1, len(df_hist) // 5)])

        plt.tight_layout()
        st.pyplot(fig_g)
        plt.close()


# ---------------------------------------------------------------------------
# TAB 2: FORMATTED HUMAN TIMELINE HISTORY LOG
# ---------------------------------------------------------------------------
with tab_timeline:
    st.subheader("📋 Formatted Event & Timeline History Log")
    st.markdown("Displays human-readable event history formatted as: `Hazard detected at 3:05 PM... Trip activated... Returned to normal conditions at 3:10 PM`.")

    if st.session_state.timeline_logs:
        df_tl = pd.DataFrame(st.session_state.timeline_logs).iloc[::-1]
        st.dataframe(df_tl, use_container_width=True)
    else:
        st.caption("No timeline history logged yet.")


# ---------------------------------------------------------------------------
# TAB 3: APPLIANCE LOAD & CONTROLS
# ---------------------------------------------------------------------------
with tab_loads:
    st.subheader("🔌 Household Appliance Load Breakdown & Controls")
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        st.markdown("### Active Household Appliances:")
        st.write(f"- Lights & Fans (150W): **{'ON' if load_lights else 'OFF'}**")
        st.write(f"- Refrigerator (350W): **{'ON' if load_fridge else 'OFF'}**")
        st.write(f"- TV & Electronics (250W): **{'ON' if load_tv else 'OFF'}**")
        st.write(f"- Washing Machine (1200W): **{'ON' if load_washer else 'OFF'}**")
        st.write(f"- Air Conditioner (2200W): **{'ON' if load_ac else 'OFF'}**")
        st.write(f"- Electric Water Heater (3000W): **{'ON' if load_heater else 'OFF'}**")
    with col_l2:
        st.markdown("### Total Active Power Demand:")
        st.markdown(f"# **{active_power_w:.0f} Watts**")
        st.write(f"Cumulative Energy Consumption: **{st.session_state.cumulative_energy_kwh:.2f} kWh**")


# ---------------------------------------------------------------------------
# TAB 4: TECHNICAL ML DIAGNOSTICS (ML CONFIDENCE KEPT STRICTLY HERE)
# ---------------------------------------------------------------------------
with tab_tech:
    st.subheader("🔬 Technical ML Prediction & Confidence Diagnostics")
    st.markdown(
        f"""
    <div style="background:#1e293b; border:1px solid #334155; border-radius:10px; padding:18px; margin-bottom:16px;">
        <h3 style="margin:0 0 8px 0; color:#38bdf8;">Random Forest ML Classifier Diagnostics</h3>
        <p style="margin:0; font-size:1.05rem; color:#f8fafc;">
            <strong>Predicted Hazard Class:</strong> <code>{pred_class}</code><br>
            <strong>ML Model Confidence Score:</strong> <code style="color:#34d399; font-size:1.15rem; font-weight:bold;">{confidence_pct:.1f}%</code>
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.write("### Full Class Probability Distribution:")
    df_p = pd.DataFrame(
        {"Hazard Class": list(probs.keys()), "Probability (%)": [v * 100 for v in probs.values()]}
    ).sort_values("Probability (%)", ascending=False)
    st.dataframe(df_p, use_container_width=True)

    st.write("### Top Contributing Feature Rationale:")
    for feat_str in pred_res.get("top_contributing_features", pred_res.get("top_features", [])):
        st.markdown(f"• **{feat_str}**")


# Disclaimer Footer
st.markdown(
    """
<div class="footer-text">
    ⚠️ <strong>DISCLAIMER:</strong> AMPIX Software Simulation Only — This application represents a software simulation of a smart household electrical monitoring device for capstone demonstration and viva evaluation. It is not connected to physical high-voltage electrical hardware.
</div>
""",
    unsafe_allow_html=True,
)


# Real-time auto-refresh loop (1.0 second interval) so physical parameters continuously change live
time.sleep(1.0)
st.rerun()
