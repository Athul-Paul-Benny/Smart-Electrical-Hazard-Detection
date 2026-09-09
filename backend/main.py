"""
AMPIX FastAPI Backend Server (backend/main.py)
Supports both Simulation Mode and Hardware Mode (ESP32 / MCU Wi-Fi Ingestion)

Launch:
  python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

import time
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    SensorReadingInput,
    PredictionOutput,
    DeviceStatusOutput,
    HistoryResponse,
)
from backend.model_service import model_service
from backend.feature_service import feature_service
from backend.risk_service import risk_service

app = FastAPI(
    title="AMPIX AI Electrical Safety API",
    description="FastAPI Backend for AMPIX (Ampere + Intelligence) Smart Electrical Hazard & Arc-Fault Detection System",
    version="2.4-PRO",
)

# Enable CORS for Streamlit frontend and local Wi-Fi clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State Memory for Hardware/Simulation Ingestion
state = {
    "latest_reading": None,
    "last_seen_timestamp": None,
    "last_seen_time_sec": 0,
    "mcb_state": "ON",
    "active_device_id": "SMCB-001",
    "active_data_source": "hardware",
    "history_log": [],
}


@app.get("/")
def read_root():
    return {
        "system": "AMPIX (Ampere + Intelligence)",
        "status": "ONLINE",
        "api_docs": "/docs",
        "version": "v2.4-PRO",
    }


@app.post("/api/v1/readings", response_model=PredictionOutput)
def receive_sensor_readings(reading: SensorReadingInput):
    """
    Primary ingestion endpoint for ESP32 hardware or simulation readings.
    Passes data through feature processing, StandardScaler, Random Forest classifier, and risk engine.
    """
    try:
        raw_dict = reading.model_dump()
        device_id = reading.device_id
        data_source = reading.data_source
        iso_time = reading.timestamp or time.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Process and align 17 features
        feats = feature_service.process_and_align_features(raw_dict, model_service.feature_names)

        # Execute ML prediction
        pred_res = model_service.predict(feats)

        # Risk assessment & MCB trip state
        risk_info = risk_service.evaluate_risk(pred_res["prediction"])

        if risk_info["should_trip"]:
            state["mcb_state"] = "TRIPPED"
        elif raw_dict.get("mcb_state") == "ON" or reading.simulated_condition == "normal":
            state["mcb_state"] = "ON"

        mcb_state_out = "TRIPPED" if state["mcb_state"] == "TRIPPED" else "ON"

        # Electrical summary
        electrical_readings = {
            "voltage_V": 230.0,
            "rms_line_A": feats["rms_line_A"],
            "rms_neutral_A": feats["rms_neutral_A"],
            "active_power_W": round(230.0 * feats["rms_line_A"] * 0.95, 1),
            "thd_pct": feats["thd_pct"],
            "leakage_rms_A": feats["leakage_rms_A"],
            "peak_A": feats["peak_A"],
        }

        output = PredictionOutput(
            success=True,
            device_id=device_id,
            timestamp=iso_time,
            data_source=data_source,
            prediction=pred_res["prediction"],
            confidence=pred_res["confidence"],
            risk_score=risk_info["risk_score"],
            risk_level=risk_info["risk_level"],
            mcb_state=mcb_state_out,
            probabilities=pred_res["probabilities"],
            electrical_readings=electrical_readings,
            top_features=pred_res["top_features"],
        )

        # Update server memory
        state["latest_reading"] = output.model_dump()
        state["last_seen_timestamp"] = iso_time
        state["last_seen_time_sec"] = time.time()
        state["active_device_id"] = device_id
        state["active_data_source"] = data_source

        # Append to rolling history
        state["history_log"].append({
            "timestamp": iso_time,
            "time_formatted": time.strftime("%I:%M %p"),
            "device_id": device_id,
            "prediction": pred_res["prediction"],
            "confidence": pred_res["confidence"],
            "risk_score": risk_info["risk_score"],
            "mcb_state": mcb_state_out,
            "data_source": data_source,
        })
        if len(state["history_log"]) > 100:
            state["history_log"].pop(0)

        return output

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"API Ingestion Error: {str(e)}")


@app.get("/api/v1/readings/latest")
def get_latest_reading():
    """Returns the most recent processed electrical reading and ML prediction."""
    if state["latest_reading"] is None:
        raise HTTPException(status_code=444, detail="No sensor readings received yet.")
    return state["latest_reading"]


@app.get("/api/v1/device/status", response_model=DeviceStatusOutput)
def get_device_status():
    """
    Returns device connection status (ONLINE if reading received in last 10 seconds),
    data source, last seen timestamp, and MCB state.
    """
    time_diff = time.time() - state["last_seen_time_sec"]
    is_online = time_diff < 10.0 and state["latest_reading"] is not None

    return DeviceStatusOutput(
        device_id=state["active_device_id"],
        status="ONLINE" if is_online else "OFFLINE",
        last_seen=state["last_seen_timestamp"],
        data_source=state["active_data_source"] if is_online else "none",
        mcb_state=state["mcb_state"],
    )


@app.get("/api/v1/readings/history")
def get_readings_history():
    """Returns rolling event history of processed readings."""
    return {
        "count": len(state["history_log"]),
        "readings": list(reversed(state["history_log"])),
    }
