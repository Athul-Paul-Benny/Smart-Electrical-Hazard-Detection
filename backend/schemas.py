"""
AMPIX FastAPI Backend — Pydantic Schemas (backend/schemas.py)
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class SensorReadingInput(BaseModel):
    device_id: str = Field(default="SMCB-001", description="Smart MCB Device ID")
    timestamp: Optional[str] = Field(default=None, description="ISO timestamp of reading")
    data_source: str = Field(default="hardware", description="Data source: 'hardware' or 'simulation'")
    
    # Precomputed / Transmitted Electrical Features
    rated_current_A: float = Field(default=16.0, description="MCB Nameplate Rated Current in Amperes")
    rms_line_A: float = Field(default=4.25, description="Line Conductor RMS Current in Amperes")
    rms_neutral_A: float = Field(default=4.21, description="Neutral Conductor RMS Current in Amperes")
    over_rated_ratio: Optional[float] = Field(default=None, description="Line RMS / Rated Current Ratio")
    peak_A: float = Field(default=6.12, description="Peak Current Amplitude in Amperes")
    crest_factor: float = Field(default=1.44, description="Crest Factor Peak/RMS Ratio")
    thd_pct: float = Field(default=2.5, description="Total Harmonic Distortion Percentage")
    h3_ratio: float = Field(default=0.02, description="3rd Harmonic Ratio")
    h5_ratio: float = Field(default=0.01, description="5th Harmonic Ratio")
    h7_ratio: float = Field(default=0.0003, description="7th Harmonic Ratio")
    h9_ratio: float = Field(default=0.0002, description="9th Harmonic Ratio")
    hf_band_energy_1p5_2p5kHz: float = Field(default=0.0005, description="Broadband 1.5-2.5 kHz HF noise energy")
    di_dt_max_A_per_s: float = Field(default=2500.0, description="Max rate of current change di/dt in A/s")
    zero_crossing_noise_std: float = Field(default=0.85, description="Noise std dev near zero crossing")
    leakage_rms_A: float = Field(default=0.02, description="Line - Neutral CT differential leakage RMS in A")
    kurtosis: float = Field(default=-1.48, description="Waveform Kurtosis")
    skew: float = Field(default=0.0, description="Waveform Skewness")

    simulated_condition: Optional[str] = Field(default=None, description="Optional target condition for simulation")


class PredictionOutput(BaseModel):
    success: bool = True
    device_id: str
    timestamp: str
    data_source: str
    prediction: str
    confidence: float
    risk_score: float
    risk_level: str
    mcb_state: str
    probabilities: Dict[str, float]
    electrical_readings: Dict[str, float]
    top_features: List[str]


class DeviceStatusOutput(BaseModel):
    device_id: str
    status: str  # ONLINE or OFFLINE
    last_seen: Optional[str]
    data_source: str
    mcb_state: str
    firmware_version: str = "v2.4-PRO"
    sampling_rate_khz: float = 5.0


class HistoryResponse(BaseModel):
    count: int
    readings: List[Dict]
