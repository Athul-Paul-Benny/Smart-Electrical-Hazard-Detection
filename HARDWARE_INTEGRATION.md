# AMPIX Hardware Integration Guide
## Connecting ESP32 / STM32 / Microcontrollers to AMPIX API

---

## 1. Network & System Architecture

```
   ┌───────────────────────┐
   │ Household Circuit CTs │
   └───────────┬───────────┘
               │ Analog CT Signals (Line + Neutral)
   ┌───────────▼───────────┐
   │   ESP32 / MCU ADC     │
   │  Sampling & Features  │
   └───────────┬───────────┘
               │ Wi-Fi / HTTP POST JSON
   ┌───────────▼───────────┐
   │ AMPIX FastAPI Backend │
   │ (http://<IP>:8000)    │
   └───────────┬───────────┘
               │ Scaling & Inference
   ┌───────────▼───────────┐
   │  Random Forest Model  │
   │ (best_model.joblib)   │
   └───────────┬───────────┘
               │ Real-Time Prediction & Risk
   ┌───────────▼───────────┐
   │   AMPIX Dashboard     │
   │  (Streamlit app.py)   │
   └───────────────────────┘
```

The ESP32 microcontroller samples dual CT sensors, calculates electrical feature metrics, and transmits a JSON payload over Wi-Fi via HTTP POST to the AMPIX FastAPI backend running on your computer.

---

## 2. API Endpoints Specification

### Base API URL:
`http://<YOUR-COMPUTER-IP>:8000`

### Primary Ingestion Endpoint:
- **URL**: `POST /api/v1/readings`
- **Header**: `Content-Type: application/json`

---

## 3. JSON Request Payload Schema

```json
{
  "device_id": "SMCB-001",
  "timestamp": "2026-09-10T12:30:45Z",
  "data_source": "hardware",
  "rated_current_A": 16.0,
  "rms_line_A": 4.25,
  "rms_neutral_A": 4.21,
  "over_rated_ratio": 0.265,
  "peak_A": 6.12,
  "crest_factor": 1.44,
  "thd_pct": 2.5,
  "h3_ratio": 0.02,
  "h5_ratio": 0.01,
  "h7_ratio": 0.0003,
  "h9_ratio": 0.0002,
  "hf_band_energy_1p5_2p5kHz": 0.0005,
  "di_dt_max_A_per_s": 2500.0,
  "zero_crossing_noise_std": 0.85,
  "leakage_rms_A": 0.02,
  "kurtosis": -1.48,
  "skew": 0.0
}
```

---

## 4. Expected API JSON Response

```json
{
  "success": true,
  "device_id": "SMCB-001",
  "timestamp": "2026-09-10T12:30:45Z",
  "data_source": "hardware",
  "prediction": "normal",
  "confidence": 98.7,
  "risk_score": 5.0,
  "risk_level": "LOW",
  "mcb_state": "ON",
  "probabilities": {
    "normal": 0.987,
    "arc_fault_series": 0.005,
    "arc_fault_parallel": 0.002,
    "short_circuit": 0.0,
    "ground_fault_leakage": 0.003,
    "overcurrent_overload": 0.002,
    "harmonic_distortion": 0.001
  },
  "electrical_readings": {
    "voltage_V": 230.0,
    "rms_line_A": 4.25,
    "rms_neutral_A": 4.21,
    "active_power_W": 929.7,
    "thd_pct": 2.5,
    "leakage_rms_A": 0.02,
    "peak_A": 6.12
  },
  "top_features": [
    "di_dt_max_A_per_s = 2500.0000",
    "thd_pct = 2.5000",
    "crest_factor = 1.4400"
  ]
}
```

---

## 5. Arduino / C++ Code Snippet for ESP32

```cpp
#include <WiFi.h>
#include <HTTPClient.h>

const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
const char* api_url = "http://192.168.1.100:8000/api/v1/readings"; // Replace with your laptop IP

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected!");
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(api_url);
    http.addHeader("Content-Type", "application/json");

    // Construct JSON Payload
    String jsonPayload = "{"
      "\"device_id\":\"SMCB-001\","
      "\"data_source\":\"hardware\","
      "\"rated_current_A\":16.0,"
      "\"rms_line_A\":4.25,"
      "\"rms_neutral_A\":4.21,"
      "\"peak_A\":6.12,"
      "\"crest_factor\":1.44,"
      "\"thd_pct\":2.5,"
      "\"h3_ratio\":0.02,"
      "\"h5_ratio\":0.01,"
      "\"h7_ratio\":0.0003,"
      "\"h9_ratio\":0.0002,"
      "\"hf_band_energy_1p5_2p5kHz\":0.0005,"
      "\"di_dt_max_A_per_s\":2500.0,"
      "\"zero_crossing_noise_std\":0.85,"
      "\"leakage_rms_A\":0.02,"
      "\"kurtosis\":-1.48,"
      "\"skew\":0.0"
    "}";

    int httpResponseCode = http.POST(jsonPayload);
    if (httpResponseCode > 0) {
      String response = http.getString();
      Serial.println(httpResponseCode);
      Serial.println(response);
    } else {
      Serial.print("Error sending POST: ");
      Serial.println(httpResponseCode);
    }
    http.end();
  }
  delay(1000); // Send readings every 1 second
}
```

---

## 6. How to Test Hardware Mode Without Hardware

You can test Hardware Mode directly using the provided `hardware_test.py` script:

1. Start the FastAPI backend:
   ```powershell
   python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```
2. Start the AMPIX dashboard in another terminal:
   ```powershell
   python -m streamlit run app.py
   ```
3. Run the test client in a third terminal:
   ```powershell
   python hardware_test.py --fault arc_fault_series
   ```
4. On the AMPIX dashboard, select **`○ Hardware Mode (ESP32)`** in the sidebar. You will see:
   `🟢 HARDWARE ONLINE | Device: SMCB-001` updating live!

---

## 7. Safety Disclaimer

> ⚠️ **ELECTRICAL SAFETY DISCLAIMER**: AMPIX is an AI monitoring and hazard classification prototype. Simulation Mode uses physics-based mathematical models. Hardware Mode is designed for low-voltage, safely isolated sensor signals. Do not connect development microcontrollers directly to high-voltage mains electricity without certified isolation transformers, optocouplers, qualified supervision, and strict electrical safety compliance.
