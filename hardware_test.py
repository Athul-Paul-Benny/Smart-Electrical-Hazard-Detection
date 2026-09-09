"""
AMPIX Hardware Mode Test Client (hardware_test.py)
Simulates an ESP32 microcontroller sending HTTP POST JSON payloads to AMPIX FastAPI Backend
(http://localhost:8000/api/v1/readings)

Usage:
  python hardware_test.py
  python hardware_test.py --fault arc_fault_series
"""

import time
import argparse
import requests
import numpy as np

import simulate

API_URL = "http://localhost:8000/api/v1/readings"


def main():
    parser = argparse.ArgumentParser(description="AMPIX ESP32 Simulated Hardware Client")
    parser.add_argument("--fault", type=str, default="normal", help="Fault condition to simulate")
    parser.add_argument("--rated", type=float, default=16.0, help="Rated MCB current in Amperes")
    parser.add_argument("--device", type=str, default="SMCB-001", help="Device ID")
    parser.add_argument("--interval", type=float, default=1.0, help="Sending interval in seconds")
    args = parser.parse_args()

    print(f"=======================================================")
    print(f" ⚡ AMPIX Hardware Mode Test Client (ESP32 Simulator)")
    print(f"=======================================================")
    print(f" Target API URL : {API_URL}")
    print(f" Device ID      : {args.device}")
    print(f" Fault Condition: {args.fault.upper()}")
    print(f" Rated MCB      : {args.rated} A")
    print(f" Interval       : {args.interval}s")
    print(f" Press Ctrl+C to stop.\n")

    counter = 0

    while True:
        try:
            # Generate physical waveform window
            gen_fn = simulate.CLASS_GENERATORS.get(args.fault, simulate.gen_normal)
            line, neutral = gen_fn(args.rated)
            feats = simulate.extract_features(line, neutral, args.rated)

            # Build ESP32 JSON payload
            payload = {
                "device_id": args.device,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "data_source": "hardware",
                "simulated_condition": args.fault,
                **feats,
            }

            resp = requests.post(API_URL, json=payload, timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                counter += 1
                print(
                    f"[{counter:03d}] Sent payload | Status: {resp.status_code} | "
                    f"Prediction: {data['prediction'].upper():20s} | "
                    f"Risk: {data['risk_score']}/100 ({data['risk_level']}) | "
                    f"MCB: {data['mcb_state']}"
                )
            else:
                print(f"[ERROR] API returned status code {resp.status_code}: {resp.text}")

        except requests.exceptions.ConnectionError:
            print("[ERROR] Cannot connect to AMPIX API backend at http://localhost:8000. Is FastAPI server running?")
        except Exception as e:
            print(f"[ERROR] Unexpected exception: {e}")

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
