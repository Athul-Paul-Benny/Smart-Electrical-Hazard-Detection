"""
Synthetic Current-Waveform Dataset Generator
AI-Based Predictive Electrical Hazard Detection System

Simulates realistic single-phase (Line + Neutral CT) and three-phase current
waveforms for a healthy circuit and 7 hazard conditions, based on documented
real-world electrical signatures:

  - Normal load: fundamental + small natural harmonics + sensor noise
  - Overcurrent/Overload: sustained RMS above rated threshold, mild thermal drift
  - Short circuit: fast current rise + exponentially decaying DC offset
    (classic asymmetrical fault current, IEEE C37 fault-current theory)
  - Series arc fault: current "flattening/shouldering" near zero-crossing +
    broadband high-freq noise bursts at reignition (UL1699 / IEEE 1458 signature)
  - Parallel arc fault: random high-current broadband noise spikes not
    locked to zero-crossing, higher average magnitude
  - Ground fault / leakage: line CT and neutral CT no longer match -
    the difference is the leakage current (this is literally how a
    digital RCD/ELCB works)
  - Harmonic distortion: elevated 3rd/5th/7th/9th harmonics that GROW across
    the window, modeling a degrading motor/compressor/appliance
  - Phase imbalance (3-phase only): unequal magnitudes/phase angles across
    L1/L2/L3, per NEMA MG1 imbalance definition

A separate long-duration low-frequency generator produces "sustained abnormal
loading pattern" data (duty-cycle behaviour of a load over tens of minutes).
"""

import numpy as np
import pandas as pd
from scipy import signal as sig
from scipy.stats import kurtosis, skew

RNG = np.random.default_rng(42)

F0 = 50.0                 # mains frequency (India)
FS = 5000.0                # sampling rate (Hz) - realistic for embedded CT + ADC arc/ground fault work
WINDOW_S = 0.5              # 25 cycles per window
N = int(FS * WINDOW_S)
T = np.arange(N) / FS


# ---------------------------------------------------------------------------
# low level helpers
# ---------------------------------------------------------------------------

def base_sine(amplitude, freq=F0, phase=0.0, t=T):
    return amplitude * np.sin(2 * np.pi * freq * t + phase)


def add_harmonics(wave, amplitude, h_ratios, t=T, freq=F0):
    """h_ratios: dict {harmonic_order: ratio_of_fundamental}, random phase each."""
    out = wave.copy()
    for h, ratio in h_ratios.items():
        ph = RNG.uniform(0, 2 * np.pi)
        out += amplitude * ratio * np.sin(2 * np.pi * freq * h * t + ph)
    return out


def sensor_noise(n=N, snr_db=42, ref_amp=1.0):
    """White measurement noise from CT + ADC quantization."""
    noise_power = ref_amp**2 / (2 * 10 ** (snr_db / 10))
    return RNG.normal(0, np.sqrt(noise_power), n)


def bandlimited_burst(n, fs, f_lo, f_hi, envelope):
    """Broadband high-frequency noise burst shaped by an envelope array (len n)."""
    white = RNG.normal(0, 1, n)
    sos = sig.butter(4, [f_lo, f_hi], btype="band", fs=fs, output="sos")
    filt = sig.sosfilt(sos, white)
    filt = filt / (np.max(np.abs(filt)) + 1e-9)
    return filt * envelope


def zero_crossing_mask(t, freq=F0, width_ms=1.2):
    """1.0 near each zero crossing of the fundamental, tapering to 0 elsewhere."""
    phase = (2 * np.pi * freq * t) % np.pi
    dist_rad = np.minimum(phase, np.pi - phase)
    dist_s = dist_rad / (2 * np.pi * freq)
    width_s = width_ms / 1000.0
    return np.clip(1 - dist_s / width_s, 0, 1)


# ---------------------------------------------------------------------------
# class generators - single phase, two CTs (line, neutral)
# ---------------------------------------------------------------------------

def gen_normal(rated):
    Irms = RNG.uniform(0.30, 0.85) * rated
    A = Irms * np.sqrt(2)
    line = base_sine(A)
    line = add_harmonics(line, A, {3: RNG.uniform(0.01, 0.03),
                                    5: RNG.uniform(0.005, 0.02)})
    line += sensor_noise(snr_db=RNG.uniform(38, 46), ref_amp=A)
    # neutral tracks line almost perfectly (tiny CT mismatch, no real leakage)
    neutral = line * RNG.uniform(0.995, 1.0) + sensor_noise(snr_db=48, ref_amp=A)
    return line, neutral


def gen_overcurrent(rated):
    mult = RNG.uniform(1.10, 1.55)          # sustained overload factor
    Irms = mult * rated
    A0 = Irms * np.sqrt(2)
    # mild thermal drift: amplitude creeps up slightly across the window
    drift = 1 + np.linspace(0, RNG.uniform(0.0, 0.04), N)
    line = base_sine(A0) * drift
    line = add_harmonics(line, A0, {3: RNG.uniform(0.02, 0.05),
                                     5: RNG.uniform(0.01, 0.03)})
    line += sensor_noise(snr_db=RNG.uniform(36, 44), ref_amp=A0)
    neutral = line * RNG.uniform(0.995, 1.0) + sensor_noise(snr_db=46, ref_amp=A0)
    return line, neutral


def gen_short_circuit(rated):
    """Fault current with classic exponentially decaying DC offset (asymmetrical fault)."""
    normal_A = RNG.uniform(0.4, 0.7) * rated * np.sqrt(2)
    fault_mult = RNG.uniform(6, 20)          # relative to rated
    fault_A = fault_mult * rated * np.sqrt(2)
    onset = RNG.uniform(0.15, 0.30)          # fault starts partway through window
    onset_idx = int(onset * FS)
    tau = RNG.uniform(0.01, 0.03)            # X/R decay time constant (s)

    line = base_sine(normal_A).astype(float)
    t_fault = T[onset_idx:] - T[onset_idx]
    dc_offset = fault_A * np.exp(-t_fault / tau)          # decaying DC bias
    ac_fault = base_sine(fault_A, t=T[onset_idx:] - T[onset_idx] + T[onset_idx])
    line[onset_idx:] = ac_fault + dc_offset
    # breaker typically clears in <100ms; simulate current collapsing to ~0 after trip
    trip_delay = RNG.uniform(0.02, 0.08)
    trip_idx = onset_idx + int(trip_delay * FS)
    if trip_idx < N:
        collapse = np.exp(-np.linspace(0, 8, N - trip_idx))
        line[trip_idx:] *= collapse
    line += sensor_noise(snr_db=30, ref_amp=fault_A)
    neutral = line * RNG.uniform(0.99, 1.0) + sensor_noise(snr_db=30, ref_amp=fault_A)
    return line, neutral


def gen_arc_series(rated):
    """Series arc: current shoulders/flattens near zero-crossing, reduced peak,
    broadband noise concentrated at reignition points."""
    Irms = RNG.uniform(0.3, 0.8) * rated
    A = Irms * np.sqrt(2)
    line = base_sine(A)
    line = add_harmonics(line, A, {3: RNG.uniform(0.03, 0.06)})
    # flatten near zero crossing (arc extinguishes, re-strikes at higher voltage)
    zc = zero_crossing_mask(T)
    flatten_depth = RNG.uniform(0.35, 0.65)
    line = line * (1 - flatten_depth * zc)
    # high-frequency noise bursts at each reignition (zero crossing)
    burst_env = zc * RNG.uniform(0.15, 0.35) * A
    hf = bandlimited_burst(N, FS, 2000, 2499, burst_env)
    line = line + hf
    line += sensor_noise(snr_db=RNG.uniform(30, 38), ref_amp=A)
    neutral = line * RNG.uniform(0.995, 1.0) + sensor_noise(snr_db=40, ref_amp=A)
    return line, neutral


def gen_arc_parallel(rated):
    """Parallel arc: elevated + erratic current with random broadband spikes
    anywhere in the cycle (not locked to zero crossing), larger magnitude jumps."""
    Irms = RNG.uniform(0.7, 1.3) * rated
    A = Irms * np.sqrt(2)
    line = base_sine(A)
    line = add_harmonics(line, A, {3: RNG.uniform(0.02, 0.05), 5: RNG.uniform(0.01, 0.03)})
    n_bursts = RNG.integers(6, 18)
    env = np.zeros(N)
    for _ in range(n_bursts):
        center = RNG.integers(0, N)
        width = RNG.integers(int(0.001 * FS), int(0.004 * FS))
        idx = np.arange(max(0, center - width), min(N, center + width))
        local = np.hanning(len(idx))
        env[idx] += local * RNG.uniform(0.3, 0.7) * A
    hf = bandlimited_burst(N, FS, 1500, 2499, env)
    line = line + hf
    line += sensor_noise(snr_db=RNG.uniform(28, 36), ref_amp=A)
    neutral = line * RNG.uniform(0.99, 1.0) + sensor_noise(snr_db=38, ref_amp=A)
    return line, neutral


def gen_ground_fault(rated):
    """Leakage current flows to earth instead of returning through neutral ->
    line CT and neutral CT diverge. This divergence IS the fault signature
    (digital equivalent of an RCD's differential transformer)."""
    Irms = RNG.uniform(0.3, 0.8) * rated
    A = Irms * np.sqrt(2)
    line = base_sine(A)
    line = add_harmonics(line, A, {3: RNG.uniform(0.01, 0.03)})
    line += sensor_noise(snr_db=42, ref_amp=A)

    leak_rms = RNG.choice([RNG.uniform(0.03, 0.1), RNG.uniform(0.1, 0.5)]) \
        if RNG.random() < 0.5 else RNG.uniform(0.03, 0.5)   # amps, 30mA-500mA typical range
    leak_A = leak_rms * np.sqrt(2)
    leak_phase = RNG.uniform(-0.6, 0.6)                      # leakage often not in-phase (capacitive/resistive mix)
    leakage = base_sine(leak_A, phase=leak_phase)
    if RNG.random() < 0.3:
        leakage = add_harmonics(leakage, leak_A, {3: RNG.uniform(0.1, 0.3)})  # electronic-load leakage

    neutral = line - leakage + sensor_noise(snr_db=44, ref_amp=A)
    return line, neutral


def gen_harmonic_distortion(rated):
    """Degrading motor/compressor: THD grows progressively across the window."""
    Irms = RNG.uniform(0.4, 0.9) * rated
    A = Irms * np.sqrt(2)
    line = base_sine(A)
    ramp = np.linspace(RNG.uniform(0.05, 0.1), RNG.uniform(0.25, 0.45), N)  # THD growth
    for h, base_ratio in {3: 0.5, 5: 0.3, 7: 0.15, 9: 0.08}.items():
        ph = RNG.uniform(0, 2 * np.pi)
        line += A * (ramp * base_ratio) * np.sin(2 * np.pi * F0 * h * T + ph)
    line += sensor_noise(snr_db=38, ref_amp=A)
    neutral = line * RNG.uniform(0.99, 1.0) + sensor_noise(snr_db=42, ref_amp=A)
    return line, neutral


CLASS_GENERATORS = {
    "normal": gen_normal,
    "overcurrent_overload": gen_overcurrent,
    "short_circuit": gen_short_circuit,
    "arc_fault_series": gen_arc_series,
    "arc_fault_parallel": gen_arc_parallel,
    "ground_fault_leakage": gen_ground_fault,
    "harmonic_distortion": gen_harmonic_distortion,
}


# ---------------------------------------------------------------------------
# feature extraction (for classical ML: RF / XGBoost / SVM)
# ---------------------------------------------------------------------------

def bandpower(x, fs, f_lo, f_hi):
    freqs, psd = sig.welch(x, fs=fs, nperseg=min(1024, len(x)))
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz_fn(psd[mask], freqs[mask])) if mask.any() else 0.0


def thd_and_harmonics(x, fs, f0=F0, n_harm=9):
    N_ = len(x)
    win = np.hanning(N_)
    X = np.fft.rfft(x * win)
    freqs = np.fft.rfftfreq(N_, 1 / fs)
    mag = np.abs(X)

    def harm_mag(h):
        idx = np.argmin(np.abs(freqs - f0 * h))
        return mag[idx]

    fund = harm_mag(1) + 1e-9
    harms = {h: harm_mag(h) / fund for h in [3, 5, 7, 9]}
    thd = 100 * np.sqrt(sum((harm_mag(h)) ** 2 for h in range(2, n_harm + 1))) / fund
    return thd, harms


def extract_features(line, neutral, rated, fs=FS):
    rms_line = float(np.sqrt(np.mean(line**2)))
    rms_neutral = float(np.sqrt(np.mean(neutral**2)))
    peak_line = float(np.max(np.abs(line)))
    crest_factor = peak_line / (rms_line + 1e-9)
    thd, harms = thd_and_harmonics(line, fs)
    hf_energy = bandpower(line, fs, 1500, 2499)
    di_dt = np.diff(line) * fs
    di_dt_max = float(np.max(np.abs(di_dt)))
    zc = zero_crossing_mask(T)
    zc_region = line[zc > 0.5]
    zc_noise_std = float(np.std(zc_region - np.mean(zc_region))) if len(zc_region) else 0.0
    leakage = line - neutral
    leakage_rms = float(np.sqrt(np.mean(leakage**2)))
    kurt = float(kurtosis(line))
    sk = float(skew(line))
    over_rated_ratio = rms_line / rated

    return {
        "rated_current_A": rated,
        "rms_line_A": rms_line,
        "rms_neutral_A": rms_neutral,
        "over_rated_ratio": over_rated_ratio,
        "peak_A": peak_line,
        "crest_factor": crest_factor,
        "thd_pct": thd,
        "h3_ratio": harms[3],
        "h5_ratio": harms[5],
        "h7_ratio": harms[7],
        "h9_ratio": harms[9],
        "hf_band_energy_1p5_2p5kHz": hf_energy,
        "di_dt_max_A_per_s": di_dt_max,
        "zero_crossing_noise_std": zc_noise_std,
        "leakage_rms_A": leakage_rms,
        "kurtosis": kurt,
        "skew": sk,
    }


# ---------------------------------------------------------------------------
# dataset assembly - single phase
# ---------------------------------------------------------------------------

def build_single_phase_dataset(samples_per_class=150):
    rows = []
    raw_line = []
    raw_neutral = []
    sample_id = 0
    for label, gen_fn in CLASS_GENERATORS.items():
        for _ in range(samples_per_class):
            rated = RNG.choice([6, 10, 16, 20, 25, 32, 40, 63])  # common MCB ratings (A)
            line, neutral = gen_fn(float(rated))
            feats = extract_features(line, neutral, float(rated))
            feats["sample_id"] = sample_id
            feats["label"] = label
            rows.append(feats)
            raw_line.append(line.astype(np.float32))
            raw_neutral.append(neutral.astype(np.float32))
            sample_id += 1
    df = pd.DataFrame(rows)
    cols = ["sample_id", "label"] + [c for c in df.columns if c not in ("sample_id", "label")]
    df = df[cols]
    return df, np.array(raw_line), np.array(raw_neutral)


# ---------------------------------------------------------------------------
# three-phase: phase imbalance dataset
# ---------------------------------------------------------------------------

def gen_three_phase_normal(rated):
    Irms = RNG.uniform(0.3, 0.85) * rated
    A = Irms * np.sqrt(2)
    phases = {}
    for i, base_phase in enumerate([0, -2 * np.pi / 3, 2 * np.pi / 3]):
        jitter = RNG.uniform(-0.02, 0.02)
        amp = A * RNG.uniform(0.97, 1.03)
        w = base_sine(amp, phase=base_phase + jitter)
        w = add_harmonics(w, amp, {3: RNG.uniform(0.01, 0.03), 5: RNG.uniform(0.005, 0.02)})
        w += sensor_noise(snr_db=42, ref_amp=amp)
        phases[f"I{['A','B','C'][i]}"] = w
    return phases


def gen_three_phase_imbalance(rated):
    Irms = RNG.uniform(0.3, 0.85) * rated
    A = Irms * np.sqrt(2)
    imbalance_pct = RNG.uniform(0.15, 0.6)  # one or two phases heavily off
    mults = [1.0, 1.0, 1.0]
    faulty = RNG.choice([0, 1, 2], size=RNG.integers(1, 3), replace=False)
    for f in faulty:
        mults[f] = RNG.choice([1 - imbalance_pct, 1 + imbalance_pct])
    phases = {}
    for i, base_phase in enumerate([0, -2 * np.pi / 3, 2 * np.pi / 3]):
        angle_dev = RNG.uniform(-0.25, 0.25) if i in faulty else RNG.uniform(-0.02, 0.02)
        amp = A * mults[i]
        w = base_sine(amp, phase=base_phase + angle_dev)
        w = add_harmonics(w, amp, {3: RNG.uniform(0.01, 0.04)})
        w += sensor_noise(snr_db=40, ref_amp=amp)
        phases[f"I{['A','B','C'][i]}"] = w
    return phases


def nema_imbalance_pct(rms_a, rms_b, rms_c):
    avg = (rms_a + rms_b + rms_c) / 3
    max_dev = max(abs(rms_a - avg), abs(rms_b - avg), abs(rms_c - avg))
    return 100 * max_dev / (avg + 1e-9)


def build_three_phase_dataset(samples_per_class=150):
    rows = []
    sample_id = 0
    gens = {"normal_3phase": gen_three_phase_normal, "phase_imbalance": gen_three_phase_imbalance}
    for label, gen_fn in gens.items():
        for _ in range(samples_per_class):
            rated = RNG.choice([16, 20, 25, 32, 40, 63, 100])
            ph = gen_fn(float(rated))
            rms = {k: float(np.sqrt(np.mean(v**2))) for k, v in ph.items()}
            imb = nema_imbalance_pct(rms["IA"], rms["IB"], rms["IC"])
            thdA, _ = thd_and_harmonics(ph["IA"], FS)
            rows.append({
                "sample_id": sample_id, "label": label, "rated_current_A": rated,
                "rms_IA": rms["IA"], "rms_IB": rms["IB"], "rms_IC": rms["IC"],
                "nema_imbalance_pct": imb, "thd_IA_pct": thdA,
            })
            sample_id += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# sustained abnormal loading pattern (low-frequency, long duration)
# ---------------------------------------------------------------------------

def build_load_pattern_dataset(n_profiles_per_class=80, duration_min=30):
    rows = []
    raw = []
    sample_id = 0
    seconds = duration_min * 60
    for label in ["normal_duty_cycle", "sustained_abnormal_loading"]:
        for _ in range(n_profiles_per_class):
            rated = RNG.choice([6, 10, 16, 20, 32])
            t = np.arange(seconds)
            if label == "normal_duty_cycle":
                # e.g. fridge/AC compressor: ~8-15 min ON, ~10-20 min OFF, moderate load
                on_dur = RNG.integers(8 * 60, 15 * 60)
                off_dur = RNG.integers(10 * 60, 20 * 60)
                level_on = RNG.uniform(0.5, 0.75) * rated
            else:
                # short-cycling / near-continuous overrun (compressor failing, contactor chatter, etc.)
                mode = RNG.choice(["short_cycle", "continuous_overrun"])
                if mode == "short_cycle":
                    on_dur = RNG.integers(30, 120)
                    off_dur = RNG.integers(20, 90)
                    level_on = RNG.uniform(0.7, 1.1) * rated
                else:
                    on_dur = seconds
                    off_dur = 1
                    level_on = RNG.uniform(0.95, 1.3) * rated

            profile = np.zeros(seconds)
            idx = 0
            state_on = True
            while idx < seconds:
                dur = on_dur if state_on else off_dur
                dur = min(dur, seconds - idx)
                val = level_on if state_on else 0.0
                profile[idx:idx + dur] = val
                idx += dur
                state_on = not state_on
            profile += RNG.normal(0, 0.02 * rated, seconds)
            profile = np.clip(profile, 0, None)

            on_mask = profile > (0.1 * rated)
            duty_pct = 100 * on_mask.mean()
            transitions = np.sum(np.abs(np.diff(on_mask.astype(int))))
            cycles_per_hour = transitions / 2 / (duration_min / 60)
            max_continuous_on_min = 0
            run = 0
            for v in on_mask:
                run = run + 1 if v else 0
                max_continuous_on_min = max(max_continuous_on_min, run)
            max_continuous_on_min /= 60.0

            rows.append({
                "sample_id": sample_id, "label": label, "rated_current_A": rated,
                "mean_rms_A": float(np.mean(profile)), "std_rms_A": float(np.std(profile)),
                "duty_cycle_pct": duty_pct, "cycles_per_hour": cycles_per_hour,
                "max_continuous_on_min": max_continuous_on_min,
            })
            raw.append(profile.astype(np.float32))
            sample_id += 1
    return pd.DataFrame(rows), np.array(raw)


if __name__ == "__main__":
    print("Building single-phase waveform dataset...")
    df_sp, raw_line, raw_neutral = build_single_phase_dataset(samples_per_class=150)
    df_sp.to_csv("features_single_phase.csv", index=False)
    np.savez_compressed("raw_waveforms_single_phase.npz",
                         line=raw_line, neutral=raw_neutral,
                         labels=df_sp["label"].values, fs=FS, window_s=WINDOW_S)
    print(df_sp["label"].value_counts())

    print("\nBuilding three-phase imbalance dataset...")
    df_3p = build_three_phase_dataset(samples_per_class=150)
    df_3p.to_csv("features_three_phase_imbalance.csv", index=False)
    print(df_3p["label"].value_counts())

    print("\nBuilding sustained load-pattern dataset...")
    df_lp, raw_lp = build_load_pattern_dataset(n_profiles_per_class=80, duration_min=30)
    df_lp.to_csv("features_load_pattern.csv", index=False)
    np.savez_compressed("raw_load_profiles.npz", profiles=raw_lp,
                         labels=df_lp["label"].values, seconds=30 * 60)
    print(df_lp["label"].value_counts())

    print("\nDone.")
