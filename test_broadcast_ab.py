"""A/B test: current mastering chain vs broadcast chain vs raw ML output.

Loads recording.m4a, runs ML enhancement, then applies both mastering
chains and compares metrics side by side.

Usage:
    uv run python test_broadcast_ab.py
"""

import time

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch
import torchaudio
from scipy import signal

# Compat shim must be imported before engine (side-effect import).
import phonepod._compat  # noqa: F401
from phonepod.engine import Engine, OUTPUT_SR

INPUT_FILE = "recording.m4a"
OUTPUT_RAW = "ab_raw_ml.wav"
OUTPUT_CURRENT = "ab_old_additive.wav"
OUTPUT_HYBRID = "ab_hybrid_subtractive.wav"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def measure_lufs(audio: np.ndarray, sr: int) -> float:
    """Integrated LUFS of a 1D float array."""
    meter = pyln.Meter(sr)
    mono = np.ascontiguousarray(audio.astype(np.float64))
    loudness = meter.integrated_loudness(mono)
    return float(loudness) if np.isfinite(loudness) else float("nan")


def measure_peak_db(audio: np.ndarray) -> float:
    """True peak in dBFS."""
    peak = np.max(np.abs(audio))
    if peak == 0:
        return float("-inf")
    return float(20.0 * np.log10(peak))


def measure_rms_db(audio: np.ndarray) -> float:
    """RMS level in dBFS."""
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
    if rms == 0:
        return float("-inf")
    return float(20.0 * np.log10(rms))


def measure_spectral_bands(audio: np.ndarray, sr: int) -> dict[str, float]:
    """Relative energy (dB) in frequency bands.

    Returns energy in dB for each band, referenced to total energy.
    """
    freqs, psd = signal.welch(audio.astype(np.float64), fs=sr, nperseg=4096)

    bands = {
        "sub (0-100)": (0, 100),
        "low-mid (100-500)": (100, 500),
        "mid (500-2k)": (500, 2000),
        "presence (2k-5k)": (2000, 5000),
        "air (5k-10k)": (5000, 10000),
        "brilliance (10k+)": (10000, sr / 2),
    }

    total_energy = np.sum(psd)
    if total_energy == 0:
        return {name: float("-inf") for name in bands}

    result = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        band_energy = np.sum(psd[mask])
        if band_energy > 0:
            result[name] = float(10.0 * np.log10(band_energy / total_energy))
        else:
            result[name] = float("-inf")

    return result


def collect_metrics(audio: np.ndarray, sr: int) -> dict:
    """Collect all metrics for a single audio signal."""
    lufs = measure_lufs(audio, sr)
    peak = measure_peak_db(audio)
    rms = measure_rms_db(audio)
    crest = peak - rms if np.isfinite(peak) and np.isfinite(rms) else float("nan")
    bands = measure_spectral_bands(audio, sr)
    return {
        "LUFS": lufs,
        "Peak dB": peak,
        "RMS dB": rms,
        "Crest factor": crest,
        **bands,
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_comparison(results: dict[str, dict]) -> None:
    """Print a formatted comparison table with winner annotations."""
    names = list(results.keys())
    metrics = list(results[names[0]].keys())

    # Column widths
    label_w = max(len(m) for m in metrics) + 2
    col_w = 20

    # Header
    header = f"{'Metric':<{label_w}}"
    for name in names:
        header += f"{name:>{col_w}}"
    header += f"{'Winner':>{col_w}}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    # Target-based metrics where closer to target is better
    broadcast_targets = {
        "LUFS": -16.0,  # broadcast standard
        "Peak dB": -1.0,  # headroom
    }

    # Higher is better for these
    higher_better = {"RMS dB"}
    # Lower is better for these
    lower_better = {"Crest factor"}

    for metric in metrics:
        row = f"{metric:<{label_w}}"
        values = {}
        for name in names:
            val = results[name][metric]
            values[name] = val
            if np.isfinite(val):
                row += f"{val:>{col_w}.2f}"
            else:
                row += f"{'N/A':>{col_w}}"

        # Determine winner (skip raw ML for mastering comparisons)
        mastering_names = [n for n in names if n != "Raw ML"]
        winner = ""

        finite_vals = {n: v for n, v in values.items() if n in mastering_names and np.isfinite(v)}
        if len(finite_vals) >= 2:
            if metric in broadcast_targets:
                target = broadcast_targets[metric]
                winner = min(finite_vals, key=lambda n: abs(finite_vals[n] - target))
            elif metric in higher_better:
                winner = max(finite_vals, key=lambda n: finite_vals[n])
            elif metric in lower_better:
                winner = min(finite_vals, key=lambda n: finite_vals[n])
            else:
                # Spectral bands: closer to 0 dB (more balanced) is better
                winner = max(finite_vals, key=lambda n: finite_vals[n])

        row += f"{winner:>{col_w}}"
        print(row)

    print("=" * len(header))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_audio(path: str) -> tuple[torch.Tensor, int]:
    """Load audio file, convert to mono 1D tensor."""
    wav, sr = torchaudio.load(path)
    mono = wav.mean(dim=0).flatten()
    if mono.numel() == 0:
        raise ValueError(f"Input audio is empty: {path}")
    return mono, sr


def main() -> None:
    print(f"Loading {INPUT_FILE}...")
    mono, sr = load_audio(INPUT_FILE)
    print(f"  Loaded: {mono.numel()} samples at {sr} Hz ({mono.numel() / sr:.1f}s)")

    if sr != OUTPUT_SR:
        print(f"  Resampling {sr} -> {OUTPUT_SR} Hz...")
        mono = torchaudio.functional.resample(mono, sr, OUTPUT_SR)
        sr = OUTPUT_SR

    print("\nInitializing Engine (loading ML models)...")
    engine = Engine()

    print("\nRunning ML enhancement pipeline (DeepFilterNet + MossFormer2)...")
    t0 = time.perf_counter()
    enhanced, out_sr = engine.enhance(mono, sr)
    ml_time = time.perf_counter() - t0
    print(f"  ML enhancement complete in {ml_time:.1f}s")

    # Convert to numpy for mastering chains
    ml_output = enhanced.numpy().astype(np.float32)

    # Save raw ML output
    sf.write(OUTPUT_RAW, ml_output, out_sr, subtype="FLOAT")
    print(f"  Saved raw ML output -> {OUTPUT_RAW}")

    # --- Old additive chain (hardcoded for comparison) ---
    from phonepod.profile import MasteringParams
    from pedalboard import Compressor, HighpassFilter, HighShelfFilter, Limiter, NoiseGate, PeakFilter, Pedalboard as PB, Reverb

    old_chain = PB([
        NoiseGate(threshold_db=-50.0, attack_ms=5.0, release_ms=200.0),
        HighpassFilter(cutoff_frequency_hz=80.0),
        PeakFilter(cutoff_frequency_hz=300.0, gain_db=-2.0, q=1.0),
        Compressor(threshold_db=-16.0, ratio=1.8, attack_ms=20.0, release_ms=150.0),
        Compressor(threshold_db=-10.0, ratio=2.5, attack_ms=10.0, release_ms=100.0),
        PeakFilter(cutoff_frequency_hz=6000.0, gain_db=-3.0, q=2.0),
        PeakFilter(cutoff_frequency_hz=3000.0, gain_db=2.0, q=0.8),  # presence BOOST
        HighShelfFilter(cutoff_frequency_hz=10000.0, gain_db=1.5, q=0.7),  # air BOOST
    ])
    old_limiter = PB([Limiter(threshold_db=-1.5)])

    print("\nRunning OLD additive chain (hardcoded from previous defaults)...")
    t0 = time.perf_counter()
    old_input = ml_output.copy()
    if old_input.ndim == 1:
        old_input = old_input[np.newaxis, :]
    old_mastered = old_chain(old_input, sample_rate=out_sr, reset=True)
    old_mono = np.ascontiguousarray(old_mastered[0].astype(np.float64))
    meter = pyln.Meter(out_sr)
    old_loud = meter.integrated_loudness(old_mono)
    if np.isfinite(old_loud):
        old_norm = pyln.normalize.loudness(old_mono, old_loud, -18.0).astype(np.float32)
        old_limited = old_limiter(old_norm[np.newaxis, :], sample_rate=out_sr, reset=True)
        ceiling_lin = 10.0 ** (-1.5 / 20.0)
        old_limited = np.clip(old_limited, -ceiling_lin, ceiling_lin)
    else:
        old_limited = old_mastered
    old_result = np.ascontiguousarray(old_limited[0])
    old_time = time.perf_counter() - t0
    print(f"  Old additive mastering complete in {old_time:.3f}s")
    sf.write(OUTPUT_CURRENT, old_result, out_sr, subtype="FLOAT")
    print(f"  Saved -> {OUTPUT_CURRENT}")

    # --- New hybrid subtractive chain (current engine default) ---
    print("\nRunning NEW hybrid subtractive chain (master_only)...")
    t0 = time.perf_counter()
    hybrid_result = engine.master_only(ml_output.copy())
    hybrid_time = time.perf_counter() - t0
    print(f"  Hybrid subtractive mastering complete in {hybrid_time:.3f}s")

    sf.write(OUTPUT_HYBRID, hybrid_result, out_sr, subtype="FLOAT")
    print(f"  Saved -> {OUTPUT_HYBRID}")

    # --- Collect and compare metrics ---
    print("\nCollecting metrics...")
    results = {
        "Raw ML": collect_metrics(ml_output, out_sr),
        "Old Additive": collect_metrics(old_result, out_sr),
        "Hybrid Sub": collect_metrics(hybrid_result, out_sr),
    }

    print_comparison(results)

    # Timing summary
    print(f"\nTiming:")
    print(f"  ML enhancement:          {ml_time:.1f}s")
    print(f"  Old additive mastering:  {old_time:.3f}s")
    print(f"  Hybrid sub mastering:    {hybrid_time:.3f}s")

    print(f"\nOutput files:")
    print(f"  {OUTPUT_RAW}                  - ML output, no mastering")
    print(f"  {OUTPUT_CURRENT}          - Old additive chain (presence+air boosts)")
    print(f"  {OUTPUT_HYBRID}  - NEW hybrid subtractive chain (cuts only)")


if __name__ == "__main__":
    main()
