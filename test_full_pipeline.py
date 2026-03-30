"""Test the full pipeline: DeepFilterNet denoise → pedalboard mastering → LUFS normalize."""
import types
import sys
import subprocess

import numpy as np
import torch
import torchaudio

# Monkey-patch torchaudio.backend for DeepFilterNet compatibility
if not hasattr(torchaudio, 'backend'):
    backend_module = types.ModuleType('torchaudio.backend')
    common_module = types.ModuleType('torchaudio.backend.common')
    class AudioMetaData:
        def __init__(self, sample_rate=0, num_frames=0, num_channels=0, bits_per_sample=0, encoding=""):
            self.sample_rate = sample_rate
            self.num_frames = num_frames
            self.num_channels = num_channels
            self.bits_per_sample = bits_per_sample
            self.encoding = encoding
    common_module.AudioMetaData = AudioMetaData
    backend_module.common = common_module
    sys.modules['torchaudio.backend'] = backend_module
    sys.modules['torchaudio.backend.common'] = common_module
    torchaudio.backend = backend_module

from df.enhance import init_df, enhance as df_enhance
from pedalboard import Pedalboard, HighpassFilter, PeakFilter, HighShelfFilter, Compressor, Limiter
import pyloudnorm as pyln

# --- Init ---
print("Loading DeepFilterNet3...")
model, df_state, _ = init_df()
target_sr = df_state.sr()  # 48000

# --- Load audio ---
subprocess.run(
    ["ffmpeg", "-i", "recording.m4a", "-ar", str(target_sr), "-ac", "1", "/tmp/pipeline_input.wav", "-y", "-loglevel", "error"],
    check=True,
)
wav, sr = torchaudio.load("/tmp/pipeline_input.wav")
print(f"Loaded: {wav.shape[1]/sr:.1f}s at {sr}Hz")

# --- Step 1: DeepFilterNet denoise (full file, it's fast) ---
print("Step 1: DeepFilterNet noise suppression...")
denoised = df_enhance(model, df_state, wav)
torchaudio.save("pipeline_step1_denoised.wav", denoised, target_sr)
print(f"  Saved step1 ({denoised.shape[1]/target_sr:.1f}s)")

# --- Step 2-6: Pedalboard mastering chain ---
print("Step 2-6: Mastering chain (HPF → EQ → Compression → De-ess → Presence/Air)...")
mastering = Pedalboard([
    HighpassFilter(cutoff_frequency_hz=80),
    PeakFilter(cutoff_frequency_hz=300, gain_db=-3.0, q=1.0),       # cut mud
    Compressor(threshold_db=-20, ratio=2.0, attack_ms=15, release_ms=100),  # gentle
    Compressor(threshold_db=-15, ratio=3.0, attack_ms=10, release_ms=80),   # tighter
    PeakFilter(cutoff_frequency_hz=6000, gain_db=-4.0, q=2.0),      # de-ess
    PeakFilter(cutoff_frequency_hz=3000, gain_db=2.5, q=0.8),       # presence
    HighShelfFilter(cutoff_frequency_hz=10000, gain_db=2.0, q=0.7),  # air
])

audio_np = denoised.numpy().astype(np.float32)
mastered = mastering(audio_np, sample_rate=target_sr, reset=True)
torchaudio.save("pipeline_step6_mastered.wav", torch.from_numpy(mastered), target_sr)
print(f"  Saved step6")

# --- Step 7: LUFS normalization on FULL audio ---
print("Step 7: LUFS normalization to -16 LUFS...")
meter = pyln.Meter(target_sr)
# pyloudnorm wants (samples,) for mono
mono_for_lufs = mastered[0].astype(np.float64)
loudness = meter.integrated_loudness(mono_for_lufs)
print(f"  Current loudness: {loudness:.1f} LUFS")
normalized = pyln.normalize.loudness(mono_for_lufs, loudness, -16.0).astype(np.float32)

# --- Step 8: Brick-wall limiter ---
print("Step 8: Brick-wall limiter at -1.5dB...")
limiter = Pedalboard([Limiter(threshold_db=-1.5)])
final = limiter(normalized[np.newaxis, :], sample_rate=target_sr, reset=True)

torchaudio.save("pipeline_final.wav", torch.from_numpy(final), target_sr)
print(f"\nDone! Output: pipeline_final.wav ({final.shape[1]/target_sr:.1f}s at {target_sr}Hz)")
print("\nListen to each step to hear the progression:")
print("  1. pipeline_step1_denoised.wav  — noise removed")
print("  2. pipeline_step6_mastered.wav  — EQ + compression applied")
print("  3. pipeline_final.wav           — normalized + limited (final output)")
