"""Diagnose muffled output — isolate each pipeline stage.

Saves intermediate WAVs so you can A/B compare in any audio player.
Outputs:
  diagnose_0_input_48k.wav     — raw input resampled to 48kHz (baseline)
  diagnose_1_deepfilter.wav    — after DeepFilterNet only
  diagnose_2_mossformer.wav    — after DeepFilterNet + MossFormer2
  diagnose_3_mastered.wav      — after full pipeline (current defaults)
  diagnose_2b_mossformer_only.wav — MossFormer2 directly on raw (skip DeepFilterNet)
"""

import sys
import logging

import numpy as np
import torch
import torchaudio

import phonepod._compat  # noqa: F401
from phonepod.engine import OUTPUT_SR
from phonepod.profile import MasteringParams

logging.basicConfig(level=logging.INFO, format="%(message)s")

INPUT = sys.argv[1] if len(sys.argv) > 1 else "recording.m4a"
PREFIX = "diagnose"


def save(name: str, audio: np.ndarray, sr: int = OUTPUT_SR) -> None:
    t = torch.from_numpy(audio).unsqueeze(0) if audio.ndim == 1 else torch.from_numpy(audio)
    path = f"{PREFIX}_{name}.wav"
    torchaudio.save(path, t, sr)
    print(f"  Saved: {path} ({audio.shape[-1] / sr:.1f}s)")


def main() -> None:
    print(f"\n=== Loading {INPUT} ===")

    # Load and convert to 48kHz mono
    wav, sr = torchaudio.load(INPUT)
    mono = wav.mean(dim=0).flatten().to(dtype=torch.float32)
    if sr != OUTPUT_SR:
        mono_48k = torchaudio.functional.resample(mono, sr, OUTPUT_SR)
    else:
        mono_48k = mono
    save("0_input_48k", mono_48k.numpy())

    # --- Stage 1: DeepFilterNet only ---
    print("\n=== Stage 1: DeepFilterNet3 ===")
    from df.enhance import enhance as df_enhance, init_df

    df_model, df_state, _ = init_df()
    df_sr = df_state.sr()

    audio_for_df = mono.clone()
    if sr != df_sr:
        audio_for_df = torchaudio.functional.resample(audio_for_df, sr, df_sr)

    denoised = df_enhance(df_model, df_state, audio_for_df.unsqueeze(0))
    denoised_1d = denoised.squeeze(0).detach().cpu()

    # Resample denoised to 48kHz for saving
    if df_sr != OUTPUT_SR:
        denoised_48k = torchaudio.functional.resample(denoised_1d, df_sr, OUTPUT_SR)
    else:
        denoised_48k = denoised_1d
    save("1_deepfilter", denoised_48k.numpy())

    # --- Stage 2: DeepFilterNet + MossFormer2 ---
    print("\n=== Stage 2: MossFormer2 (after DeepFilterNet) ===")
    from clearvoice import ClearVoice

    cv = ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])

    # MossFormer2 expects 48kHz
    cv_input = denoised_48k.numpy().astype(np.float32)[np.newaxis, :]
    with torch.no_grad():
        cv_result = cv(cv_input)

    if isinstance(cv_result, dict):
        enhanced_np = list(cv_result.values())[0]
    elif isinstance(cv_result, torch.Tensor):
        enhanced_np = cv_result.numpy()
    else:
        enhanced_np = np.asarray(cv_result, dtype=np.float32)

    enhanced_np = np.ascontiguousarray(enhanced_np.flatten().astype(np.float32))
    save("2_mossformer", enhanced_np)

    # --- Stage 2b: MossFormer2 ONLY (skip DeepFilterNet) ---
    print("\n=== Stage 2b: MossFormer2 only (no DeepFilterNet) ===")
    cv_raw_input = mono_48k.numpy().astype(np.float32)[np.newaxis, :]
    with torch.no_grad():
        cv_raw_result = cv(cv_raw_input)

    if isinstance(cv_raw_result, dict):
        raw_enhanced = list(cv_raw_result.values())[0]
    elif isinstance(cv_raw_result, torch.Tensor):
        raw_enhanced = cv_raw_result.numpy()
    else:
        raw_enhanced = np.asarray(cv_raw_result, dtype=np.float32)

    raw_enhanced = np.ascontiguousarray(raw_enhanced.flatten().astype(np.float32))
    save("2b_mossformer_only", raw_enhanced)

    # --- Stage 3: Full mastering on stage 2 output ---
    print("\n=== Stage 3: Mastering (default params) ===")
    from pedalboard import (
        Compressor, HighpassFilter, HighShelfFilter,
        Limiter, PeakFilter, Pedalboard,
    )
    import pyloudnorm as pyln

    p = MasteringParams()
    mastering = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=p.hpf_cutoff_hz),
        PeakFilter(cutoff_frequency_hz=p.low_mid_freq_hz, gain_db=p.low_mid_gain_db, q=p.low_mid_q),
        Compressor(threshold_db=p.comp1_threshold_db, ratio=p.comp1_ratio, attack_ms=p.comp1_attack_ms, release_ms=p.comp1_release_ms),
        Compressor(threshold_db=p.comp2_threshold_db, ratio=p.comp2_ratio, attack_ms=p.comp2_attack_ms, release_ms=p.comp2_release_ms),
        PeakFilter(cutoff_frequency_hz=p.deess_freq_hz, gain_db=p.deess_gain_db, q=p.deess_q),
        PeakFilter(cutoff_frequency_hz=p.presence_freq_hz, gain_db=p.presence_gain_db, q=p.presence_q),
        HighShelfFilter(cutoff_frequency_hz=p.air_freq_hz, gain_db=p.air_gain_db, q=p.air_q),
    ])
    limiter = Pedalboard([Limiter(threshold_db=p.limiter_ceiling_db)])

    buf = enhanced_np[np.newaxis, :]
    mastered = mastering(buf, sample_rate=OUTPUT_SR, reset=True)

    mono_64 = np.ascontiguousarray(mastered[0].astype(np.float64))
    meter = pyln.Meter(OUTPUT_SR)
    loudness = meter.integrated_loudness(mono_64)
    if np.isfinite(loudness):
        normalized = pyln.normalize.loudness(mono_64, loudness, p.lufs_target).astype(np.float32)
    else:
        normalized = mono_64.astype(np.float32)

    limited = limiter(normalized[np.newaxis, :], sample_rate=OUTPUT_SR, reset=True)
    save("3_mastered", limited[0])

    # --- Stage 3b: Mastering on MossFormer2-only output ---
    print("\n=== Stage 3b: Mastering on MossFormer2-only ===")
    buf_raw = raw_enhanced[np.newaxis, :]
    mastered_raw = mastering(buf_raw, sample_rate=OUTPUT_SR, reset=True)
    mono_64_raw = np.ascontiguousarray(mastered_raw[0].astype(np.float64))
    loudness_raw = meter.integrated_loudness(mono_64_raw)
    if np.isfinite(loudness_raw):
        norm_raw = pyln.normalize.loudness(mono_64_raw, loudness_raw, p.lufs_target).astype(np.float32)
    else:
        norm_raw = mono_64_raw.astype(np.float32)
    limited_raw = limiter(norm_raw[np.newaxis, :], sample_rate=OUTPUT_SR, reset=True)
    save("3b_mastered_mossformer_only", limited_raw[0])

    # --- Summary ---
    print("\n=== Listen & Compare ===")
    print("  0_input_48k           — raw input (baseline)")
    print("  1_deepfilter          — DeepFilterNet only")
    print("  2_mossformer          — DeepFilterNet + MossFormer2")
    print("  2b_mossformer_only    — MossFormer2 only (no denoise)")
    print("  3_mastered            — full pipeline (current)")
    print("  3b_mastered_mossformer_only — MossFormer2 + mastering (no DeepFilterNet)")
    print("\nCompare 1 vs 2 to hear what MossFormer2 does.")
    print("Compare 2 vs 2b to hear if double-processing is the problem.")
    print("Compare 2 vs 3 to hear what mastering does.")
    print("Compare 0 vs 3b to hear the best single-model path.")


if __name__ == "__main__":
    main()
