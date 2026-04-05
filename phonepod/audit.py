"""Pipeline audit - proves what each stage does with measurements and spectrograms.

Generates an HTML report showing before/after for every stage:
- Spectral energy by band
- LUFS, peak, crest factor
- Frequency response delta (what changed)
- Pass/fail against podcast quality targets

Usage:
    from phonepod.audit import audit_pipeline
    report = audit_pipeline("recording.m4a")
    # returns path to HTML report
"""

import logging
import base64
import io
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyloudnorm as pyln

from .engine import OUTPUT_SR

logger = logging.getLogger(__name__)

# Podcast quality targets
TARGETS = {
    "lufs": (-19.0, -16.0),       # acceptable range
    "peak_db": (-3.0, -0.5),      # should not clip, should not be too quiet
    "crest_db": (12.0, 30.0),     # dynamic range preserved
    "noise_floor_db": (-60.0, None),  # below -60dB is clean
}

BANDS = [
    ("Sub", 0, 100),
    ("Low", 100, 300),
    ("Low-Mid", 300, 1000),
    ("Mid", 1000, 3000),
    ("Presence", 3000, 6000),
    ("Brilliance", 6000, 12000),
    ("Air", 12000, 24000),
]


@dataclass(frozen=True)
class StageMetrics:
    name: str
    lufs: float
    peak_db: float
    rms_db: float
    crest_db: float
    band_energy: dict[str, float]  # band name -> dB relative to total


def measure(audio: np.ndarray, name: str, sr: int = OUTPUT_SR) -> StageMetrics:
    """Measure audio quality metrics."""
    audio_64 = audio.astype(np.float64)
    meter = pyln.Meter(sr)
    lufs = meter.integrated_loudness(audio_64)
    peak = 20 * np.log10(np.abs(audio_64).max() + 1e-10)
    rms = 20 * np.log10(np.sqrt(np.mean(audio_64 ** 2)) + 1e-10)
    crest = peak - rms

    fft = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)
    power = np.abs(fft) ** 2
    total = power.sum() + 1e-10

    band_energy = {}
    for band_name, lo, hi in BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        band_db = 10 * np.log10(power[mask].sum() / total + 1e-10)
        band_energy[band_name] = round(band_db, 1)

    return StageMetrics(
        name=name,
        lufs=round(lufs, 1) if np.isfinite(lufs) else -99.0,
        peak_db=round(peak, 1),
        rms_db=round(rms, 1),
        crest_db=round(crest, 1),
        band_energy=band_energy,
    )


def _check(value: float, target_range: tuple) -> str:
    lo, hi = target_range
    if lo is not None and value < lo:
        return "fail-low"
    if hi is not None and value > hi:
        return "fail-high"
    return "pass"


def _make_spectrogram_png(audio: np.ndarray, sr: int, title: str) -> str:
    """Generate a spectrogram as base64 PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 2.5), dpi=100)
        fig.patch.set_facecolor("#18181b")
        ax.set_facecolor("#18181b")

        # Use a 30s window max for readability
        max_samples = sr * 30
        segment = audio[:max_samples] if len(audio) > max_samples else audio

        ax.specgram(segment, NFFT=2048, Fs=sr, noverlap=1024,
                    cmap="magma", vmin=-80, vmax=0)
        ax.set_ylabel("Hz", color="#a1a1aa", fontsize=9)
        ax.set_xlabel("Time (s)", color="#a1a1aa", fontsize=9)
        ax.set_title(title, color="#fafafa", fontsize=10, pad=4)
        ax.tick_params(colors="#71717a", labelsize=8)
        ax.set_ylim(0, 16000)
        for spine in ax.spines.values():
            spine.set_color("#3f3f46")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight",
                    facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except ImportError:
        return ""


def generate_report(stages: list[tuple[str, np.ndarray]], sr: int = OUTPUT_SR) -> str:
    """Generate an HTML audit report from a list of (name, audio_array) pairs.

    Args:
        stages: list of (stage_name, audio_numpy_array) tuples in pipeline order.
        sr: sample rate.

    Returns:
        Path to generated HTML report.
    """
    metrics = [measure(audio, name, sr) for name, audio in stages]
    spectrograms = [_make_spectrogram_png(audio, sr, name) for name, audio in stages]

    final = metrics[-1]
    checks = {
        "LUFS": _check(final.lufs, TARGETS["lufs"]),
        "Peak": _check(final.peak_db, TARGETS["peak_db"]),
        "Crest (dynamics)": _check(final.crest_db, TARGETS["crest_db"]),
    }

    # Build HTML
    rows_html = ""
    for i, m in enumerate(metrics):
        spec_img = ""
        if spectrograms[i]:
            spec_img = f'<img src="data:image/png;base64,{spectrograms[i]}" style="width:100%;border-radius:6px;margin-top:8px;">'

        band_bars = ""
        for band_name, _ , _ in BANDS:
            val = m.band_energy.get(band_name, -40)
            width = max(0, min(100, (val + 40) * 2.5))
            color = "#f94839" if band_name in ("Presence", "Brilliance", "Air") else "#71717a"
            band_bars += f'''
                <div style="display:flex;align-items:center;gap:8px;margin:2px 0;">
                    <span style="width:80px;font-size:11px;color:#a1a1aa;text-align:right;">{band_name}</span>
                    <div style="flex:1;height:12px;background:#27272a;border-radius:3px;overflow:hidden;">
                        <div style="width:{width}%;height:100%;background:{color};border-radius:3px;"></div>
                    </div>
                    <span style="width:45px;font-size:11px;color:#a1a1aa;">{val} dB</span>
                </div>'''

        delta_html = ""
        if i > 0:
            prev = metrics[i - 1]
            d_lufs = m.lufs - prev.lufs
            d_crest = m.crest_db - prev.crest_db
            d_peak = m.peak_db - prev.peak_db
            delta_html = f'''
                <div style="margin-top:8px;padding:8px;background:#1f1f23;border-radius:6px;font-size:12px;">
                    <strong style="color:#a1a1aa;">Delta from {prev.name}:</strong>
                    <span style="color:{"#4ade80" if abs(d_lufs) < 2 else "#f94839"};">LUFS {d_lufs:+.1f}</span> |
                    <span style="color:{"#4ade80" if d_crest > -3 else "#f94839"};">Crest {d_crest:+.1f}</span> |
                    <span style="color:{"#4ade80" if d_peak < 1 else "#f94839"};">Peak {d_peak:+.1f}</span>
                </div>'''

        rows_html += f'''
        <div style="background:#27272a;border:1px solid #3f3f46;border-radius:8px;padding:16px;margin-bottom:12px;">
            <h3 style="margin:0 0 8px 0;color:#fafafa;font-size:15px;">{m.name}</h3>
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:8px;">
                <div style="text-align:center;">
                    <div style="font-size:20px;font-weight:700;color:#fafafa;">{m.lufs}</div>
                    <div style="font-size:11px;color:#a1a1aa;">LUFS</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:20px;font-weight:700;color:#fafafa;">{m.peak_db}</div>
                    <div style="font-size:11px;color:#a1a1aa;">Peak dB</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:20px;font-weight:700;color:#fafafa;">{m.rms_db}</div>
                    <div style="font-size:11px;color:#a1a1aa;">RMS dB</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:20px;font-weight:700;color:#fafafa;">{m.crest_db}</div>
                    <div style="font-size:11px;color:#a1a1aa;">Crest dB</div>
                </div>
            </div>
            {band_bars}
            {delta_html}
            {spec_img}
        </div>'''

    # Pass/fail summary
    verdict_html = ""
    all_pass = True
    for check_name, result in checks.items():
        icon = "PASS" if result == "pass" else "FAIL"
        color = "#4ade80" if result == "pass" else "#f94839"
        if result != "pass":
            all_pass = False
        verdict_html += f'<span style="color:{color};font-weight:600;margin-right:16px;">{icon} {check_name}</span>'

    overall_color = "#4ade80" if all_pass else "#f94839"
    overall_text = "Pipeline output meets podcast quality targets" if all_pass else "Pipeline output needs adjustment"

    html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>phonepod audit</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: "Space Grotesk", system-ui, sans-serif; background: #18181b; color: #fafafa; padding: 24px; max-width: 800px; margin: 0 auto; }}
    h1 {{ font-size: 1.6rem; font-weight: 700; color: #f94839; margin-bottom: 4px; }}
    h2 {{ font-size: 1.1rem; font-weight: 500; color: #a1a1aa; margin-bottom: 20px; }}
</style>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
</head><body>
    <h1>phonepod audit report</h1>
    <h2>Stage-by-stage pipeline analysis</h2>

    <div style="background:{overall_color}22;border:1px solid {overall_color};border-radius:8px;padding:12px 16px;margin-bottom:20px;">
        <div style="font-weight:600;color:{overall_color};margin-bottom:4px;">{overall_text}</div>
        <div>{verdict_html}</div>
    </div>

    {rows_html}

    <div style="text-align:center;color:#52525b;font-size:12px;margin-top:24px;">
        phonepod pipeline audit | targets: LUFS {TARGETS["lufs"]}, Peak {TARGETS["peak_db"]}, Crest {TARGETS["crest_db"]}
    </div>
</body></html>'''

    out = tempfile.NamedTemporaryFile(suffix=".html", delete=False, prefix="phonepod_audit_")
    out.write(html.encode())
    out.close()
    logger.info("Audit report: %s", out.name)
    return out.name


def audit_pipeline(input_path: str) -> str:
    """Run the full pipeline with audit, saving intermediate results.

    Returns path to HTML audit report.
    """
    import torch
    import torchaudio
    from .engine import Engine

    wav, sr = torchaudio.load(input_path)
    mono = wav.mean(dim=0).flatten().to(torch.float32)

    if sr != OUTPUT_SR:
        mono_48k = torchaudio.functional.resample(mono, sr, OUTPUT_SR)
    else:
        mono_48k = mono

    stages = [("0. Input (48kHz)", mono_48k.numpy())]

    engine = Engine()

    # Stage 1: DeepFilterNet
    from df.enhance import enhance as df_enhance
    audio_for_df = mono.clone()
    if sr != engine._df_sr:
        audio_for_df = torchaudio.functional.resample(audio_for_df, sr, engine._df_sr)
    denoised = df_enhance(engine._df_model, engine._df_state, audio_for_df.unsqueeze(0))
    denoised_1d = denoised.squeeze(0).detach().cpu()
    if engine._df_sr != OUTPUT_SR:
        denoised_48k = torchaudio.functional.resample(denoised_1d, engine._df_sr, OUTPUT_SR)
    else:
        denoised_48k = denoised_1d
    stages.append(("1. DeepFilterNet (denoise)", denoised_48k.numpy()))

    # Stage 2: MossFormer2
    cv_input = denoised_48k.numpy().astype(np.float32)[np.newaxis, :]
    with torch.no_grad():
        cv_result = engine._clearvoice(cv_input)
    if isinstance(cv_result, dict):
        enhanced_np = list(cv_result.values())[0]
    elif isinstance(cv_result, torch.Tensor):
        enhanced_np = cv_result.numpy()
    else:
        enhanced_np = np.asarray(cv_result, dtype=np.float32)
    enhanced_flat = np.ascontiguousarray(enhanced_np.flatten().astype(np.float32))
    stages.append(("2. MossFormer2 (enhance)", enhanced_flat))

    # Stage 3: Mastering DSP
    buf = enhanced_flat[np.newaxis, :]
    mastered = engine._mastering(buf, sample_rate=OUTPUT_SR, reset=True)
    stages.append(("3. Mastering (EQ + compression)", mastered[0]))

    # Stage 4: Reverb
    if engine._params.reverb_wet > 0:
        reverbed = engine._reverb(mastered, sample_rate=OUTPUT_SR, reset=True)
        stages.append(("4. Room reverb", reverbed[0]))
    else:
        reverbed = mastered

    # Stage 5: LUFS
    mono_64 = np.ascontiguousarray(reverbed[0].astype(np.float64))
    meter = pyln.Meter(OUTPUT_SR)
    loudness = meter.integrated_loudness(mono_64)
    if np.isfinite(loudness):
        normalized = pyln.normalize.loudness(mono_64, loudness, engine._params.lufs_target).astype(np.float32)
    else:
        normalized = mono_64.astype(np.float32)
    stages.append(("5. LUFS normalization", normalized))

    # Stage 6: Limiter
    limited = engine._limiter(normalized[np.newaxis, :], sample_rate=OUTPUT_SR, reset=True)
    stages.append(("6. Final output (limited)", limited[0]))

    return generate_report(stages)
