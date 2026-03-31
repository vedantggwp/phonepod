# Project Resonance — Knowledge Base

> Index of all research findings, model evaluations, and engineering decisions.

## Table of Contents

1. [Model Evaluations](#model-evaluations) — every model tested, verdict, API
2. [The Professional Podcast Chain](#the-professional-podcast-chain) — what real engineers do
3. [The Final Architecture](#the-final-architecture) — our 4-stage pipeline
4. [Compatibility Notes](#compatibility-notes) — version issues, workarounds
5. [Parameter Reference](#parameter-reference) — every tunable knob and what it does

---

## Model Evaluations

### DeepFilterNet3 (CHOSEN — noise suppression)
- **Repo:** github.com/Rikorose/DeepFilterNet (4,007 stars)
- **Size:** ~1M parameters
- **Speed:** Real-time on CPU, no GPU needed
- **Sample rate:** 48kHz native
- **What it does:** Spectral envelope + quasi-periodic fine structure decomposition. Removes background noise (fans, AC, keyboard, room ambience) while preserving voice character.
- **What it doesn't do:** Does NOT enhance, upsample, or change the voice. Only subtracts noise.
- **API:**
  ```python
  from df.enhance import init_df, enhance
  model, df_state, _ = init_df()  # loads DeepFilterNet3
  enhanced = enhance(model, df_state, audio_tensor)  # shape: (channels, samples)
  ```
- **Compatibility:** Requires torchaudio.backend monkey-patch for torchaudio 2.9+
- **Verdict:** Best-in-class noise suppression. Intel's Audacity plugin chose this. Confirmed working on our recording.

### AudioSR (NEXT — super-resolution)
- **Repo:** github.com/haoheliu/versatile_audio_super_resolution (1,793 stars)
- **What it does:** Upsamples audio from any input sample rate to 48kHz. Generates missing high-frequency content that phone mics can't capture. Has a dedicated `speech` model.
- **Why we need it:** Phone mics typically capture up to 8-12kHz. Podcast audio needs content up to 20kHz+ for that "studio" feel. AudioSR fills the gap.
- **CLI:** `audiosr -i file.wav --model_name speech`
- **API:**
  ```python
  import audiosr
  audiosr.super_resolution(input_file, output_file, model_name="speech")
  ```
- **Intel validation:** Intel's OpenVINO Audacity plugin uses AudioSR for its super-resolution effect — confirms the model works for voice.
- **Status:** NOT YET TESTED in our pipeline.

### Resemble-Enhance (EVALUATED — rejected for main pipeline)
- **Repo:** github.com/resemble-ai/resemble-enhance (2,232 stars)
- **Size:** 356M parameters (10M denoiser + 346M CFM enhancer)
- **Two stages:**
  - Denoiser (UNet): Works on MPS, moderate quality
  - Enhancer (CFM flow matching): BROKEN on MPS (pure noise), robotic on CPU
- **API:**
  ```python
  from resemble_enhance.enhancer.inference import enhance
  from resemble_enhance.denoiser.inference import denoise
  ```
- **Parameters:** nfe (1-128), lambd (0-1), tau (0-1), solver (midpoint/rk4/euler)
- **Verdict:** CFM stage over-processes clean audio. Denoiser alone is inferior to DeepFilterNet. Rejected.
- **Evidence:** 7-config parameter sweep, all had tearing artifacts. MPS test = pure noise.

### Spotify Pedalboard (CHOSEN — DSP mastering)
- **Repo:** github.com/spotify/pedalboard (6,042 stars)
- **What it does:** Production-grade audio effects. Compressor, EQ, Gain, Limiter, etc. Can also load VST3/AU plugins.
- **API:**
  ```python
  from pedalboard import Pedalboard, HighpassFilter, Compressor, Limiter, PeakFilter, HighShelfFilter
  board = Pedalboard([...effects...])
  output = board(audio_np, sample_rate=sr, reset=True)
  ```
- **Shape:** Expects numpy array (channels, samples). Returns same shape.
- **Apple Silicon:** Native arm64 wheel, works perfectly.

### pyloudnorm (CHOSEN — LUFS normalization)
- **Repo:** github.com/csteinmetz1/pyloudnorm
- **What it does:** Measures integrated loudness (LUFS) and normalizes to target.
- **API:**
  ```python
  import pyloudnorm as pyln
  meter = pyln.Meter(sample_rate)
  loudness = meter.integrated_loudness(audio)  # shape: (samples,) or (samples, channels)
  normalized = pyln.normalize.loudness(audio, loudness, target_lufs)
  ```
- **Shape warning:** Expects (samples, channels) — transposed from pedalboard's (channels, samples).
- **Podcast standard:** -16 LUFS for stereo, -19 LUFS for mono (Spotify/Apple standard).

### ClearVoice / MossFormer2 (EVALUATED — viable alternative)
- **Repo:** github.com/modelscope/ClearerVoice-Studio (4,007 stars)
- **API:** `pip install clearvoice`, supports 48kHz, numpy array I/O
- **Verdict:** Good alternative to DeepFilterNet. Not tested yet. Keep as backup.

### Silero-VAD (NOT an enhancer)
- **Repo:** github.com/snakers4/silero-vad
- **What it actually is:** Voice Activity Detector. Classifies speech vs silence. 1.8MB model.
- **Useful for:** Pre-processing (trimming silence), NOT enhancement.

---

## The Professional Podcast Chain

Source: r/podcasting, r/audioengineering, professional podcast editors.

**The canonical order (each step feeds the next):**

| Step | Effect | Settings | Purpose |
|------|--------|----------|---------|
| 1 | Noise Reduction | ML-based (DeepFilterNet) | Remove room noise, AC, fans |
| 2 | High-Pass Filter | 80Hz cutoff | Remove rumble, handling noise, AC hum |
| 3 | Subtractive EQ | -3dB at 200-400Hz, Q=1.0 | Cut "mud" that makes voice muffled |
| 4 | Compressor | 3:1 ratio, -20dB threshold, 15ms attack, 100ms release | Tame dynamic range |
| 5 | De-Esser | -4dB at 4-10kHz, narrow Q | Reduce sibilance (harsh "s" sounds) |
| 6 | Additive EQ | +2-3dB at 2-4kHz (presence), +2dB shelf at 8-12kHz (air) | Add clarity and brightness |
| 7 | Loudness Normalization | -16 LUFS (stereo) or -19 LUFS (mono) | Industry standard loudness |
| 8 | Brick-Wall Limiter | -1.5dB ceiling | Prevent clipping |

**Pro techniques:**
- Use 2-3 light compressors in series rather than 1 heavy compressor
- Order matters: EQ before compression ≠ compression before EQ
- Normalize LUFS on the FULL audio, not per-chunk
- The limiter goes LAST, always

---

## The Final Architecture

```
Phone Recording (any format: m4a, mp3, wav, etc.)
    │
    ▼
[0] ffmpeg — convert to 48kHz mono WAV
    │
    ▼
[1] DeepFilterNet3 — ML noise suppression (1M params, real-time CPU)
    │
    ▼
[2] AudioSR (speech model) — super-resolution to 48kHz, fills missing frequencies
    │
    ▼
[3] Pedalboard DSP chain:
    │   ├── High-Pass Filter (80Hz)
    │   ├── Subtractive EQ (-3dB at 300Hz)
    │   ├── Compressor 1 (2:1, -20dB threshold)
    │   ├── Compressor 2 (3:1, -15dB threshold)
    │   ├── De-Esser (-4dB at 6kHz)
    │   ├── Presence (+2.5dB at 3kHz)
    │   └── Air (+2dB shelf at 10kHz)
    │
    ▼
[4] LUFS Normalization — target -16 LUFS (or -19 for mono)
    │
    ▼
[5] Brick-Wall Limiter — -1.5dB ceiling
    │
    ▼
Podcast-Quality Output (48kHz WAV)
```

---

## Compatibility Notes

### torchaudio 2.9+ breaking change
- `torchaudio.backend` module removed entirely
- `torchaudio.info()` function removed
- DeepFilterNet imports `from torchaudio.backend.common import AudioMetaData` — crashes
- **Workaround:** Monkey-patch a stub module before importing df.enhance
- **Proper fix needed:** Create a compatibility shim module

### NumPy 2.0+ breaking change
- Implicit scalar conversion removed (only 0-d arrays → scalar, not 1-element arrays)
- resemble-enhance crashes with `only 0-dimensional arrays can be converted to Python scalars`
- **Fix:** Pin `numpy<2.0` (1.26.4 confirmed working)

### MPS + iterative solvers
- MPS float32 precision issues compound across sequential ODE/SDE steps
- Single forward-pass models (DeepFilterNet, UNet denoiser): work fine on MPS
- Multi-step flow matching (resemble-enhance CFM, nfe=64): pure noise on MPS
- **Rule:** Use MPS for single-pass inference, CPU for iterative/sequential computation

---

## Parameter Reference

### DeepFilterNet3
| Parameter | Default | Notes |
|-----------|---------|-------|
| atten_lim_db | 100 | Max attenuation in dB |
| min_db_thresh | -10 | Minimum threshold |
| post_filter | false | Additional spectral flooring |

### Pedalboard Effects
| Effect | Key Params | Our Settings |
|--------|-----------|--------------|
| HighpassFilter | cutoff_frequency_hz | 80 |
| PeakFilter (mud cut) | cutoff_frequency_hz, gain_db, q | 300, -3.0, 1.0 |
| Compressor 1 | threshold_db, ratio, attack_ms, release_ms | -20, 2.0, 15, 100 |
| Compressor 2 | threshold_db, ratio, attack_ms, release_ms | -15, 3.0, 10, 80 |
| PeakFilter (de-ess) | cutoff_frequency_hz, gain_db, q | 6000, -4.0, 2.0 |
| PeakFilter (presence) | cutoff_frequency_hz, gain_db, q | 3000, +2.5, 0.8 |
| HighShelfFilter (air) | cutoff_frequency_hz, gain_db, q | 10000, +2.0, 0.7 |
| Limiter | threshold_db | -1.5 |

### LUFS Targets
| Platform | Target | Notes |
|----------|--------|-------|
| Spotify | -14 LUFS | Loud |
| Apple Podcasts | -16 LUFS | Standard |
| YouTube | -14 LUFS | Matches Spotify |
| General podcast | -16 to -19 LUFS | -19 for mono is safer |
