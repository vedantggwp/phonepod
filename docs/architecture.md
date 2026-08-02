# System Architecture: phonepod

> Built under the working title "Project Resonance" — `JOURNEY.md` and
> `docs/knowledge-base.md` still use that name. Same project.

## Tech Stack
* **Python Manager:** `uv`
* **Noise Suppression:** DeepFilterNet3 (1M params, real-time on CPU)
* **Speech Enhancement:** MossFormer2_SE_48K via ClearerVoice-Studio (Alibaba)
* **DSP Mastering:** Spotify Pedalboard (gate, HPF, EQ, compression, de-essing, reverb, limiting)
* **Loudness:** pyloudnorm (LUFS measurement and normalization)
* **Audio I/O:** torchaudio + soundfile + ffmpeg
* **UI:** Gradio

Everything runs on **CPU**. The engine explicitly moves its input to CPU; MPS is
touched only to release cached memory in `processor.py`. This is deliberate —
iterative ODE solvers produced pure noise on Apple MPS during Phase 3, and the
single-pass models used here are fast enough on CPU (~7s for 2 minutes on Apple
Silicon). Do not add device routing to the pipeline.

## Pipeline (6 stages)

```
Input (.wav/.m4a/.mp3/.mp4/.flac/.ogg/.aac)
  → ffmpeg convert to 48kHz mono WAV        (CLI only, non-WAV input)
  → torchaudio load, channel-mean to mono
  → Stage 1: DeepFilterNet3 noise suppression
       (input resampled to the model's rate if needed)
  → Stage 2: MossFormer2_SE_48K speech enhancement
       (numpy mode, wrapped in torch.no_grad())
  → Stage 3: Pedalboard DSP mastering chain
      → Noise Gate (-50dB, 5ms attack / 200ms release)
      → High-Pass Filter (80Hz)
      → Mud cut (-2.5dB at 200Hz, Q 0.7)
      → Boxiness cut (-4dB at 500Hz, Q 1.0)
      → Nasal cut (-3dB at 1500Hz, Q 1.0)
      → Compressor 1 (1.8:1 at -16dB)
      → Compressor 2 (2.5:1 at -10dB)
      → De-ess / harshness cut (-3dB at 6500Hz, Q 1.5)
  → Stage 4: Studio room reverb (3% wet — skipped entirely when reverb_wet is 0)
  → Stage 5/6: LUFS normalization (-18) → Limiter → hard ceiling (-1.5dB)
Output (.wav, 48kHz mono)
```

All defaults above live in `MasteringParams` (`phonepod/profile.py`) and are
overridable per-run via a saved profile.

### Two properties of the chain worth knowing

**Subtractive EQ only.** Every EQ move is a cut. The ML models have already
shaped the spectrum; boosting on top of that adds harshness and artifacts, so
the chain removes problem frequencies (mud, box, nasal, harshness) instead of
boosting good ones. This replaced an earlier additive chain with presence and
air boosts after A/B testing in Sprint 2.5. Adding a boost back is a regression.

**Stages 5 and 6 are one loop, not two steps.** Limiting raises measured
loudness, so normalizing and then limiting overshoots the LUFS target. The
engine normalizes, limits, re-measures, and retries with an adjusted target up
to 3 times until it lands within 0.5 dB. Overshoot is always compared against
the *original* desired target, not the adjusted one. Separately, Pedalboard's
JUCE `Limiter` outputs at 0 dBFS regardless of `threshold_db`, so an explicit
`np.clip` (`_apply_ceiling`) after it is what actually enforces the ceiling.

## Module Definitions (Strict Separation)

The package lives in `phonepod/`. The flat root-level `engine.py`,
`processor.py`, `cli.py`, and `app.py` are superseded predecessors kept for
history — do not edit or import them.

### 1. `phonepod/engine.py` (The AI + DSP Core)
* **Purpose:** Wraps both ML models and the DSP mastering chain.
* **Input:** A 1D mono `torch.Tensor` and sample rate.
* **Process:** Runs the 6-stage pipeline. ClearVoice runs in numpy mode — no
  temp files, 3.2x faster than file-I/O mode, and no OOM when wrapped in
  `torch.no_grad()`. Both models segment internally.
* **Output:** A podcast-quality `torch.Tensor` at 48kHz (`OUTPUT_SR`).
* **Constraint:** Zero filesystem I/O, user-facing or internal.
* **Also provides:** `master_only()`, which runs stages 3-6 on already-denoised
  audio so the tuner can preview mastering changes without re-running ML
  inference. It duplicates the LUFS loop from `enhance()` — change both together.

### 2. `phonepod/processor.py` (The Audio Loader)
* **Purpose:** Loads audio, converts to mono, passes to engine, saves output.
* **Input:** File path to any supported audio format.
* **Process:** Loads via torchaudio, averages channels to mono, calls
  `engine.enhance()`, saves result.
* **Output:** Enhanced `.wav` file on disk.
* **Owns the engine singleton.** `_ENGINE` is module-global so the models load
  once per process; `shutdown_engine()` frees it.
* **Note:** OLA chunking was removed in v6 — both models handle segmentation
  internally, and chunking created boundary artifacts.

### 3. `phonepod/profile.py` (Tunable Parameters)
* **Purpose:** Every tunable DSP value, and named presets that persist them.
* **Contents:** `MasteringParams` (frozen dataclass of raw DSP values),
  `Profile` (named preset saved as JSON in `~/.phonepod/profiles/`), and
  `params_from_semantic()`, which maps 0-100 semantic sliders onto raw
  parameters with 50 as the default.

### 4. `phonepod/cli.py` (The CLI)
* **Purpose:** Command-line interface with format conversion and cleanup.
* **Process:** Validates input, converts non-WAV via ffmpeg to 48kHz mono, calls
  `processor.process_audio()`, reports duration and sample rate. Installs signal
  handlers so the engine is released and temp files removed on interrupt.

### 5. `phonepod/app.py` (The Web UI)
* **Purpose:** Drag-and-drop Gradio interface with A/B comparison player.
* **Process:** Accepts upload, passes to processor, displays original vs
  enhanced audio players. Port 7860. Requires the `[ui]` extra.

### 6. `phonepod/tuner.py` (The Voice Tuner UI)
* **Purpose:** Tune the mastering chain by ear and save the result as a profile.
* **Process:** Runs stages 1-2 once, caches the denoised audio, then re-renders
  only stages 3-6 through `master_only()` on every slider move. Exposes 5
  semantic sliders plus an accordion of raw parameters. Port 7861.
* **Constraint:** Slider moves must never re-run ML inference — the cache is what
  makes tuning feel instant.
* **See also:** `tuner_minimal.py` at the repo root, a consumer-facing reskin
  with a custom theme and a sixth "Room" slider. It duplicates rather than
  imports this module, and the two have drifted.

### 7. `phonepod/audit.py` (Pipeline Audit)
* **Purpose:** Proves what each stage does, rather than asserting it.
* **Process:** Re-runs the pipeline stage by stage, capturing intermediate
  audio, and emits an HTML report with per-stage spectrograms, LUFS / peak /
  RMS / crest metrics, band energy, deltas between stages, and pass/fail against
  podcast targets. matplotlib is optional — spectrograms are skipped if absent.

### 8. `phonepod/_compat.py` (Compatibility Shim)
* **Purpose:** Recreates `torchaudio.backend.common.AudioMetaData`, which newer
  torchaudio removed but DeepFilterNet still imports.
* **Constraint:** Must be imported before any model code. `phonepod/__init__.py`
  imports it first for exactly this reason — preserve that ordering.
