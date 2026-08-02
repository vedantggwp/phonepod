# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**phonepod** — a local, privacy-first audio restoration pipeline. Phone recording in, podcast-quality WAV out. Two neural models (noise suppression + speech enhancement) followed by a DSP mastering chain, all running on CPU on the user's machine.

Shipped as a pip-installable package (`pip install phonepod`) with a CLI, a Python API, and two Gradio UIs. Currently `0.1.0b2` / beta.

Earlier docs (`docs/architecture.md`, `docs/knowledge-base.md`, `JOURNEY.md`) call the project "Project Resonance" — that was the pre-rename working title. Same project.

## Non-negotiables

1. **No external APIs.** 100% local. Never send audio, telemetry, or file contents to any network service. This is the product's core promise, not a preference.
2. **`uv` only.** Never suggest `pip install` (for dev), `conda`, or `poetry` in project workflows. Runtime install docs say `pip install phonepod` because that's what end users run — that's the one exception.
3. **Respect module boundaries.** `engine.py` never touches the filesystem. `processor.py` never touches the model internals. `cli.py` / UIs never touch tensors. See "Architecture" below.
4. **Edit the package, not the legacy roots.** Changes go in `phonepod/`. The flat root-level `engine.py`, `processor.py`, `cli.py`, `app.py` are dead history (see "Repo layout").
5. **Don't re-litigate settled decisions.** The "Settled decisions" section lists approaches already tried and rejected with evidence. Read it before proposing a model swap or a chain change.

## Repo layout

### The package — `phonepod/` (this is the real code)

| File | Role |
|---|---|
| `__init__.py` | Public API: `enhance()`, `Engine`, `process_audio`, `shutdown_engine`, `OUTPUT_SR`, `MasteringParams`, `Profile`, `params_from_semantic`. Imports `_compat` **first** — order matters. |
| `_compat.py` | Shim recreating `torchaudio.backend.common.AudioMetaData`, which newer torchaudio removed but DeepFilterNet still imports. Must load before any model code. |
| `engine.py` | The 6-stage pipeline. Tensor/numpy in, tensor out. Zero file I/O. Also `master_only()` for fast DSP-only re-renders. |
| `processor.py` | All file I/O: load → mono → engine → save. Owns the module-level singleton engine (`_get_engine` / `shutdown_engine`). |
| `profile.py` | `MasteringParams` (frozen dataclass, every tunable DSP value), `Profile` (named JSON preset in `~/.phonepod/profiles/`), `params_from_semantic()` (0–100 sliders → raw params). |
| `cli.py` | `phonepod in.m4a out.wav [--profile NAME]`. ffmpeg conversion for non-WAV, signal handlers, cleanup. |
| `app.py` | Simple Gradio A/B player. Port 7860. Needs the `[ui]` extra. |
| `tuner.py` | Voice tuner UI: 5 semantic sliders + expert accordion of raw params. Port 7861. Tested by `tests/test_tuner.py`. |
| `audit.py` | `audit_pipeline(path)` → HTML report with per-stage spectrograms, LUFS/peak/crest metrics, band energy, pass/fail vs podcast targets. matplotlib is optional (spectrograms silently skipped if absent). |

### Root-level Python files — legacy and experiments, not the package

- `engine.py`, `processor.py`, `cli.py`, `app.py` — **superseded** by their `phonepod/` counterparts (root `engine.py` still defines `PodcastEngine`, the old class name). Left for history. Do not edit or import them; do not "fix" them to match the package.
- `tuner_minimal.py` — **not legacy.** The consumer-facing reskin of the tuner: custom Gradio theme (coral, Space Grotesk), 6 semantic sliders (adds **Room**), playful microcopy. Its theme class is still called `CleanfeedTheme` — a leftover from the pre-rename; `MANIFEST.md` calls it `PhonepodTheme`, which is wrong. Run directly (`uv run python tuner_minimal.py`, port 7861). It duplicates rather than imports `phonepod/tuner.py`, and the two have drifted — `phonepod/tuner.py` has the expert-parameter accordion and the tests; `tuner_minimal.py` has the theme and the Room slider. Know which one you're changing, and note the fork if you change shared behavior.
- `test_*.py`, `benchmark_*.py`, `sweep.py`, `diagnose*.py` — one-off research scripts from the build, **not** part of the pytest suite (`testpaths = ["tests"]`). They document experiments; they aren't maintained. Don't wire them into CI.

### Docs

| File | Contents |
|---|---|
| `JOURNEY.md` | Full build log, phases 1–8, version history v1→v6, 11 numbered learnings. Read before proposing architecture changes. |
| `docs/benchmarks.md` | Decision record: DeepFilterNet3 vs DPDFNet, ClearVoice file-I/O vs numpy mode. Numbers, not opinions. |
| `docs/knowledge-base.md` | Model evaluations, professional podcast chain reference, compatibility notes, parameter guide. |
| `docs/references.md` | 30+ models/repos/papers evaluated. |
| `docs/architecture.md` | System design. **Partly stale** — its stage-3 EQ listing still shows the old additive chain (presence/air boosts) that subtractive EQ replaced. |
| `docs/system-architecture.html` | Visual architecture diagram. |
| `TODOS.md` | Sprint plan 1–5, dependency graph, research findings. The live roadmap. |
| `MANIFEST.md` | File-by-file inventory + dated changelog. **Update this when adding files.** |
| `.impeccable.md` | Design context for any UI work: users, brand personality, aesthetic direction, anti-references, 5 design principles. Read before touching a UI. |

## Architecture

```
Input (.wav/.m4a/.mp3/.mp4/.flac/.ogg/.aac)
  → ffmpeg → 48kHz mono WAV                     (cli.py, non-WAV only)
  → torchaudio load, channel-mean to mono       (processor.py)
  → Stage 1: DeepFilterNet3 — noise suppression (engine.py)
  → Stage 2: MossFormer2_SE_48K — speech enhancement, numpy mode
  → Stage 3: Pedalboard mastering
       NoiseGate(-50dB) → HPF(80Hz)
       → mud cut(200Hz) → box cut(500Hz) → nasal cut(1500Hz)
       → Compressor(1.8:1) → Compressor(2.5:1)
       → de-ess/harshness cut(6500Hz)
  → Stage 4: Reverb — subtle room, skipped when reverb_wet == 0
  → Stage 5/6: iterative LUFS normalize (-18) → Limiter → hard ceiling (-1.5 dB)
Output: 48kHz mono WAV
```

`OUTPUT_SR = 48000` throughout. Input at any rate is resampled for DeepFilterNet; everything after stage 1 is 48kHz.

**The hard boundary.** `engine.py` never opens a file. `processor.py` never reaches into the model. `cli.py` and the UIs never handle tensors. This is the rule that keeps the engine testable and the package embeddable — enforce it in review.

**Engine is a singleton.** `processor._ENGINE` is module-global; the models load once per process (~seconds). `shutdown_engine()` frees it. The tuner keeps its own `_engine` and a `_denoised_cache` so slider moves re-run only stages 3–6 via `master_only()` — that's what makes tuning feel instant. Never make the tuner re-run ML inference on a slider move.

## Settled decisions

Each of these was tried, measured, and closed. Don't reopen without new evidence.

- **CPU, not MPS.** The engine explicitly `.cpu()`s its input. MPS is touched only for `empty_cache()` in `processor.py`. Iterative ODE solvers produced pure noise on MPS during Phase 3; single-pass models run fine and fast on CPU (~7s for 2 minutes on Apple Silicon). *Do not add MPS/CUDA device routing to the pipeline* — an older version of this file demanded it, and it was wrong.
- **Discriminative, not generative.** resemble-enhance (CFM) and FINALLY (Samsung, NeurIPS 2024) were both rejected: artifacts, no weights, and — the dealbreaker — generative models can shift accent and voice identity. Voice fidelity beats MOS score here.
- **Subtractive EQ only.** All EQ moves are **cuts**. Cut mud (200Hz), box (500Hz), nasal (1500Hz), harshness (6500Hz). The ML models already shaped the spectrum; boosts fight them and add harshness. A/B-validated in Sprint 2.5. Adding a presence or air boost is a regression.
- **No saturation, no proximity effect.** Softclip sounds sleepy; tube warmth reads as coloration even at conservative settings. Don't layer enhancement on enhancement.
- **No OLA chunking.** Both models segment internally. Chunking created boundary artifacts. Removed in v6 — don't reintroduce it.
- **ClearVoice numpy mode, wrapped in `torch.no_grad()`.** 3.2× faster than file-I/O mode with no OOM on a 116s clip. (`JOURNEY.md` learning #11 warns t2t mode OOMs — that predates the `no_grad` fix; numpy mode is the current, benchmarked answer.)
- **DeepFilterNet3 over DPDFNet.** 2.5× faster end-to-end, preferred in blind A/B, and DPDFNet's `librosa>=0.11` conflicts with ClearVoice's pin.
- **`_apply_ceiling()` is load-bearing.** Pedalboard's JUCE `Limiter` outputs at 0 dBFS regardless of `threshold_db`. The explicit `np.clip` after it is what actually enforces the ceiling. Removing it reintroduces clipping.
- **LUFS normalization iterates (3 attempts).** Limiting raises measured loudness, so the target is adjusted and retried until within 0.5 dB. Compare overshoot against the *original* desired value, not the mutated `target` — that exact bug was found and fixed once already.
- **Ceiling: the DSP chain is not where quality comes from.** The ML models determine ~95% of output character; A/B tests across mastering variants were near-identical. The pipeline cleans phone audio well but cannot manufacture condenser-mic qualities that were never captured. Set expectations accordingly rather than chasing it through DSP.

## Commands

```bash
# Setup (installs everything, including dev group)
uv sync

# Fast tests — no model loading, seconds
uv run pytest -m "not slow"

# Full suite — loads DeepFilterNet + MossFormer2, downloads weights on first run
uv run pytest

# Process a file
uv run phonepod recording.m4a output.wav
uv run phonepod recording.m4a output.wav --profile my-voice

# UIs
uv run python -m phonepod.app       # simple A/B player, :7860
uv run python -m phonepod.tuner     # tuner + expert params, :7861
uv run python tuner_minimal.py      # themed consumer tuner, :7861

# Pipeline audit → HTML report
uv run python -c "from phonepod.audit import audit_pipeline; print(audit_pipeline('recording.m4a'))"
```

`setup.sh` is **stale** — it still installs resemble-enhance and predates `pyproject.toml`'s dependency list. Use `uv sync`. Don't cite setup.sh in docs; fix or delete it if you touch that area.

ffmpeg is a hard external requirement for any non-WAV input (`brew install ffmpeg`).

## Testing

- Suite lives in `tests/` only (`testpaths = ["tests"]`). 
- **`slow` marker** = loads ML models. Everything touching `Engine` must be marked `@pytest.mark.slow`. Default local loop is `-m "not slow"`; run the full suite before shipping engine/processor/mastering/model changes.
- `tests/conftest.py` does two things at import time you should know about: it wraps `init_df` to disable DeepFilterNet's file logger (so tests run sandboxed), and it repoints `profile.PROFILES_DIR` at `.test-profiles/` so tests never write to `~/.phonepod/`. That directory is **not** in `.gitignore` — don't commit it.
- Fixtures: `test_wav_48k` synthesizes 5s of speech-like audio (150Hz fundamental + harmonics + noise). `recording_wav` converts `recording.m4a` if present and returns `None` otherwise — those tests self-skip, since the file is gitignored as a personal recording.
- Tests import `soundfile`, which is not declared in `pyproject.toml` — it arrives transitively via the model deps. Add it to the dev group if that ever breaks.
- **CI does not run the tests.** `.github/workflows/ci.yml` only does `compileall` + `python -m build` on ubuntu. Green CI means "it imports and packages," nothing about behavior. Test locally.

## Gotchas

- **Import order.** `_compat` must be imported before anything that pulls in DeepFilterNet. `phonepod/__init__.py` handles this; preserve it, and keep the `# noqa: F401`.
- **`numpy<2.0` is pinned.** Model deps still require NumPy 1.x. Don't relax it casually.
- **Version lives in three places and they currently disagree:** `pyproject.toml` (`0.1.0b2`), `phonepod/__init__.py` (`__version__ = "0.1.0"`, asserted by `tests/test_public_api.py`), and README (`0.1.0-beta.1`). Bumping means updating all of them plus the test.
- **`*.wav` is gitignored** (with a `demo/` exception), as is `recording.m4a`. Never commit generated audio, model caches, or personal recordings.
- **`Engine` docstring says 5 stages, the module docstring says 6.** The code runs 6 (reverb was added later). Cosmetic, but don't let it mislead you.
- **`master_only()` duplicates the LUFS loop from `enhance()`.** If you change loudness behavior, change both.

## Conventions

- Python ≥3.11. Modern type hints (`X | None`, builtin generics). Type-annotate public functions.
- Module-level `logger = logging.getLogger(__name__)`; `logger.info("msg %s", var)` lazy formatting, never f-strings in log calls. Stage completions are logged — keep that trace.
- Private helpers prefixed `_`. Tunable-parameter containers are frozen dataclasses.
- No linter or formatter is configured. Match surrounding style; don't reformat files you're not changing.
- Docstrings: one-line summary, then Args/Returns for public API. Comment *why*, not *what* — the DSP constants especially.
- Commits follow conventional-commit prefixes: `feat:`, `fix:`, `docs:`, `test:`, `chore:`.
- PRs use `.github/PULL_REQUEST_TEMPLATE.md`: Summary / Verification checkboxes / Notes.
- When adding a file, update `MANIFEST.md` (inventory + dated entry under Recent Changes). When finishing a task, tick it in `TODOS.md`.
