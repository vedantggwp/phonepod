# Manifest

## Package (phonepod/)
- `phonepod/__init__.py` — Public API: enhance(), Engine, process_audio, shutdown_engine
- `phonepod/_compat.py` — Torchaudio backend compatibility shim
- `phonepod/engine.py` — 5-stage pipeline: DeepFilterNet → MossFormer2 → Pedalboard → LUFS → Limiter. Subtractive EQ philosophy (cuts only, no boosts). _apply_ceiling fix for peak clipping.
- `phonepod/processor.py` — Audio loader, mono conversion, engine passthrough, file save
- `phonepod/cli.py` — CLI interface with ffmpeg format conversion
- `phonepod/app.py` — Gradio web UI with A/B comparison player

## Legacy (flat files, superseded by phonepod/)
- `engine.py` — Old engine (now in phonepod/engine.py)
- `processor.py` — Old processor (now in phonepod/processor.py)
- `cli.py` — Old CLI (now in phonepod/cli.py)
- `app.py` — Old app (now in phonepod/app.py)
- `JOURNEY.md` — Full build log from idea to working pipeline (Phases 1-8)
- `CLAUDE.md` — Agent identity, architecture rules, execution protocol
- `docs/architecture.md` — System architecture and module definitions
- `docs/tasks.md` — Execution roadmap and phase status
- `docs/references.md` — 30+ models/repos/papers evaluated
- `docs/knowledge-base.md` — Model evaluations, API reference, parameter guide
- `setup.sh` — First-time setup script (uv dependencies)
- `recording.m4a` — Test input (voice memo)
- `podcast_v6_final.wav` — Current best output

## Test Scripts
- `test_full_pipeline.py` — Original v2 pipeline test (DeepFilterNet + Pedalboard, no MossFormer2)
- `test_deepfilter.py` — DeepFilterNet isolation test
- `test_studio_character.py` — A/B/C test: none vs softclip vs tube saturation
- `test_tube_sweep.py` — Parameter sweep for tube saturation (6 configs)
- `test_processor.py` — Original processor test (sine wave)
- `diagnose.py` — Layer-by-layer diagnostic (MPS vs CPU, denoise vs enhance)
- `sweep.py` — resemble-enhance 7-config parameter sweep
- `test_broadcast_ab.py` — A/B test: subtractive EQ (hybrid) vs additive EQ chain comparison

## Benchmark Scripts
- `benchmark_denoisers.py` — DPDFNet vs DeepFilterNet3 isolated comparison
- `benchmark_clearvoice_numpy.py` — ClearVoice file I/O vs numpy mode comparison
- `benchmark_pipeline.py` — Full pipeline A/B: DeepFilterNet3 vs DPDFNet-2 48kHz

## Tuner UI
- `tuner_minimal.py` — Voice tuner Gradio app with custom PhonepodTheme, semantic sliders, preset save/load
- `.impeccable.md` — Design context: users, brand personality, aesthetic direction, design principles
- `phonepod/profile.py` — MasteringParams dataclass, Profile save/load, params_from_semantic(). Subtractive EQ: mud/box/nasal cuts replace presence/air boosts.

## Planning & Documentation
- `TODOS.md` — Full task list: Sprints 1-5, dependency graph, research findings
- `docs/system-architecture.html` — Visual system architecture (open in browser)
- `docs/benchmarks.md` — Benchmark results and decision record (Sprint 1)

## New Files (2026-04-04)
- `phonepod/audit.py` - Pipeline audit tool: generates HTML report with per-stage spectrograms, metrics, pass/fail
- `diagnose_muffled.py` - Diagnostic script: isolates each pipeline stage into separate WAV files

## Recent Changes
- 2026-04-04: RESOLVED muffled audio blocker - root cause was mastering chain crushing dynamics (LUFS overshooting -18 by 3.6dB, crest halved). Fixed with iterative LUFS normalization, gentler compression defaults
- 2026-04-04: Added noise gate (Pedalboard NoiseGate, -50dB threshold) - silences artifacts between speech
- 2026-04-04: Added studio room reverb (Pedalboard Reverb, 3% wet default) - subtle early reflections
- 2026-04-04: Added Room slider to tuner UI + signal health metrics display
- 2026-04-04: Fixed LUFS convergence bug in enhance() found by Codex audit - was comparing against mutable target instead of original
- 2026-04-04: Softened noise gate from -40dB to -50dB, longer release (200ms) to preserve speech tails
- 2026-04-04: Added input clamping to lerp() in params_from_semantic()
- 2026-04-04: KEY FINDING - mastering DSP has minimal audible impact; ML models determine 95% of output character. Pipeline cleans well but cannot manufacture condenser mic qualities from phone recordings.
- 2026-04-04: Switched to subtractive EQ philosophy (cuts only, no boosts). Fixed peak clipping bug (_apply_ceiling). Hybrid chain validated via A/B test.
- 2026-04-03: Reskinned tuner UI - custom coral theme, Space Grotesk, fun microcopy, accordion layout, WCAG fixes
- 2026-04-03: Created `.impeccable.md` - design context for all future UI work
- 2026-04-01: Sprint 2 - restructured into `phonepod/` package, public API, pyproject.toml, 19 tests passing
- 2026-04-01: Sprint 1 complete - benchmarked DPDFNet (rejected), switched ClearVoice to numpy mode (3.2x faster)
- 2026-04-01: Product named `phonepod` - PyPI available, no trademark conflicts
