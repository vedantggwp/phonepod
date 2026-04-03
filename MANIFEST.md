# Manifest

## Package (cleanfeed/)
- `cleanfeed/__init__.py` — Public API: enhance(), Engine, process_audio, shutdown_engine
- `cleanfeed/_compat.py` — Torchaudio backend compatibility shim
- `cleanfeed/engine.py` — 5-stage pipeline: DeepFilterNet → MossFormer2 → Pedalboard → LUFS → Limiter
- `cleanfeed/processor.py` — Audio loader, mono conversion, engine passthrough, file save
- `cleanfeed/cli.py` — CLI interface with ffmpeg format conversion
- `cleanfeed/app.py` — Gradio web UI with A/B comparison player

## Legacy (flat files, superseded by cleanfeed/)
- `engine.py` — Old engine (now in cleanfeed/engine.py)
- `processor.py` — Old processor (now in cleanfeed/processor.py)
- `cli.py` — Old CLI (now in cleanfeed/cli.py)
- `app.py` — Old app (now in cleanfeed/app.py)
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

## Benchmark Scripts
- `benchmark_denoisers.py` — DPDFNet vs DeepFilterNet3 isolated comparison
- `benchmark_clearvoice_numpy.py` — ClearVoice file I/O vs numpy mode comparison
- `benchmark_pipeline.py` — Full pipeline A/B: DeepFilterNet3 vs DPDFNet-2 48kHz

## Tuner UI
- `tuner_minimal.py` — Voice tuner Gradio app with custom CleanfeedTheme, semantic sliders, preset save/load
- `.impeccable.md` — Design context: users, brand personality, aesthetic direction, design principles
- `cleanfeed/profile.py` — MasteringParams dataclass, Profile save/load, params_from_semantic()

## Planning & Documentation
- `TODOS.md` — Full task list: Sprints 1-5, dependency graph, research findings
- `docs/system-architecture.html` — Visual system architecture (open in browser)
- `docs/benchmarks.md` — Benchmark results and decision record (Sprint 1)

## Recent Changes
- 2026-04-01: Product named `cleanfeed` — PyPI available, no trademark conflicts
- 2026-04-01: Created `TODOS.md` — 5-sprint roadmap from benchmark to real-time
- 2026-04-01: Created `docs/system-architecture.html` — full visual architecture
- 2026-04-01: Rewrote `engine.py` — FlashSR replaced with MossFormer2, LUFS -18, clean 5-stage pipeline
- 2026-04-01: Simplified `processor.py` — removed OLA chunking (115 → 55 lines)
- 2026-04-01: Fixed `cli.py` — ffmpeg converts to 48kHz (was 16kHz)
- 2026-04-01: Updated `docs/architecture.md` — reflects v6 pipeline
- 2026-04-01: Updated `docs/tasks.md` — Phase 8 complete
- 2026-04-01: Updated `JOURNEY.md` — Phase 8 with FINALLY research and saturation experiments
- 2026-04-01: Created `MANIFEST.md`
- 2026-04-01: Sprint 1 complete — benchmarked DPDFNet (rejected), switched ClearVoice to numpy mode (3.2x faster)
- 2026-04-01: Updated `engine.py` — removed temp file I/O, uses ClearVoice numpy mode with torch.no_grad()
- 2026-04-01: Created `docs/benchmarks.md` — full decision record with timing data
- 2026-04-01: Sprint 2 — restructured into `cleanfeed/` package, public API, pyproject.toml, 19 tests passing
- 2026-04-03: Reskinned tuner UI — custom coral theme, Space Grotesk, fun microcopy, accordion layout, WCAG fixes
- 2026-04-03: Created `.impeccable.md` — design context for all future UI work
- 2026-04-03: BLOCKER — output sounds muffled across ~10 test recordings, pipeline quality needs investigation
