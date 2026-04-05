# Phonepod - Task List

> Shipping as `phonepod` on PyPI. Local, privacy-first audio restoration.

## Sprint 1: Benchmark + Validate (DONE)

- [x] **1.1** Benchmark DPDFNet vs DeepFilterNet3 on recording.m4a - DeepFilterNet3 wins (2.5x faster, preferred in A/B)
- [x] **1.2** Test ClearVoice Numpy2Numpy API - 3.2x faster, no OOM, switched to numpy mode
- [x] **1.3** Write `docs/benchmarks.md` - decision record for denoiser choice
- [x] **1.4** Update engine.py with winning stack - removed temp file I/O, added torch.no_grad()

## Sprint 2: Package as phonepod

- [x] **2.1** Restructure flat files -> `phonepod/` package (engine, processor, cli, app, _compat)
- [x] **2.2** Public API: `phonepod.enhance()`, `phonepod.Engine()`, `__init__.py`
- [x] **2.3** pyproject.toml - name, entry points, classifiers, dependency cleanup, gradio optional
- [x] **2.4** Lower Python floor to 3.11 - done in pyproject.toml
- [x] **2.5** pytest suite - 19 tests (3 unit, 11 integration, 5 E2E), all passing
- [x] **2.8** Pipeline audit + mastering fix - diagnosed muffled audio (mastering chain was crushing dynamics), fixed iterative LUFS normalization, added noise gate + studio reverb, Codex-audited for bugs
- [x] **2.9** Audit tool - `phonepod.audit.audit_pipeline()` generates HTML report with per-stage spectrograms and pass/fail metrics
- [ ] **2.6** GitHub repo `ved-labs/phonepod` + README with before/after audio demos
- [ ] **2.7** First PyPI release: `pip install phonepod` v0.1.0

## Sprint 2.5: Audacity-style Enhancement (DONE)

- [x] **2.5.1** Research + implement Audacity podcast workflow as alternative/additional processing path - implemented as hybrid (subtractive EQ from Audacity approach + gentle compression from current)
- [x] **2.5.2** A/B test Audacity-style chain vs current ML pipeline on diverse recordings - result: hybrid subtractive sounds cleaner and more natural
- [x] **2.5.3** Integrate winning techniques into default pipeline - subtractive EQ is now the default

## Sprint 3: HuggingFace Space + Feedback Loop

- [ ] **3.1** Deploy Gradio app to HuggingFace Spaces (`vedant/phonepod`)
- [ ] **3.2** Feedback widget - star rating, issue tags, 3 consent tiers, HuggingFaceDatasetSaver
- [ ] **3.3** Quality metrics - DNSMOS/SpeechScore, spectral analysis, SNR estimation
- [ ] **3.4** CLI opt-in feedback - voluntary post-processing prompt, offline fallback

## Sprint 4: Semantic Controls (after feedback data)

- [x] **4.3** Build semantic control layer - done early in tuner UI (Warmth, Clarity, Punch, De-ess, Volume, Room)
- [x] **4.5** Consumer-facing UI - tuner_minimal.py with custom PhonepodTheme, reskinned
- [ ] **4.1** Cluster feedback submissions by spectral profile
- [ ] **4.2** Derive presets from real-world clusters (not guesswork)
- [ ] **4.4** AI profile suggestion from input spectral analysis

## Sprint 5+: Real-Time On-Device (future)

- [ ] **5.1** DPDFNet TFLite for streaming denoise
- [ ] **5.2** Pedalboard real-time DSP chain
- [ ] **5.3** System audio filter integration (CoreAudio on Mac, PipeWire on Linux)
- [ ] **5.4** Latency optimization - target <50ms end-to-end

## Key Findings (2026-04-04)

The mastering DSP chain has minimal audible impact. The ML models (DeepFilterNet + MossFormer2) determine 95% of the output character. A/B testing 4 variants (no processing, default mastering, heavy reverb, no compression) produced nearly identical results. The pipeline cleans and normalizes well, but cannot manufacture condenser mic qualities that were never in the phone recording. This is a fundamental limitation, not a bug.

**Subtractive vs Additive EQ (Sprint 2.5):** On ML-processed audio, subtractive EQ (cutting mud at 200Hz, box at 500Hz, nasal at 1500Hz) works significantly better than additive boosts (presence/air shelves). ML models already optimize the spectral balance - boosting on top of that introduces harshness and artifacts. Cutting problem frequencies instead sounds cleaner and more natural. Subtractive EQ is now the default chain. Peak clipping bug also found and fixed: Pedalboard Limiter always outputs 0 dBFS, so _apply_ceiling was added to enforce the actual target ceiling.

## Dependency Graph

```
Sprint 1 (benchmark)
    │
    ▼
Sprint 2.1 (restructure) ──────────────────────────┐
    │                                               │
    ├──→ 2.2 (public API)                           │
    │       │                                       │
    ├──→ 2.3 (pyproject) ──┐                        │
    │                      │                        │
    ├──→ 2.4 (python ver) ─┤                        │
    │                      │                        │
    ├──→ 2.5 (tests) ──────┤     Sprint 3.3 (metrics)
    │                      │         │
    │                      ▼         │
    │                  2.6 (GitHub)  │
    │                      │         │
    │                      ▼         │
    │                  2.7 (PyPI) ───┤
    │                      │         │
    │                      ▼         ▼
    │                  3.1 (Space)  3.4 (CLI feedback)
    │                      │
    │                      ▼
    │                  3.2 (feedback widget)
    │
    └──→ 3.3 (metrics) — can start in parallel with 2.2-2.5
```

## Research Findings (2026-04-01)

| Finding | Impact | Action |
|---------|--------|--------|
| DPDFNet (ceva-ip/DPDFNet) — DeepFilterNet successor, ONNX+TFLite | Potential denoiser swap, cleaner API, enables real-time | Benchmark in Sprint 1 |
| ClearVoice Numpy2Numpy API (June 2025) | Eliminates temp file I/O in engine.py | Test in Sprint 1 |
| DNSMOS Pro + ClearVoice SpeechScore | Local quality metrics, no Azure | Add in Sprint 3.3 |
| No full-pipeline competitors exist | Market gap confirmed | Ship fast |
| `phonepod` available on PyPI + GitHub | Name is clear | Use it |
