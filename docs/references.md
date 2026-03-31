# Project Resonance — References Index

> Every model, repo, paper, and tool we've evaluated or referenced. Maintained for future development and launch documentation.

## Models — Evaluated & Used

| Model | Repo | Stars | Status in Our Pipeline | Notes |
|-------|------|-------|----------------------|-------|
| DeepFilterNet3 | [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) | 4,007 | ACTIVE — Stage 1 noise suppression | 1M params, real-time CPU, 48kHz native |
| FlashSR (ONNX) | [ysharma3501/FlashSR](https://github.com/ysharma3501/FlashSR) | 204 | ACTIVE — Stage 2 super-resolution | 500KB ONNX, 16kHz→48kHz, based on HierSpeech++ |
| FlashSR (Official) | [jakeoneijk/FlashSR_Inference](https://github.com/jakeoneijk/FlashSR_Inference) | — | REFERENCE | Full diffusion model, script-based, conda env |
| Resemble Enhance | [resemble-ai/resemble-enhance](https://github.com/resemble-ai/resemble-enhance) | 2,232 | REJECTED | CFM breaks on MPS, over-processes clean audio |
| Spotify Pedalboard | [spotify/pedalboard](https://github.com/spotify/pedalboard) | 6,042 | ACTIVE — Stage 3 DSP mastering | HPF, EQ, compression, de-essing, limiting |
| pyloudnorm | [csteinmetz1/pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) | — | ACTIVE — Stage 4 LUFS normalization | -16 LUFS targeting |

## Models — Candidates for Next Iteration

| Model | Repo | Stars | What It Does | Why It Matters |
|-------|------|-------|-------------|----------------|
| ClearerVoice-Studio | [modelscope/ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) | 4,007 | Enhancement + separation + super-resolution + speaker extraction | All-in-one, pip installable, 48kHz, numpy I/O |
| MossFormer2_SE_48K | [HuggingFace](https://huggingface.co/alibabasglab/MossFormer2_SE_48K) | — | Speech enhancement at 48kHz | Inside ClearerVoice, discriminative (no hallucination) |
| FINALLY | [SamsungLabs/FINALLY-page](https://github.com/SamsungLabs/FINALLY-page) | — | Any degraded speech → studio quality (NeurIPS 2024) | SOTA but Samsung hasn't released official weights |
| FINALLY (unofficial) | [inverse-ai/FINALLY-Speech-Enhancement](https://github.com/inverse-ai/FINALLY-Speech-Enhancement) | — | Community implementation of FINALLY | Needs conda, Python 3.10, script-based |
| VoiceFixer | [haoheliu/voicefixer](https://github.com/haoheliu/voicefixer) | — | Restoration + super-resolution in one model | Abandoned Nov 2023, broken on Python 3.13 |
| VoiceFixer2 | [Render-AI-Team/voicefixer2](https://github.com/Render-AI-Team/voicefixer2) | — | Updated fork with MPS support | Community maintained, updated deps |
| AudioSR | [haoheliu/versatile_audio_super_resolution](https://github.com/haoheliu/versatile_audio_super_resolution) | 1,793 | Audio upsampling to 48kHz (speech model) | Stale deps (numpy 1.23, librosa 0.9), won't install on Python 3.13 |
| AudioLBM | [AudioLBM project page](https://audiolbm.github.io/) | — | SOTA super-resolution, any→48kHz and any→192kHz (NeurIPS 2025) | No code released yet |

## Models — Noise Suppression Alternatives

| Model | Repo | Stars | Notes |
|-------|------|-------|-------|
| RNNoise VST | [werman/noise-suppression-for-voice](https://github.com/werman/noise-suppression-for-voice) | 6,451 | VST2/VST3 plugin, loadable via pedalboard |
| RNNoise (original) | [xiph/rnnoise](https://github.com/xiph/rnnoise) | 5,468 | C library, no Python API |
| Facebook Denoiser | [facebookresearch/denoiser](https://github.com/facebookresearch/denoiser) | 1,883 | Archived Oct 2023, real-time CPU |
| noisereduce | [timsainb/noisereduce](https://github.com/timsainb/noisereduce) | 1,828 | Classical spectral gating, not deep learning |
| SpeechBrain SepFormer | [HuggingFace](https://huggingface.co/speechbrain/sepformer-wham16k-enhancement) | 444 likes | 16kHz only |

## Tools — Audio Processing

| Tool | Repo | Stars | What It Does |
|------|------|-------|-------------|
| ffmpeg-normalize | [slhck/ffmpeg-normalize](https://github.com/slhck/ffmpeg-normalize) | 1,495 | CLI LUFS normalization, podcast preset |
| matchering | [sergree/matchering](https://github.com/sergree/matchering) | 2,460 | Match audio to reference track (EQ, loudness, stereo) |
| Intel OpenVINO Audacity | [intel/openvino-plugins-ai-audacity](https://github.com/intel/openvino-plugins-ai-audacity) | 1,916 | AI effects for Audacity: DeepFilterNet + AudioSR + Whisper + Demucs |

## Existing Pipelines (Closest to What We're Building)

| Repo | Stars | Pipeline | Gap |
|------|-------|----------|-----|
| [jcherranz/audio-restorer](https://github.com/jcherranz/audio-restorer) | 0 | DeepFilterNet → loudness norm → DNSMOS report | No super-resolution, no EQ/compression |
| [rioharper/VocalForge](https://github.com/rioharper/VocalForge) | 130 | Download → silence removal → speaker sep → DeepFilterNet → normalize | Training dataset tool, not podcast output |
| [chuck1z/AudioCleaner](https://github.com/chuck1z/AudioCleaner) | 25 | DeepFilterNet via Streamlit | Simple denoising only |
| [omeryusufyagci/fast-music-remover](https://github.com/omeryusufyagci/fast-music-remover) | 708 | C++ DeepFilterNet for YouTube videos | Music removal, not voice enhancement |

## Research Papers

| Paper | Venue | Year | Key Finding |
|-------|-------|------|-------------|
| FINALLY: Fast and Universal Speech Enhancement | NeurIPS | 2024 | GAN + WavLM → studio quality. SOTA but no official code |
| DeepFilterNet: Perceptually Motivated Real-Time SE | INTERSPEECH | 2023 | Spectral envelope + fine structure decomposition |
| FlashSR: One-step Audio Super-Resolution | arXiv | 2025 | 22x faster than AudioSR, single diffusion step |
| AudioLBM: Audio SR with Latent Bridge Models | NeurIPS | 2025 | First model doing any→192kHz. Current SOTA metrics |
| VoiceFixer: General Speech Restoration | INTERSPEECH | 2022 | Neural vocoder for joint restoration + SR |
| FlowSE: Efficient Speech Enhancement via Flow Matching | INTERSPEECH | 2025 | Mel-spectrogram domain flow matching |
| URGENT Challenge: Universal Speech Enhancement | INTERSPEECH | 2025 | 32 submissions benchmark. Generative preferred in subjective tests |

## What Makes Studio Audio Sound Different (5 Factors)

1. **Frequency range**: Studio condensers capture 20Hz-20kHz+. Phone MEMS mics roll off below ~100Hz and above ~8-12kHz.
2. **Proximity effect**: Directional studio mics boost bass at close range (80-250Hz warmth). Phone omni mics have zero proximity effect — physically impossible.
3. **Harmonic richness**: Large diaphragm captures subtle harmonics and transients. Small MEMS capsule cannot resolve them.
4. **Room acoustics**: Studios have absorption/diffusion. Phones capture reflections, HVAC, ambient.
5. **Signal chain**: XLR → preamp → 24-bit/96kHz vs MEMS → compressed AGC → 16-bit.

Source: r/audioengineering, Sound on Sound, DPA Microphones technical docs
