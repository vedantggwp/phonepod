# phonepod

Local AI audio restoration. Phone recording → podcast quality.

**Zero cloud. Zero uploads. Everything runs on your machine.**

phonepod transforms noisy voice memos into broadcast-ready audio. It combines neural noise suppression (DeepFilterNet3 + MossFormer2) with a subtractive DSP mastering chain - all running locally on CPU. No cloud, no uploads, no subscription.

> Status: `0.1.0-beta.1` - works well, API may change. Feedback welcome.

## Before / After

> Audio demos coming soon — record on your phone, run `phonepod`, hear the difference.

<!-- TODO: Add audio player embeds once demo files are hosted -->

## Install

```bash
pip install phonepod
```

Requires Python 3.11+ and ffmpeg (`brew install ffmpeg` on macOS).

## Usage

### CLI (simplest)

```bash
phonepod recording.m4a podcast.wav
```

### Python API

```python
import phonepod

# One-liner: file in, file out
phonepod.enhance("recording.m4a", "podcast.wav")

# Advanced: tensor-level control
engine = phonepod.Engine()
enhanced_tensor, sample_rate = engine.enhance(audio_tensor, input_sr)
```

### Web UI

```bash
pip install phonepod[ui]
python -m phonepod.app
# Opens at http://localhost:7860
```

## What it does

| Stage | Model / Tool | What it does |
|-------|-------------|-------------|
| 1 | DeepFilterNet3 | Neural noise suppression - removes background noise |
| 2 | MossFormer2 (48kHz) | Speech enhancement - fills frequencies phones can't capture |
| 3 | Pedalboard DSP | Subtractive mastering - gate, HPF, EQ cuts (mud/box/nasal), 2x compression, de-ess |
| 4 | Pedalboard Reverb | Optional studio room ambience |
| 5 | pyloudnorm | Loudness normalization to -18 LUFS (podcast standard) |
| 6 | Limiter + ceiling | Prevents clipping at -1.5 dB ceiling |

**Subtractive philosophy**: all EQ moves are cuts, not boosts. Remove mud (200Hz), boxiness (500Hz), nasal honk (1500Hz), and harshness (6500Hz). The ML models already shaped the frequency balance - cuts work with them, boosts fight them.

Processing a 2-minute recording takes ~7 seconds on Apple Silicon.

## How it started

phonepod began as a personal problem: voice memos recorded on a phone sound terrible in a podcast. The AI models that exist are research demos, not products. Professional mastering chains exist but don't denoise. Nothing combines both into a single, local pipeline.

So I built it. The full build story — from first prototype to production pipeline, every dead end and breakthrough — is in [JOURNEY.md](JOURNEY.md).

## Architecture

```
Input (any format)
  -> ffmpeg -> 48kHz mono WAV
  -> Stage 1: DeepFilterNet3 (noise suppression)
  -> Stage 2: MossFormer2_SE_48K (speech enhancement)
  -> Stage 3: Pedalboard mastering (gate -> HPF -> mud/box/nasal cuts -> 2x compression -> de-ess)
  -> Stage 4: Reverb (subtle room ambience, optional)
  -> Stage 5: LUFS normalization (-18 LUFS)
  -> Stage 6: Limiter + hard ceiling (-1.5 dB)
Output: podcast-quality 48kHz WAV
```

Hard boundaries: the engine never touches the filesystem. The processor never touches the model. The CLI never touches tensors.

## Development

```bash
# Clone and setup
git clone https://github.com/vedantggwp/phonepod.git
cd phonepod
uv sync

# Run tests (fast unit tests only)
uv run pytest -m "not slow"

# Run full test suite (loads ML models, ~30s)
uv run pytest

# Run on a file
uv run phonepod recording.m4a output.wav
```

## License

MIT
