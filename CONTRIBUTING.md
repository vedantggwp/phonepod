# Contributing to phonepod

Thanks for helping improve phonepod. The project is a local-first audio restoration pipeline for turning noisy phone recordings into podcast-quality WAV output without uploading audio.

## Good First Areas

- CLI error messages and format handling
- Documentation and examples
- Fast tests that do not load ML models
- Audio-profile tuning and mastering parameter coverage
- Packaging and install reliability

## Local Setup

```bash
git clone https://github.com/vedantggwp/phonepod.git
cd phonepod
uv sync
uv run pytest -m "not slow"
```

The full test suite loads ML models and may download or use cached model weights:

```bash
uv run pytest
```

## Pull Request Checklist

- Keep the change focused on one bug, feature, or doc improvement.
- Add or update tests when behavior changes.
- Run `uv run pytest -m "not slow"` for fast checks.
- Run the full test suite when changing model, engine, processor, or mastering behavior.
- Do not commit generated audio, private recordings, model caches, or local tuning artifacts.

## Reporting Issues

Please include:

- operating system;
- Python version;
- install method;
- input file format;
- command or API call used;
- expected behavior and actual behavior.

Do not attach private voice recordings unless you are comfortable publishing them publicly.
