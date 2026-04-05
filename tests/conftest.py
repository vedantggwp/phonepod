"""Shared fixtures for phonepod tests."""

import importlib
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

PROJECT_ROOT = Path(__file__).parent.parent
RECORDING = PROJECT_ROOT / "recording.m4a"
TEST_PROFILES_DIR = PROJECT_ROOT / ".test-profiles"


# Keep DeepFilterNet on the existing cached model, but disable its default file
# logger so engine-backed tests can run inside the sandbox.
_engine_module = importlib.import_module("phonepod.engine")
_original_init_df = _engine_module.init_df


def _init_df_without_file_logging(*args, **kwargs):
    kwargs.setdefault("log_file", None)
    return _original_init_df(*args, **kwargs)


_engine_module.init_df = _init_df_without_file_logging

# Redirect profile persistence to a writable project-local directory.
_profile_module = importlib.import_module("phonepod.profile")
TEST_PROFILES_DIR.mkdir(exist_ok=True)
_profile_module.PROFILES_DIR = TEST_PROFILES_DIR


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests that load ML models (deselect with '-m not slow')")


@pytest.fixture(scope="session")
def test_wav_48k(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a 5-second test WAV at 48kHz with speech-like content."""
    out = tmp_path_factory.mktemp("audio") / "test_48k.wav"
    sr = 48000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)

    # Simulate voiced speech: fundamental + harmonics + noise
    signal = (
        0.3 * np.sin(2 * np.pi * 150 * t)   # fundamental
        + 0.15 * np.sin(2 * np.pi * 300 * t)  # 1st harmonic
        + 0.08 * np.sin(2 * np.pi * 450 * t)  # 2nd harmonic
        + 0.05 * np.random.randn(len(t)).astype(np.float32)  # noise floor
    ).astype(np.float32)

    sf.write(str(out), signal, sr)
    return out


@pytest.fixture(scope="session")
def recording_wav(tmp_path_factory: pytest.TempPathFactory) -> Path | None:
    """Convert recording.m4a to WAV if it exists. Returns None if absent."""
    if not RECORDING.exists():
        return None
    out = tmp_path_factory.mktemp("audio") / "recording_48k.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(RECORDING), "-ar", "48000", "-ac", "1", str(out)],
        capture_output=True, check=True,
    )
    return out
