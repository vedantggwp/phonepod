"""End-to-end tests for the tuner UI callbacks.

Tests the actual functions that Gradio calls when the user interacts with the UI.
These catch issues that unit tests miss — format handling, state management, file I/O.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# Generate test fixtures before importing tuner (which imports engine)
from cleanfeed import _compat as _compat  # noqa: F401


@pytest.fixture(scope="module")
def test_wav(tmp_path_factory) -> str:
    """Create a 3-second test WAV at 48kHz."""
    out = tmp_path_factory.mktemp("tuner") / "test.wav"
    sr = 48000
    t = np.linspace(0, 3.0, sr * 3, dtype=np.float32)
    signal = (0.3 * np.sin(2 * np.pi * 150 * t) + 0.05 * np.random.randn(len(t)).astype(np.float32))
    sf.write(str(out), signal, sr)
    return str(out)


@pytest.fixture(scope="module")
def test_m4a(tmp_path_factory, test_wav) -> str | None:
    """Convert test WAV to M4A if ffmpeg is available."""
    import shutil
    import subprocess
    if not shutil.which("ffmpeg"):
        return None
    out = tmp_path_factory.mktemp("tuner") / "test.m4a"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", test_wav, str(out)],
        check=True,
    )
    return str(out)


@pytest.mark.slow
class TestCleanAudio:
    """Tests for the clean_audio callback."""

    def test_clean_wav_returns_valid_audio(self, test_wav):
        from cleanfeed.tuner import clean_audio
        result = clean_audio(test_wav)

        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 3, f"Expected 3 outputs, got {len(result)}"

        # First output: path to cleaned audio
        audio_path = result[0]
        assert isinstance(audio_path, str), f"Expected str path, got {type(audio_path)}"
        assert os.path.exists(audio_path), f"Output file does not exist: {audio_path}"

        # Verify the audio file is valid
        audio, sr = sf.read(audio_path)
        assert sr == 48000, f"Expected 48kHz, got {sr}"
        assert len(audio) > 0, "Output audio is empty"
        assert np.isfinite(audio).all(), "Output contains NaN/Inf"

    def test_clean_m4a_returns_valid_audio(self, test_m4a):
        if test_m4a is None:
            pytest.skip("ffmpeg not available")

        from cleanfeed.tuner import clean_audio
        result = clean_audio(test_m4a)

        audio_path = result[0]
        assert os.path.exists(audio_path)
        audio, sr = sf.read(audio_path)
        assert sr == 48000
        assert len(audio) > 0

    def test_clean_none_raises(self):
        from cleanfeed.tuner import clean_audio
        import gradio as gr
        with pytest.raises(gr.Error):
            clean_audio(None)

    def test_clean_nonexistent_raises(self):
        from cleanfeed.tuner import clean_audio
        with pytest.raises(Exception):
            clean_audio("/nonexistent/file.wav")


@pytest.mark.slow
class TestPreviewSemantic:
    """Tests for the preview_semantic callback."""

    def test_preview_after_clean(self, test_wav):
        from cleanfeed.tuner import clean_audio, preview_semantic

        # Must clean first to populate _denoised_cache
        clean_audio(test_wav)

        result = preview_semantic(70, 60, 50, 40, 55)
        assert isinstance(result, tuple)
        assert len(result) == 2

        audio_path = result[0]
        assert isinstance(audio_path, str)
        assert os.path.exists(audio_path)

        audio, sr = sf.read(audio_path)
        assert sr == 48000
        assert len(audio) > 0

    def test_preview_different_params_produce_different_audio(self, test_wav):
        from cleanfeed.tuner import clean_audio, preview_semantic

        clean_audio(test_wav)

        result_a = preview_semantic(0, 50, 50, 50, 50)
        result_b = preview_semantic(100, 50, 50, 50, 50)

        audio_a, _ = sf.read(result_a[0])
        audio_b, _ = sf.read(result_b[0])

        # Different warmth values should produce different audio
        assert not np.array_equal(audio_a, audio_b), "Different params produced identical audio"

    def test_preview_without_clean_skips(self):
        from cleanfeed.tuner import preview_semantic, _denoised_cache
        import cleanfeed.tuner as tuner

        # Temporarily clear cache
        old_cache = tuner._denoised_cache
        tuner._denoised_cache = None
        try:
            import gradio as gr
            result = preview_semantic(50, 50, 50, 50, 50)
            # Should return gr.skip() values (not crash)
        finally:
            tuner._denoised_cache = old_cache


@pytest.mark.slow
class TestSavePreset:
    def test_save_creates_file(self, test_wav):
        from cleanfeed.tuner import clean_audio, save_preset
        from cleanfeed.profile import PROFILES_DIR

        clean_audio(test_wav)
        save_preset("tuner-test-save")

        assert (PROFILES_DIR / "tuner-test-save.json").exists()

        # Cleanup
        (PROFILES_DIR / "tuner-test-save.json").unlink(missing_ok=True)

    def test_save_empty_name_raises(self):
        import gradio as gr
        from cleanfeed.tuner import save_preset

        with pytest.raises(gr.Error):
            save_preset("")

        with pytest.raises(gr.Error):
            save_preset("   ")
