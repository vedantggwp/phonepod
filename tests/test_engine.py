"""Integration tests for the Engine class. Loads real ML models."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

import phonepod

ENGINE_OUTPUT_SR = phonepod.OUTPUT_SR


@pytest.fixture(scope="module")
def engine():
    """Shared engine instance for the module (models are expensive to load)."""
    eng = phonepod.Engine()
    yield eng
    del eng
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


@pytest.mark.slow
class TestEngine:
    def test_enhance_returns_tensor_and_sr(self, engine, test_wav_48k):
        audio, sr = sf.read(str(test_wav_48k), dtype="float32")
        tensor = torch.from_numpy(audio)

        result, output_sr = engine.enhance(tensor, sr)

        assert isinstance(result, torch.Tensor)
        assert result.ndim == 1
        assert output_sr == ENGINE_OUTPUT_SR

    def test_enhance_output_is_finite(self, engine, test_wav_48k):
        audio, sr = sf.read(str(test_wav_48k), dtype="float32")
        tensor = torch.from_numpy(audio)

        result, _ = engine.enhance(tensor, sr)

        assert torch.isfinite(result).all(), "Output contains NaN or Inf"

    def test_enhance_output_not_silent(self, engine, test_wav_48k):
        audio, sr = sf.read(str(test_wav_48k), dtype="float32")
        tensor = torch.from_numpy(audio)

        result, _ = engine.enhance(tensor, sr)
        rms = float(torch.sqrt(torch.mean(result**2)))

        assert rms > 0.001, f"Output appears silent (RMS={rms})"

    def test_enhance_output_not_clipped(self, engine, test_wav_48k):
        audio, sr = sf.read(str(test_wav_48k), dtype="float32")
        tensor = torch.from_numpy(audio)

        result, _ = engine.enhance(tensor, sr)
        peak = float(torch.max(torch.abs(result)))

        assert peak <= 1.0, f"Output is clipped (peak={peak})"

    def test_enhance_rejects_stereo(self, engine):
        stereo = torch.randn(2, 48000)
        with pytest.raises(ValueError, match="1D mono"):
            engine.enhance(stereo, 48000)

    def test_enhance_with_recording(self, engine, recording_wav):
        if recording_wav is None:
            pytest.skip("recording.m4a not found")

        audio, sr = sf.read(str(recording_wav), dtype="float32")
        tensor = torch.from_numpy(audio)

        result, output_sr = engine.enhance(tensor, sr)

        assert result.ndim == 1
        assert output_sr == ENGINE_OUTPUT_SR
        assert torch.isfinite(result).all()
        assert float(torch.max(torch.abs(result))) <= 1.0
