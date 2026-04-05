"""Integration tests for process_audio (file I/O layer)."""

import tempfile
from pathlib import Path

import pytest
import soundfile as sf

import phonepod
from phonepod.processor import process_audio, shutdown_engine


@pytest.mark.slow
class TestProcessor:
    def test_process_audio_creates_output(self, test_wav_48k):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name

        try:
            process_audio(str(test_wav_48k), out_path)
            assert Path(out_path).exists()
            assert Path(out_path).stat().st_size > 0
        finally:
            Path(out_path).unlink(missing_ok=True)
            shutdown_engine()

    def test_process_audio_output_is_48k_mono(self, test_wav_48k):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name

        try:
            process_audio(str(test_wav_48k), out_path)
            audio, sr = sf.read(out_path, dtype="float32")
            assert sr == phonepod.OUTPUT_SR
            assert audio.ndim == 1  # mono
        finally:
            Path(out_path).unlink(missing_ok=True)
            shutdown_engine()

    def test_process_audio_with_recording(self, recording_wav):
        if recording_wav is None:
            pytest.skip("recording.m4a not found")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name

        try:
            process_audio(str(recording_wav), out_path)
            audio, sr = sf.read(out_path, dtype="float32")
            assert sr == phonepod.OUTPUT_SR
            assert audio.ndim == 1
            assert len(audio) > 0
        finally:
            Path(out_path).unlink(missing_ok=True)
            shutdown_engine()
