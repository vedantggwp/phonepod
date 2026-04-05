"""E2E tests for the public API and CLI."""

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import soundfile as sf

import phonepod


class TestPublicAPI:
    def test_version_exists(self):
        assert hasattr(phonepod, "__version__")
        assert phonepod.__version__ == "0.1.0"

    def test_exports(self):
        assert callable(phonepod.enhance)
        assert callable(phonepod.Engine)
        assert callable(phonepod.process_audio)
        assert callable(phonepod.shutdown_engine)
        assert phonepod.OUTPUT_SR == 48000

    @pytest.mark.slow
    def test_enhance_function(self, test_wav_48k):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name

        try:
            result = phonepod.enhance(str(test_wav_48k), out_path)
            assert result == out_path
            assert Path(out_path).exists()

            audio, sr = sf.read(out_path, dtype="float32")
            assert sr == 48000
            assert len(audio) > 0
        finally:
            Path(out_path).unlink(missing_ok=True)
            phonepod.shutdown_engine()


class TestCLI:
    def test_help_flag(self):
        result = subprocess.run(
            [sys.executable, "-m", "phonepod.cli", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "phonepod" in result.stdout
        assert "input" in result.stdout

    def test_missing_input_file(self):
        result = subprocess.run(
            [sys.executable, "-m", "phonepod.cli", "nonexistent.wav", "out.wav"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "does not exist" in result.stderr

    def test_unsupported_format(self, tmp_path):
        fake = tmp_path / "test.xyz"
        fake.write_text("not audio")

        result = subprocess.run(
            [sys.executable, "-m", "phonepod.cli", str(fake), "out.wav"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "unsupported" in result.stderr.lower()

    @pytest.mark.slow
    def test_cli_e2e(self, test_wav_48k, monkeypatch, capsys):
        from phonepod import cli

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name

        try:
            monkeypatch.setattr(sys, "argv", ["phonepod", str(test_wav_48k), out_path])
            cli.main()

            output = capsys.readouterr()
            assert Path(out_path).exists()
            assert "Duration:" in output.out
            assert "Sample rate:" in output.out
            assert output.err == ""
        finally:
            Path(out_path).unlink(missing_ok=True)
