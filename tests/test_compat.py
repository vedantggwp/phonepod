"""Unit tests for the torchaudio compatibility shim."""

import sys


def test_torchaudio_backend_module_exists():
    """The shim must register torchaudio.backend in sys.modules."""
    import phonepod._compat  # noqa: F401
    assert "torchaudio.backend" in sys.modules
    assert "torchaudio.backend.common" in sys.modules


def test_audiometadata_class_accessible():
    """DeepFilterNet imports AudioMetaData from torchaudio.backend.common."""
    import phonepod._compat  # noqa: F401
    from torchaudio.backend.common import AudioMetaData

    meta = AudioMetaData(sample_rate=48000, num_frames=1000, num_channels=1)
    assert meta.sample_rate == 48000
    assert meta.num_frames == 1000
    assert meta.num_channels == 1


def test_shim_is_idempotent():
    """Importing _compat multiple times must not break anything."""
    import phonepod._compat  # noqa: F401
    import phonepod._compat  # noqa: F401, F811
    from torchaudio.backend.common import AudioMetaData
    assert AudioMetaData is not None
