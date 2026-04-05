"""phonepod — Local AI audio restoration. Phone recording → podcast quality.

Usage:
    # Simple: file in, file out
    import phonepod
    phonepod.enhance("recording.m4a", "podcast.wav")

    # Advanced: tensor-level control
    from phonepod import Engine
    engine = Engine()
    enhanced, sr = engine.enhance(audio_tensor, sample_rate)
"""

# Torchaudio compatibility shim — MUST be imported before any model code.
from . import _compat as _compat  # noqa: F401

from .engine import Engine, OUTPUT_SR
from .processor import process_audio, shutdown_engine
from .profile import MasteringParams, Profile, params_from_semantic

__version__ = "0.1.0"
__all__ = [
    "Engine", "enhance", "process_audio", "shutdown_engine",
    "OUTPUT_SR", "MasteringParams", "Profile", "params_from_semantic",
]


def enhance(input_path: str, output_path: str, profile: str | None = None) -> str:
    """Enhance an audio file. The simplest way to use phonepod.

    Args:
        input_path: Path to input audio file (WAV, M4A, MP3, FLAC, OGG, AAC).
        output_path: Path for enhanced output WAV file.
        profile: Optional profile name (saved via tuner UI).

    Returns:
        The output_path, for chaining.
    """
    process_audio(input_path, output_path, profile=profile)
    return output_path
