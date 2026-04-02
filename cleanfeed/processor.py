"""Audio file loader, processor, and saver.

Handles all file I/O. Loads audio, converts to mono, passes to engine,
saves output. The engine never touches the filesystem.
"""

import logging

import torch
import torchaudio

from .engine import Engine
from .profile import Profile

logger = logging.getLogger(__name__)

_ENGINE: Engine | None = None


def _get_engine(profile: str | None = None) -> Engine:
    global _ENGINE
    if _ENGINE is None:
        params = None
        if profile:
            p = Profile.load_by_name(profile)
            params = p.params
            logger.info("Using profile: %s", profile)
        _ENGINE = Engine(params=params)
    elif profile:
        p = Profile.load_by_name(profile)
        _ENGINE.set_params(p.params)
        logger.info("Switched to profile: %s", profile)
    return _ENGINE


def shutdown_engine() -> None:
    global _ENGINE
    if _ENGINE is not None:
        del _ENGINE
        _ENGINE = None
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        logger.info("Engine unloaded and MPS cache cleared.")


def process_audio(input_path: str, output_path: str, profile: str | None = None) -> None:
    """Load audio file, enhance it, and save the result.

    Args:
        input_path: Path to input audio file (WAV format).
        output_path: Path for enhanced output WAV file.
        profile: Optional profile name for mastering settings.
    """
    logger.info("Loading audio from %s", input_path)
    wav, sample_rate = torchaudio.load(input_path)

    mono = wav.mean(dim=0).flatten()

    if mono.numel() == 0:
        raise ValueError("Input audio is empty.")

    logger.info(
        "Processing %d samples at %d Hz (%.1fs)",
        mono.numel(),
        sample_rate,
        mono.numel() / sample_rate,
    )

    engine = _get_engine(profile=profile)
    enhanced, output_sr = engine.enhance(mono, sample_rate)

    if enhanced.ndim != 1:
        raise ValueError("Engine output must be a 1D mono waveform tensor.")

    enhanced = enhanced.detach().cpu()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    logger.info("Saving enhanced audio to %s", output_path)
    torchaudio.save(output_path, enhanced.unsqueeze(0), output_sr)
