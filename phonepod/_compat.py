"""Compatibility shim for torchaudio.backend.common.AudioMetaData.

Newer versions of torchaudio removed this module, but DeepFilterNet still
imports it. This shim must be imported before any DeepFilterNet code loads.
"""

import sys
import types

import torchaudio

if not hasattr(torchaudio, "backend"):
    _backend = types.ModuleType("torchaudio.backend")
    _common = types.ModuleType("torchaudio.backend.common")

    class AudioMetaData:
        def __init__(
            self,
            sample_rate: int = 0,
            num_frames: int = 0,
            num_channels: int = 0,
            bits_per_sample: int = 0,
            encoding: str = "",
        ) -> None:
            self.sample_rate = sample_rate
            self.num_frames = num_frames
            self.num_channels = num_channels
            self.bits_per_sample = bits_per_sample
            self.encoding = encoding

    _common.AudioMetaData = AudioMetaData
    _backend.common = _common
    sys.modules["torchaudio.backend"] = _backend
    sys.modules["torchaudio.backend.common"] = _common
    torchaudio.backend = _backend
