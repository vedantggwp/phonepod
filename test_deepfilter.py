"""Quick test: can we use DeepFilterNet's core model without its broken I/O layer?"""
import torch
import torchaudio
import numpy as np

# Monkey-patch the missing torchaudio.backend module
import types
import torchaudio

if not hasattr(torchaudio, 'backend'):
    backend_module = types.ModuleType('torchaudio.backend')
    common_module = types.ModuleType('torchaudio.backend.common')

    class AudioMetaData:
        def __init__(self, sample_rate=0, num_frames=0, num_channels=0, bits_per_sample=0, encoding=""):
            self.sample_rate = sample_rate
            self.num_frames = num_frames
            self.num_channels = num_channels
            self.bits_per_sample = bits_per_sample
            self.encoding = encoding

    common_module.AudioMetaData = AudioMetaData
    backend_module.common = common_module
    import sys
    sys.modules['torchaudio.backend'] = backend_module
    sys.modules['torchaudio.backend.common'] = common_module
    torchaudio.backend = backend_module

# Now try importing DeepFilterNet
from df.enhance import init_df, enhance
from df.io import load_audio, save_audio

print("DeepFilterNet imported successfully!")

model, df_state, _ = init_df()
print(f"Model loaded. Sample rate: {df_state.sr()}")

# Load our test file using torchaudio directly (bypass df.io)
import subprocess
target_sr = df_state.sr()  # 48000
subprocess.run(
    ["ffmpeg", "-i", "recording.m4a", "-ar", str(target_sr), "-ac", "1", "/tmp/df_test_input.wav", "-y", "-loglevel", "error"],
    check=True,
)

wav, sr = torchaudio.load("/tmp/df_test_input.wav")
print(f"Audio loaded: shape={wav.shape}, sr={sr}")

# Take first 15 seconds
chunk = wav[:, :target_sr * 15]
print(f"Processing chunk: shape={chunk.shape}")

enhanced = enhance(model, df_state, chunk)
print(f"Enhanced: shape={enhanced.shape}")

torchaudio.save("deepfilter_test.wav", enhanced, target_sr)
print("Saved deepfilter_test.wav — listen to this!")
