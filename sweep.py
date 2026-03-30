"""Parameter sweep — generates multiple enhanced versions for A/B comparison."""
import logging
import subprocess
import sys

import torch
import torchaudio
from resemble_enhance.enhancer.inference import enhance

logging.basicConfig(level=logging.INFO, format="%(message)s")

INPUT_PATH = "recording.m4a"
SR = 16000
CHUNK_SECONDS = 15  # first 15 seconds only, for speed

# Parameter grid — each combo produces a different output
CONFIGS = [
    {"nfe": 32, "lambd": 0.1, "tau": 0.1, "label": "light_clean"},
    {"nfe": 32, "lambd": 0.5, "tau": 0.1, "label": "medium_clean"},
    {"nfe": 32, "lambd": 0.9, "tau": 0.1, "label": "heavy_denoise_low_temp"},
    {"nfe": 64, "lambd": 0.5, "tau": 0.5, "label": "original_settings"},
    {"nfe": 64, "lambd": 0.5, "tau": 0.1, "label": "original_low_temp"},
    {"nfe": 64, "lambd": 0.9, "tau": 0.1, "label": "heavy_denoise_more_steps"},
    {"nfe": 128, "lambd": 0.5, "tau": 0.1, "label": "max_steps_low_temp"},
]


def main() -> None:
    # Convert m4a to wav
    tmp_wav = "/tmp/sweep_input.wav"
    subprocess.run(
        ["ffmpeg", "-i", INPUT_PATH, "-ar", str(SR), "-ac", "1", tmp_wav, "-y", "-loglevel", "error"],
        check=True,
    )

    wav, sr = torchaudio.load(tmp_wav)
    wav = wav.mean(dim=0).flatten()

    # Take first N seconds only
    chunk = wav[: sr * CHUNK_SECONDS]
    print(f"Input: {chunk.shape[0]/sr:.1f}s at {sr}Hz")
    print(f"Device: cpu (MPS broken for CFM)")
    print(f"Running {len(CONFIGS)} configurations...\n")

    for i, cfg in enumerate(CONFIGS, 1):
        label = cfg["label"]
        outfile = f"sweep_{i}_{label}.wav"
        print(f"[{i}/{len(CONFIGS)}] nfe={cfg['nfe']}, lambd={cfg['lambd']}, tau={cfg['tau']} → {outfile}")

        try:
            result, out_sr = enhance(
                chunk,
                sr,
                device="cpu",
                nfe=cfg["nfe"],
                solver="midpoint",
                lambd=cfg["lambd"],
                tau=cfg["tau"],
                run_dir=None,
            )
            result = result.detach().cpu()
            torchaudio.save(outfile, result.unsqueeze(0), out_sr)
            print(f"    ✓ saved ({result.shape[0]/out_sr:.1f}s at {out_sr}Hz)")
        except Exception as exc:
            print(f"    ✗ failed: {exc}")

        # Flush MPS/memory between runs
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print("\n=== DONE ===")
    print("Listen to each file and tell me which sounds best.")
    print("Files are in the project root: sweep_1_*.wav through sweep_7_*.wav")


if __name__ == "__main__":
    main()
