"""Diagnostic script to isolate where the noise is coming from."""
import logging
import sys

import torch
import torchaudio

logging.basicConfig(level=logging.INFO, format="%(message)s")

INPUT_PATH = "recording.m4a"
SR = 16000


def convert_to_wav(input_path: str, output_path: str) -> None:
    import subprocess
    subprocess.run(
        ["ffmpeg", "-i", input_path, "-ar", str(SR), "-ac", "1", output_path, "-y", "-loglevel", "error"],
        check=True,
    )


def test_1_denoise_only():
    """Test: denoise only (no CFM enhancement) — rules out the ODE solver."""
    from resemble_enhance.denoiser.inference import denoise

    print("\n=== TEST 1: Denoise only (no CFM) ===")
    convert_to_wav(INPUT_PATH, "/tmp/diag_input.wav")
    wav, sr = torchaudio.load("/tmp/diag_input.wav")
    wav = wav.mean(dim=0)

    # Take just 10 seconds
    chunk = wav[:sr * 10]
    device = "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu"
    print(f"Device: {device}")
    print(f"Input stats: min={chunk.min():.4f}, max={chunk.max():.4f}, mean={chunk.mean():.4f}")

    result, out_sr = denoise(chunk, sr, run_dir=None, device=device)
    result = result.detach().cpu()
    print(f"Output stats: min={result.min():.4f}, max={result.max():.4f}, mean={result.mean():.4f}")
    torchaudio.save("diag_test1_denoise_only.wav", result.unsqueeze(0), out_sr)
    print(f"Saved: diag_test1_denoise_only.wav ({result.shape[0]/out_sr:.2f}s at {out_sr}Hz)")


def test_2_enhance_cpu():
    """Test: full enhance on CPU — rules out MPS as the problem."""
    from resemble_enhance.enhancer.inference import enhance

    print("\n=== TEST 2: Full enhance on CPU (bypasses MPS) ===")
    convert_to_wav(INPUT_PATH, "/tmp/diag_input.wav")
    wav, sr = torchaudio.load("/tmp/diag_input.wav")
    wav = wav.mean(dim=0)

    chunk = wav[:sr * 10]
    print(f"Input stats: min={chunk.min():.4f}, max={chunk.max():.4f}, mean={chunk.mean():.4f}")

    result, out_sr = enhance(chunk, sr, device="cpu", nfe=64, solver="midpoint", lambd=0.5, tau=0.5)
    result = result.detach().cpu()
    print(f"Output stats: min={result.min():.4f}, max={result.max():.4f}, mean={result.mean():.4f}")
    torchaudio.save("diag_test2_enhance_cpu.wav", result.unsqueeze(0), out_sr)
    print(f"Saved: diag_test2_enhance_cpu.wav ({result.shape[0]/out_sr:.2f}s at {out_sr}Hz)")


def test_3_enhance_mps():
    """Test: full enhance on MPS — if CPU works but this doesn't, MPS is the problem."""
    from resemble_enhance.enhancer.inference import enhance

    if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
        print("\n=== TEST 3: SKIPPED (MPS not available) ===")
        return

    print("\n=== TEST 3: Full enhance on MPS ===")
    convert_to_wav(INPUT_PATH, "/tmp/diag_input.wav")
    wav, sr = torchaudio.load("/tmp/diag_input.wav")
    wav = wav.mean(dim=0)

    chunk = wav[:sr * 10]
    print(f"Input stats: min={chunk.min():.4f}, max={chunk.max():.4f}, mean={chunk.mean():.4f}")

    result, out_sr = enhance(chunk, sr, device="mps", nfe=64, solver="midpoint", lambd=0.5, tau=0.5)
    result = result.detach().cpu()
    print(f"Output stats: min={result.min():.4f}, max={result.max():.4f}, mean={result.mean():.4f}")
    torchaudio.save("diag_test3_enhance_mps.wav", result.unsqueeze(0), out_sr)
    print(f"Saved: diag_test3_enhance_mps.wav ({result.shape[0]/out_sr:.2f}s at {out_sr}Hz)")


if __name__ == "__main__":
    test = sys.argv[1] if len(sys.argv) > 1 else "all"

    if test in ("1", "all"):
        test_1_denoise_only()
    if test in ("2", "all"):
        test_2_enhance_cpu()
    if test in ("3", "all"):
        test_3_enhance_mps()

    print("\n=== DONE ===")
    print("Listen to each output file to isolate where noise is introduced.")
    print("If test2 (CPU) sounds good but test3 (MPS) is noise → MPS is the problem.")
    print("If test1 (denoise only) sounds good but test2 (full enhance) is noise → CFM solver is the problem.")
