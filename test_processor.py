import math
import sys


INPUT_PATH = "test_input.wav"
OUTPUT_PATH = "test_output.wav"
SAMPLE_RATE = 16000
DURATION_SECONDS = 30
FREQUENCY_HZ = 440.0


def main() -> None:
    # Phase 1: fixture setup
    try:
        import torch
        import torchaudio
        time_axis = torch.arange(DURATION_SECONDS * SAMPLE_RATE, dtype=torch.float32) / SAMPLE_RATE
        waveform = 0.2 * torch.sin(2 * math.pi * FREQUENCY_HZ * time_axis)
        torchaudio.save(INPUT_PATH, waveform.unsqueeze(0), SAMPLE_RATE)
    except Exception as exc:
        print(f'Fixture setup failed. Run setup.sh first. Details: {exc}')
        sys.exit(1)

    # Phase 2: processing
    try:
        from processor import process_audio
        process_audio(INPUT_PATH, OUTPUT_PATH)
    except Exception as exc:
        print(f'Processing failed: {exc}')
        sys.exit(1)

    # Phase 3: verification
    output_wav, output_sr = torchaudio.load(OUTPUT_PATH)
    duration_seconds = output_wav.shape[-1] / output_sr
    print(f'Output duration: {duration_seconds:.2f}s')
    print(f'Output sample rate: {output_sr}')


if __name__ == "__main__":
    main()
