"""Command-line interface for phonepod.

Usage: phonepod input.m4a output.wav
"""

import argparse
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile

import torchaudio

from .engine import OUTPUT_SR
from .processor import process_audio, shutdown_engine

SUPPORTED_EXTENSIONS = {".wav", ".m4a", ".mp3", ".mp4", ".flac", ".ogg", ".aac"}


def _cleanup() -> None:
    shutdown_engine()


def _handle_signal(signum: int, frame: object | None) -> None:
    _cleanup()
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="phonepod",
        description="phonepod — Local AI audio restoration. Phone recording → podcast quality.",
    )
    parser.add_argument(
        "input", help="path to the input audio file (wav, m4a, mp3, flac, ogg, aac)"
    )
    parser.add_argument("output", help="path for the enhanced output .wav file")
    parser.add_argument(
        "--profile", "-p",
        help="voice profile name (created via tuner UI)",
        default=None,
    )

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    args = None
    processing_input = None
    try:
        args = parser.parse_args()

        if not os.path.isfile(args.input):
            print(f"Error: input file does not exist: {args.input}", file=sys.stderr)
            sys.exit(1)

        extension = os.path.splitext(args.input)[1].lower()
        if extension not in SUPPORTED_EXTENSIONS:
            supported_formats = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            print(
                f"Error: unsupported input format. Supported formats: {supported_formats}",
                file=sys.stderr,
            )
            sys.exit(1)

        if extension != ".wav":
            if shutil.which("ffmpeg") is None:
                print(
                    "Error: ffmpeg is required to convert non-wav files. Install with: brew install ffmpeg",
                    file=sys.stderr,
                )
                sys.exit(1)

            temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_wav.close()
            subprocess.run(
                [
                    "ffmpeg", "-i", args.input,
                    "-ar", str(OUTPUT_SR), "-ac", "1",
                    temp_wav.name, "-y", "-loglevel", "error",
                ],
                check=True,
            )
            processing_input = temp_wav.name
            logging.info("Converted %s to wav for processing", extension)
        else:
            processing_input = args.input

        process_audio(processing_input, args.output, profile=args.profile)

        waveform, sample_rate = torchaudio.load(args.output)
        duration_seconds = waveform.shape[-1] / sample_rate
        print(f"Duration: {duration_seconds:.2f} seconds")
        print(f"Sample rate: {sample_rate} Hz")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        _cleanup()
        if (
            args is not None
            and processing_input is not None
            and processing_input != args.input
            and os.path.exists(processing_input)
        ):
            os.unlink(processing_input)


if __name__ == "__main__":
    main()
