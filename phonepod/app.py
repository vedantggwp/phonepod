"""Gradio web UI for phonepod.

Drag-and-drop audio restoration with A/B comparison player.
"""

import atexit
import logging
import os
import signal
import tempfile

import gradio as gr

from .processor import process_audio, shutdown_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_TEMP_FILES: list[str] = []


def _cleanup_temp_files() -> None:
    for path in _TEMP_FILES:
        if os.path.exists(path):
            os.unlink(path)


atexit.register(_cleanup_temp_files)


def _shutdown(signum: int, frame: object) -> None:
    logger.info("Received signal %s, shutting down...", signal.Signals(signum).name)
    _cleanup_temp_files()
    try:
        shutdown_engine()
    except Exception:
        pass
    logger.info("Cleanup complete. Exiting.")
    raise SystemExit(0)


signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


def set_processing_status() -> str:
    return "Processing audio..."


def restore_audio(input_path: str | None) -> tuple[str | None, str | None, str]:
    if not input_path:
        gr.Warning("Upload an audio file before starting restoration.")
        return None, None, "Waiting for audio."

    output_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            output_path = temp_file.name

        logger.info("Starting restoration for %s", input_path)
        process_audio(input_path, output_path)
        logger.info("Restoration finished for %s", input_path)
        _TEMP_FILES.append(output_path)
        return input_path, output_path, "Restoration complete."
    except Exception:
        logger.exception("Audio restoration failed for %s", input_path)
        if output_path and os.path.exists(output_path):
            os.unlink(output_path)
        gr.Warning("Audio restoration failed. Check the logs and try again.")
        return input_path, None, "Restoration failed."


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="phonepod — Audio Restoration") as demo:
        gr.Markdown("# phonepod — Audio Restoration")
        gr.Markdown(
            "Drop a degraded audio file to restore it using AI enhancement. 100% local, privacy-first."
        )

        input_audio = gr.Audio(
            label="Upload Audio",
            sources=["upload"],
            type="filepath",
        )
        restore_button = gr.Button("Restore Audio", variant="primary")
        status = gr.Textbox(label="Status", value="Waiting for audio.", interactive=False)

        with gr.Row():
            original_audio = gr.Audio(label="Original", type="filepath", interactive=False)
            enhanced_audio = gr.Audio(label="Enhanced", type="filepath", interactive=False)

        restore_button.click(
            fn=set_processing_status,
            outputs=status,
            queue=False,
        ).then(
            fn=restore_audio,
            inputs=input_audio,
            outputs=[original_audio, enhanced_audio, status],
        )

    return demo


def main() -> None:
    demo = build_ui()
    try:
        demo.queue().launch(server_name="127.0.0.1", server_port=7860)
    finally:
        logger.info("Gradio server stopped. Running final cleanup...")
        _cleanup_temp_files()
        try:
            shutdown_engine()
        except Exception:
            pass
        logger.info("All resources released.")


if __name__ == "__main__":
    main()
