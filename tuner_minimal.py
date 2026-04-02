"""Minimal tuner — the simplest thing that could possibly work."""

import cleanfeed._compat  # noqa: F401

import os
import shutil
import subprocess
import tempfile

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torchaudio

from cleanfeed.engine import Engine, OUTPUT_SR
from cleanfeed.profile import MasteringParams, Profile, params_from_semantic

_engine = None
_denoised = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = Engine()
    return _engine


def to_wav(path):
    """Convert any audio to 48kHz mono WAV. Returns new path (always WAV)."""
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-ar", "48000", "-ac", "1", out],
        check=True,
    )
    return out


def convert_on_upload(filepath):
    """Convert uploaded file to WAV so the browser can play it back."""
    if not filepath:
        return None
    wav = to_wav(filepath)
    return wav


def clean(filepath, progress=gr.Progress()):
    global _denoised
    if not filepath:
        raise gr.Error("Upload a file first.")

    progress(0.1, desc="Loading...")
    wav = to_wav(filepath)
    audio, sr = torchaudio.load(wav)
    mono = audio.mean(dim=0).flatten().to(torch.float32).cpu()

    engine = get_engine()

    progress(0.2, desc="Removing noise (DeepFilterNet)...")
    if sr != engine._df_sr:
        mono = torchaudio.functional.resample(mono, int(sr), engine._df_sr)
    from df.enhance import enhance as df_enhance
    denoised = df_enhance(engine._df_model, engine._df_state, mono.unsqueeze(0))
    denoised = denoised.squeeze(0).detach().cpu().contiguous()

    progress(0.5, desc="Enhancing speech (MossFormer2)...")
    cv_in = denoised.numpy().astype(np.float32)[np.newaxis, :]
    with torch.no_grad():
        cv_out = engine._clearvoice(cv_in)
    if isinstance(cv_out, dict):
        cv_out = list(cv_out.values())[0]
    elif isinstance(cv_out, torch.Tensor):
        cv_out = cv_out.numpy()
    _denoised = np.ascontiguousarray(cv_out.flatten().astype(np.float32))

    progress(0.85, desc="Mastering...")
    engine.set_params(MasteringParams())
    mastered = engine.master_only(_denoised.copy())

    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    sf.write(out, mastered, OUTPUT_SR)
    progress(1.0, desc="Done!")
    return out


def remaster(warmth, clarity, punch, de_ess, vol):
    if _denoised is None:
        raise gr.Error("Clean your audio first.")
    engine = get_engine()
    engine.set_params(params_from_semantic(warmth, clarity, punch, de_ess, vol))
    mastered = engine.master_only(_denoised.copy())
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    sf.write(out, mastered, OUTPUT_SR)
    return out


def save(name, warmth, clarity, punch, de_ess, vol):
    if not name or not name.strip():
        raise gr.Error("Enter a name.")
    slug = name.strip().lower().replace(" ", "-")
    engine = get_engine()
    engine.set_params(params_from_semantic(warmth, clarity, punch, de_ess, vol))
    Profile(name=slug, params=engine.params).save()
    gr.Info(f'Saved "{slug}". CLI: cleanfeed input.m4a out.wav --profile {slug}')


with gr.Blocks(title="cleanfeed tuner") as demo:
    gr.Markdown("# cleanfeed tuner")
    gr.Markdown("Upload → Clean → Tune → Download. **100% local.**")

    with gr.Row():
        original = gr.Audio(label="Original", type="filepath", sources=["upload", "microphone"], format="wav")
        cleaned = gr.Audio(label="Cleaned", type="filepath", interactive=False, sources=[])

    clean_btn = gr.Button("Clean it up", variant="primary", size="lg")

    gr.Markdown("---")
    gr.Markdown("### Tune (optional)")
    warmth = gr.Slider(0, 100, 50, step=1, label="Warmth")
    clarity = gr.Slider(0, 100, 50, step=1, label="Clarity")
    punch = gr.Slider(0, 100, 50, step=1, label="Punch")
    de_ess = gr.Slider(0, 100, 50, step=1, label="De-ess")
    vol = gr.Slider(0, 100, 50, step=1, label="Volume")

    sliders = [warmth, clarity, punch, de_ess, vol]

    gr.Markdown("---")
    with gr.Row():
        preset_name = gr.Textbox(label="Preset name", placeholder="my-voice", scale=3)
        save_btn = gr.Button("Save preset", scale=1)

    # Wiring — convert on upload so browser can play the original
    original.upload(fn=convert_on_upload, inputs=original, outputs=original)
    original.stop_recording(fn=convert_on_upload, inputs=original, outputs=original)

    clean_btn.click(fn=clean, inputs=original, outputs=cleaned)

    for s in sliders:
        s.release(fn=remaster, inputs=sliders, outputs=cleaned)

    save_btn.click(fn=save, inputs=[preset_name] + sliders)

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7861)
