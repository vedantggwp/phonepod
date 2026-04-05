"""Voice Tuner — clean your recording, fine-tune by ear, save as a preset.

Run with: uv run python -m phonepod.tuner
Opens at http://localhost:7861
"""

import atexit
import logging
import os
import shutil
import subprocess
import tempfile

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torchaudio

from .engine import Engine, OUTPUT_SR
from .profile import MasteringParams, Profile, params_from_semantic

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# --- Session state ---
_engine: Engine | None = None
_denoised_cache: np.ndarray | None = None
_temp_files: list[str] = []


def _cleanup_temp_files() -> None:
    for f in _temp_files:
        try:
            if os.path.exists(f):
                os.unlink(f)
        except OSError:
            pass
    _temp_files.clear()


atexit.register(_cleanup_temp_files)


def _get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = Engine()
    return _engine


def _make_temp_wav(audio: np.ndarray, sr: int) -> str:
    """Write audio to a tracked temp file."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sr)
    _temp_files.append(tmp.name)
    return tmp.name


def _convert_to_wav(audio_path: str) -> str:
    """Convert non-WAV to 48kHz mono WAV."""
    ext = os.path.splitext(audio_path)[1].lower()
    if ext == ".wav":
        return audio_path
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is required for non-WAV files. Install: brew install ffmpeg"
        )
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    _temp_files.append(tmp.name)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", audio_path,
             "-ar", str(OUTPUT_SR), "-ac", "1", tmp.name],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Could not convert this file. Try WAV or M4A. (ffmpeg: {e.stderr[:200]})"
        ) from None
    return tmp.name


# --- Callbacks ---

def clean_audio(filepath, progress=gr.Progress()):
    """Run full pipeline: denoise + enhance + default mastering."""
    global _denoised_cache

    if not filepath:
        raise gr.Error("Upload a recording first.")

    progress(0.05, desc="Loading audio...")
    wav_path = _convert_to_wav(filepath)
    wav, sr = torchaudio.load(wav_path)
    mono = wav.mean(dim=0).flatten()

    engine = _get_engine()

    progress(0.15, desc="Removing background noise...")
    audio_tensor = mono.to(dtype=torch.float32).cpu().contiguous()
    if sr != engine._df_sr:
        audio_tensor = torchaudio.functional.resample(audio_tensor, int(sr), engine._df_sr)

    from df.enhance import enhance as df_enhance
    denoised = df_enhance(engine._df_model, engine._df_state, audio_tensor.unsqueeze(0))
    denoised_1d = denoised.squeeze(0).detach().cpu().contiguous()

    progress(0.50, desc="Enhancing speech clarity...")
    cv_input = denoised_1d.numpy().astype(np.float32)[np.newaxis, :]
    with torch.no_grad():
        cv_result = engine._clearvoice(cv_input)

    if isinstance(cv_result, dict):
        enhanced_np = list(cv_result.values())[0]
    elif isinstance(cv_result, torch.Tensor):
        enhanced_np = cv_result.numpy()
    else:
        enhanced_np = np.asarray(cv_result, dtype=np.float32)

    _denoised_cache = np.ascontiguousarray(enhanced_np.flatten().astype(np.float32))

    progress(0.85, desc="Applying studio polish...")
    engine.set_params(MasteringParams())
    mastered = engine.master_only(_denoised_cache.copy())
    output_path = _make_temp_wav(mastered, OUTPUT_SR)

    progress(1.0, desc="Done!")
    return (
        output_path,                                                         # cleaned_audio
        gr.DownloadButton(value=output_path, visible=True),                  # download_btn
        gr.Group(visible=True),                                              # tuning_section
    )


def preview_semantic(warmth, clarity, punch, de_ess, volume):
    """Re-master with semantic sliders. Called on slider release."""
    if _denoised_cache is None:
        return gr.skip(), gr.skip()
    params = params_from_semantic(warmth, clarity, punch, de_ess, volume)
    engine = _get_engine()
    engine.set_params(params)
    mastered = engine.master_only(_denoised_cache.copy())
    path = _make_temp_wav(mastered, OUTPUT_SR)
    return path, gr.DownloadButton(value=path)


def preview_raw(
    hpf_cutoff,
    low_mid_freq, low_mid_gain, low_mid_q,
    comp1_thresh, comp1_ratio, comp1_attack, comp1_release,
    comp2_thresh, comp2_ratio, comp2_attack, comp2_release,
    deess_freq, deess_gain, deess_q,
    presence_freq, presence_gain, presence_q,
    air_freq, air_gain, air_q,
    lufs_target, limiter_ceiling,
):
    """Re-master with raw parameters. Called on slider release."""
    if _denoised_cache is None:
        return gr.skip(), gr.skip()
    params = MasteringParams(
        hpf_cutoff_hz=hpf_cutoff,
        low_mid_freq_hz=low_mid_freq, low_mid_gain_db=low_mid_gain, low_mid_q=low_mid_q,
        comp1_threshold_db=comp1_thresh, comp1_ratio=comp1_ratio,
        comp1_attack_ms=comp1_attack, comp1_release_ms=comp1_release,
        comp2_threshold_db=comp2_thresh, comp2_ratio=comp2_ratio,
        comp2_attack_ms=comp2_attack, comp2_release_ms=comp2_release,
        deess_freq_hz=deess_freq, deess_gain_db=deess_gain, deess_q=deess_q,
        presence_freq_hz=presence_freq, presence_gain_db=presence_gain, presence_q=presence_q,
        air_freq_hz=air_freq, air_gain_db=air_gain, air_q=air_q,
        lufs_target=lufs_target, limiter_ceiling_db=limiter_ceiling,
    )
    engine = _get_engine()
    engine.set_params(params)
    mastered = engine.master_only(_denoised_cache.copy())
    path = _make_temp_wav(mastered, OUTPUT_SR)
    return path, gr.DownloadButton(value=path)


def save_preset(name):
    """Save current engine params as a named preset."""
    if not name or not name.strip():
        raise gr.Error("Give your preset a name first.")
    slug = name.strip().lower().replace(" ", "-")
    existing = Profile.list_profiles()
    verb = "updated" if slug in existing else "saved"
    engine = _get_engine()
    Profile(name=slug, params=engine.params).save()
    gr.Info(f'Preset "{slug}" {verb}. CLI: phonepod input.m4a out.wav --profile {slug}')
    return ""


# --- Build UI ---

def build_tuner_ui() -> gr.Blocks:
    with gr.Blocks(
        title="phonepod",
        theme=gr.themes.Soft(primary_hue="stone", neutral_hue="stone", font=gr.themes.GoogleFont("Inter")),
        css=".gradio-container { max-width: 760px !important; margin: auto !important; }",
    ) as demo:

        gr.Markdown("# phonepod")
        gr.Markdown("Your phone recording, made podcast-ready. **100% local — your audio never leaves this computer.**")

        # --- Row 1: Original + Cleaned side by side ---
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Original")
                input_audio = gr.Audio(
                    type="filepath",
                    sources=["upload", "microphone"],
                    label="Upload or record",
                    format="wav",
                )
            with gr.Column():
                gr.Markdown("#### Cleaned")
                cleaned_audio = gr.Audio(
                    label="Cleaned audio appears here",
                    interactive=False,
                    type="filepath",
                    sources=[],
                )

        # --- Action buttons ---
        with gr.Row():
            clean_btn = gr.Button("Clean it up", variant="primary", size="lg", scale=2)
            download_btn = gr.DownloadButton("Download", visible=False, variant="secondary", scale=1)

        # --- Tuning section ---
        with gr.Group(visible=False) as tuning_section:
            gr.Markdown("### Fine-tune the sound")
            gr.Markdown("Adjust a slider and release — the cleaned audio updates instantly.")

            warmth = gr.Slider(0, 100, 50, step=1, label="Warmth",
                               info="Adds richness and body. Sounds like a close-up mic.")
            clarity = gr.Slider(0, 100, 50, step=1, label="Clarity",
                                info="Makes your words cut through. Presence without harshness.")
            punch = gr.Slider(0, 100, 50, step=1, label="Punch",
                              info="Tightens dynamics. More punch = more consistent volume.")
            de_ess = gr.Slider(0, 100, 50, step=1, label="De-ess",
                               info="Softens sharp 's' and 'sh' sounds.")
            volume = gr.Slider(0, 100, 50, step=1, label="Volume",
                               info="Overall loudness. 50 is podcast standard (-18 LUFS).")

            with gr.Accordion("Advanced controls — for audio engineers", open=False):
                gr.Markdown("*These override the simple sliders above.*")

                gr.Markdown("**Rumble filter**")
                hpf_cutoff = gr.Slider(20, 200, 80, step=1, label="Cutoff (Hz)",
                                       info="Removes low rumble below this frequency")

                gr.Markdown("**Low-mid shaping**")
                low_mid_freq = gr.Slider(100, 600, 300, step=10, label="Frequency (Hz)")
                low_mid_gain = gr.Slider(-12, 3, -3, step=0.5, label="Gain (dB)")
                low_mid_q = gr.Slider(0.3, 3.0, 1.0, step=0.1, label="Width")

                gr.Markdown("**Compressor — gentle**")
                comp1_thresh = gr.Slider(-40, 0, -20, step=1, label="Threshold (dB)")
                comp1_ratio = gr.Slider(1, 10, 2, step=0.5, label="Ratio")
                comp1_attack = gr.Slider(1, 100, 15, step=1, label="Attack (ms)")
                comp1_release = gr.Slider(10, 500, 100, step=5, label="Release (ms)")

                gr.Markdown("**Compressor — tight**")
                comp2_thresh = gr.Slider(-40, 0, -15, step=1, label="Threshold (dB)")
                comp2_ratio = gr.Slider(1, 10, 3, step=0.5, label="Ratio")
                comp2_attack = gr.Slider(1, 100, 10, step=1, label="Attack (ms)")
                comp2_release = gr.Slider(10, 500, 80, step=5, label="Release (ms)")

                gr.Markdown("**De-esser**")
                deess_freq = gr.Slider(3000, 10000, 6000, step=100, label="Frequency (Hz)")
                deess_gain = gr.Slider(-12, 0, -4, step=0.5, label="Gain (dB)")
                deess_q = gr.Slider(0.5, 5.0, 2.0, step=0.1, label="Width")

                gr.Markdown("**Presence**")
                presence_freq = gr.Slider(1000, 6000, 3000, step=100, label="Frequency (Hz)")
                presence_gain = gr.Slider(-3, 8, 2.5, step=0.5, label="Gain (dB)")
                presence_q = gr.Slider(0.3, 3.0, 0.8, step=0.1, label="Width")

                gr.Markdown("**Air**")
                air_freq = gr.Slider(6000, 16000, 10000, step=500, label="Frequency (Hz)")
                air_gain = gr.Slider(-3, 6, 2, step=0.5, label="Gain (dB)")
                air_q = gr.Slider(0.3, 2.0, 0.7, step=0.1, label="Width")

                gr.Markdown("**Loudness**")
                lufs_target = gr.Slider(-24, -14, -18, step=0.5, label="Target (LUFS)")
                limiter_ceiling = gr.Slider(-6, 0, -1.5, step=0.5, label="Limiter ceiling (dB)")

            with gr.Row():
                preset_name = gr.Textbox(label="Preset name", placeholder="e.g. my-podcast-voice",
                                         max_lines=1, scale=3)
                save_btn = gr.Button("Save as preset", variant="secondary", scale=1)

        # --- Wiring ---

        # Clean button
        clean_btn.click(
            fn=clean_audio,
            inputs=input_audio,
            outputs=[cleaned_audio, download_btn, tuning_section],
        )

        # Semantic sliders — preview on release
        semantic_sliders = [warmth, clarity, punch, de_ess, volume]
        for s in semantic_sliders:
            s.release(fn=preview_semantic, inputs=semantic_sliders, outputs=[cleaned_audio, download_btn])

        # Raw sliders — preview on release
        raw_sliders = [
            hpf_cutoff,
            low_mid_freq, low_mid_gain, low_mid_q,
            comp1_thresh, comp1_ratio, comp1_attack, comp1_release,
            comp2_thresh, comp2_ratio, comp2_attack, comp2_release,
            deess_freq, deess_gain, deess_q,
            presence_freq, presence_gain, presence_q,
            air_freq, air_gain, air_q,
            lufs_target, limiter_ceiling,
        ]
        for s in raw_sliders:
            s.release(fn=preview_raw, inputs=raw_sliders, outputs=[cleaned_audio, download_btn])

        # Save preset
        save_btn.click(fn=save_preset, inputs=preset_name, outputs=preset_name)

    return demo


def main() -> None:
    demo = build_tuner_ui()
    demo.queue().launch(server_name="127.0.0.1", server_port=7861)


if __name__ == "__main__":
    main()
