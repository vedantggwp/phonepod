"""cleanfeed tuner — drop trash audio, get podcast gold."""

import cleanfeed._compat  # noqa: F401

import random
import subprocess
import tempfile

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torchaudio
from gradio.themes import Base
from gradio.themes.utils import colors, fonts, sizes

from cleanfeed.engine import Engine, OUTPUT_SR
from cleanfeed.profile import MasteringParams, Profile, params_from_semantic

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

_coral = colors.Color(
    name="coral",
    c50="#fff1f0",
    c100="#ffe0de",
    c200="#ffc2bd",
    c300="#ff9e96",
    c400="#ff6f63",
    c500="#f94839",
    c600="#e7322a",
    c700="#c4241e",
    c800="#a2201b",
    c900="#861e1a",
    c950="#4a0c09",
)


class CleanfeedTheme(Base):
    def __init__(self):
        super().__init__(
            primary_hue=_coral,
            secondary_hue=_coral,
            neutral_hue=colors.zinc,
            text_size=sizes.text_md,
            spacing_size=sizes.spacing_md,
            radius_size=sizes.radius_md,
            font=[
                fonts.GoogleFont("Space Grotesk", weights=(400, 500, 600, 700)),
                "system-ui",
                "sans-serif",
            ],
            font_mono=[
                fonts.GoogleFont("JetBrains Mono", weights=(400, 500)),
                "monospace",
            ],
        )
        self.name = "cleanfeed"
        super().set(
            # Body
            body_background_fill="#18181b",
            body_background_fill_dark="#18181b",
            body_text_color="#fafafa",
            body_text_color_dark="#fafafa",
            body_text_color_subdued="#acacb8",
            body_text_color_subdued_dark="#a1a1aa",
            # Blocks
            block_background_fill="#27272a",
            block_background_fill_dark="#27272a",
            block_border_color="#3f3f46",
            block_border_color_dark="#3f3f46",
            block_border_width="1px",
            block_label_background_fill="#27272a",
            block_label_background_fill_dark="#27272a",
            block_label_text_color="#fafafa",
            block_label_text_color_dark="#fafafa",
            block_label_radius="*radius_sm",
            block_title_text_color="#fafafa",
            block_title_text_color_dark="#fafafa",
            # Inputs
            input_background_fill="#3f3f46",
            input_background_fill_dark="#3f3f46",
            input_background_fill_focus="#52525b",
            input_background_fill_focus_dark="#52525b",
            input_border_color="#52525b",
            input_border_color_dark="#52525b",
            input_border_color_focus="*primary_500",
            input_border_color_focus_dark="*primary_500",
            input_placeholder_color="#a1a1aa",
            input_placeholder_color_dark="#71717a",
            # Buttons — primary
            button_primary_background_fill="*primary_500",
            button_primary_background_fill_dark="*primary_500",
            button_primary_background_fill_hover="*primary_400",
            button_primary_background_fill_hover_dark="*primary_400",
            button_primary_text_color="#ffffff",
            button_primary_text_color_dark="#ffffff",
            button_primary_border_color="*primary_600",
            button_primary_border_color_dark="*primary_600",
            button_primary_shadow="0 4px 14px 0 rgba(249, 72, 57, 0.3)",
            button_primary_shadow_dark="0 4px 14px 0 rgba(249, 72, 57, 0.3)",
            button_primary_shadow_hover="0 6px 20px 0 rgba(249, 72, 57, 0.4)",
            button_primary_shadow_hover_dark="0 6px 20px 0 rgba(249, 72, 57, 0.4)",
            button_transform_hover="translateY(-1px)",
            button_transform_active="translateY(0px)",
            button_transition="all 0.15s cubic-bezier(0.25, 1, 0.5, 1)",
            button_large_text_weight="600",
            button_large_radius="*radius_md",
            # Buttons — secondary
            button_secondary_background_fill="#3f3f46",
            button_secondary_background_fill_dark="#3f3f46",
            button_secondary_background_fill_hover="#52525b",
            button_secondary_background_fill_hover_dark="#52525b",
            button_secondary_text_color="#fafafa",
            button_secondary_text_color_dark="#fafafa",
            button_secondary_border_color="#52525b",
            button_secondary_border_color_dark="#52525b",
            # Slider
            slider_color="*primary_500",
            slider_color_dark="*primary_500",
            # Shadows
            shadow_drop="0 2px 8px rgba(0,0,0,0.3)",
            shadow_drop_lg="0 8px 24px rgba(0,0,0,0.4)",
            shadow_spread="4px",
            shadow_spread_dark="4px",
            # Links
            link_text_color="*primary_400",
            link_text_color_dark="*primary_400",
            link_text_color_hover="*primary_300",
            link_text_color_hover_dark="*primary_300",
            # Accent
            color_accent="*primary_500",
            color_accent_soft="*primary_950",
            color_accent_soft_dark="*primary_950",
            border_color_accent="*primary_500",
            border_color_accent_dark="*primary_500",
            # Backgrounds
            background_fill_primary="#27272a",
            background_fill_primary_dark="#27272a",
            background_fill_secondary="#1f1f23",
            background_fill_secondary_dark="#1f1f23",
        )


_CUSTOM_CSS = """
/* Container */
.gradio-container {
    max-width: 780px !important;
    margin: 0 auto !important;
}

/* Hero heading */
#hero-title {
    text-align: center;
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}
#hero-title h1 {
    font-size: 2.4rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em !important;
    color: #f94839;
}

#hero-sub {
    text-align: center;
    margin-top: 0 !important;
    padding-top: 0 !important;
    opacity: 0.7;
}
#hero-sub p {
    font-size: 1.05rem !important;
    font-weight: 400 !important;
}

/* Audio panels — force equal width */
#col-original, #col-cleaned {
    min-width: 0 !important;
    flex: 1 1 0% !important;
    max-width: 50% !important;
}

/* CTA button */
#clean-btn {
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em;
    padding: 14px 0 !important;
}

/* Hide Gradio branding footer */
footer {
    display: none !important;
}

/* Accordion styling */
.accordion {
    border-color: var(--block-border-color) !important;
    background: var(--block-background-fill) !important;
}

/* Slider tweaks */
input[type="range"] {
    accent-color: var(--slider-color);
}

/* Preset row */
#preset-row {
    margin-top: 4px;
}

/* Footer */
#footer {
    text-align: center;
    opacity: 0.5;
    margin-top: 8px !important;
}
#footer p {
    font-size: 0.8rem !important;
}

/* Responsive — stack panels on mobile */
@media (max-width: 640px) {
    #col-original, #col-cleaned {
        max-width: 100% !important;
        flex: 1 1 100% !important;
    }
    .row {
        flex-direction: column !important;
    }
    #hero-title h1 {
        font-size: 1.8rem !important;
    }
    #clean-btn {
        font-size: 1rem !important;
    }
}
"""

# ---------------------------------------------------------------------------
# Engine singleton + state
# ---------------------------------------------------------------------------

_engine = None
_denoised = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = Engine()
    return _engine


# ---------------------------------------------------------------------------
# Microcopy
# ---------------------------------------------------------------------------

_PROGRESS_NOISE = [
    "Hunting down background noise...",
    "Shushing the hum...",
    "Evicting unwanted guests from your audio...",
    "Teaching noise to be quiet...",
]

_PROGRESS_ENHANCE = [
    "Polishing your voice...",
    "Making you sound expensive...",
    "Adding that podcast sparkle...",
    "Giving your voice the glow-up...",
]

_PROGRESS_MASTER = [
    "Mastering the final mix...",
    "Adding the chef's kiss...",
    "Putting on the finishing touches...",
    "Making it radio-ready...",
]

_PROGRESS_DONE = [
    "Boom. Listen to that.",
    "Done! You sound incredible.",
    "That's a wrap. Hit play.",
    "Fresh out the oven.",
]


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

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
    return to_wav(filepath)


def clean(filepath, progress=gr.Progress()):
    global _denoised
    if not filepath:
        raise gr.Error("Drop something in first!")

    progress(0.1, desc="Loading your audio...")
    wav = to_wav(filepath)
    audio, sr = torchaudio.load(wav)
    mono = audio.mean(dim=0).flatten().to(torch.float32).cpu()

    engine = get_engine()

    progress(0.2, desc=random.choice(_PROGRESS_NOISE))
    if sr != engine._df_sr:
        mono = torchaudio.functional.resample(mono, int(sr), engine._df_sr)
    from df.enhance import enhance as df_enhance
    denoised = df_enhance(engine._df_model, engine._df_state, mono.unsqueeze(0))
    denoised = denoised.squeeze(0).detach().cpu().contiguous()

    progress(0.5, desc=random.choice(_PROGRESS_ENHANCE))
    cv_in = denoised.numpy().astype(np.float32)[np.newaxis, :]
    with torch.no_grad():
        cv_out = engine._clearvoice(cv_in)
    if isinstance(cv_out, dict):
        cv_out = list(cv_out.values())[0]
    elif isinstance(cv_out, torch.Tensor):
        cv_out = cv_out.numpy()
    _denoised = np.ascontiguousarray(cv_out.flatten().astype(np.float32))

    progress(0.85, desc=random.choice(_PROGRESS_MASTER))
    engine.set_params(MasteringParams())
    mastered = engine.master_only(_denoised.copy())

    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    sf.write(out, mastered, OUTPUT_SR)
    progress(1.0, desc=random.choice(_PROGRESS_DONE))
    return out


def remaster(warmth, clarity, punch, de_ess, vol):
    if _denoised is None:
        raise gr.Error("Clean your audio first, then tweak away.")
    engine = get_engine()
    engine.set_params(params_from_semantic(warmth, clarity, punch, de_ess, vol))
    mastered = engine.master_only(_denoised.copy())
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    sf.write(out, mastered, OUTPUT_SR)
    return out


def save(name, warmth, clarity, punch, de_ess, vol):
    if not name or not name.strip():
        raise gr.Error("Give your preset a name first!")
    slug = name.strip().lower().replace(" ", "-")
    engine = get_engine()
    engine.set_params(params_from_semantic(warmth, clarity, punch, de_ess, vol))
    Profile(name=slug, params=engine.params).save()
    gr.Info(f'Saved "{slug}"! Use it from CLI: cleanfeed input.m4a out.wav --profile {slug}')


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

theme = CleanfeedTheme()

with gr.Blocks(title="cleanfeed", theme=theme, css=_CUSTOM_CSS) as demo:
    gr.Markdown("# cleanfeed", elem_id="hero-title")
    gr.Markdown("Drop your audio. Get it back sounding good. **Runs 100% on your machine.**", elem_id="hero-sub")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, elem_id="col-original"):
            original = gr.Audio(
                label="Your recording",
                type="filepath",
                sources=["upload", "microphone"],
                format="wav",
            )
        with gr.Column(scale=1, elem_id="col-cleaned"):
            cleaned = gr.Audio(
                label="Cleaned up — hit the button and it'll appear here",
                type="filepath",
                interactive=False,
                sources=[],
            )

    clean_btn = gr.Button("Clean it up", variant="primary", size="lg", elem_id="clean-btn")

    with gr.Accordion("Want more control?", open=False):
        gr.Markdown("*Drag the sliders, hear the difference instantly.*")
        warmth = gr.Slider(0, 100, 50, step=1, label="Warmth")
        clarity = gr.Slider(0, 100, 50, step=1, label="Clarity")
        punch = gr.Slider(0, 100, 50, step=1, label="Punch")
        de_ess = gr.Slider(0, 100, 50, step=1, label="De-ess")
        vol = gr.Slider(0, 100, 50, step=1, label="Volume")

        sliders = [warmth, clarity, punch, de_ess, vol]

        gr.Markdown("---")
        with gr.Row(elem_id="preset-row"):
            preset_name = gr.Textbox(
                label="Save as preset",
                placeholder="e.g. my-podcast-voice",
                scale=3,
            )
            save_btn = gr.Button("Save", scale=1)

    gr.Markdown("made with ears, not eyes", elem_id="footer")

    # Wiring
    original.upload(fn=convert_on_upload, inputs=original, outputs=original)
    original.stop_recording(fn=convert_on_upload, inputs=original, outputs=original)
    clean_btn.click(fn=clean, inputs=original, outputs=cleaned)

    for s in sliders:
        s.release(fn=remaster, inputs=sliders, outputs=cleaned)

    save_btn.click(fn=save, inputs=[preset_name] + sliders)

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7861, show_error=True)
