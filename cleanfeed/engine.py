"""5-stage audio enhancement pipeline.

DeepFilterNet3 → MossFormer2 → Pedalboard DSP → LUFS → Limiter.
Pure tensor/numpy in, tensor out. Zero filesystem I/O.
"""

import logging

import numpy as np
import pyloudnorm as pyln
import torch
import torchaudio
from clearvoice import ClearVoice
from df.enhance import enhance as df_enhance
from df.enhance import init_df
from pedalboard import (
    Compressor,
    HighpassFilter,
    HighShelfFilter,
    Limiter,
    PeakFilter,
    Pedalboard,
)

from .profile import MasteringParams

logger = logging.getLogger(__name__)

OUTPUT_SR = 48000


def _build_mastering_chain(p: MasteringParams) -> tuple[Pedalboard, Pedalboard]:
    """Build mastering + limiter chains from parameters. Cheap to call."""
    mastering = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=p.hpf_cutoff_hz),
        PeakFilter(cutoff_frequency_hz=p.low_mid_freq_hz, gain_db=p.low_mid_gain_db, q=p.low_mid_q),
        Compressor(threshold_db=p.comp1_threshold_db, ratio=p.comp1_ratio, attack_ms=p.comp1_attack_ms, release_ms=p.comp1_release_ms),
        Compressor(threshold_db=p.comp2_threshold_db, ratio=p.comp2_ratio, attack_ms=p.comp2_attack_ms, release_ms=p.comp2_release_ms),
        PeakFilter(cutoff_frequency_hz=p.deess_freq_hz, gain_db=p.deess_gain_db, q=p.deess_q),
        PeakFilter(cutoff_frequency_hz=p.presence_freq_hz, gain_db=p.presence_gain_db, q=p.presence_q),
        HighShelfFilter(cutoff_frequency_hz=p.air_freq_hz, gain_db=p.air_gain_db, q=p.air_q),
    ])
    limiter = Pedalboard([Limiter(threshold_db=p.limiter_ceiling_db)])
    return mastering, limiter


class Engine:
    """5-stage pipeline: DeepFilterNet → MossFormer2 → Pedalboard DSP → LUFS → Limiter."""

    def __init__(self, params: MasteringParams | None = None) -> None:
        self._params = params or MasteringParams()

        logger.info("Loading DeepFilterNet3...")
        self._df_model, self._df_state, _ = init_df()
        self._df_sr = self._df_state.sr()
        logger.info("DeepFilterNet3 loaded (sr=%d)", self._df_sr)

        logger.info("Loading MossFormer2_SE_48K...")
        self._clearvoice = ClearVoice(
            task="speech_enhancement",
            model_names=["MossFormer2_SE_48K"],
        )
        logger.info("MossFormer2_SE_48K loaded")

        self._mastering, self._limiter = _build_mastering_chain(self._params)
        logger.info("Mastering chain ready")

    @property
    def params(self) -> MasteringParams:
        return self._params

    def set_params(self, params: MasteringParams) -> None:
        """Swap mastering parameters without reloading ML models."""
        self._params = params
        self._mastering, self._limiter = _build_mastering_chain(params)
        logger.info("Mastering chain rebuilt with new parameters")

    def enhance(self, audio_tensor: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, int]:
        """Run the full enhancement pipeline.

        Args:
            audio_tensor: 1D mono float32 tensor.
            sample_rate: Input sample rate.

        Returns:
            (enhanced_tensor, output_sample_rate) — 1D mono tensor at 48kHz.
        """
        if audio_tensor.ndim != 1:
            raise ValueError("audio_tensor must be 1D mono")

        audio_tensor = audio_tensor.detach().flatten().to(dtype=torch.float32).cpu().contiguous()
        logger.info(
            "Starting enhancement: %d samples at %d Hz (%.1fs)",
            audio_tensor.numel(),
            sample_rate,
            audio_tensor.numel() / sample_rate,
        )

        # --- Stage 1: DeepFilterNet3 noise suppression ---
        if sample_rate != self._df_sr:
            logger.info("Resampling %d → %d Hz for DeepFilterNet", sample_rate, self._df_sr)
            audio_tensor = torchaudio.functional.resample(audio_tensor, sample_rate, self._df_sr)

        denoised = df_enhance(self._df_model, self._df_state, audio_tensor.unsqueeze(0))
        denoised_1d = denoised.squeeze(0).detach().cpu().contiguous()
        logger.info("Stage 1 complete: DeepFilterNet denoise")

        # --- Stage 2: MossFormer2 speech enhancement ---
        cv_input = denoised_1d.numpy().astype(np.float32)[np.newaxis, :]
        with torch.no_grad():
            cv_result = self._clearvoice(cv_input)

        if isinstance(cv_result, dict):
            enhanced_np = list(cv_result.values())[0]
        elif isinstance(cv_result, torch.Tensor):
            enhanced_np = cv_result.numpy()
        else:
            enhanced_np = np.asarray(cv_result, dtype=np.float32)

        enhanced_np = np.ascontiguousarray(enhanced_np.astype(np.float32, copy=False))
        if enhanced_np.ndim == 1:
            enhanced_np = enhanced_np[np.newaxis, :]
        logger.info("Stage 2 complete: MossFormer2 enhance")

        # --- Stage 3: Pedalboard DSP mastering ---
        mastered = self._mastering(enhanced_np, sample_rate=OUTPUT_SR, reset=True)
        logger.info("Stage 3 complete: DSP mastering")

        # --- Stage 4: LUFS normalization ---
        mono = np.ascontiguousarray(mastered[0].astype(np.float64, copy=False))
        meter = pyln.Meter(OUTPUT_SR)
        loudness = meter.integrated_loudness(mono)
        target = self._params.lufs_target
        if np.isfinite(loudness):
            normalized = pyln.normalize.loudness(mono, loudness, target).astype(np.float32)
            logger.info("Stage 4 complete: LUFS %.1f → %.1f", loudness, target)
        else:
            logger.warning("LUFS non-finite (%.2f), skipping normalization", loudness)
            normalized = mono.astype(np.float32)

        # --- Stage 5: Brick-wall limiter ---
        limited = self._limiter(normalized[np.newaxis, :], sample_rate=OUTPUT_SR, reset=True)
        logger.info("Stage 5 complete: Limiter at %.1f dB", self._params.limiter_ceiling_db)

        result = torch.from_numpy(np.ascontiguousarray(limited[0]))
        return result, OUTPUT_SR

    def master_only(self, enhanced_np: np.ndarray) -> np.ndarray:
        """Run only stages 3-5 (mastering + LUFS + limiter) on pre-denoised audio.

        Used by the tuning UI to preview mastering changes instantly
        without re-running the slow ML inference stages.

        Args:
            enhanced_np: 1D float32 numpy array at 48kHz (output of stages 1+2).

        Returns:
            Mastered 1D float32 numpy array at 48kHz.
        """
        if enhanced_np.ndim == 1:
            enhanced_np = enhanced_np[np.newaxis, :]

        mastered = self._mastering(enhanced_np, sample_rate=OUTPUT_SR, reset=True)

        mono = np.ascontiguousarray(mastered[0].astype(np.float64, copy=False))
        meter = pyln.Meter(OUTPUT_SR)
        loudness = meter.integrated_loudness(mono)
        target = self._params.lufs_target
        if np.isfinite(loudness):
            normalized = pyln.normalize.loudness(mono, loudness, target).astype(np.float32)
        else:
            normalized = mono.astype(np.float32)

        limited = self._limiter(normalized[np.newaxis, :], sample_rate=OUTPUT_SR, reset=True)
        return np.ascontiguousarray(limited[0])
