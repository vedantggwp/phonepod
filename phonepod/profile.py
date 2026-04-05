"""Voice profiles — mastering chain parameters that can be tuned and saved.

A profile captures the DSP settings that sound best for a particular voice
or recording setup. Profiles are JSON files that can be loaded by the engine.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

PROFILES_DIR = Path.home() / ".phonepod" / "profiles"


@dataclass(frozen=True)
class MasteringParams:
    """All tunable parameters for the Stage 3-5 mastering chain.

    Subtractive EQ philosophy: all EQ moves are cuts. Remove what's bad
    rather than boosting what's good. The voice is already there - just
    clean the window.
    """

    # Noise gate (silences artifacts between speech, soft to preserve tails)
    gate_threshold_db: float = -50.0

    # High-pass filter
    hpf_cutoff_hz: float = 80.0

    # Subtractive EQ - mud cut (low-mid resonance)
    mud_freq_hz: float = 200.0
    mud_gain_db: float = -2.5
    mud_q: float = 0.7

    # Subtractive EQ - boxiness cut (hollow/boxy room tone)
    box_freq_hz: float = 500.0
    box_gain_db: float = -4.0
    box_q: float = 1.0

    # Subtractive EQ - nasal cut (honky midrange)
    nasal_freq_hz: float = 1500.0
    nasal_gain_db: float = -3.0
    nasal_q: float = 1.0

    # Compressor 1 (gentle leveling - only catches peaks)
    comp1_threshold_db: float = -16.0
    comp1_ratio: float = 1.8
    comp1_attack_ms: float = 20.0
    comp1_release_ms: float = 150.0

    # Compressor 2 (safety net - barely touches most audio)
    comp2_threshold_db: float = -10.0
    comp2_ratio: float = 2.5
    comp2_attack_ms: float = 10.0
    comp2_release_ms: float = 100.0

    # De-esser (tames sibilance + harshness, post-compression)
    deess_freq_hz: float = 6500.0
    deess_gain_db: float = -3.0
    deess_q: float = 1.5

    # Studio room reverb (subtle early reflections)
    reverb_room_size: float = 0.15
    reverb_damping: float = 0.7
    reverb_wet: float = 0.03
    reverb_dry: float = 1.0

    # Loudness
    lufs_target: float = -18.0
    limiter_ceiling_db: float = -1.5


@dataclass(frozen=True)
class Profile:
    """A named voice profile with mastering parameters."""

    name: str
    params: MasteringParams = field(default_factory=MasteringParams)

    def save(self, path: Path | None = None) -> Path:
        """Save profile to JSON file."""
        if path is None:
            PROFILES_DIR.mkdir(parents=True, exist_ok=True)
            path = PROFILES_DIR / f"{self.name}.json"

        data = {"name": self.name, "params": asdict(self.params)}
        path.write_text(json.dumps(data, indent=2))
        logger.info("Profile saved: %s", path)
        return path

    @staticmethod
    def load(path: Path) -> "Profile":
        """Load profile from JSON file."""
        data = json.loads(path.read_text())
        params = MasteringParams(**data["params"])
        return Profile(name=data["name"], params=params)

    @staticmethod
    def load_by_name(name: str) -> "Profile":
        """Load a profile by name from the default profiles directory."""
        path = PROFILES_DIR / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Profile not found: {path}")
        return Profile.load(path)

    @staticmethod
    def list_profiles() -> list[str]:
        """List available profile names."""
        if not PROFILES_DIR.exists():
            return []
        return sorted(p.stem for p in PROFILES_DIR.glob("*.json"))


def params_from_semantic(
    warmth: float = 50.0,
    clarity: float = 50.0,
    compression: float = 50.0,
    de_ess: float = 50.0,
    loudness: float = 50.0,
    room: float = 30.0,
) -> MasteringParams:
    """Map semantic sliders (0-100) to raw mastering parameters.

    50 = current defaults. Below 50 = less effect. Above 50 = more effect.
    Room defaults to 30 (subtle studio ambience).
    """

    def lerp(low: float, mid: float, high: float, value: float) -> float:
        """Linear interpolation: 0->low, 50->mid, 100->high. Clamped to [0, 100]."""
        value = max(0.0, min(100.0, value))
        if value <= 50:
            return low + (mid - low) * (value / 50)
        return mid + (high - mid) * ((value - 50) / 50)

    return MasteringParams(
        # Warmth: controls depth of low-freq cuts (more warmth = shallower cuts)
        mud_gain_db=lerp(-5.0, -2.5, 0.0, warmth),
        box_gain_db=lerp(-6.0, -4.0, -1.5, warmth),

        # Clarity: controls mid-freq cleanup (more clarity = deeper nasal cut)
        nasal_gain_db=lerp(-1.0, -3.0, -5.0, clarity),
        deess_freq_hz=lerp(6000, 6500, 7500, clarity),

        # Compression: controls both compressor stages (gentler range)
        comp1_threshold_db=lerp(-10.0, -16.0, -24.0, compression),
        comp1_ratio=lerp(1.2, 1.8, 3.5, compression),
        comp2_threshold_db=lerp(-6.0, -10.0, -18.0, compression),
        comp2_ratio=lerp(1.2, 2.5, 5.0, compression),

        # De-ess: controls sibilance reduction
        deess_gain_db=lerp(0.0, -3.0, -7.0, de_ess),
        deess_q=lerp(2.5, 1.5, 0.8, de_ess),

        # Room: studio reverb amount (0 = bone dry, 100 = noticeable room)
        reverb_room_size=lerp(0.05, 0.15, 0.35, room),
        reverb_damping=lerp(0.8, 0.7, 0.5, room),
        reverb_wet=lerp(0.0, 0.03, 0.10, room),

        # Loudness: controls LUFS target
        lufs_target=lerp(-14.0, -18.0, -24.0, 100 - loudness),
        limiter_ceiling_db=lerp(-0.5, -1.5, -3.0, 100 - loudness),
    )
