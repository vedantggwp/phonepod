"""Voice profiles — mastering chain parameters that can be tuned and saved.

A profile captures the DSP settings that sound best for a particular voice
or recording setup. Profiles are JSON files that can be loaded by the engine.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

PROFILES_DIR = Path.home() / ".cleanfeed" / "profiles"


@dataclass(frozen=True)
class MasteringParams:
    """All tunable parameters for the Stage 3-5 mastering chain."""

    # High-pass filter
    hpf_cutoff_hz: float = 80.0

    # Low-mid EQ (subtractive — tames muddiness)
    low_mid_freq_hz: float = 300.0
    low_mid_gain_db: float = -3.0
    low_mid_q: float = 1.0

    # Compressor 1 (gentle leveling)
    comp1_threshold_db: float = -20.0
    comp1_ratio: float = 2.0
    comp1_attack_ms: float = 15.0
    comp1_release_ms: float = 100.0

    # Compressor 2 (tighter control)
    comp2_threshold_db: float = -15.0
    comp2_ratio: float = 3.0
    comp2_attack_ms: float = 10.0
    comp2_release_ms: float = 80.0

    # De-esser (tames sibilance)
    deess_freq_hz: float = 6000.0
    deess_gain_db: float = -4.0
    deess_q: float = 2.0

    # Presence boost (vocal clarity)
    presence_freq_hz: float = 3000.0
    presence_gain_db: float = 2.5
    presence_q: float = 0.8

    # Air boost (high shelf — openness)
    air_freq_hz: float = 10000.0
    air_gain_db: float = 2.0
    air_q: float = 0.7

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
) -> MasteringParams:
    """Map semantic sliders (0-100) to raw mastering parameters.

    50 = current defaults. Below 50 = less effect. Above 50 = more effect.
    """

    def lerp(low: float, mid: float, high: float, value: float) -> float:
        """Linear interpolation: 0→low, 50→mid, 100→high."""
        if value <= 50:
            return low + (mid - low) * (value / 50)
        return mid + (high - mid) * ((value - 50) / 50)

    return MasteringParams(
        # Warmth: controls low-mid cut and air boost (inverse relationship)
        low_mid_gain_db=lerp(0.0, -3.0, -6.0, 100 - warmth),  # more warmth = less cut
        air_gain_db=lerp(4.0, 2.0, 0.0, warmth),  # more warmth = less air

        # Clarity: controls presence boost and high shelf
        presence_gain_db=lerp(0.0, 2.5, 5.0, clarity),
        air_freq_hz=lerp(12000, 10000, 8000, clarity),  # more clarity = lower air shelf

        # Compression: controls both compressor stages
        comp1_threshold_db=lerp(-10.0, -20.0, -30.0, compression),
        comp1_ratio=lerp(1.5, 2.0, 4.0, compression),
        comp2_threshold_db=lerp(-8.0, -15.0, -25.0, compression),
        comp2_ratio=lerp(1.5, 3.0, 6.0, compression),

        # De-ess: controls sibilance reduction
        deess_gain_db=lerp(0.0, -4.0, -8.0, de_ess),

        # Loudness: controls LUFS target
        lufs_target=lerp(-14.0, -18.0, -24.0, 100 - loudness),  # more loudness = higher LUFS
        limiter_ceiling_db=lerp(-0.5, -1.5, -3.0, 100 - loudness),
    )
