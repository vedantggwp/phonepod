"""Tests for the profile system and parameterized engine."""

import json
from pathlib import Path

import pytest

from cleanfeed.profile import MasteringParams, Profile, params_from_semantic


class TestMasteringParams:
    def test_defaults(self):
        p = MasteringParams()
        assert p.hpf_cutoff_hz == 80.0
        assert p.lufs_target == -18.0
        assert p.limiter_ceiling_db == -1.5
        assert p.comp1_ratio == 2.0

    def test_immutable(self):
        p = MasteringParams()
        with pytest.raises(AttributeError):
            p.hpf_cutoff_hz = 100.0  # type: ignore[misc]

    def test_custom_values(self):
        p = MasteringParams(hpf_cutoff_hz=120.0, lufs_target=-16.0)
        assert p.hpf_cutoff_hz == 120.0
        assert p.lufs_target == -16.0


class TestSemanticMapping:
    def test_defaults_at_50(self):
        p = params_from_semantic(50, 50, 50, 50, 50)
        assert p.presence_gain_db == 2.5
        assert p.deess_gain_db == -4.0
        assert p.lufs_target == -18.0

    def test_warmth_high(self):
        p = params_from_semantic(warmth=100, clarity=50, compression=50, de_ess=50, loudness=50)
        # High warmth = less low-mid cut (closer to 0) + less air
        assert p.low_mid_gain_db > -3.0  # less cut than default
        assert p.air_gain_db < 2.0       # less air than default

    def test_warmth_low(self):
        p = params_from_semantic(warmth=0, clarity=50, compression=50, de_ess=50, loudness=50)
        # Low warmth = more low-mid cut + more air
        assert p.low_mid_gain_db < -3.0
        assert p.air_gain_db > 2.0

    def test_compression_high(self):
        p = params_from_semantic(warmth=50, clarity=50, compression=100, de_ess=50, loudness=50)
        assert p.comp1_threshold_db < -20.0  # lower threshold = more compression
        assert p.comp1_ratio > 2.0

    def test_loudness_high(self):
        p = params_from_semantic(warmth=50, clarity=50, compression=50, de_ess=50, loudness=100)
        assert p.lufs_target > -18.0  # louder = higher LUFS (less negative)

    def test_all_zeros(self):
        p = params_from_semantic(0, 0, 0, 0, 0)
        assert isinstance(p, MasteringParams)

    def test_all_hundreds(self):
        p = params_from_semantic(100, 100, 100, 100, 100)
        assert isinstance(p, MasteringParams)


class TestProfile:
    def test_save_and_load(self, tmp_path):
        params = MasteringParams(hpf_cutoff_hz=120.0, lufs_target=-16.0)
        profile = Profile(name="test-voice", params=params)

        path = profile.save(path=tmp_path / "test-voice.json")
        assert path.exists()

        loaded = Profile.load(path)
        assert loaded.name == "test-voice"
        assert loaded.params.hpf_cutoff_hz == 120.0
        assert loaded.params.lufs_target == -16.0

    def test_save_creates_valid_json(self, tmp_path):
        profile = Profile(name="check", params=MasteringParams())
        path = profile.save(path=tmp_path / "check.json")

        data = json.loads(path.read_text())
        assert data["name"] == "check"
        assert "params" in data
        assert data["params"]["hpf_cutoff_hz"] == 80.0

    def test_load_by_name(self, tmp_path, monkeypatch):
        monkeypatch.setattr("cleanfeed.profile.PROFILES_DIR", tmp_path)

        profile = Profile(name="my-voice", params=MasteringParams(lufs_target=-16.0))
        profile.save(path=tmp_path / "my-voice.json")

        loaded = Profile.load_by_name("my-voice")
        assert loaded.params.lufs_target == -16.0

    def test_load_missing_profile_raises(self):
        with pytest.raises(FileNotFoundError, match="Profile not found"):
            Profile.load_by_name("nonexistent-profile-xyz")

    def test_list_profiles(self, tmp_path, monkeypatch):
        monkeypatch.setattr("cleanfeed.profile.PROFILES_DIR", tmp_path)

        Profile(name="a", params=MasteringParams()).save(path=tmp_path / "a.json")
        Profile(name="b", params=MasteringParams()).save(path=tmp_path / "b.json")

        names = Profile.list_profiles()
        assert names == ["a", "b"]


class TestEngineParams:
    @pytest.mark.slow
    def test_engine_accepts_params(self):
        from cleanfeed import Engine
        params = MasteringParams(lufs_target=-16.0, hpf_cutoff_hz=120.0)
        engine = Engine(params=params)
        assert engine.params.lufs_target == -16.0
        assert engine.params.hpf_cutoff_hz == 120.0

    @pytest.mark.slow
    def test_set_params_without_reload(self):
        from cleanfeed import Engine
        engine = Engine()
        original_lufs = engine.params.lufs_target

        new_params = MasteringParams(lufs_target=-14.0)
        engine.set_params(new_params)

        assert engine.params.lufs_target == -14.0
        assert engine.params.lufs_target != original_lufs

    @pytest.mark.slow
    def test_master_only(self, test_wav_48k):
        import numpy as np
        import soundfile as sf
        import torch
        from cleanfeed import Engine

        engine = Engine()

        # Get denoised audio through full pipeline stages 1+2
        audio, sr = sf.read(str(test_wav_48k), dtype="float32")
        tensor = torch.from_numpy(audio)
        full_result, _ = engine.enhance(tensor, sr)

        # Now test master_only with a simple array
        test_audio = np.random.randn(48000 * 5).astype(np.float32) * 0.1
        mastered = engine.master_only(test_audio)

        assert isinstance(mastered, np.ndarray)
        assert mastered.ndim == 1
        assert len(mastered) == len(test_audio)
        assert np.isfinite(mastered).all()
