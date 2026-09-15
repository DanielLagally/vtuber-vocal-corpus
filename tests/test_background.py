"""How much non-target sound a clip actually contained.

The separator emits a vocals stem and a residual stem, so their energy ratio is
a direct per-clip measure of competing sound — music, game audio, a film being
watched, a second speaker. The corpus already has both stems on disk, so this
costs nothing beyond reading them.

It answers three things nothing else can:

- whether a clip needed separating at all, since separating clean speech only
  adds error (see reference/measurement.md);
- which clips carry the contamination that produces implausible pitch readings;
- whether a stream type is actually noisy, instead of assuming it. Watchalongs
  were treated as the worst case on no evidence; many carry no background audio
  at all.

Tests use synthetic audio only.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vvc import background


def _write(path, amplitude: float, sr: int = 16000, seconds: float = 1.0, freq: float = 200.0):
    import soundfile as sf

    t = np.arange(int(seconds * sr)) / sr
    sf.write(path, (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr)
    return path


class TestBackgroundRatio:
    def test_equal_energy_stems_report_zero_db(self, tmp_path):
        vocals = _write(tmp_path / "v.wav", 0.5)
        other = _write(tmp_path / "o.wav", 0.5, freq=90.0)
        assert background.background_ratio_db(vocals, other) == pytest.approx(0.0, abs=0.5)

    def test_a_quiet_residual_reports_a_negative_ratio(self, tmp_path):
        """Voice dominant = clean clip = separation had little to remove."""
        vocals = _write(tmp_path / "v.wav", 0.5)
        other = _write(tmp_path / "o.wav", 0.05, freq=90.0)
        assert background.background_ratio_db(vocals, other) == pytest.approx(-20.0, abs=1.0)

    def test_a_loud_residual_reports_a_positive_ratio(self, tmp_path):
        vocals = _write(tmp_path / "v.wav", 0.05)
        other = _write(tmp_path / "o.wav", 0.5, freq=90.0)
        assert background.background_ratio_db(vocals, other) == pytest.approx(20.0, abs=1.0)

    def test_a_silent_vocal_stem_reports_nan_not_a_huge_ratio(self, tmp_path):
        """No voice is a QC problem, not an infinitely noisy clip."""
        import soundfile as sf

        silent = tmp_path / "v.wav"
        sf.write(silent, np.zeros(16000, dtype=np.float32), 16000)
        other = _write(tmp_path / "o.wav", 0.5)
        assert math.isnan(background.background_ratio_db(silent, other))

    def test_a_missing_residual_stem_reports_nan_rather_than_clean(self, tmp_path):
        """Absent evidence is not evidence of a clean clip."""
        vocals = _write(tmp_path / "v.wav", 0.5)
        assert math.isnan(background.background_ratio_db(vocals, tmp_path / "missing.wav"))


class TestLevel:
    def test_measures_the_whole_file(self, tmp_path):
        """Bursty audio is the normal case for a residual stem: long
        near-silence broken by loud passages. Estimating the level from part of
        the clip misreads exactly those, and this level decides which audio
        gets measured, so the cheap estimate is not a safe trade."""
        import numpy as np
        import soundfile as sf

        sr = 16000
        quiet = 0.001 * np.ones(int(9 * sr))
        loud = 0.5 * np.ones(int(1 * sr))
        path = tmp_path / "bursty.wav"
        sf.write(path, np.concatenate([quiet, loud]).astype("float32"), sr)

        expected = math.sqrt((9 * 0.001 ** 2 + 1 * 0.5 ** 2) / 10)
        assert background._rms(path) == pytest.approx(expected, rel=0.02)


class TestRatiosFor:
    def test_returns_a_ratio_for_every_requested_id(self, tmp_path):
        stems = tmp_path / "stems"
        stems.mkdir()
        _write(stems / "a_raw90_(vocals)_m.wav", 0.5)
        _write(stems / "a_raw90_(other)_m.wav", 0.05, freq=90.0)
        got = background.ratios_for(["a", "missing"], stems)
        assert set(got) == {"a", "missing"}
        assert got["a"] == pytest.approx(-20.0, abs=1.0)
        assert math.isnan(got["missing"])


class TestNeedsSeparation:
    def test_a_clip_dominated_by_voice_does_not_need_separation(self):
        assert background.needs_separation(-25.0, threshold_db=-15.0) is False

    def test_a_clip_with_substantial_background_does(self):
        assert background.needs_separation(-5.0, threshold_db=-15.0) is True

    def test_an_unknown_ratio_defaults_to_separating(self):
        """When we cannot tell, do what v1 did. Silently skipping separation on
        an unmeasured clip would be a change nobody chose."""
        assert background.needs_separation(math.nan, threshold_db=-15.0) is True


class TestStemPaths:
    def test_finds_the_residual_stem_beside_the_vocal_stem(self, tmp_path):
        stems = tmp_path / "stems_fast"
        stems.mkdir()
        (stems / "vid1_raw90_(vocals)_model.wav").touch()
        other = stems / "vid1_raw90_(other)_model.wav"
        other.touch()
        assert background.other_stem_path("vid1", stems) == other

    def test_returns_none_when_there_is_no_residual_stem(self, tmp_path):
        stems = tmp_path / "stems_fast"
        stems.mkdir()
        (stems / "vid1_raw90_(vocals)_model.wav").touch()
        assert background.other_stem_path("vid1", stems) is None

    def test_a_cached_listing_can_be_invalidated_after_new_stems_are_written(
        self, tmp_path
    ):
        """Listings are cached per process because globbing a directory of tens
        of thousands of stems per lookup is the dominant cost of resolving a
        corpus. Anything that writes new stems must clear the cache."""
        stems = tmp_path / "stems_fast"
        stems.mkdir()
        assert background.other_stem_path("vid2", stems) is None

        (stems / "vid2_raw90_(other)_model.wav").touch()
        assert background.other_stem_path("vid2", stems) is None, "expected stale cache"

        background.clear_listing_cache()
        assert background.other_stem_path("vid2", stems) is not None
