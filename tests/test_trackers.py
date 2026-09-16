"""Pitch trackers behind one interface, so they can be compared on evidence.

The corpus committed to a tracker once, early, on a comparison against exactly
one alternative. "Best accuracy wins" needs more than that: several independent
implementations, measured against signals whose true pitch is known, and then
against each other on real audio where it is not.

Every tracker here returns the same thing — a frame track at a common hop, 0.0
for unvoiced — so nothing downstream has to know which one produced it.

Tests use synthetic tones only, per the repo rule: no Cover audio in tests.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vvc import trackers


def _tone(f0: float, seconds: float = 2.0, sr: int = 22050, harmonics: int = 6) -> np.ndarray:
    """A harmonic-rich tone — a bare sine is easier to track than any voice,
    so it would not distinguish the trackers at all."""
    t = np.arange(int(seconds * sr)) / sr
    wave = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, harmonics + 1))
    return (0.3 * wave / np.max(np.abs(wave))).astype(np.float32)


@pytest.fixture
def tone_file(tmp_path):
    def _write(f0: float, **kwargs):
        import soundfile as sf

        path = tmp_path / f"tone_{f0:.0f}.wav"
        sf.write(path, _tone(f0, **kwargs), 22050)
        return path

    return _write


class TestRegistry:
    def test_every_registered_tracker_is_callable_by_name(self):
        assert trackers.TRACKERS, "no trackers registered"
        for name, fn in trackers.TRACKERS.items():
            assert callable(fn), f"{name} is not callable"

    def test_the_production_v1_configuration_is_registered_as_the_baseline(self):
        """A bake-off without the incumbent measures nothing."""
        assert "praat_ac" in trackers.TRACKERS


class TestPitchTrack:
    def test_median_ignores_unvoiced_frames(self):
        track = trackers.PitchTrack(
            freqs=np.array([0.0, 200.0, 0.0, 300.0]), hop_s=0.02, tracker="t", params={}
        )
        assert track.median_f0 == pytest.approx(250.0)

    def test_voiced_fraction_counts_every_frame(self):
        track = trackers.PitchTrack(
            freqs=np.array([0.0, 200.0, 0.0, 300.0]), hop_s=0.02, tracker="t", params={}
        )
        assert track.voiced_fraction == pytest.approx(0.5)

    def test_an_all_unvoiced_track_reports_nan_not_zero(self):
        """Zero Hz is a claim about a voice. Silence is the absence of one."""
        track = trackers.PitchTrack(
            freqs=np.zeros(10), hop_s=0.02, tracker="t", params={}
        )
        assert math.isnan(track.median_f0)

    def test_spread_is_reported_in_semitones_so_voices_compare(self):
        """200->400 Hz is one octave whatever the register it sits in."""
        freqs = np.array([200.0] * 25 + [400.0] * 25)
        track = trackers.PitchTrack(freqs=freqs, hop_s=0.02, tracker="t", params={})
        assert track.iqr_semitones == pytest.approx(12.0, abs=0.01)

    def test_low_frequency_mass_measures_contamination_below_a_band(self):
        """The real failure in this corpus is non-target audio, not octave
        errors: a clip carrying another speaker shows voiced mass far below
        the talent's own range."""
        freqs = np.array([100.0] * 30 + [300.0] * 70)
        track = trackers.PitchTrack(freqs=freqs, hop_s=0.02, tracker="t", params={})
        assert track.fraction_below(160.0) == pytest.approx(0.30)


@pytest.mark.parametrize("name", sorted(trackers.TRACKERS))
@pytest.mark.parametrize("f0", [180.0, 330.0])
def test_each_tracker_recovers_a_known_pitch(name, f0, tone_file):
    """The accuracy criterion. A tracker that cannot find a synthetic tone
    cannot be trusted on a voice."""
    track = trackers.TRACKERS[name](tone_file(f0), floor=60.0, ceiling=800.0)
    assert track.voiced_fraction > 0.5, f"{name} found almost no voiced frames"
    assert track.median_f0 == pytest.approx(f0, rel=0.03), (
        f"{name} reported {track.median_f0:.1f} Hz for a {f0} Hz tone"
    )


@pytest.mark.parametrize("name", sorted(trackers.TRACKERS))
def test_each_tracker_reports_its_identity_and_parameters(name, tone_file):
    """A measurement whose configuration is not recorded cannot be reproduced,
    and a threshold derived from a tracker parameter cannot be re-derived."""
    track = trackers.TRACKERS[name](tone_file(220.0), floor=60.0, ceiling=800.0)
    assert track.tracker == name
    assert track.params.get("floor") == 60.0


class TestCrepeDeviceSelection:
    """CREPE is only worth its cost with acceleration. The separator already
    picks CUDA/MPS/CoreML per platform (see isolate.py); the tracker's own
    device choice must not leave Apple Silicon stuck on CPU by omission."""

    def test_prefers_cuda_when_available(self):
        assert trackers._select_device(cuda_available=True, mps_available=True) == "cuda"

    def test_falls_back_to_mps_on_apple_silicon(self):
        assert (
            trackers._select_device(cuda_available=False, mps_available=True) == "mps"
        )

    def test_falls_back_to_cpu_when_no_accelerator_is_available(self):
        assert (
            trackers._select_device(cuda_available=False, mps_available=False)
            == "cpu"
        )


@pytest.mark.parametrize("name", sorted(trackers.TRACKERS))
def test_each_tracker_returns_a_track_on_silence_without_raising(name, tmp_path):
    """Silence is ordinary input here — clips get windowed on imperfect audio."""
    import soundfile as sf

    path = tmp_path / "silence.wav"
    sf.write(path, np.zeros(22050, dtype=np.float32), 22050)
    track = trackers.TRACKERS[name](path, floor=60.0, ceiling=800.0)
    assert track.voiced_fraction < 0.2
