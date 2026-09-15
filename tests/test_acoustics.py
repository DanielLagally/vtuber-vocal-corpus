"""The v2 feature extractor: one pitch track, gated features, recorded config.

Every defect this fixes was real in v1 and is documented in
docs/v1/LIMITATIONS.md:

- formants averaged over unvoiced frames, so consonants, breath and silence sat
  inside every published F1-F4;
- the spectral centroid computed after an un-anti-aliased downsample, folding
  everything above the new Nyquist straight into the statistic, and averaged
  over silence as well as speech;
- the pitch tracker re-run once per feature, which is why a full re-measure
  cost seconds per clip;
- spread published in Hz, which is a weaker constraint on a high voice than a
  low one;
- no record of which tracker or parameters produced a number.

v1's extractor is untouched: it still has to reproduce the published corpus.

Tests use synthetic audio only, per the repo rule.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vvc import acoustics, trackers


def _voice(f0: float, sr: int, seconds: float, harmonics: int = 12) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    wave = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, harmonics + 1))
    return wave / np.max(np.abs(wave))


def _write(path, data, sr):
    import soundfile as sf

    sf.write(path, (0.3 * data / np.max(np.abs(data))).astype(np.float32), sr)
    return path


@pytest.fixture
def voiced_then_hiss(tmp_path):
    """Half harmonic voice, half bright broadband hiss.

    The hiss is deliberately much brighter than the voice: an ungated average
    is pulled toward it, a gated one is not. That difference is the test.
    """
    sr = 22050
    voiced = _voice(220.0, sr, 2.0)
    rng = np.random.default_rng(0)
    hiss = rng.standard_normal(int(2.0 * sr))
    hiss = hiss / np.max(np.abs(hiss))
    return _write(tmp_path / "half.wav", np.concatenate([voiced, hiss]), sr)


@pytest.fixture
def voice_only(tmp_path):
    sr = 22050
    return _write(tmp_path / "voice.wav", _voice(220.0, sr, 2.0), sr)


class TestSingleExtraction:
    def test_accepts_a_precomputed_pitch_track(self, voice_only):
        """v1 re-ran the tracker once per feature. The track is an input."""
        track = trackers.praat_ac(voice_only, floor=60.0, ceiling=800.0)
        result = acoustics.clip_features(voice_only, pitch_track=track)
        assert result["median_f0"] == pytest.approx(track.median_f0, rel=1e-6)

    def test_reports_the_tracker_and_its_full_search_range(self, voice_only):
        """A gate derived from a tracker parameter must be re-derivable."""
        result = acoustics.clip_features(voice_only)
        assert result["tracker"] == acoustics.DEFAULT_CONFIG.tracker
        assert result["tracker_floor_hz"] == acoustics.DEFAULT_CONFIG.floor
        assert result["tracker_ceiling_hz"] == acoustics.DEFAULT_CONFIG.ceiling


class TestPitchFeatures:
    def test_stores_quartiles_not_just_a_spread(self, voice_only):
        """Stored quartiles let a gate be re-derived without re-reading audio."""
        result = acoustics.clip_features(voice_only)
        assert result["f0_q25"] <= result["f0_q50"] <= result["f0_q75"]

    def test_spread_is_reported_in_semitones(self, voice_only):
        result = acoustics.clip_features(voice_only)
        assert "f0_iqr_semitones" in result

    def test_reports_voiced_and_total_frame_counts(self, voice_only):
        result = acoustics.clip_features(voice_only)
        assert 0 < result["voiced_frames"] <= result["total_frames"]

    def test_silence_yields_nan_never_zero(self, tmp_path):
        import soundfile as sf

        path = tmp_path / "silence.wav"
        sf.write(path, np.zeros(22050, dtype=np.float32), 22050)
        result = acoustics.clip_features(path)
        assert math.isnan(result["median_f0"])


class TestVoicedGating:
    def test_formants_are_gated_to_voiced_frames(self, voiced_then_hiss, voice_only):
        """The v1 defect. Ungated, the hiss half drags the formant average;
        gated, the answer stays close to the voice-only reading."""
        gated = acoustics.clip_features(voiced_then_hiss)["f1_hz"]
        ungated = acoustics.clip_features(voiced_then_hiss, gate_formants=False)["f1_hz"]
        reference = acoustics.clip_features(voice_only)["f1_hz"]
        assert abs(gated - reference) < abs(ungated - reference)

    def test_records_how_many_frames_the_gate_kept(self, voiced_then_hiss):
        """Gating that cannot be audited is a claim, not a method."""
        result = acoustics.clip_features(voiced_then_hiss)
        assert result["formant_voiced_frames"] < result["formant_total_frames"]

    def test_brightness_is_gated_to_voiced_frames(self, voiced_then_hiss, voice_only):
        gated = acoustics.clip_features(voiced_then_hiss)["brightness_hz"]
        ungated = acoustics.clip_features(voiced_then_hiss, gate_brightness=False)["brightness_hz"]
        reference = acoustics.clip_features(voice_only)["brightness_hz"]
        assert abs(gated - reference) < abs(ungated - reference)


class TestAntiAliasing:
    """v1 resampled by plain interpolation, so a 10 kHz component reappeared at
    6 kHz after downsampling to 16 kHz — and the spectral centroid *is* a
    weighted mean frequency, so that ghost was added straight to the answer.

    Rejection is never total; the meaningful question is how much survives
    relative to a legitimate in-band signal of the same amplitude.
    """

    @staticmethod
    def _peak(path, target_sr=16_000):
        audio, out_sr = acoustics.load_mono(path, target_sr=target_sr)
        spectrum = np.abs(np.fft.rfft(audio * np.hanning(audio.size)))
        freqs = np.fft.rfftfreq(audio.size, 1.0 / out_sr)
        return float(spectrum.max()), float(freqs[np.argmax(spectrum)])

    def test_out_of_band_energy_is_strongly_suppressed(self, tmp_path):
        sr = 44100
        t = np.arange(int(1.0 * sr)) / sr
        in_band = _write(tmp_path / "in.wav", np.sin(2 * np.pi * 3_000 * t), sr)
        out_band = _write(tmp_path / "out.wav", np.sin(2 * np.pi * 10_000 * t), sr)

        in_peak, in_freq = self._peak(in_band)
        out_peak, _ = self._peak(out_band)

        assert 2_900 < in_freq < 3_100, "an in-band tone must survive intact"
        assert out_peak < in_peak / 100.0, (
            f"out-of-band energy only {in_peak / max(out_peak, 1e-12):.0f}x down; "
            "the resample is not anti-aliased"
        )

    def test_plain_interpolation_would_fail_this(self, tmp_path):
        """Proves the test discriminates: v1's method aliases badly here."""
        sr = 44100
        t = np.arange(int(1.0 * sr)) / sr
        tone = (0.3 * np.sin(2 * np.pi * 10_000 * t)).astype(np.float32)

        target_sr = 16_000
        naive = np.interp(
            np.arange(0, tone.size, sr / target_sr), np.arange(tone.size), tone
        )
        spectrum = np.abs(np.fft.rfft(naive * np.hanning(naive.size)))
        freqs = np.fft.rfftfreq(naive.size, 1.0 / target_sr)
        alias_peak = float(spectrum.max())
        alias_freq = float(freqs[np.argmax(spectrum)])

        assert 5_500 < alias_freq < 6_500, "expected the 10 kHz tone to fold to ~6 kHz"

        out_band = _write(tmp_path / "out.wav", np.sin(2 * np.pi * 10_000 * t), sr)
        filtered_peak, _ = self._peak(out_band)
        assert filtered_peak < alias_peak / 50.0, (
            "the anti-aliased path must suppress what plain interpolation lets through"
        )

    def test_a_tone_inside_the_band_survives_resampling(self, tmp_path):
        sr = 44100
        t = np.arange(int(1.0 * sr)) / sr
        path = _write(tmp_path / "lf.wav", np.sin(2 * np.pi * 1_000 * t), sr)
        audio, out_sr = acoustics.load_mono(path, target_sr=16_000)
        spectrum = np.abs(np.fft.rfft(audio * np.hanning(audio.size)))
        freqs = np.fft.rfftfreq(audio.size, 1.0 / out_sr)
        assert 900 < freqs[np.argmax(spectrum)] < 1_100


class TestContaminationSignal:
    def test_reports_the_share_of_voiced_frames_below_a_low_band(self, tmp_path):
        """The real failure in this corpus is another speaker in the clip, not
        an octave error. A low-frequency voiced mass is what reveals it."""
        sr = 22050
        mixed = np.concatenate([_voice(110.0, sr, 1.0), _voice(330.0, sr, 2.0)])
        path = _write(tmp_path / "two.wav", mixed, sr)
        result = acoustics.clip_features(path)
        assert 0.0 < result["fraction_below_160hz"] < 1.0


class TestConfig:
    def test_config_round_trips_into_the_record(self, voice_only):
        config = acoustics.FeatureConfig(tracker="praat_ac", floor=80.0, ceiling=500.0)
        result = acoustics.clip_features(voice_only, config=config)
        assert result["tracker_floor_hz"] == 80.0
        assert result["tracker_ceiling_hz"] == 500.0
        assert result["formant_ceiling_hz"] == config.formant_ceiling_hz
