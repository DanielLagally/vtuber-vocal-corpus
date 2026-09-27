"""Voice-source and timbre features: what makes a voice airy, full, or fast.

Pitch and formants say how high a voice is and roughly how long the vocal
tract is; they cannot tell a breathy, "fluffy" voice from a pressed, full one
at the same pitch. These features can, and each test below states the
analytic answer a synthetic signal must produce:

- a harmonic series whose k-th harmonic has amplitude k^-p has H1-H2 of
  20*log10(2^p) = 6.02p dB, and a harmonic tilt of -6.02p dB per octave;
- adding aspiration noise lowers cepstral peak prominence (CPP);
- a steeper tilt moves energy below 1 kHz, lowering the alpha ratio;
- a resonance sitting on H2 distorts raw H1-H2, and the formant correction
  undoes it;
- syllable-rate amplitude modulation sets the speaking rate.

Synthetic signals only.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vvc import voice_source

SR = 16_000
HOP = 0.02


def _harmonics(f0: float, p: float, seconds: float = 2.0, noise: float = 0.0, seed: int = 0):
    t = np.arange(int(seconds * SR)) / SR
    wave = np.zeros_like(t)
    k = 1
    while k * f0 < SR / 2 - 200:
        wave += np.sin(2 * np.pi * f0 * k * t) * k ** (-p)
        k += 1
    wave /= np.max(np.abs(wave))
    if noise:
        rng = np.random.default_rng(seed)
        wave = wave + noise * rng.standard_normal(t.size)
    return wave.astype(np.float64)


def _grid(audio, f0: float, voiced: bool = True):
    times = np.arange(0.05, audio.size / SR - 0.05, HOP)
    freqs = np.full(times.size, f0 if voiced else 0.0)
    return times, freqs, freqs > 0


def _features(audio, f0, **kwargs):
    times, freqs, voiced = _grid(audio, f0)
    return voice_source.voice_source_features(audio, SR, times, freqs, voiced, **kwargs)


def _resonate(audio, freq: float, bandwidth: float):
    """Second-order digital resonator (one formant), unity gain at DC."""
    r = math.exp(-math.pi * bandwidth / SR)
    theta = 2 * math.pi * freq / SR
    a1, a2 = -2 * r * math.cos(theta), r * r
    gain = 1 + a1 + a2
    out = np.zeros_like(audio)
    y1 = y2 = 0.0
    for i, x in enumerate(audio):
        y = gain * x - a1 * y1 - a2 * y2
        out[i] = y
        y2, y1 = y1, y
    return out


class TestHarmonicAmplitudes:
    @pytest.mark.parametrize("p", [1.0, 2.0])
    def test_h1_h2_matches_the_source_rolloff(self, p):
        result = _features(_harmonics(200.0, p), 200.0)
        assert result["h1h2_raw_db"] == pytest.approx(6.02 * p, abs=0.5)

    @pytest.mark.parametrize("p", [1.0, 2.0])
    def test_harmonic_tilt_matches_the_source_rolloff(self, p):
        result = _features(_harmonics(200.0, p), 200.0)
        assert result["harmonic_tilt_db_per_octave"] == pytest.approx(-6.02 * p, abs=0.7)

    def test_a_formant_on_h2_is_corrected_back_to_the_source(self):
        f0, p = 250.0, 1.0
        filtered = _resonate(_harmonics(f0, p), freq=500.0, bandwidth=80.0)
        times, freqs, voiced = _grid(filtered, f0)
        n = times.size
        result = voice_source.voice_source_features(
            filtered,
            SR,
            times,
            freqs,
            voiced,
            formants={1: np.full(n, 500.0), 2: np.full(n, 2500.0)},
            bandwidths={1: np.full(n, 80.0), 2: np.full(n, 150.0)},
        )
        # Raw H1-H2 is dragged far below the source's 6 dB by the boosted H2 ...
        assert result["h1h2_raw_db"] < 6.02 * p - 5.0
        # ... and the correction recovers the source value.
        assert result["h1h2_db"] == pytest.approx(6.02 * p, abs=1.5)

    def test_reports_how_often_h2_sits_near_f1(self):
        f0 = 250.0
        audio = _harmonics(f0, 1.0)
        times, freqs, voiced = _grid(audio, f0)
        n = times.size
        near = voice_source.voice_source_features(
            audio, SR, times, freqs, voiced,
            formants={1: np.full(n, 520.0), 2: np.full(n, 2500.0)},
            bandwidths={1: np.full(n, 80.0), 2: np.full(n, 150.0)},
        )
        far = voice_source.voice_source_features(
            audio, SR, times, freqs, voiced,
            formants={1: np.full(n, 900.0), 2: np.full(n, 2500.0)},
            bandwidths={1: np.full(n, 80.0), 2: np.full(n, 150.0)},
        )
        assert near["h2_near_f1_fraction"] == pytest.approx(1.0)
        assert far["h2_near_f1_fraction"] == pytest.approx(0.0)

    def test_without_formants_the_corrected_value_is_absent_not_copied(self):
        result = _features(_harmonics(200.0, 1.0), 200.0)
        assert math.isnan(result["h1h2_db"])


class TestNoiseAndTilt:
    def test_aspiration_noise_lowers_cpp(self):
        clean = _features(_harmonics(220.0, 1.0), 220.0)
        breathy = _features(_harmonics(220.0, 1.0, noise=0.3), 220.0)
        assert breathy["cpp_db"] < clean["cpp_db"] - 3.0

    def test_steeper_tilt_lowers_the_alpha_ratio(self):
        full = _features(_harmonics(220.0, 0.5), 220.0)
        soft = _features(_harmonics(220.0, 2.0), 220.0)
        assert soft["alpha_ratio_db"] < full["alpha_ratio_db"] - 6.0

    def test_steeper_tilt_raises_the_hammarberg_index(self):
        full = _features(_harmonics(220.0, 0.5), 220.0)
        soft = _features(_harmonics(220.0, 2.0), 220.0)
        assert soft["hammarberg_db"] > full["hammarberg_db"] + 6.0


class TestSpeakingRate:
    @pytest.mark.parametrize("rate", [3.0, 6.0])
    def test_syllable_rate_modulation_sets_the_speaking_rate(self, rate):
        seconds = 6.0
        base = _harmonics(200.0, 1.0, seconds=seconds)
        t = np.arange(base.size) / SR
        envelope = 0.5 * (1 - np.cos(2 * np.pi * rate * t))
        audio = base * envelope
        times, freqs, voiced = _grid(audio, 200.0)
        result = voice_source.voice_source_features(audio, SR, times, freqs, voiced)
        assert result["speaking_rate_syl_per_s"] == pytest.approx(rate, rel=0.2)


class TestGaps:
    def test_unvoiced_audio_yields_nan_never_zero(self):
        rng = np.random.default_rng(1)
        hiss = rng.standard_normal(2 * SR)
        times, freqs, voiced = _grid(hiss, 200.0, voiced=False)
        result = voice_source.voice_source_features(hiss, SR, times, freqs, voiced)
        for key in (
            "cpp_db", "h1h2_raw_db", "h1h2_db", "harmonic_tilt_db_per_octave",
            "alpha_ratio_db", "hammarberg_db", "speaking_rate_syl_per_s",
        ):
            assert math.isnan(result[key]), key

    def test_formant_dispersion_is_the_mean_spacing(self):
        audio = _harmonics(200.0, 1.0)
        times, freqs, voiced = _grid(audio, 200.0)
        n = times.size
        result = voice_source.voice_source_features(
            audio, SR, times, freqs, voiced,
            formants={1: np.full(n, 700.0), 2: np.full(n, 1700.0),
                      3: np.full(n, 2800.0), 4: np.full(n, 3700.0)},
        )
        assert result["formant_dispersion_hz"] == pytest.approx(1000.0)
