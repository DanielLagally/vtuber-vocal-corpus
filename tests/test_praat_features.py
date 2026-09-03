"""Product tests for the Praat-backed tracker (tracker diagnostic,
PLAN "Why the Luna series disappointed" lever 4).

User-visible rules (synthetic audio ONLY — tones/silence; never Cover/
hololive audio, never downloads):

1. Median F0 on a clean tone lands within 5 Hz of the synthesized
   frequency, at two different pitches (matches vvc.features'
   own 220 Hz tolerance so the two trackers are comparable).
2. F0 IQR on a clean, steady tone is near zero (Praat should not itself
   manufacture the kind of large within-clip spread the numpy tracker
   flags as junk).
3. Silence returns math.nan for median_f0, never an invented Hz value
   (same no-bogus-confidence rule as vvc.features).
4. stem_features(path) returns median_f0, f0_iqr, voiced_fraction,
   brightness_hz, dynamism_semitones, jitter_local, shimmer_local,
   hnr_db, loudness_dynamics_db.
5. dynamism_semitones = mean absolute semitone change between
   consecutive VOICED frames. A steady tone is near 0 (no frame-to-
   frame movement). Critically, a silence GAP between two differently-
   pitched voiced blocks must NOT count as a jump — only frame pairs
   that are BOTH voiced (and therefore adjacent in time, not just
   adjacent after unvoiced frames are filtered out) contribute.
6. jitter_local / shimmer_local / hnr_db (voice quality; PLAN caveat:
   calibrated for a sustained vowel, not conversational speech, and
   sensitive to residual vocal-isolation artifact — trust the relative
   comparison within this pipeline, not the absolute number against a
   clinical reference). A clean steady tone (a synthetic sine has
   essentially zero real cycle-to-cycle irregularity) has very low
   jitter/shimmer and very high HNR — this is a sanity floor, not a
   claim the numbers are clinically meaningful on real speech.
7. formants_hz(path) returns f1_hz/f2_hz/f3_hz/f4_hz — mean vocal-tract
   resonance frequency per formant, over frames where Praat's Burg
   tracker returns a defined value (undefined frames excluded, never
   averaged in as 0, same no-bogus-confidence convention as median_f0).
   On a synthetic two-formant "vowel" (glottal pulse train through two
   cascaded resonators at known F1/F2), the measured f1_hz/f2_hz land
   within 80 Hz of the synthesized targets, and f3_hz/f4_hz are still
   finite and strictly increasing (f1 < f2 < f3 < f4) — Praat reports
   formants in ascending-frequency order by construction. On true
   digital silence, every frame is undefined and all four are math.nan
   (unlike pitch tracking, Praat's LPC/Burg fit still finds *some* pole
   in ordinary low-amplitude noise, so this specifically needs zero
   samples, not the amp=4 dithered noise used elsewhere). stem_features
   includes all four alongside the existing nine keys.
"""

import array
import math
import random
import sys
import wave
from pathlib import Path

import pytest

from vvc import praat_features

SR = 16_000
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def _samples_to_bytes(samples: list[int]) -> bytes:
    buf = array.array("h", samples)
    if sys.byteorder == "big":
        buf.byteswap()
    return buf.tobytes()


def _write_wav(path: Path, samples: list[int], sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(_samples_to_bytes(samples))


def _sine(freq_hz: float, seconds: float, sr: int = SR, amp: float = 0.6) -> list[int]:
    peak = amp * 32767.0
    n = int(seconds * sr)
    fade = int(0.005 * sr)
    out = []
    for i in range(n):
        env = min(1.0, i / fade, (n - 1 - i) / fade)
        out.append(int(round(peak * env * math.sin(2.0 * math.pi * freq_hz * i / sr))))
    return out


def _noise(seconds: float, amp: float, rng: random.Random, sr: int = SR) -> list[int]:
    return [rng.randint(-amp, amp) for _ in range(int(seconds * sr))]


def _resonator(x, freq_hz: float, bandwidth_hz: float, sr: int = SR):
    """A single 2-pole resonant (all-pole) filter — the standard way to
    put a controllable vocal-tract-like resonance peak at freq_hz into a
    source signal. Cascading two of these onto a glottal pulse train
    synthesizes a minimal two-formant 'vowel' with known F1/F2."""
    import numpy as np
    from scipy.signal import lfilter

    r = math.exp(-math.pi * bandwidth_hz / sr)
    theta = 2.0 * math.pi * freq_hz / sr
    a1 = 2.0 * r * math.cos(theta)
    a2 = -r * r
    return lfilter([1.0 - a1 - a2], [1.0, -a1, -a2], x)


def _synthetic_vowel(f0: float, f1: float, f2: float, seconds: float, sr: int = SR) -> list[int]:
    """Glottal pulse train (rich harmonic source) at f0, filtered through
    cascaded resonators at f1/f2 — genuine, controllable formant
    structure, unlike a pure sine (which has none)."""
    import numpy as np

    n = int(seconds * sr)
    period = sr / f0
    source = np.zeros(n)
    t = 0.0
    while t < n:
        idx = int(round(t))
        if idx < n:
            source[idx] = 1.0
        t += period
    shaped = _resonator(_resonator(source, f1, 80.0, sr), f2, 100.0, sr)
    peak = float(np.max(np.abs(shaped))) + 1e-9
    samples = (shaped / peak * 0.8 * 32767.0).astype(np.int16)
    return samples.tolist()


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    rng = random.Random(0x5EED)
    _write_wav(FIXTURES_DIR / "praat_tone220.wav", _sine(220.0, 1.0))
    _write_wav(FIXTURES_DIR / "praat_tone330.wav", _sine(330.0, 1.0))
    _write_wav(FIXTURES_DIR / "praat_silence.wav", _noise(1.0, amp=4, rng=rng))
    _write_wav(
        FIXTURES_DIR / "praat_gap_220_330.wav",
        _sine(220.0, 0.15) + [0] * int(0.15 * SR) + _sine(330.0, 0.15),
    )
    _write_wav(
        FIXTURES_DIR / "praat_vowel_f1_500_f2_1500.wav",
        _synthetic_vowel(f0=150.0, f1=500.0, f2=1500.0, seconds=1.0),
    )
    # Literal zero samples, not the amp=4 dithered noise praat_silence.wav
    # uses: Praat's Burg/LPC formant fit still finds *some* pole in random
    # noise (unlike pitch tracking, which needs real periodicity and
    # correctly comes back undefined on either fixture) — true digital
    # silence is what actually drives every formant frame undefined.
    _write_wav(FIXTURES_DIR / "praat_pure_zero_silence.wav", [0] * SR)
    return FIXTURES_DIR


def test_median_f0_220hz_tone_within_5hz(fixtures_dir: Path) -> None:
    f0 = praat_features.median_f0(fixtures_dir / "praat_tone220.wav")
    assert math.isfinite(f0)
    assert abs(f0 - 220.0) < 5.0, f"median F0 {f0} not within 5 Hz of 220"


def test_median_f0_330hz_tone_within_5hz(fixtures_dir: Path) -> None:
    f0 = praat_features.median_f0(fixtures_dir / "praat_tone330.wav")
    assert math.isfinite(f0)
    assert abs(f0 - 330.0) < 5.0, f"median F0 {f0} not within 5 Hz of 330"


def test_f0_iqr_steady_tone_near_zero(fixtures_dir: Path) -> None:
    iqr = praat_features.f0_iqr(fixtures_dir / "praat_tone220.wav")
    assert math.isfinite(iqr)
    assert iqr < 5.0, f"steady-tone IQR {iqr} should be near zero"


def test_median_f0_silence_returns_nan(fixtures_dir: Path) -> None:
    f0 = praat_features.median_f0(fixtures_dir / "praat_silence.wav")
    assert isinstance(f0, float)
    assert math.isnan(f0), f"median_f0 on silence must be nan, got {f0!r}"


def test_stem_features_shape(fixtures_dir: Path) -> None:
    features = praat_features.stem_features(fixtures_dir / "praat_tone220.wav")
    assert set(features) == {
        "median_f0", "f0_iqr", "voiced_fraction", "brightness_hz",
        "dynamism_semitones", "jitter_local", "shimmer_local", "hnr_db",
        "loudness_dynamics_db", "f1_hz", "f2_hz", "f3_hz", "f4_hz",
    }
    assert math.isfinite(features["median_f0"])
    assert 0.0 <= features["voiced_fraction"] <= 1.0
    assert math.isfinite(features["brightness_hz"])
    assert math.isfinite(features["dynamism_semitones"])
    assert math.isfinite(features["jitter_local"])
    assert math.isfinite(features["shimmer_local"])
    assert math.isfinite(features["hnr_db"])
    assert math.isfinite(features["loudness_dynamics_db"])


def test_dynamism_steady_tone_near_zero(fixtures_dir: Path) -> None:
    dynamism = praat_features.dynamism_semitones(fixtures_dir / "praat_tone220.wav")
    assert math.isfinite(dynamism)
    assert dynamism < 0.3, f"steady-tone dynamism {dynamism} should be near zero"


def test_dynamism_does_not_count_a_silence_gap_as_a_jump(fixtures_dir: Path) -> None:
    """Rule 5: a 220 Hz block, a silence gap, then a 330 Hz block — the
    ~7-semitone jump BETWEEN the blocks must never be counted (it
    spans an unvoiced gap, not two adjacent voiced frames). Each block
    is individually steady, so the correct result stays near zero."""
    dynamism = praat_features.dynamism_semitones(fixtures_dir / "praat_gap_220_330.wav")
    assert math.isfinite(dynamism)
    assert dynamism < 0.3, (
        f"dynamism {dynamism} suggests the cross-gap 220->330 Hz jump was "
        "counted; only within-block frame pairs may contribute"
    )


def test_formants_synthetic_vowel_f1_f2_within_80hz(fixtures_dir: Path) -> None:
    """Rule 7: measured F1/F2 land close to the synthesized targets on a
    genuine two-formant synthetic vowel (a pure sine has no resonance
    structure at all, so this needs its own fixture)."""
    formants = praat_features.formants_hz(fixtures_dir / "praat_vowel_f1_500_f2_1500.wav")
    assert math.isfinite(formants["f1_hz"])
    assert math.isfinite(formants["f2_hz"])
    assert abs(formants["f1_hz"] - 500.0) < 80.0, f"F1 {formants['f1_hz']} not within 80 Hz of 500"
    assert abs(formants["f2_hz"] - 1500.0) < 80.0, f"F2 {formants['f2_hz']} not within 80 Hz of 1500"


def test_formants_ascending_and_finite(fixtures_dir: Path) -> None:
    """Rule 7: Praat reports formants in ascending-frequency order by
    construction — F3/F4 aren't independently controlled by this
    fixture's two resonators, but they must still be present and
    strictly increasing past F1 < F2."""
    formants = praat_features.formants_hz(fixtures_dir / "praat_vowel_f1_500_f2_1500.wav")
    assert math.isfinite(formants["f3_hz"])
    assert math.isfinite(formants["f4_hz"])
    assert formants["f1_hz"] < formants["f2_hz"] < formants["f3_hz"] < formants["f4_hz"]


def test_formants_silence_returns_nan(fixtures_dir: Path) -> None:
    """Rule 7: every frame is undefined on true digital silence — no
    bogus confidence, same convention as median_f0 on silence. (Ordinary
    low-amplitude noise isn't quiet enough: Praat's LPC/Burg fit still
    finds some pole in it, unlike pitch tracking's need for real
    periodicity — see praat_pure_zero_silence.wav's fixture comment.)"""
    formants = praat_features.formants_hz(fixtures_dir / "praat_pure_zero_silence.wav")
    assert math.isnan(formants["f1_hz"])
    assert math.isnan(formants["f2_hz"])
    assert math.isnan(formants["f3_hz"])
    assert math.isnan(formants["f4_hz"])


def test_voice_quality_clean_tone_low_jitter_shimmer_high_hnr(fixtures_dir: Path) -> None:
    """Rule 6: a synthetic sine has essentially zero real cycle-to-cycle
    irregularity — low jitter/shimmer, high HNR is a sanity floor
    (does not claim these are clinically meaningful on real speech)."""
    jitter = praat_features.jitter_local(fixtures_dir / "praat_tone220.wav")
    shimmer = praat_features.shimmer_local(fixtures_dir / "praat_tone220.wav")
    hnr = praat_features.hnr_db(fixtures_dir / "praat_tone220.wav")
    assert math.isfinite(jitter) and jitter < 0.01, f"jitter {jitter} too high for a clean tone"
    assert math.isfinite(shimmer) and shimmer < 0.01, f"shimmer {shimmer} too high for a clean tone"
    assert math.isfinite(hnr) and hnr > 40.0, f"HNR {hnr} too low for a clean tone"
