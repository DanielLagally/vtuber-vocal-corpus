"""Product tests for the Praat-backed tracker (tracker diagnostic,
PLAN "Why the Luna series disappointed" lever 4).

User-visible rules (synthetic audio ONLY — tones/silence; never Cover/
hololive audio, never downloads):

1. Median F0 on a clean tone lands within 5 Hz of the synthesized
   frequency, at two different pitches (matches vanalysis.features'
   own 220 Hz tolerance so the two trackers are comparable).
2. F0 IQR on a clean, steady tone is near zero (Praat should not itself
   manufacture the kind of large within-clip spread the numpy tracker
   flags as junk).
3. Silence returns math.nan for median_f0, never an invented Hz value
   (same no-bogus-confidence rule as vanalysis.features).
4. stem_features(path) returns median_f0, f0_iqr, voiced_fraction,
   brightness_hz, dynamism_semitones.
5. dynamism_semitones = mean absolute semitone change between
   consecutive VOICED frames. A steady tone is near 0 (no frame-to-
   frame movement). Critically, a silence GAP between two differently-
   pitched voiced blocks must NOT count as a jump — only frame pairs
   that are BOTH voiced (and therefore adjacent in time, not just
   adjacent after unvoiced frames are filtered out) contribute.
"""

import array
import math
import random
import sys
import wave
from pathlib import Path

import pytest

from vanalysis import praat_features

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
        "dynamism_semitones",
    }
    assert math.isfinite(features["median_f0"])
    assert 0.0 <= features["voiced_fraction"] <= 1.0
    assert math.isfinite(features["brightness_hz"])
    assert math.isfinite(features["dynamism_semitones"])


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
