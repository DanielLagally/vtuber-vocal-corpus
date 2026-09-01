"""Product tests for vanalysis v1 feature measurements.

User-visible rules (v1 measures only, synthetic audio ONLY — tones/noise;
never Cover/hololive audio, never downloads):

1. Median F0: a 220 Hz voiced-like tone (~1 s, 16 kHz mono wav) has a
   median F0 within 5 Hz of 220.
2. Silence / white noise: voiced fraction is near 0 (< 0.1), and the
   value is always in [0, 1].
3. Brightness: a lowpassed (dark) noise file has a strictly lower
   spectral centroid than a highpassed (bright) noise file.
4. No bogus confidence: median_f0 on silence must NOT invent an F0 —
   it returns math.nan when there are no voiced frames.

Fixtures are tiny wavs (16 kHz, mono, 16-bit, ~1 s) synthesized here with
the stdlib `wave` module into <repo>/fixtures/ — deterministic (seeded RNG),
no third-party test deps.
"""

import array
import math
import random
import sys
import wave
from pathlib import Path

import pytest

from vanalysis import features

SR = 16_000
NYQUIST = SR / 2
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"

# ---------------------------------------------------------------- synthesis


def _samples_to_bytes(samples: list[int]) -> bytes:
    buf = array.array("h", samples)
    if sys.byteorder == "big":
        buf.byteswap()
    return buf.tobytes()


def _write_wav(path: Path, samples: list[int], sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit PCM
        w.setframerate(sr)
        w.writeframes(_samples_to_bytes(samples))


def _sine(freq_hz: float, seconds: float, sr: int = SR, amp: float = 0.6) -> list[int]:
    peak = amp * 32767.0
    n = int(seconds * sr)
    fade = int(0.005 * sr)  # 5 ms fade to avoid clicks
    out = []
    for i in range(n):
        env = min(1.0, i / fade, (n - 1 - i) / fade)
        out.append(int(round(peak * env * math.sin(2.0 * math.pi * freq_hz * i / sr))))
    return out


def _noise(seconds: float, amp: float, rng: random.Random, sr: int = SR) -> list[int]:
    return [rng.randint(-amp, amp) for _ in range(int(seconds * sr))]


def _lowpassed(samples: list[int], cutoff_hz: float, sr: int = SR) -> list[int]:
    """One-pole IIR lowpass — the 'dark' fixture."""
    a = 1.0 - math.exp(-2.0 * math.pi * cutoff_hz / sr)
    out, y = [], 0.0
    for x in samples:
        y += a * (x - y)
        out.append(int(round(y)))
    return out


def _highpassed(samples: list[int]) -> list[int]:
    """First-difference highpass — the 'bright' fixture."""
    return [x - p for x, p in zip(samples, [0, *samples[:-1]])]


def _clip16(samples: list[int], peak: int = 32000) -> list[int]:
    m = max(1, max(abs(s) for s in samples))
    scale = peak / m
    return [max(-32768, min(32767, int(round(s * scale)))) for s in samples]


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Synthesize the tiny fixture wavs into <repo>/fixtures/ (deterministic)."""
    rng = random.Random(0x5EED)
    _write_wav(FIXTURES_DIR / "tone220.wav", _sine(220.0, 1.0))
    _write_wav(FIXTURES_DIR / "silence.wav", _noise(1.0, amp=4, rng=rng))
    _write_wav(FIXTURES_DIR / "noise_white.wav", _noise(1.0, amp=3000, rng=rng))
    _write_wav(FIXTURES_DIR / "noise_dark.wav", _clip16(_lowpassed(_noise(1.0, amp=30000, rng=rng), 400.0)))
    _write_wav(FIXTURES_DIR / "noise_bright.wav", _clip16(_highpassed(_noise(1.0, amp=3000, rng=rng))))
    return FIXTURES_DIR


# ------------------------------------------------------------------- tests


def test_median_f0_220hz_tone_within_5hz(fixtures_dir: Path) -> None:
    """Rule 1: median F0 of a 220 Hz tone is within 5 Hz of 220."""
    f0 = features.median_f0(fixtures_dir / "tone220.wav")
    assert isinstance(f0, float)
    assert math.isfinite(f0), f"median_f0 must return a real Hz value, got {f0!r}"
    assert abs(f0 - 220.0) < 5.0, f"median F0 {f0} not within 5 Hz of 220"


def test_median_f0_accepts_str_path(fixtures_dir: Path) -> None:
    """Signature rule: median_f0 accepts Path | str."""
    f0 = features.median_f0(str(fixtures_dir / "tone220.wav"))
    assert math.isfinite(f0)
    assert abs(f0 - 220.0) < 5.0


def test_median_f0_silence_returns_nan(fixtures_dir: Path) -> None:
    """Rule 4: no voiced frames -> math.nan, never a confident invented F0."""
    f0 = features.median_f0(fixtures_dir / "silence.wav")
    assert isinstance(f0, float)
    assert math.isnan(f0), f"median_f0 on silence must be nan, got {f0!r}"


def test_voiced_fraction_silence_near_zero(fixtures_dir: Path) -> None:
    """Rule 2: near-silence has voiced fraction < 0.1, within [0, 1]."""
    v = features.voiced_fraction(fixtures_dir / "silence.wav")
    assert 0.0 <= v <= 1.0
    assert v < 0.1, f"silence voiced fraction {v} not < 0.1"


def test_voiced_fraction_white_noise_near_zero(fixtures_dir: Path) -> None:
    """Rule 2: white noise is not voiced speech — voiced fraction < 0.1."""
    v = features.voiced_fraction(fixtures_dir / "noise_white.wav")
    assert 0.0 <= v <= 1.0
    assert v < 0.1, f"white-noise voiced fraction {v} not < 0.1"


def test_voiced_fraction_tone_is_voiced(fixtures_dir: Path) -> None:
    """Consistency: the 'voiced-like' tone must actually register as voiced
    for a majority of frames — otherwise median_f0 could never find 220 Hz."""
    v = features.voiced_fraction(fixtures_dir / "tone220.wav")
    assert 0.0 <= v <= 1.0
    assert v > 0.5, f"220 Hz tone voiced fraction {v} unexpectedly low"


def test_spectral_centroid_brighter_file_is_higher(fixtures_dir: Path) -> None:
    """Rule 3: the highpassed fixture has a strictly higher centroid than
    the lowpassed one, separated well beyond any binning slack."""
    dark = features.spectral_centroid_hz(fixtures_dir / "noise_dark.wav")
    bright = features.spectral_centroid_hz(fixtures_dir / "noise_bright.wav")
    assert math.isfinite(dark) and math.isfinite(bright)
    assert dark > 0.0, "centroid is a frequency in Hz — must be positive"
    assert bright <= NYQUIST + 1.0, f"centroid {bright} exceeds Nyquist ({NYQUIST})"
    assert bright > dark, f"bright ({bright}) must exceed dark ({dark})"
    assert bright - dark > 500.0, (
        f"fixtures are strongly separated; got dark={dark} bright={bright}"
    )
