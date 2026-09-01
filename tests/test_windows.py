"""Product tests for vanalysis speech-window selection.

User-visible rules (synthetic audio ONLY — tones/noise; never Cover/hololive
audio, never downloads):

1. From a longer recording, best_speech_window picks the shorter speech
   window: high voiced fraction AND low F0 IQR (unstable F0 = BGM/melody,
   junk). A 4 s fixture of noise + 220 Hz tone (middle) + noise must
   select a 2 s window that covers the tone, not the noise.
2. A file shorter than window_s returns the whole file: (0.0, duration).
3. Signature accepts Path | str.
4. slice_wav(src, dest, start_s, end_s) cuts a synthetic 16 kHz mono
   16-bit wav into dest: the dest sample count is exactly the requested
   range, the samples are exactly the source slice, and sr/channels/
   sample width are preserved.
5. slice_wav clamps end_s beyond EOF to the end of file; a start_s at or
   after the duration raises ValueError.

Fixtures are tiny wavs (16 kHz, mono, 16-bit) synthesized here with the
stdlib `wave` module into <repo>/fixtures/ — deterministic (seeded RNG),
no third-party test deps. The synthesis helpers are copied from
tests/test_features.py on purpose: test modules stay independent.
"""

import array
import math
import random
import sys
import wave
from pathlib import Path

import pytest

from vanalysis import windows

SR = 16_000
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


def _read_wav(path: Path) -> tuple[list[int], int, int, int]:
    """Read a 16-bit wav back as samples plus (sr, channels, sampwidth)."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        channels = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    buf = array.array("h")
    buf.frombytes(raw)
    if sys.byteorder == "big":
        buf.byteswap()
    return list(buf), sr, channels, width


@pytest.fixture(scope="module")
def fixtures_dir() -> Path:
    """Synthesize the tiny fixture wavs (deterministic, synthetic only)."""
    rng = random.Random(0x5EED)
    # 1 s noise + 2 s 220 Hz tone + 1 s noise: a speech-like island in junk.
    _write_wav(
        FIXTURES_DIR / "window_tone_mid.wav",
        _noise(1.0, amp=3000, rng=rng) + _sine(220.0, 2.0) + _noise(1.0, amp=3000, rng=rng),
    )
    # 1 s tone — shorter than any useful window.
    _write_wav(FIXTURES_DIR / "window_short.wav", _sine(220.0, 1.0))
    # 3 s tone for slice_wav: 48000 samples to cut ranges out of.
    _write_wav(FIXTURES_DIR / "slice_src.wav", _sine(220.0, 3.0))
    return FIXTURES_DIR


# ------------------------------------------------------------------- tests


def test_best_window_lands_on_tone_not_noise(fixtures_dir: Path) -> None:
    """Rule 1: the chosen 2 s window covers the tone island, not the noise.

    With hop 0.5 s over a 4 s file, the windows that overlap the tone more
    than the noise start at 0.5, 1.0 or 1.5 s; tone-poor windows (start
    0.0 or 2.0) must lose.
    """
    start_s, end_s = windows.best_speech_window(
        fixtures_dir / "window_tone_mid.wav", window_s=2.0, hop_s=0.5
    )
    assert end_s - start_s == 2.0, f"window must be exactly window_s long, got {end_s - start_s}"
    # 1e-6 slack is float dust only; candidates sit on 0.5 s hop multiples.
    assert 0.5 - 1e-6 <= start_s <= 1.5 + 1e-6, (
        f"best window start {start_s} does not cover the tone island"
    )


def test_file_shorter_than_window_returns_whole_file(fixtures_dir: Path) -> None:
    """Rule 2: 1 s file with window_s=90 -> the whole file, (0.0, 1.0)."""
    start_s, end_s = windows.best_speech_window(
        fixtures_dir / "window_short.wav", window_s=90.0
    )
    assert start_s == 0.0, f"short file must start at 0.0, got {start_s}"
    assert end_s == pytest.approx(1.0), f"short file must end at its duration, got {end_s}"


def test_signature_accepts_path_and_str(fixtures_dir: Path) -> None:
    """Rule 3: best_speech_window accepts Path | str, same answer for both."""
    p = fixtures_dir / "window_tone_mid.wav"
    from_path = windows.best_speech_window(p, window_s=2.0, hop_s=0.5)
    from_str = windows.best_speech_window(str(p), window_s=2.0, hop_s=0.5)
    assert from_path == from_str
    start_s, end_s = from_str
    assert end_s - start_s == 2.0


# ---------------------------------------------------------------- slice_wav


def test_slice_wav_exact_samples_and_format(fixtures_dir: Path, tmp_path: Path) -> None:
    """Rule 4: slicing 1.0–2.0 s of a 3 s source gives dest exactly
    source_samples[16000:32000] with sr/channels/sample width preserved."""
    src = fixtures_dir / "slice_src.wav"
    src_samples, src_sr, _, _ = _read_wav(src)
    dest = tmp_path / "sliced.wav"

    windows.slice_wav(src, dest, 1.0, 2.0)

    got, sr, channels, width = _read_wav(dest)
    assert len(got) == SR, f"dest must hold exactly 1 s of samples ({SR}), got {len(got)}"
    assert got == src_samples[SR : 2 * SR], "dest samples must equal the source slice"
    assert sr == src_sr, f"sample rate must be preserved, got {sr}"
    assert channels == 1, f"channels must be preserved, got {channels}"
    assert width == 2, f"16-bit sample width must be preserved, got {width}"


def test_slice_wav_clamps_end_beyond_duration_to_eof(
    fixtures_dir: Path, tmp_path: Path
) -> None:
    """Rule 5a: end_s past EOF clamps to the end of file — the tail from
    2.0 s onward (16000 samples), not silence padding."""
    src = fixtures_dir / "slice_src.wav"
    src_samples, _, _, _ = _read_wav(src)
    dest = tmp_path / "tail.wav"

    windows.slice_wav(src, dest, 2.0, 999.0)

    got, _, _, _ = _read_wav(dest)
    assert got == src_samples[2 * SR :], "dest must be the clamped EOF tail"
    assert len(got) == SR, f"clamped tail must hold {SR} samples, got {len(got)}"


@pytest.mark.parametrize(
    ("start_s", "end_s"), [(3.0, 4.0), (3.5, 4.0)]
)
def test_slice_wav_start_at_or_after_duration_raises(
    fixtures_dir: Path, tmp_path: Path, start_s: float, end_s: float
) -> None:
    """Rule 5b: a start_s at (==) or beyond (>) the 3 s duration raises
    ValueError — never a silent empty wav."""
    with pytest.raises(ValueError):
        windows.slice_wav(fixtures_dir / "slice_src.wav", tmp_path / "never.wav", start_s, end_s)
