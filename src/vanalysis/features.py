from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

_FMIN = 75.0
_FMAX = 600.0
_FRAME_S = 0.04
_HOP_S = 0.02
_VOICING = 0.55
_QUIET_RMS = 1e-3
_TARGET_SR = 16_000


def _resample(y: np.ndarray, sr: int, target: int) -> tuple[np.ndarray, int]:
    if sr == target:
        return y, sr
    n = max(1, int(round(y.size * target / sr)))
    old = np.linspace(0.0, 1.0, y.size, endpoint=False)
    new = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.interp(new, old, y), target


def _load_mono(path: Path | str) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        sr = wav.getframerate()
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        n_frames = wav.getnframes()
        raw = wav.readframes(n_frames)
    if width != 2:
        raise ValueError("only 16-bit PCM wav is supported")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return _resample(samples, sr, _TARGET_SR)


def _frame_f0(frame: np.ndarray, sr: int) -> float:
    x = frame - np.mean(frame)
    rms = float(np.sqrt(np.mean(x * x)))
    if rms < _QUIET_RMS:
        return 0.0
    corr = np.correlate(x, x, mode="full")
    corr = corr[corr.size // 2 :]
    if corr[0] <= 0:
        return 0.0
    min_lag = max(1, int(sr / _FMAX))
    max_lag = min(int(sr / _FMIN), corr.size - 1)
    if max_lag <= min_lag:
        return 0.0
    segment = corr[min_lag : max_lag + 1]
    peak = int(np.argmax(segment))
    if segment[peak] / corr[0] < _VOICING:
        return 0.0
    lag = min_lag + peak
    if 0 < peak < segment.size - 1:
        y0, y1, y2 = (float(segment[peak + d]) for d in (-1, 0, 1))
        denom = y0 - 2.0 * y1 + y2
        if denom != 0:
            lag = lag + 0.5 * (y0 - y2) / denom
    return float(sr / lag)


def _f0_track(path: Path | str) -> np.ndarray:
    y, sr = _load_mono(path)
    frame = max(1, int(_FRAME_S * sr))
    hop = max(1, int(_HOP_S * sr))
    if y.size < frame:
        f0 = _frame_f0(y, sr)
        return np.array([f0], dtype=np.float64)
    n = 1 + (y.size - frame) // hop
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        start = i * hop
        out[i] = _frame_f0(y[start : start + frame], sr)
    return out


def median_f0(path: Path | str) -> float:
    voiced = _f0_track(path)
    voiced = voiced[voiced > 0]
    if voiced.size == 0:
        return math.nan
    return float(np.median(voiced))


def f0_iqr(path: Path | str) -> float:
    voiced = _f0_track(path)
    voiced = voiced[voiced > 0]
    if voiced.size < 2:
        return math.nan
    q75, q25 = np.percentile(voiced, [75, 25])
    return float(q75 - q25)


def voiced_fraction(path: Path | str) -> float:
    track = _f0_track(path)
    if track.size == 0:
        return 0.0
    return float(np.mean(track > 0))


def spectral_centroid_hz(path: Path | str) -> float:
    y, sr = _load_mono(path)
    frame = max(1, int(0.05 * sr))
    hop = max(1, int(0.025 * sr))
    if y.size < frame:
        window = np.hanning(y.size)
        mag = np.abs(np.fft.rfft(y * window))
        freqs = np.fft.rfftfreq(y.size, 1.0 / sr)
        total = float(np.sum(mag))
        if total <= 0:
            return math.nan
        return float(np.sum(freqs * mag) / total)
    cents: list[float] = []
    win = np.hanning(frame)
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    for start in range(0, y.size - frame + 1, hop):
        mag = np.abs(np.fft.rfft(y[start : start + frame] * win))
        total = float(np.sum(mag))
        if total <= 0:
            continue
        cents.append(float(np.sum(freqs * mag) / total))
    if not cents:
        return math.nan
    return float(np.mean(cents))


def loudness_dynamics_db(path: Path | str) -> float:
    """Standard deviation of frame RMS-in-dB across the clip — how much
    the VOLUME moves (loud/excited vs quiet moments) versus a flat,
    even delivery. Tracker-independent (no pitch tracking), unlike
    jitter/shimmer/HNR not designed for a sustained vowel, so it
    doesn't inherit their measurement-mismatch caveat on conversational
    speech. Frames below a near-silence RMS floor are excluded (a
    silent frame's dB is not a loudness data point, just absence of
    signal)."""
    y, sr = _load_mono(path)
    frame = max(1, int(0.05 * sr))
    hop = max(1, int(0.025 * sr))
    if y.size < frame:
        return math.nan
    dbs: list[float] = []
    for start in range(0, y.size - frame + 1, hop):
        seg = y[start : start + frame]
        rms = float(np.sqrt(np.mean(seg * seg)))
        if rms > 1e-6:
            dbs.append(20.0 * math.log10(rms))
    if len(dbs) < 2:
        return math.nan
    return float(np.std(dbs))
