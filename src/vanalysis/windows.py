from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

from .features import _FRAME_S, _HOP_S, _frame_f0, _load_mono


def _slice_f0_track(seg: np.ndarray, sr: int) -> np.ndarray:
    frame = max(1, int(_FRAME_S * sr))
    hop = max(1, int(_HOP_S * sr))
    if seg.size < frame:
        return np.array([_frame_f0(seg, sr)], dtype=np.float64)
    n = 1 + (seg.size - frame) // hop
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        start = i * hop
        out[i] = _frame_f0(seg[start : start + frame], sr)
    return out


def _score_window(seg: np.ndarray, sr: int) -> tuple[float, float]:
    track = _slice_f0_track(seg, sr)
    voiced = track[track > 0]
    frac = float(np.mean(track > 0)) if track.size else 0.0
    if voiced.size < 2:
        return frac, math.inf
    q75, q25 = np.percentile(voiced, [75, 25])
    return frac, float(q75 - q25)


def _grid_starts(duration: float, window_s: float, hop_s: float) -> list[float]:
    """Candidate window starts: every k*hop_s that fits, plus a final
    candidate at exactly max_start when the grid does not land on it."""
    max_start = duration - window_s
    starts: list[float] = []
    k = 0
    while k * hop_s <= max_start + 1e-9:
        starts.append(k * hop_s)
        k += 1
    if starts and max_start - starts[-1] > 1e-9:
        starts.append(max_start)
    return starts


def best_speech_window(
    path: Path | str, *, window_s: float = 90.0, hop_s: float = 15.0
) -> tuple[float, float]:
    y, sr = _load_mono(path)
    duration = y.size / sr
    if duration <= window_s:
        return 0.0, duration
    starts = _grid_starts(duration, window_s, hop_s)
    best_start = 0.0
    best_key = (-1.0, math.inf)
    for s in starts:
        i0 = int(round(s * sr))
        i1 = min(int(round((s + window_s) * sr)), y.size)
        frac, iqr = _score_window(y[i0:i1], sr)
        key = (frac, -iqr)
        if key > best_key:
            best_key = key
            best_start = s
    return best_start, best_start + window_s


def second_speech_window(
    path: Path | str,
    first: tuple[float, float],
    *,
    window_s: float = 90.0,
    hop_s: float = 15.0,
) -> tuple[float, float]:
    """Hunt the 2nd-best speech window on the same grid as
    best_speech_window, excluding every candidate that overlaps the
    half-open first window (overlap iff s < first_end and first_start <
    s + window_s). Same scoring, same (frac, -iqr) key, earliest wins
    ties. Raises ValueError when no non-overlapping window fits."""
    y, sr = _load_mono(path)
    duration = y.size / sr
    first_start, first_end = float(first[0]), float(first[1])
    if first_start < 0:
        raise ValueError(f"first window start {first_start} is negative")
    if first_end <= first_start:
        raise ValueError(
            f"first window end {first_end} must be greater than start {first_start}"
        )
    if first_end > duration + 1e-9:
        raise ValueError(
            f"first window end {first_end} is beyond file duration {duration:.3f}s"
        )
    best_start: float | None = None
    best_key = (-1.0, math.inf)
    for s in _grid_starts(duration, window_s, hop_s):
        if s < first_end and first_start < s + window_s:
            continue
        i0 = int(round(s * sr))
        i1 = min(int(round((s + window_s) * sr)), y.size)
        frac, iqr = _score_window(y[i0:i1], sr)
        key = (frac, -iqr)
        if key > best_key:
            best_key = key
            best_start = s
    if best_start is None:
        raise ValueError(
            f"no non-overlapping {window_s:g}s window fits in {duration:.3f}s of "
            f"audio without overlapping the first window "
            f"{first_start:g}s-{first_end:g}s (grid hop {hop_s:g}s)"
        )
    return best_start, best_start + window_s


def slice_wav(src: Path | str, dest: Path | str, start_s: float, end_s: float) -> None:
    with wave.open(str(src), "rb") as wav:
        sr = wav.getframerate()
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        n_frames = wav.getnframes()
        raw = wav.readframes(n_frames)
    frame_bytes = width * channels
    start_i = int(round(start_s * sr))
    if start_i >= n_frames:
        raise ValueError(
            f"start_s {start_s} is at or after duration {n_frames / sr:.3f}s"
        )
    end_i = min(int(round(end_s * sr)), n_frames)
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dest_path), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(width)
        out.setframerate(sr)
        out.writeframes(raw[start_i * frame_bytes : end_i * frame_bytes])
