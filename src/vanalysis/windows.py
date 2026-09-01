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


def best_speech_window(
    path: Path | str, *, window_s: float = 90.0, hop_s: float = 15.0
) -> tuple[float, float]:
    y, sr = _load_mono(path)
    duration = y.size / sr
    if duration <= window_s:
        return 0.0, duration
    max_start = duration - window_s
    starts: list[float] = []
    k = 0
    while k * hop_s <= max_start + 1e-9:
        starts.append(k * hop_s)
        k += 1
    if starts and max_start - starts[-1] > 1e-9:
        starts.append(max_start)
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
