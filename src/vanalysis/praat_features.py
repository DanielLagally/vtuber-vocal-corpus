from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import parselmouth

from .features import spectral_centroid_hz

_FMIN = 75.0
_FMAX = 600.0
_HOP_S = 0.02


def _pitch_track(path: Path | str) -> np.ndarray:
    """Full time-ordered frequency track, 0.0 for unvoiced frames — the
    RAW array, not filtered. Callers that need to respect adjacency
    (dynamism) must use this, not the filtered ``_voiced_track``: once
    zeros are dropped, formerly non-adjacent voiced frames on either
    side of a pause become adjacent by array position, silently
    fabricating a frame-to-frame jump across a silence gap."""
    sound = parselmouth.Sound(str(path))
    pitch = sound.to_pitch_ac(time_step=_HOP_S, pitch_floor=_FMIN, pitch_ceiling=_FMAX)
    return pitch.selected_array["frequency"]


def _voiced_track(path: Path | str) -> np.ndarray:
    freqs = _pitch_track(path)
    return freqs[freqs > 0]


def median_f0(path: Path | str) -> float:
    voiced = _voiced_track(path)
    if voiced.size == 0:
        return math.nan
    return float(np.median(voiced))


def f0_iqr(path: Path | str) -> float:
    voiced = _voiced_track(path)
    if voiced.size < 2:
        return math.nan
    q75, q25 = np.percentile(voiced, [75, 25])
    return float(q75 - q25)


def voiced_fraction(path: Path | str) -> float:
    freqs = _pitch_track(path)
    if freqs.size == 0:
        return 0.0
    return float(np.mean(freqs > 0))


def dynamism_semitones(path: Path | str) -> float:
    """Mean absolute semitone change between consecutive VOICED frames —
    how much the pitch actually MOVES over time, distinct from
    f0_iqr's static spread (two clips can share an IQR while differing
    sharply here). A frame pair contributes only when BOTH frames are
    voiced; a pause between two voiced stretches contributes nothing,
    never a fabricated jump across the gap. No voiced pair at all (< 2
    voiced frames, or every voiced frame is isolated by pauses) is
    ``math.nan``, same no-invented-value convention as the other
    features here."""
    freqs = _pitch_track(path)
    both_voiced = (freqs[:-1] > 0) & (freqs[1:] > 0)
    if not np.any(both_voiced):
        return math.nan
    ratios = freqs[1:][both_voiced] / freqs[:-1][both_voiced]
    semitone_deltas = np.abs(12.0 * np.log2(ratios))
    return float(np.mean(semitone_deltas))


def stem_features(path: Path | str) -> dict:
    """Same shape as measure.stem_features / features.{median_f0,f0_iqr,
    voiced_fraction}, backed by Praat autocorrelation instead of numpy
    ACF, same 75-600 Hz bounds — for tracker-vs-tracker comparison on the
    same audio (see diagnose.py). Plus two tracker-independent extras:
    brightness_hz (features.spectral_centroid_hz, pure FFT, reused as-is)
    and dynamism_semitones (this module)."""
    return {
        "median_f0": median_f0(path),
        "f0_iqr": f0_iqr(path),
        "voiced_fraction": voiced_fraction(path),
        "brightness_hz": spectral_centroid_hz(path),
        "dynamism_semitones": dynamism_semitones(path),
    }
