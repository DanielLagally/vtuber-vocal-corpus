from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import parselmouth

_FMIN = 75.0
_FMAX = 600.0
_HOP_S = 0.02


def _voiced_track(path: Path | str) -> np.ndarray:
    sound = parselmouth.Sound(str(path))
    pitch = sound.to_pitch_ac(time_step=_HOP_S, pitch_floor=_FMIN, pitch_ceiling=_FMAX)
    freqs = pitch.selected_array["frequency"]
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
    sound = parselmouth.Sound(str(path))
    pitch = sound.to_pitch_ac(time_step=_HOP_S, pitch_floor=_FMIN, pitch_ceiling=_FMAX)
    freqs = pitch.selected_array["frequency"]
    if freqs.size == 0:
        return 0.0
    return float(np.mean(freqs > 0))


def stem_features(path: Path | str) -> dict:
    """Same shape as measure.stem_features / features.{median_f0,f0_iqr,
    voiced_fraction}, backed by Praat autocorrelation instead of numpy
    ACF, same 75-600 Hz bounds — for tracker-vs-tracker comparison on the
    same audio (see diagnose.py)."""
    return {
        "median_f0": median_f0(path),
        "f0_iqr": f0_iqr(path),
        "voiced_fraction": voiced_fraction(path),
    }
