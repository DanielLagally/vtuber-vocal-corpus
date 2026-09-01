from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import parselmouth

from .features import loudness_dynamics_db, spectral_centroid_hz

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


def _point_process(sound: parselmouth.Sound, pitch: parselmouth.Pitch):
    return parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")


def jitter_local(path: Path | str) -> float:
    """Cycle-to-cycle variation in the TIMING between successive pitch
    periods (fraction, e.g. 0.01 = 1%). PLAN caveat: Praat's local-
    jitter algorithm is calibrated for a sustained vowel, not
    conversational speech, and is sensitive to residual vocal-isolation
    artifact — trust the relative comparison within this pipeline
    (same isolation model throughout), not the absolute number against
    a clinical reference."""
    sound = parselmouth.Sound(str(path))
    pitch = sound.to_pitch_ac(time_step=_HOP_S, pitch_floor=_FMIN, pitch_ceiling=_FMAX)
    try:
        point_process = _point_process(sound, pitch)
        value = parselmouth.praat.call(
            point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
        )
    except Exception:
        return math.nan
    return float(value) if value == value else math.nan


def shimmer_local(path: Path | str) -> float:
    """Cycle-to-cycle variation in AMPLITUDE between successive pitch
    periods (fraction). Same sustained-vowel-calibration /
    isolation-artifact caveat as jitter_local."""
    sound = parselmouth.Sound(str(path))
    pitch = sound.to_pitch_ac(time_step=_HOP_S, pitch_floor=_FMIN, pitch_ceiling=_FMAX)
    try:
        point_process = _point_process(sound, pitch)
        value = parselmouth.praat.call(
            [sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
        )
    except Exception:
        return math.nan
    return float(value) if value == value else math.nan


def hnr_db(path: Path | str) -> float:
    """Harmonics-to-noise ratio in dB — high = clear/tonal, low =
    breathy/noisy. Same sustained-vowel-calibration / isolation-
    artifact caveat as jitter_local/shimmer_local."""
    sound = parselmouth.Sound(str(path))
    harmonicity = sound.to_harmonicity_cc(time_step=0.01, minimum_pitch=_FMIN)
    value = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
    return float(value) if value == value else math.nan


def stem_features(path: Path | str) -> dict:
    """Same shape as measure.stem_features / features.{median_f0,f0_iqr,
    voiced_fraction}, backed by Praat autocorrelation instead of numpy
    ACF, same 75-600 Hz bounds — for tracker-vs-tracker comparison on the
    same audio (see diagnose.py). Plus tracker-independent extras:
    brightness_hz and loudness_dynamics_db (features.py, pure FFT/RMS,
    reused as-is), dynamism_semitones, and the voice-quality trio
    jitter_local/shimmer_local/hnr_db (this module — see their
    docstrings for the sustained-vowel/isolation-artifact caveat)."""
    return {
        "median_f0": median_f0(path),
        "f0_iqr": f0_iqr(path),
        "voiced_fraction": voiced_fraction(path),
        "brightness_hz": spectral_centroid_hz(path),
        "dynamism_semitones": dynamism_semitones(path),
        "jitter_local": jitter_local(path),
        "shimmer_local": shimmer_local(path),
        "hnr_db": hnr_db(path),
        "loudness_dynamics_db": loudness_dynamics_db(path),
    }
