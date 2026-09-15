"""The v2 feature extractor.

One pitch track per clip, computed once and handed to everything that needs it.
Every frame-based feature is evaluated at *that track's* frame times, which
makes voiced-gating exact by construction rather than correct to within a frame
of alignment slop.

What this fixes, all of it real in v1 (see docs/v1/LIMITATIONS.md):

- **Formants were never gated to voiced frames.** Praat's Burg method returns
  values for unvoiced and noisy frames too, so consonants, breath, silence and
  separator artifact sat inside every published F1-F4.
- **Brightness was computed after an un-anti-aliased downsample**, folding
  everything above the new Nyquist back into a statistic that *is* a weighted
  mean frequency — and averaged over silence as well as speech.
- **The pitch tracker ran once per feature**, six times per clip.
- **Spread was published in Hz**, a weaker constraint on a high voice than a
  low one, which is backwards for this corpus.
- **Nothing recorded which tracker or parameters produced a number**, so a gate
  derived from a tracker parameter could not be re-derived.

v1's extractor (`praat_features.py`) is deliberately untouched: it still has to
reproduce the published corpus byte for byte.

See reference/measurement.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .trackers import TRACKERS, PitchTrack

#: Frequency under which voiced energy is counted as likely contamination.
#: Not a tracker bound — a diagnostic. The talents in this corpus do not speak
#: here, so a substantial voiced mass below it means something else is in the
#: clip: another speaker, game or video dialogue, or music.
CONTAMINATION_BAND_HZ = 160.0


@dataclass(frozen=True)
class FeatureConfig:
    """Everything that can vary between measurements, in one recorded object.

    A corpus whose treatment differed between clips is usable when the
    treatment is known and unusable when it is not, so this travels onto every
    record rather than living as module constants.
    """

    #: CREPE, chosen on measured evidence rather than convention. Against
    #: synthetic tones with a competing periodic sound mixed under the target
    #: voice — the failure mode that actually corrupts this corpus — every
    #: Praat variant (autocorrelation, cross-correlation, and the newer
    #: filtered autocorrelation) locks onto the competitor and lands roughly 19
    #: semitones out, while CREPE stays within a tenth of a semitone. On clean
    #: and noisy tones the Praat family is exact and CREPE carries a ~0.09
    #: semitone bias, which is far below the corpus's own measurement noise
    #: floor. On real audio CREPE sits closest to the cross-tracker consensus.
    tracker: str = "crepe"
    floor: float = 60.0
    ceiling: float = 800.0
    hop_s: float = 0.02
    formant_ceiling_hz: float = 5500.0
    formant_window_s: float = 0.025
    formant_pre_emphasis_hz: float = 50.0
    spectral_window_s: float = 0.05
    spectral_sr: int = 16_000
    contamination_band_hz: float = CONTAMINATION_BAND_HZ

    #: Harmonicity is analysed on its own grid because its window length scales
    #: with 1/minimum_pitch, and the tracker's deliberately wide 60 Hz floor
    #: would make it several times more expensive for no gain on voices that
    #: never go near it. Recorded on every record, because it changes the
    #: value: these are not free parameters to leave implicit.
    hnr_time_step_s: float = 0.02
    hnr_min_pitch_hz: float = 75.0

    def as_record(self) -> dict:
        return {
            "tracker": self.tracker,
            "tracker_floor_hz": self.floor,
            "tracker_ceiling_hz": self.ceiling,
            "hop_s": self.hop_s,
            "formant_ceiling_hz": self.formant_ceiling_hz,
            "hnr_time_step_s": self.hnr_time_step_s,
            "hnr_min_pitch_hz": self.hnr_min_pitch_hz,
        }


DEFAULT_CONFIG = FeatureConfig()


def load_mono(path: Path | str, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Mono float32, anti-aliased on any resample.

    Polyphase resampling applies the low-pass filter that plain interpolation
    omits. Without it, every component above the new Nyquist folds back into
    the band — and for a spectral centroid, that contaminant is not noise
    around the answer, it is added directly to the answer.
    """
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if target_sr is None or sr == target_sr or mono.size == 0:
        return mono, sr
    from math import gcd

    from scipy.signal import resample_poly

    divisor = gcd(int(sr), int(target_sr))
    resampled = resample_poly(mono, target_sr // divisor, sr // divisor)
    return resampled.astype(np.float32), int(target_sr)


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def _nanmedian(values: np.ndarray) -> float:
    finite = _finite(values)
    return float(np.median(finite)) if finite.size else math.nan


# --------------------------------------------------------------------------
# Frame-aligned analyses
# --------------------------------------------------------------------------


def _formants_at_times(
    path: Path | str, times: np.ndarray, config: FeatureConfig
) -> dict[int, np.ndarray]:
    """F1-F4 sampled at the pitch track's own frame times.

    Querying Praat at these times, rather than reading its formant frames and
    aligning afterwards, is what makes the voiced mask exact.
    """
    import parselmouth

    sound = parselmouth.Sound(str(path))
    formant = sound.to_formant_burg(
        time_step=config.hop_s,
        max_number_of_formants=5,
        maximum_formant=config.formant_ceiling_hz,
        window_length=config.formant_window_s,
        pre_emphasis_from=config.formant_pre_emphasis_hz,
    )
    return {
        number: np.array(
            [formant.get_value_at_time(number, float(t)) for t in times], dtype=float
        )
        for number in (1, 2, 3, 4)
    }


def _spectral_centroid_at_times(
    audio: np.ndarray, sr: int, times: np.ndarray, window_s: float
) -> np.ndarray:
    """Per-frame magnitude-weighted mean frequency, centred on ``times``."""
    half = max(1, int(round(window_s * sr / 2)))
    window = np.hanning(2 * half)
    freqs = np.fft.rfftfreq(2 * half, 1.0 / sr)
    out = np.full(times.size, np.nan)
    for index, t in enumerate(times):
        centre = int(round(t * sr))
        start, stop = centre - half, centre + half
        if start < 0 or stop > audio.size:
            continue
        spectrum = np.abs(np.fft.rfft(audio[start:stop] * window))
        total = spectrum.sum()
        if total <= 0:
            continue
        out[index] = float((freqs * spectrum).sum() / total)
    return out


def _loudness_dynamics_db(audio: np.ndarray, sr: int, times: np.ndarray, window_s: float) -> float:
    half = max(1, int(round(window_s * sr / 2)))
    levels = []
    for t in times:
        centre = int(round(t * sr))
        start, stop = centre - half, centre + half
        if start < 0 or stop > audio.size:
            continue
        rms = float(np.sqrt(np.mean(audio[start:stop] ** 2)))
        if rms > 1e-6:
            levels.append(20.0 * math.log10(rms))
    return float(np.std(levels)) if len(levels) > 1 else math.nan


def _voice_quality(path: Path | str, config: FeatureConfig) -> dict:
    """Jitter, shimmer and HNR, via Praat.

    Kept for continuity with v1 and labelled experimental everywhere it is
    shown: these are calibrated for sustained vowels rather than conversation,
    and HNR in particular is inflated by vocal separation in a systematic
    direction, so a trend in it is a candidate artifact before it is a finding.
    """
    import parselmouth

    sound = parselmouth.Sound(str(path))
    out = {"jitter_local": math.nan, "shimmer_local": math.nan, "hnr_db": math.nan}
    try:
        pitch = sound.to_pitch_ac(
            time_step=config.hop_s, pitch_floor=config.floor, pitch_ceiling=config.ceiling
        )
        point_process = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")
        out["jitter_local"] = float(
            parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        )
        out["shimmer_local"] = float(
            parselmouth.praat.call(
                [sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
            )
        )
    except Exception:
        pass
    try:
        harmonicity = sound.to_harmonicity_cc(
            time_step=config.hnr_time_step_s, minimum_pitch=config.hnr_min_pitch_hz
        )
        out["hnr_db"] = float(parselmouth.praat.call(harmonicity, "Get mean", 0, 0))
    except Exception:
        pass
    return {k: (v if v == v else math.nan) for k, v in out.items()}


# --------------------------------------------------------------------------
# The extractor
# --------------------------------------------------------------------------


def clip_features(
    path: Path | str,
    *,
    config: FeatureConfig = DEFAULT_CONFIG,
    pitch_track: PitchTrack | None = None,
    gate_formants: bool = True,
    gate_brightness: bool = True,
    voice_quality: bool = True,
) -> dict:
    """Every v2 feature for one clip, from a single pitch track.

    ``gate_formants`` and ``gate_brightness`` exist to make the v1 behaviour
    reproducible for comparison. Production always gates; turning a gate off is
    how you measure what gating was worth, not a supported configuration.
    """
    path = Path(path)
    track = pitch_track or TRACKERS[config.tracker](
        path, floor=config.floor, ceiling=config.ceiling, hop_s=config.hop_s
    )
    times = track.times
    voiced = track.voiced_mask
    q25, q50, q75 = track.quartiles()

    record: dict = {
        "median_f0": track.median_f0,
        "f0_q25": q25,
        "f0_q50": q50,
        "f0_q75": q75,
        "f0_iqr_hz": track.iqr_hz,
        "f0_iqr_semitones": track.iqr_semitones,
        "voiced_fraction": track.voiced_fraction,
        "voiced_frames": int(voiced.sum()),
        "total_frames": int(voiced.size),
        "dynamism_semitones": _dynamism(track),
        "fraction_below_160hz": track.fraction_below(config.contamination_band_hz),
    }

    # Spectral work runs on an anti-aliased, fixed-rate copy so brightness is
    # comparable across clips whose sources differ in sample rate.
    audio, sr = load_mono(path, config.spectral_sr)
    centroid = _spectral_centroid_at_times(audio, sr, times, config.spectral_window_s)
    usable = voiced[: centroid.size] if gate_brightness else np.ones(centroid.size, bool)
    record["brightness_hz"] = _nanmedian(centroid[: usable.size][usable])
    record["loudness_dynamics_db"] = _loudness_dynamics_db(
        audio, sr, times, config.spectral_window_s
    )

    formants = _formants_at_times(path, times, config)
    mask = voiced if gate_formants else np.ones(times.size, bool)
    kept = 0
    total = 0
    for number, values in formants.items():
        selected = values[: mask.size][mask[: values.size]]
        finite = _finite(selected)
        record[f"f{number}_hz"] = float(np.mean(finite)) if finite.size else math.nan
        if number == 1:
            kept = int(finite.size)
            total = int(_finite(values).size)
    record["formant_voiced_frames"] = kept
    record["formant_total_frames"] = total

    if voice_quality:
        record.update(_voice_quality(path, config))

    record.update(config.as_record())
    return record


def _dynamism(track: PitchTrack) -> float:
    """Mean absolute semitone change between CONSECUTIVE voiced frames.

    A pair contributes only when both frames are voiced, so a pause never
    fabricates a jump across the gap. There is no octave-jump rejection, so a
    single tracker error contributes a full twelve semitones — which is part of
    why this is as much a measure of tracker noise as of prosody.
    """
    freqs = track.freqs
    if freqs.size < 2:
        return math.nan
    both = (freqs[:-1] > 0) & (freqs[1:] > 0)
    if not np.any(both):
        return math.nan
    ratios = freqs[1:][both] / freqs[:-1][both]
    return float(np.mean(np.abs(12.0 * np.log2(ratios))))
