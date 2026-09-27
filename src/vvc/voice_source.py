"""Voice-source and timbre features: airy vs full, clear vs breathy, fast vs slow.

Pitch and formants place a voice (how high, roughly how long the vocal tract)
but cannot tell a breathy, soft voice from a pressed, full one at the same
pitch. The features here can, and they are the standard ones for connected
speech rather than sustained vowels:

- **H1-H2** — the first harmonic's level over the second's. A breathy voice
  (vocal folds never fully closing) has a dominant fundamental and weak
  overtones; a pressed one the reverse. Reported raw and formant-corrected
  (H1*-H2*, Iseli & Alwan): a formant sitting on a harmonic boosts it, and at
  the high pitches in this corpus H2 often sits right on F1, so the raw value
  can mostly measure the vowel. ``h2_near_f1_fraction`` says how often.
- **Harmonic tilt** — slope of harmonic levels in dB per octave up to 5 kHz:
  how fast the voice's energy falls away with frequency ("body" vs "air").
- **Alpha ratio / Hammarberg index** — the same question asked of bands
  rather than harmonics (1-5 kHz over 50 Hz-1 kHz energy; strongest peak below
  2 kHz over strongest in 2-5 kHz). Cheap, and standard in eGeMAPS.
- **CPP** — cepstral peak prominence: how far the voice's periodicity stands
  above the spectral noise floor. The best-validated breathiness measure for
  conversational speech, unlike jitter/shimmer/HNR.
- **Speaking rate** — syllable nuclei (intensity peaks inside voiced speech,
  after de Jong & Wempe) per voiced second.
- **Formant dispersion** — mean spacing (F4-F1)/3, the usual single-number
  correlate of vocal-tract length, in place of four correlated formant axes.

Everything is evaluated per voiced frame on the pitch track's own grid, and the
clip value is the median over those frames. No voiced frames means ``nan``.

See reference/measurement.md.
"""

from __future__ import annotations

import math

import numpy as np

#: Analysis window per frame. Three periods at the tracker's 60 Hz floor, and
#: narrow enough in frequency (Hann main lobe ~80 Hz) to resolve harmonics of
#: every voice in this corpus.
FRAME_S = 0.05
#: Zero-padded FFT size for harmonic amplitudes, for fine frequency sampling.
HARMONIC_NFFT = 8192
#: Harmonic peaks are searched within this fraction of F0 around k*F0.
HARMONIC_SEARCH = 0.1
TILT_MAX_HZ = 5000.0
#: Cepstral peak search range: the pitch-tracker's own bounds, as quefrency.
CPP_F0_RANGE_HZ = (60.0, 800.0)
#: Quefrency range the cepstral baseline is regressed over.
CPP_BASELINE_S = (0.001, 0.025)
RATE_HOP_S = 0.01
RATE_WINDOW_S = 0.03
#: A syllable nucleus must stand this far above the dips either side of it.
RATE_PROMINENCE_DB = 2.0
#: ... and within this much of the clip's loudest frame (ignores room noise).
RATE_FLOOR_BELOW_MAX_DB = 25.0

_KEYS = (
    "cpp_db",
    "h1h2_raw_db",
    "h1h2_db",
    "h2_near_f1_fraction",
    "harmonic_tilt_db_per_octave",
    "alpha_ratio_db",
    "hammarberg_db",
    "speaking_rate_syl_per_s",
    "formant_dispersion_hz",
)


def _median(values: list[float]) -> float:
    finite = [v for v in values if v == v and math.isfinite(v)]
    return float(np.median(finite)) if finite else math.nan


def _frame(audio: np.ndarray, sr: int, t: float, length: int) -> np.ndarray | None:
    start = int(round(t * sr)) - length // 2
    if start < 0 or start + length > audio.size:
        return None
    return audio[start : start + length]


def formant_gain_db(freq: float, formant: float, bandwidth: float, sr: int) -> float:
    """Level added at ``freq`` by one resonance at ``formant`` with
    ``bandwidth``, normalised to 0 dB at DC (Iseli & Alwan's correction term)."""
    r = math.exp(-math.pi * bandwidth / sr)
    wi = 2 * math.pi * formant / sr
    w = 2 * math.pi * freq / sr
    num = (r * r + 1 - 2 * r * math.cos(wi)) ** 2
    den = (r * r + 1 - 2 * r * math.cos(wi + w)) * (r * r + 1 - 2 * r * math.cos(wi - w))
    return 10.0 * math.log10(num / den)


def _harmonic_levels(spectrum_db: np.ndarray, bin_hz: float, f0: float, max_hz: float):
    levels: list[tuple[float, float]] = []
    k = 1
    while k * f0 <= max_hz:
        lo = int(math.floor((k - HARMONIC_SEARCH) * f0 / bin_hz))
        hi = int(math.ceil((k + HARMONIC_SEARCH) * f0 / bin_hz)) + 1
        if hi > spectrum_db.size:
            break
        levels.append((k * f0, float(np.max(spectrum_db[max(lo, 0) : hi]))))
        k += 1
    return levels


def _cpp_db(frame: np.ndarray, sr: int) -> float:
    nfft = 1 << (frame.size - 1).bit_length()
    power = np.abs(np.fft.rfft(frame * np.hanning(frame.size), nfft)) ** 2
    log_spec = 10.0 * np.log10(power + 1e-12)
    ceps = np.abs(np.fft.irfft(log_spec))[: nfft // 2]
    ceps_db = 20.0 * np.log10(ceps + 1e-12)
    q = np.arange(ceps_db.size) / sr
    lo_q, hi_q = 1.0 / CPP_F0_RANGE_HZ[1], 1.0 / CPP_F0_RANGE_HZ[0]
    search = (q >= lo_q) & (q <= hi_q)
    base = (q >= CPP_BASELINE_S[0]) & (q <= min(CPP_BASELINE_S[1], q[-1]))
    if not search.any() or base.sum() < 2:
        return math.nan
    slope, intercept = np.polyfit(q[base], ceps_db[base], 1)
    peak = int(np.argmax(np.where(search, ceps_db, -np.inf)))
    return float(ceps_db[peak] - (slope * q[peak] + intercept))


def _speaking_rate(audio: np.ndarray, sr: int, times: np.ndarray, voiced: np.ndarray) -> float:
    from scipy.signal import find_peaks

    voiced_s = float(voiced.sum()) * (float(np.median(np.diff(times))) if times.size > 1 else 0.0)
    if voiced_s <= 0:
        return math.nan
    win = max(1, int(round(RATE_WINDOW_S * sr)))
    hop = max(1, int(round(RATE_HOP_S * sr)))
    n = 1 + max(0, (audio.size - win) // hop)
    frames = np.lib.stride_tricks.sliding_window_view(audio, win)[::hop][:n]
    level = 10.0 * np.log10(np.mean(frames**2, axis=1) + 1e-12)
    centres = (np.arange(level.size) * hop + win / 2) / sr
    peaks, _ = find_peaks(
        level,
        prominence=RATE_PROMINENCE_DB,
        height=float(np.max(level)) - RATE_FLOOR_BELOW_MAX_DB,
    )
    # A nucleus counts only inside voiced speech: a click or a consonant burst
    # between words is an intensity peak but not a syllable.
    nearest = np.clip(np.searchsorted(times, centres[peaks]), 0, times.size - 1)
    return float(np.count_nonzero(voiced[nearest])) / voiced_s


def voice_source_features(
    audio: np.ndarray,
    sr: int,
    times: np.ndarray,
    freqs: np.ndarray,
    voiced: np.ndarray,
    *,
    formants: dict[int, np.ndarray] | None = None,
    bandwidths: dict[int, np.ndarray] | None = None,
) -> dict:
    """Clip-level voice-source features, per voiced frame then median.

    ``formants``/``bandwidths`` are per-frame arrays on ``times`` keyed by
    formant number. Without F1/F2 and their bandwidths the corrected H1*-H2*
    is ``nan`` rather than a copy of the raw value.
    """
    out = {key: math.nan for key in _KEYS}
    if not np.any(voiced):
        return out
    audio = np.asarray(audio, dtype=np.float64)
    formants = formants or {}
    bandwidths = bandwidths or {}
    can_correct = all(k in formants for k in (1, 2)) and all(k in bandwidths for k in (1, 2))

    length = int(round(FRAME_S * sr))
    window = np.hanning(length)
    bin_hz = sr / HARMONIC_NFFT
    freq_axis = np.fft.rfftfreq(HARMONIC_NFFT, 1.0 / sr)
    low_band = (freq_axis >= 50.0) & (freq_axis < 1000.0)
    high_band = (freq_axis >= 1000.0) & (freq_axis < 5000.0)
    below_2k = freq_axis < 2000.0
    band_2k_5k = (freq_axis >= 2000.0) & (freq_axis < 5000.0)

    cpp, raw, corrected, near, tilt, alpha, hammarberg = [], [], [], [], [], [], []
    for index in np.flatnonzero(voiced):
        f0 = float(freqs[index])
        frame = _frame(audio, sr, float(times[index]), length)
        if frame is None or f0 <= 0:
            continue
        spectrum = np.abs(np.fft.rfft(frame * window, HARMONIC_NFFT))
        power = spectrum**2
        if power.sum() <= 0:
            continue
        spectrum_db = 20.0 * np.log10(spectrum + 1e-12)

        levels = _harmonic_levels(spectrum_db, bin_hz, f0, min(TILT_MAX_HZ, sr / 2 - f0))
        if len(levels) >= 2:
            h1, h2 = levels[0][1], levels[1][1]
            raw.append(h1 - h2)
            if can_correct:
                f1, f2 = float(formants[1][index]), float(formants[2][index])
                b1, b2 = float(bandwidths[1][index]), float(bandwidths[2][index])
                if all(math.isfinite(v) and v > 0 for v in (f1, f2, b1, b2)):
                    c1 = sum(formant_gain_db(f0, f, b, sr) for f, b in ((f1, b1), (f2, b2)))
                    c2 = sum(formant_gain_db(2 * f0, f, b, sr) for f, b in ((f1, b1), (f2, b2)))
                    corrected.append((h1 - c1) - (h2 - c2))
                    near.append(1.0 if abs(2 * f0 - f1) < f0 / 2 else 0.0)
        if len(levels) >= 3:
            octaves = np.log2([f for f, _ in levels])
            tilt.append(float(np.polyfit(octaves, [db for _, db in levels], 1)[0]))

        low, high = power[low_band].sum(), power[high_band].sum()
        if low > 0 and high > 0:
            alpha.append(10.0 * math.log10(high / low))
        hammarberg.append(float(spectrum_db[below_2k].max() - spectrum_db[band_2k_5k].max()))
        cpp.append(_cpp_db(frame, sr))

    out["cpp_db"] = _median(cpp)
    out["h1h2_raw_db"] = _median(raw)
    out["h1h2_db"] = _median(corrected)
    out["h2_near_f1_fraction"] = float(np.mean(near)) if near else math.nan
    out["harmonic_tilt_db_per_octave"] = _median(tilt)
    out["alpha_ratio_db"] = _median(alpha)
    out["hammarberg_db"] = _median(hammarberg)
    out["speaking_rate_syl_per_s"] = _speaking_rate(audio, sr, times, voiced)

    if all(k in formants for k in (1, 4)):
        f1 = formants[1][voiced[: formants[1].size]]
        f4 = formants[4][voiced[: formants[4].size]]
        spacing = (f4 - f1) / 3.0
        spacing = spacing[np.isfinite(spacing)]
        out["formant_dispersion_hz"] = float(np.mean(spacing)) if spacing.size else math.nan
    return out
