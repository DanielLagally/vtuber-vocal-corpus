"""Pitch trackers behind one interface, so they can be compared on evidence.

The corpus picked its tracker early, from a comparison against exactly one
alternative, and every published number has rested on that choice since. This
module exists to re-open it properly: several independent implementations —
classical and neural — measured against signals whose pitch is known, and then
against each other on real audio where it is not.

Each tracker returns a `PitchTrack`: a frame track at a common hop, 0.0 for
unvoiced, carrying its own identity and parameters. Nothing downstream needs to
know which produced it, and no measurement can end up unattributable.

A note on what these can and cannot fix. The implausibly low readings in this
corpus are *not* octave errors: Praat autocorrelation, Praat filtered
autocorrelation and CREPE independently agree on them, and the clips carry a
large voiced mass far below the talent's own range that their same-month
counterparts do not have. They are contaminated audio — a different speaker,
game or video dialogue, residual music. No tracker fixes that, and raising the
pitch floor until the number looks plausible would manufacture a result rather
than correct an error. `fraction_below` exists to detect it instead.

See reference/measurement.md.
"""

from __future__ import annotations

import math
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: One hop for every tracker, so tracks are directly comparable frame by frame.
DEFAULT_HOP_S = 0.02

#: Neural trackers run on a fixed rate; 16 kHz is CREPE's native input and is
#: ample for speech F0 (the highest F0 of interest is far below Nyquist).
NEURAL_SR = 16_000

_PRAAT_SCRIPT = Path(__file__).parent / "praat_scripts" / "pitch.praat"


@dataclass(frozen=True)
class PitchTrack:
    """A frame-by-frame pitch track. ``freqs`` is 0.0 on unvoiced frames.

    ``t0`` is the centre time of the first frame. It exists so other analyses
    can be evaluated at *these* times rather than on their own grid: sampling
    formants and spectra at the pitch frames makes voiced-gating exact by
    construction, instead of correct to within a frame of alignment slop.
    """

    freqs: np.ndarray
    hop_s: float
    tracker: str
    params: dict = field(default_factory=dict)
    t0: float = 0.0

    @property
    def times(self) -> np.ndarray:
        return self.t0 + np.arange(self.freqs.size) * self.hop_s

    @property
    def voiced_mask(self) -> np.ndarray:
        return self.freqs > 0

    @property
    def voiced(self) -> np.ndarray:
        return self.freqs[self.freqs > 0]

    @property
    def median_f0(self) -> float:
        """NaN, never 0.0, when nothing was voiced — 0 Hz is a claim about a
        voice, and silence is the absence of one."""
        voiced = self.voiced
        return float(np.median(voiced)) if voiced.size else math.nan

    @property
    def voiced_fraction(self) -> float:
        return float(np.mean(self.freqs > 0)) if self.freqs.size else 0.0

    @property
    def iqr_hz(self) -> float:
        voiced = self.voiced
        if voiced.size < 2:
            return math.nan
        q75, q25 = np.percentile(voiced, [75, 25])
        return float(q75 - q25)

    @property
    def iqr_semitones(self) -> float:
        """Scale-free spread. An absolute Hz IQR is a far weaker constraint on
        a high-pitched voice than a low one, which is backwards for a corpus
        whose subject is high-pitched voices."""
        voiced = self.voiced
        if voiced.size < 2:
            return math.nan
        q75, q25 = np.percentile(voiced, [75, 25])
        if q25 <= 0:
            return math.nan
        return float(12.0 * np.log2(q75 / q25))

    def fraction_below(self, hz: float) -> float:
        """Share of VOICED frames under ``hz`` — the contamination signal.

        A clip carrying a second, lower-pitched speaker shows substantial mass
        here while its same-month counterpart shows almost none. The median
        alone cannot distinguish that from a genuinely low reading.
        """
        voiced = self.voiced
        if voiced.size == 0:
            return 0.0
        return float(np.mean(voiced < hz))

    def quartiles(self) -> tuple[float, float, float]:
        voiced = self.voiced
        if voiced.size < 2:
            return (math.nan, math.nan, math.nan)
        q25, q50, q75 = np.percentile(voiced, [25, 50, 75])
        return (float(q25), float(q50), float(q75))


def _load_mono(path: Path | str, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Mono float32. Any resample is anti-aliased via polyphase filtering —
    plain interpolation folds everything above the new Nyquist back into the
    band, which corrupts any spectral statistic computed afterwards.
    """
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if target_sr is None or sr == target_sr:
        return mono, sr
    from math import gcd

    divisor = gcd(int(sr), int(target_sr))
    from scipy.signal import resample_poly

    resampled = resample_poly(mono, target_sr // divisor, sr // divisor)
    return resampled.astype(np.float32), target_sr


# --------------------------------------------------------------------------
# Praat via the pip binding (the v1 production path)
# --------------------------------------------------------------------------


def _parselmouth_track(
    path: Path | str, *, floor: float, ceiling: float, hop_s: float, method: str
) -> tuple[np.ndarray, float]:
    import parselmouth

    sound = parselmouth.Sound(str(path))
    if method == "cc":
        pitch = parselmouth.praat.call(
            sound, "To Pitch (cc)", hop_s, floor, 15, "no", 0.03, 0.45, 0.01, 0.35, 0.14, ceiling
        )
    else:
        pitch = sound.to_pitch_ac(time_step=hop_s, pitch_floor=floor, pitch_ceiling=ceiling)
    freqs = np.asarray(pitch.selected_array["frequency"], dtype=float)
    times = np.asarray(pitch.xs(), dtype=float)
    return freqs, float(times[0]) if times.size else 0.0


def praat_ac(
    path: Path | str, *, floor: float, ceiling: float, hop_s: float = DEFAULT_HOP_S
) -> PitchTrack:
    """Praat autocorrelation — the v1 production tracker, kept as the baseline."""
    freqs, t0 = _parselmouth_track(
        path, floor=floor, ceiling=ceiling, hop_s=hop_s, method="ac"
    )
    return PitchTrack(
        freqs=freqs,
        hop_s=hop_s,
        tracker="praat_ac",
        params={"floor": floor, "ceiling": ceiling, "hop_s": hop_s},
        t0=t0,
    )


def praat_cc(
    path: Path | str, *, floor: float, ceiling: float, hop_s: float = DEFAULT_HOP_S
) -> PitchTrack:
    """Praat cross-correlation."""
    freqs, t0 = _parselmouth_track(
        path, floor=floor, ceiling=ceiling, hop_s=hop_s, method="cc"
    )
    return PitchTrack(
        freqs=freqs,
        hop_s=hop_s,
        tracker="praat_cc",
        params={"floor": floor, "ceiling": ceiling, "hop_s": hop_s},
        t0=t0,
    )


# --------------------------------------------------------------------------
# Praat via the standalone program (newer methods the binding cannot reach)
# --------------------------------------------------------------------------


class PraatUnavailable(RuntimeError):
    """The standalone `praat` program is not on PATH."""


def _run_praat_script(
    path: Path | str, *, floor: float, ceiling: float, hop_s: float, method: str
) -> tuple[np.ndarray, float]:
    argv = [
        "praat",
        "--run",
        str(_PRAAT_SCRIPT),
        str(Path(path).resolve()),  # Praat resolves relative paths against the script
        str(floor),
        str(ceiling),
        str(hop_s),
        method,
    ]
    try:
        # check=False: the returncode is inspected below so Praat's own stderr
        # can be surfaced, which a CalledProcessError would swallow.
        done = subprocess.run(
            argv, capture_output=True, text=True, timeout=600, check=False
        )
    except FileNotFoundError as exc:  # pragma: no cover - environment-dependent
        raise PraatUnavailable("the standalone `praat` program is not on PATH") from exc
    if done.returncode != 0:
        raise RuntimeError(f"praat failed: {done.stderr.strip()[:400]}")
    lines = [line.strip() for line in done.stdout.split("\n") if line.strip()]
    t0 = 0.0
    if lines and lines[0].startswith("t0"):
        try:
            t0 = float(lines[0].split()[1])
        except (IndexError, ValueError):
            t0 = 0.0
        lines = lines[1:]
    if not lines:
        return np.zeros(0, dtype=float), t0
    return np.array([float(v) for v in lines], dtype=float), t0


def praat_filtered_ac(
    path: Path | str, *, floor: float, ceiling: float, hop_s: float = DEFAULT_HOP_S
) -> PitchTrack:
    """Praat filtered autocorrelation, the method Praat now recommends for F0.

    ``ceiling`` is passed as Praat's "pitch top" — where attenuation begins,
    not a hard cutoff. The method is parameterised differently on purpose.
    """
    freqs, t0 = _run_praat_script(
        path, floor=floor, ceiling=ceiling, hop_s=hop_s, method="filtered_ac"
    )
    return PitchTrack(
        freqs=freqs,
        hop_s=hop_s,
        tracker="praat_filtered_ac",
        params={"floor": floor, "ceiling": ceiling, "hop_s": hop_s},
        t0=t0,
    )


# --------------------------------------------------------------------------
# pYIN (librosa) — a strong classical baseline independent of Praat
# --------------------------------------------------------------------------


def pyin(
    path: Path | str, *, floor: float, ceiling: float, hop_s: float = DEFAULT_HOP_S
) -> PitchTrack:
    import librosa

    audio, sr = _load_mono(path, NEURAL_SR)
    hop_length = max(1, int(round(hop_s * sr)))
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=floor,
        fmax=ceiling,
        sr=sr,
        hop_length=hop_length,
        frame_length=hop_length * 4,
    )
    freqs = np.nan_to_num(np.where(voiced_flag, f0, 0.0), nan=0.0)
    return PitchTrack(
        freqs=freqs.astype(float),
        hop_s=hop_s,
        tracker="pyin",
        params={"floor": floor, "ceiling": ceiling, "hop_s": hop_s},
    )


# --------------------------------------------------------------------------
# CREPE (torchcrepe) — neural, GPU-accelerated
# --------------------------------------------------------------------------

#: CREPE emits a confidence per frame rather than a voiced/unvoiced decision.
#: 0.5 is torchcrepe's own conventional operating point.
CREPE_PERIODICITY_THRESHOLD = 0.5

#: Frames per forward pass. Measured on a 90-second clip: batch 1024 reserves
#: ~4.4 GB and takes 1.33 s, batch 256 reserves ~1.3 GB and takes 1.32 s, batch
#: 128 reserves ~0.8 GB and takes 1.42 s. The GPU is already saturated by 128,
#: so a larger batch buys nothing and costs several gigabytes — which matters
#: enormously when a corpus build runs several workers at once on a consumer
#: card. Memory, not batch size, is the binding constraint here.
CREPE_BATCH_SIZE = 128


def _select_device(*, cuda_available: bool, mps_available: bool) -> str:
    """CUDA, then Apple Silicon's MPS, then CPU.

    The separator (isolate.py, via audio-separator) already picks CUDA / MPS /
    CoreML per platform with no code change needed. CREPE is a from-scratch
    torchcrepe call in this repo, not delegated to a library that does this
    itself, so it needs the same three-way choice made explicitly — checking
    only `torch.cuda.is_available()` silently strands Apple Silicon on CPU,
    which is many times slower for no accuracy gain.
    """
    if cuda_available:
        return "cuda"
    if mps_available:
        return "mps"
    return "cpu"


def crepe(
    path: Path | str,
    *,
    floor: float,
    ceiling: float,
    hop_s: float = DEFAULT_HOP_S,
    periodicity_threshold: float = CREPE_PERIODICITY_THRESHOLD,
    model: str = "full",
    batch_size: int = CREPE_BATCH_SIZE,
) -> PitchTrack:
    import torch
    import torchcrepe

    audio, sr = _load_mono(path, NEURAL_SR)
    if audio.size < sr // 10:
        return PitchTrack(
            freqs=np.zeros(0),
            hop_s=hop_s,
            tracker="crepe",
            params={"floor": floor, "ceiling": ceiling, "hop_s": hop_s},
        )
    def _predict(device: str, batch_size: int):
        return torchcrepe.predict(
            torch.from_numpy(audio)[None],
            sr,
            hop_length=max(1, int(round(hop_s * sr))),
            fmin=floor,
            fmax=ceiling,
            model=model,
            device=device,
            batch_size=batch_size,
            return_periodicity=True,
        )

    device = _select_device(
        cuda_available=torch.cuda.is_available(),
        mps_available=torch.backends.mps.is_available(),
    )
    if device == "cpu":
        frequency, periodicity = _predict(device, batch_size)
    else:
        try:
            frequency, periodicity = _predict(device, batch_size)
        except RuntimeError:
            # A GPU/accelerator shared with other work can still refuse the
            # allocation. Falling back to CPU is much slower but returns the
            # same numbers, which beats failing a measurement because
            # something else was busy.
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()
            device = "cpu"
            frequency, periodicity = _predict(device, batch_size)
    freqs = frequency[0].cpu().numpy().astype(float)
    confidence = periodicity[0].cpu().numpy()
    freqs = np.where(confidence >= periodicity_threshold, freqs, 0.0)
    return PitchTrack(
        freqs=freqs,
        hop_s=hop_s,
        tracker="crepe",
        params={
            "floor": floor,
            "ceiling": ceiling,
            "hop_s": hop_s,
            "periodicity_threshold": periodicity_threshold,
            "model": model,
            "device": device,
        },
    )


TrackerFn = Callable[..., PitchTrack]

#: Every tracker the bake-off considers. `praat_ac` is the v1 incumbent and is
#: present so the comparison always includes the number currently published.
TRACKERS: dict[str, TrackerFn] = {
    "praat_ac": praat_ac,
    "praat_cc": praat_cc,
    "praat_filtered_ac": praat_filtered_ac,
    "pyin": pyin,
    "crepe": crepe,
}
