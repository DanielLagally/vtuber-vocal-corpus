"""Choosing a tracker, and sizing the recording-era confound, on evidence.

Three questions this answers, none of which the corpus had an answer to:

1. **Can a tracker find a pitch that is known?** Synthetic tones, optionally
   with competing sound mixed in, where the right answer is not in doubt.
2. **Do independent trackers agree on real audio?** Where they disagree at
   least one is wrong, and distance from their consensus ranks them without any
   ground truth being available.
3. **Does a feature survive re-encoding?** Career time and recording era are
   nearly collinear in this corpus, so a metric that moves as far under a
   bitrate change as it moves across a career cannot support a claim about
   voices. This puts a number on that in the feature's own units.

Nothing here writes to the corpus. It reads audio and emits a report.

See reference/measurement.md and reference/statistics.md.
"""

from __future__ import annotations

import math
import random
import statistics
import subprocess
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .trackers import TRACKERS, PitchTrack

#: Default pitch search range for the comparison. Deliberately wide: the point
#: is to see what each tracker does, not to constrain them all into agreement.
DEFAULT_FLOOR = 60.0
DEFAULT_CEILING = 800.0

#: Disagreement beyond this is not a difference of opinion about a voice — it
#: is one tracker being wrong about which thing it is tracking.
DEFAULT_GROSS_SEMITONES = 3.0


@dataclass(frozen=True)
class TrackerResult:
    clip_id: str
    tracker: str
    median_f0: float
    voiced_fraction: float
    iqr_semitones: float
    fraction_below_160: float
    seconds: float


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value


def stratified_sample(
    records_by_talent: dict[str, list[dict]],
    *,
    n: int,
    seed: int = 0,
    required_ids: set[str] | None = None,
) -> list[dict]:
    """A reproducible sample spread across talents.

    Round-robin across talents rather than a flat random draw, so a talent with
    many clips cannot dominate — a sample dominated by one person measures that
    person, not the corpus. ``required_ids`` are always included: the known-bad
    clips are the most informative rows in the whole comparison and a random
    draw would almost certainly miss them.
    """
    required_ids = required_ids or set()
    rng = random.Random(seed)

    pools: dict[str, list[dict]] = {}
    forced: list[dict] = []
    for talent in sorted(records_by_talent):
        shuffled = list(records_by_talent[talent])
        rng.shuffle(shuffled)
        remaining = []
        for record in shuffled:
            row = {"talent": talent, "id": record.get("id"), "month": record.get("month")}
            if row["id"] in required_ids:
                forced.append(row)
            else:
                remaining.append(row)
        pools[talent] = remaining

    chosen: list[dict] = list(forced)
    seen = {row["id"] for row in chosen}
    talents = sorted(pools)
    index = 0
    while len(chosen) < n and any(pools[t] for t in talents):
        talent = talents[index % len(talents)]
        index += 1
        if not pools[talent]:
            continue
        row = pools[talent].pop()
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        chosen.append(row)
    return chosen


def run_trackers(
    clips: Sequence[tuple[str, Path]],
    *,
    tracker_names: Iterable[str] | None = None,
    floor: float = DEFAULT_FLOOR,
    ceiling: float = DEFAULT_CEILING,
    on_error: str = "skip",
) -> list[TrackerResult]:
    """Every tracker over every clip. ``clips`` is (clip_id, path) pairs."""
    names = list(tracker_names) if tracker_names is not None else list(TRACKERS)
    results: list[TrackerResult] = []
    for clip_id, path in clips:
        for name in names:
            started = time.time()
            try:
                track = TRACKERS[name](path, floor=floor, ceiling=ceiling)
            except Exception:
                if on_error == "raise":
                    raise
                continue
            results.append(
                TrackerResult(
                    clip_id=clip_id,
                    tracker=name,
                    median_f0=track.median_f0,
                    voiced_fraction=track.voiced_fraction,
                    iqr_semitones=track.iqr_semitones,
                    fraction_below_160=track.fraction_below(160.0),
                    seconds=time.time() - started,
                )
            )
    return results


def _semitones(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return math.nan
    return abs(12.0 * math.log2(a / b))


def consensus_deviation(
    results: Sequence[TrackerResult],
    *,
    gross_semitones: float = DEFAULT_GROSS_SEMITONES,
) -> dict[str, dict]:
    """Rank trackers by their distance from the others' consensus.

    There is no ground truth for real stream audio, but there is agreement. On
    a clip where two or more trackers concur and one dissents, the dissenter is
    the suspect. Consensus is the median of *all* trackers' medians on that
    clip, including the one being scored — median, so one wild reading cannot
    drag it.

    Leaving the scored tracker out sounds more principled and is worse here:
    with only a handful of trackers, the leave-one-out consensus of an outlier
    plus one good reading lands halfway between them, which then marks the good
    trackers as deviant too. Including self costs a small flattery to whichever
    tracker happens to be the median and keeps the ranking meaningful.

    Gross disagreements are counted separately from typical ones because they
    are a different kind of risk: a tracker that is usually right but sometimes
    off by an octave corrupts a handful of clips badly, which is worse for this
    corpus than one that is consistently a little off.
    """
    by_clip: dict[str, list[TrackerResult]] = {}
    for row in results:
        by_clip.setdefault(row.clip_id, []).append(row)

    deviations: dict[str, list[float]] = {}
    gross: dict[str, int] = {}
    for rows in by_clip.values():
        usable = [r for r in rows if _finite(r.median_f0) and r.median_f0 > 0]
        if len(usable) < 3:
            # With fewer than three opinions there is no majority to dissent
            # from: two trackers disagreeing says nothing about which is wrong.
            continue
        consensus = statistics.median([r.median_f0 for r in usable])
        for row in usable:
            delta = _semitones(row.median_f0, consensus)
            if math.isnan(delta):
                continue
            deviations.setdefault(row.tracker, []).append(delta)
            gross.setdefault(row.tracker, 0)
            if delta >= gross_semitones:
                gross[row.tracker] += 1

    summary: dict[str, dict] = {}
    for tracker in sorted({r.tracker for r in results}):
        values = deviations.get(tracker, [])
        seconds = [r.seconds for r in results if r.tracker == tracker]
        summary[tracker] = {
            "n_clips": len(values),
            "median_abs_semitones": statistics.median(values) if values else None,
            "p90_abs_semitones": (
                sorted(values)[max(0, math.ceil(0.9 * len(values)) - 1)] if values else None
            ),
            "gross_disagreements": gross.get(tracker, 0),
            "gross_rate": (gross.get(tracker, 0) / len(values)) if values else None,
            "median_seconds": statistics.median(seconds) if seconds else None,
        }
    return summary


# --------------------------------------------------------------------------
# Synthetic accuracy: the one place a right answer exists
# --------------------------------------------------------------------------


def synthetic_clip(
    path: Path,
    f0: float,
    *,
    sr: int = 22050,
    seconds: float = 2.0,
    harmonics: int = 8,
    noise_db: float | None = None,
    bgm_f0: float | None = None,
    bgm_db: float = -12.0,
) -> Path:
    """A harmonic tone, optionally with noise and/or a competing tone.

    The competing tone is the interesting case: it stands in for background
    music or a second speaker, which is what actually corrupts this corpus.
    """
    import soundfile as sf

    t = np.arange(int(seconds * sr)) / sr
    wave = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, harmonics + 1))
    wave = wave / np.max(np.abs(wave))
    if bgm_f0:
        competing = sum(np.sin(2 * np.pi * bgm_f0 * k * t) / k for k in range(1, 5))
        wave = wave + (10 ** (bgm_db / 20.0)) * competing / np.max(np.abs(competing))
    if noise_db is not None:
        rng = np.random.default_rng(0)
        wave = wave + (10 ** (noise_db / 20.0)) * rng.standard_normal(t.size)
    wave = 0.3 * wave / np.max(np.abs(wave))
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wave.astype(np.float32), sr)
    return path


def synthetic_accuracy(
    work_dir: Path,
    *,
    f0s: Sequence[float] = (150.0, 200.0, 280.0, 350.0, 450.0),
    tracker_names: Iterable[str] | None = None,
    conditions: Sequence[dict] | None = None,
    floor: float = DEFAULT_FLOOR,
    ceiling: float = DEFAULT_CEILING,
) -> list[dict]:
    """Absolute error against a known F0, per tracker and condition."""
    names = list(tracker_names) if tracker_names is not None else list(TRACKERS)
    conditions = conditions or [
        {"label": "clean"},
        {"label": "noisy", "noise_db": -20.0},
        {"label": "with_bgm", "bgm_f0": 110.0, "bgm_db": -10.0},
    ]
    rows: list[dict] = []
    for condition in conditions:
        label = condition["label"]
        kwargs = {k: v for k, v in condition.items() if k != "label"}
        for f0 in f0s:
            path = synthetic_clip(work_dir / f"{label}_{f0:.0f}.wav", f0, **kwargs)
            for name in names:
                try:
                    track: PitchTrack = TRACKERS[name](path, floor=floor, ceiling=ceiling)
                    measured = track.median_f0
                except Exception:
                    measured = math.nan
                rows.append(
                    {
                        "condition": label,
                        "true_f0": f0,
                        "tracker": name,
                        "measured_f0": measured,
                        "abs_semitone_error": (
                            _semitones(measured, f0) if _finite(measured) and measured > 0 else None
                        ),
                    }
                )
    return rows


# --------------------------------------------------------------------------
# Era sensitivity
# --------------------------------------------------------------------------


def degrade(
    src: Path, out: Path, *, bitrate_kbps: int, codec: str = "libopus"
) -> Path:
    """Re-encode through a lossy codec and back to wav.

    This stands in for the recording-era differences the corpus cannot control
    for directly: the same voice, carried by a worse pipe.

    The output is forced back to the source sample rate. Opus always encodes at
    48 kHz, so without this the "degraded" file would differ from the original
    in sample rate as well as quality — and any feature that moved could be
    blamed on either. Only the encode may vary.
    """
    import soundfile as sf

    source_sr = sf.info(str(src)).samplerate
    out.parent.mkdir(parents=True, exist_ok=True)
    intermediate = out.with_suffix(".ogg" if codec == "libopus" else ".m4a")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-c:a", codec, "-b:a", f"{bitrate_kbps}k", str(intermediate)],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(intermediate),
         "-ar", str(source_sr), "-c:a", "pcm_s16le", str(out)],
        check=True, capture_output=True,
    )
    intermediate.unlink(missing_ok=True)
    return out


def era_sensitivity(
    paths: Sequence[Path],
    *,
    bitrates: Sequence[int] = (24, 48, 96, 160),
    tracker_name: str = "praat_ac",
    work_dir: Path = Path("data/logs/bakeoff_work"),
    floor: float = DEFAULT_FLOOR,
    ceiling: float = DEFAULT_CEILING,
    extra_features: bool = False,
) -> list[dict]:
    """How far each feature moves when only the encode quality changes.

    The comparison is against the *same* file measured un-degraded, so the
    speaker, the window and the tracker are all held fixed and the encode is
    the only thing that varied.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    tracker = TRACKERS[tracker_name]
    rows: list[dict] = []
    for path in paths:
        try:
            baseline = tracker(path, floor=floor, ceiling=ceiling)
        except Exception:
            continue
        base_extra = _extra_features(path) if extra_features else {}
        for bitrate in bitrates:
            out = work_dir / f"{Path(path).stem}_{bitrate}k.wav"
            try:
                degraded_path = degrade(Path(path), out, bitrate_kbps=bitrate)
                degraded = tracker(degraded_path, floor=floor, ceiling=ceiling)
            except Exception:
                continue
            row = {
                "clip": Path(path).stem,
                "bitrate_kbps": bitrate,
                "tracker": tracker_name,
                "median_f0_base": baseline.median_f0,
                "median_f0_degraded": degraded.median_f0,
                "median_f0_delta": _delta(degraded.median_f0, baseline.median_f0),
                "median_f0_delta_semitones": (
                    _semitones(degraded.median_f0, baseline.median_f0)
                    if _finite(degraded.median_f0) and _finite(baseline.median_f0)
                    else None
                ),
                "voiced_fraction_delta": _delta(
                    degraded.voiced_fraction, baseline.voiced_fraction
                ),
            }
            if extra_features:
                for key, base_value in base_extra.items():
                    degraded_value = _extra_features(degraded_path).get(key, math.nan)
                    row[f"{key}_base"] = base_value
                    row[f"{key}_delta"] = _delta(degraded_value, base_value)
            out.unlink(missing_ok=True)
            rows.append(row)
    return rows


def _delta(a: float, b: float) -> float | None:
    if not (_finite(a) and _finite(b)):
        return None
    return float(a - b)


def _extra_features(path: Path) -> dict:
    """Spectral and voice-quality features, for era sensitivity beyond pitch."""
    from . import praat_features

    return {
        "brightness_hz": praat_features.spectral_centroid_hz(path),
        "hnr_db": praat_features.hnr_db(path),
        "jitter_local": praat_features.jitter_local(path),
        "shimmer_local": praat_features.shimmer_local(path),
    }


def results_as_dicts(results: Sequence[TrackerResult]) -> list[dict]:
    return [asdict(row) for row in results]
