"""How much non-target sound a clip actually contained.

The vocal separator emits two stems — the voice it found and everything else —
so the energy ratio between them is a direct, per-clip measure of competing
sound: music, game audio, a film being watched, a second speaker. Both stems
are already on disk for most of the corpus, so this costs nothing beyond
reading them.

Why it matters more than it sounds. v1 separated every clip unconditionally so
that the transform would at least be uniform, but separation is lossy: on audio
with nothing to remove it can only add error, and the error it adds is large
for every feature except pitch. Uniform damage is not neutral, it is uniform.
This index is what makes the decision conditional and, more importantly,
*recorded* — a corpus where the treatment varies is fine if the treatment is
known, and unusable if it is not.

It also settles arguments that were previously settled by assumption. Stream
types were ranked by presumed audio quality with no measurement behind the
ranking; watchalongs in particular were treated as the worst case when many
carry no background audio at all.

See reference/measurement.md.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np

#: Below this, the residual is quiet enough relative to the voice that
#: separating costs more (in distorted formants, inflated HNR, shifted voiced
#: fraction) than the background it would remove.
DEFAULT_THRESHOLD_DB = -15.0

#: An RMS at or under this is silence, not a quiet signal.
_SILENCE_RMS = 1e-6


def _rms(path: Path | str) -> float:
    """Level of a whole stem.

    The whole file, deliberately. Estimating it from a few seconds sampled
    across the clip is ~15x cheaper and was tried: on smooth synthetic audio it
    agrees to a fraction of a dB, but real residual stems are bursty — long
    near-silence broken by loud passages — and the estimate lands up to ~8 dB
    out, flipping the separation decision for several percent of clips even
    when half the clip is sampled. Since that decision chooses which audio gets
    measured, the cheap version is not a safe trade. Parallelise the scan
    instead; do not shorten the read.
    """
    import soundfile as sf

    audio, _ = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio.mean(axis=1) ** 2)))


def ratios_for(
    video_ids: list[str],
    stems_dir: Path | str,
    *,
    workers: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, float]:
    """Background ratios for many clips, in parallel.

    The scan is pure I/O over hundreds of gigabytes of stems, so it is worth
    spreading across processes — it is otherwise the longest serial step in a
    corpus build.
    """
    jobs = [(vid, str(stems_dir)) for vid in video_ids]
    out: dict[str, float] = {}
    done = 0

    if workers <= 1:
        for job in jobs:
            vid, ratio = _ratio_job(job)
            out[vid] = ratio
            done += 1
            if progress:
                progress(done, len(jobs))
        return out

    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for vid, ratio in pool.map(_ratio_job, jobs, chunksize=16):
            out[vid] = ratio
            done += 1
            if progress:
                progress(done, len(jobs))
    return out


def _ratio_job(job: tuple[str, str]) -> tuple[str, float]:
    """Worker for the pool. Module-level so it is picklable."""
    video_id, stems_dir = job
    return video_id, clip_background_ratio_db(video_id, stems_dir)


def background_ratio_db(vocals_path: Path | str, other_path: Path | str) -> float:
    """Residual-to-voice energy, in dB. Negative means the voice dominates.

    NaN when it cannot be determined — a silent vocal stem, or a missing
    residual stem. Neither is an infinitely noisy clip, and neither is a clean
    one: absent evidence must not read as evidence of cleanliness, because a
    conditional-separation rule would then skip separation on exactly the clips
    it knows least about.
    """
    vocals_path, other_path = Path(vocals_path), Path(other_path)
    if not vocals_path.is_file() or not other_path.is_file():
        return math.nan
    voice = _rms(vocals_path)
    residual = _rms(other_path)
    if voice <= _SILENCE_RMS:
        return math.nan
    if residual <= _SILENCE_RMS:
        return -math.inf
    return float(20.0 * math.log10(residual / voice))


def needs_separation(ratio_db: float, *, threshold_db: float = DEFAULT_THRESHOLD_DB) -> bool:
    """Whether a clip carries enough competing sound to be worth separating.

    An unknown ratio separates, matching v1's behaviour. Skipping separation on
    a clip nobody measured would be a silent change of method.
    """
    if not isinstance(ratio_db, (int, float)) or ratio_db != ratio_db:
        return True
    return bool(ratio_db >= threshold_db)


#: Directory listings, cached per process.
#:
#: Every lookup used to glob the stems directory, and a glob scans the whole
#: directory — which here holds tens of thousands of entries. Resolving a
#: corpus-sized batch that way costs over a billion filename comparisons and
#: minutes of pure CPU, for information that changes only when files are
#: written. Listing each directory once turns it into a dict lookup.
_LISTING_CACHE: dict[str, dict[str, list[str]]] = {}


def _listing(stems_dir: Path | str) -> dict[str, list[str]]:
    """Filenames in ``stems_dir``, indexed by the video id they start with."""
    key = str(stems_dir)
    cached = _LISTING_CACHE.get(key)
    if cached is not None:
        return cached
    index: dict[str, list[str]] = {}
    directory = Path(stems_dir)
    if directory.is_dir():
        for name in os.listdir(directory):
            video_id = name.split("_raw90_")[0].split("_(")[0]
            index.setdefault(video_id, []).append(name)
    _LISTING_CACHE[key] = index
    return index


def clear_listing_cache() -> None:
    """Forget cached directory listings — call after writing new stems."""
    _LISTING_CACHE.clear()


def _stem_match(video_id: str, stems_dir: Path | str, kind: str) -> Path | None:
    stems_dir = Path(stems_dir)
    names = _listing(stems_dir).get(video_id)
    if not names:
        return None
    marker = f"({kind})"
    for prefix in (f"{video_id}_raw90_", f"{video_id}_("):
        matches = sorted(
            n for n in names if n.startswith(prefix) and marker in n and n.endswith(".wav")
        )
        if matches:
            return stems_dir / matches[0]
    return None


def vocals_stem_path(video_id: str, stems_dir: Path | str) -> Path | None:
    return _stem_match(video_id, stems_dir, "vocals")


def other_stem_path(video_id: str, stems_dir: Path | str) -> Path | None:
    return _stem_match(video_id, stems_dir, "other")


def clip_background_ratio_db(video_id: str, stems_dir: Path | str) -> float:
    """Convenience: resolve both stems for a video id and compare them."""
    vocals = vocals_stem_path(video_id, stems_dir)
    other = other_stem_path(video_id, stems_dir)
    if vocals is None or other is None:
        return math.nan
    return background_ratio_db(vocals, other)
