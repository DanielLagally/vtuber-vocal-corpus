"""Building the v2 corpus from local audio, without disturbing v1.

v1's measurement files are frozen. They still feed the published site and they
are the baseline every v2 comparison is made against, so nothing here writes to
them; v2 is a new corpus in a new directory.

Each clip is re-measured from the audio still on disk, replaying the window
offsets v1 recorded. That matters more than it sounds: re-fetching a stream and
hunting a window again would produce a *different* 90 seconds, so any change in
the numbers would confound the new measurement method with a new sample.
Replaying the stored window holds the sample fixed and lets the method be the
only thing that varied.

Clips whose audio is gone are marked, not dropped and not re-measured from
nothing. They keep their v1 features and carry `legacy: true`, so a corrected
analysis can exclude them explicitly instead of silently mixing two pipelines'
output — which is the failure mode that makes a corpus unusable rather than
merely incomplete.

See reference/schema.md.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import metadata as metadata_module
from .acoustics import DEFAULT_CONFIG as DEFAULT_FEATURE_CONFIG
from .acoustics import FeatureConfig, clip_features
from .quality import DEFAULT_CONFIG as DEFAULT_QUALITY_CONFIG
from .quality import QualityConfig, verdict

DEFAULT_OUT_DIR = Path("data/measurements/v2")

#: v1 separated every clip unconditionally. Recording that as a fact about each
#: record — rather than leaving it implicit in the pipeline's history — is what
#: lets a later, conditional policy coexist with these records in one corpus.
V1_SEPARATION_APPLIED = True


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_records(
    v1_records: list[dict],
    *,
    cache: dict[str, dict],
    resolve_audio: Callable[[str], Path | None],
    extract: Callable[[Path, FeatureConfig], dict] | None = None,
    background_ratio: Callable[[str], float] | None = None,
    separation_applied: Callable[[str], bool] | None = None,
    config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
    quality_config: QualityConfig = DEFAULT_QUALITY_CONFIG,
    release: str = "unreleased",
    measured_at: str | None = None,
) -> list[dict]:
    """v2 records for one talent. Never mutates ``v1_records``."""
    extract = extract or (lambda path, cfg: clip_features(path, config=cfg))
    timestamp = measured_at or _now()

    out: list[dict] = []
    for source in v1_records:
        video_id = str(source.get("id", ""))
        row: dict = {
            "id": video_id,
            "month": source.get("month"),
            "score": source.get("score"),
            "window": source.get("window"),
            "corpus_release": release,
        }

        found = metadata_module.clip_metadata(video_id, cache)
        if found is None:
            row.update(
                {
                    "title": None,
                    "topic_id": None,
                    "is_talk": None,
                    "duration_s": None,
                    "published_at": None,
                    "available_at": None,
                    "reliability_band": None,
                    "metadata_resolved": False,
                }
            )
        else:
            row.update(found)
            row["metadata_resolved"] = True

        audio = resolve_audio(video_id)
        if audio is None:
            # Keep v1's numbers verbatim rather than inventing any. The record
            # is real; only our ability to re-measure it is missing.
            row.update(
                {
                    "features": dict(source.get("features") or {}),
                    "qc": dict(source.get("qc") or {}),
                    "tracker": source.get("tracker"),
                    "model": source.get("model"),
                    "source_audio": None,
                    "background_ratio_db": None,
                    "separation_applied": V1_SEPARATION_APPLIED,
                    "measured_at": None,
                    "legacy": True,
                    "legacy_reason": "audio_unavailable",
                }
            )
            out.append(row)
            continue

        features = extract(audio, config)
        ratio = background_ratio(video_id) if background_ratio else math.nan
        passed, reason = verdict(
            features, config=quality_config, require_contamination=True
        )
        row.update(
            {
                "features": features,
                "qc": {"pass": passed, "reason": reason},
                "tracker": config.tracker,
                "model": source.get("model"),
                "source_audio": str(audio),
                "background_ratio_db": None if ratio != ratio else float(ratio),
                "separation_applied": (
                    separation_applied(video_id)
                    if separation_applied
                    else V1_SEPARATION_APPLIED
                ),
                "measured_at": timestamp,
                "legacy": False,
                "legacy_reason": None,
            }
        )
        out.append(row)
    return out


@dataclass(frozen=True)
class ExtractionResult:
    """Measured features, plus how much of the batch failed.

    The counts are not bookkeeping. A build once reported success having
    measured nothing at all — every extraction had failed because the GPU was
    full, and each failure was swallowed on its own, so the run produced an
    empty corpus, labelled every clip "legacy", and exited zero. Tolerating one
    bad clip is right; tolerating all of them silently is not, and the
    difference is only visible if somebody counts.
    """

    features: dict[str, dict]
    failed: int
    errors: dict[str, str]

    @property
    def succeeded(self) -> int:
        return len(self.features)

    @property
    def attempted(self) -> int:
        return self.succeeded + self.failed

    @property
    def all_failed(self) -> bool:
        return self.attempted > 0 and self.succeeded == 0

    @property
    def failure_rate(self) -> float:
        return self.failed / self.attempted if self.attempted else 0.0


def _extract_one(job: tuple[str, FeatureConfig]) -> tuple[str, dict | None, str]:
    """Worker for the process pool. Module-level so it is picklable.

    A clip that fails returns its error rather than raising, so one unreadable
    file cannot abort a corpus-wide run — but the error is carried back rather
    than discarded, so the caller can tell one bad clip from a broken batch.
    """
    path, config = job
    try:
        return path, clip_features(Path(path), config=config), ""
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return path, None, f"{type(exc).__name__}: {exc}"


def extract_many(
    paths: list[Path],
    *,
    config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
    workers: int = 1,
    progress: Callable[[int, int], None] | None = None,
    extract: Callable[[Path, FeatureConfig], dict] | None = None,
) -> ExtractionResult:
    """Features for many clips, keyed by path string.

    Parallel by process rather than thread: the work is CPU-bound inside Praat
    and numpy, so threads would serialise on the GIL. ``extract`` overrides the
    measurement function and forces the serial path, for tests.
    """
    jobs = [(str(p), config) for p in paths]
    out: dict[str, dict] = {}
    errors: dict[str, str] = {}
    done = 0

    def record(path: str, features: dict | None, error: str) -> None:
        if features is not None:
            out[path] = features
        else:
            errors[path] = error

    if extract is not None or workers <= 1:
        for path, cfg in jobs:
            if extract is not None:
                try:
                    record(path, extract(Path(path), cfg), "")
                except Exception as exc:  # noqa: BLE001 - reported below
                    record(path, None, f"{type(exc).__name__}: {exc}")
            else:
                record(*_extract_one((path, cfg)))
            done += 1
            if progress:
                progress(done, len(jobs))
        return ExtractionResult(out, len(errors), errors)

    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for path, features, error in pool.map(_extract_one, jobs, chunksize=4):
            record(path, features, error)
            done += 1
            if progress:
                progress(done, len(jobs))
    return ExtractionResult(out, len(errors), errors)


def write_talent(out_dir: Path | str, talent: str, records: list[dict]) -> Path:
    """Write one talent's v2 records.

    The filename deliberately drops v1's ``_monthly`` suffix so the two corpora
    cannot be confused by a glob, and lands in a separate directory so v1's
    negated gitignore rules keep applying to v1 alone.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{talent}.json"
    path.write_text(json.dumps(records, indent=1), encoding="utf-8")
    return path


def summarise(records: list[dict]) -> dict:
    """Counts a build report can quote without recomputing them elsewhere."""
    remeasured = [r for r in records if not r.get("legacy")]
    passed = [r for r in remeasured if (r.get("qc") or {}).get("pass")]
    reasons: dict[str, int] = {}
    for record in remeasured:
        reason = (record.get("qc") or {}).get("reason")
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "total": len(records),
        "remeasured": len(remeasured),
        "legacy": len(records) - len(remeasured),
        "qc_pass": len(passed),
        "qc_fail_reasons": reasons,
    }
