"""Per-clip stream metadata, recovered from the cached Holodex listings.

v1 stored a month per clip and nothing else about the stream, which leaves the
corpus unable to answer basic questions about itself. Two matter:

- **What kind of stream was this?** The corpus is not chatting-only — topic is
  a soft preference in selection, so game streams, ASMR, superchat readings and
  watchalongs are all in it. Stream type has to be a recorded covariate, not an
  assumption, and `catalog.reliability_band` has existed all along with zero
  call sites because there was nowhere to put its verdict.
- **Exactly when did it air?** A month-granularity date cannot support a
  regression on real time, and separating career time from calendar time — the
  corpus's central confound — is a regression on real time.

None of this needs the API: the cached channel listings already carry it. A
clip that cannot be resolved is marked, never guessed, because a fabricated
date corrupts a time regression silently.

See reference/schema.md and reference/sampling.md.
"""

from __future__ import annotations

from pathlib import Path

from .catalog import _TALK_TOPICS, reliability_band

DEFAULT_CACHE_DIR = Path("data/catalog/video_cache")


def load_video_cache(cache_dir: Path | str = DEFAULT_CACHE_DIR) -> dict[str, dict]:
    """Index every cached channel listing by video id.

    Later files win on a duplicate id, which is harmless: the same video cached
    under two channels carries the same metadata.
    """
    import json

    cache_dir = Path(cache_dir)
    index: dict[str, dict] = {}
    if not cache_dir.is_dir():
        return index
    for path in sorted(cache_dir.glob("*.json")):
        try:
            videos = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if not isinstance(videos, list):
            continue
        for video in videos:
            if isinstance(video, dict) and video.get("id"):
                index[video["id"]] = video
    return index


def clip_metadata(video_id: str, cache: dict[str, dict]) -> dict | None:
    """Stream facts for one measured clip, or None if the cache lacks it."""
    video = cache.get(video_id)
    if video is None:
        return None
    topic = video.get("topic_id")
    return {
        "title": video.get("title"),
        "topic_id": topic,
        "is_talk": topic in _TALK_TOPICS,
        "duration_s": video.get("duration"),
        "published_at": video.get("published_at"),
        # available_at is when the stream actually aired; published_at can be
        # the scheduling time for a stream that was set up in advance, so the
        # former is the better clock for a time regression.
        "available_at": video.get("available_at") or video.get("published_at"),
        "reliability_band": reliability_band(video),
    }


def enrich(records: list[dict], cache: dict[str, dict]) -> list[dict]:
    """Copies of ``records`` with stream metadata attached.

    Never mutates the input. v1's measurement files are frozen and are the
    baseline every v2 comparison is made against; enrichment feeds a new
    corpus, so it must not edit the old one in place.
    """
    enriched: list[dict] = []
    for record in records:
        row = dict(record)
        found = clip_metadata(str(record.get("id", "")), cache)
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
        enriched.append(row)
    return enriched
