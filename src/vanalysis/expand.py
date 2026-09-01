"""Per-talent v1 sampling plan from Holodex channel listings (metadata only).

For every roster talent the channel's video listing is fetched from
``GET /api/v2/videos?channel_id=<id>&include=mentions&limit=50&offset=N``
(same required ``X-APIKEY`` / ``User-Agent: vanalysis/0.1`` headers as the
roster), cached per talent as the raw listing at
``<cache-dir>/<channel_id>.json`` (temp file + rename), and turned into a
v1 plan with the SHARED catalog machinery: eligible = ``filter_videos``
(no collabs/singing/clips), then ``pick_for_year(eligible, year, 4)`` —
4 streams per year, one per quarter, highest score, filled from leftovers.

Pacing (live runs): ~1.5 s between API pages; on HTTP 429 the rate-limit
pause (90 s) is slept ONCE and that talent FAILS with a logged warning —
per-talent errors never abort the plan run. Listings are cached, so a
crashed run resumes without re-hitting the API for finished talents, and
B3 reuses the same cache.

Metadata only: nothing here downloads media.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from vanalysis.catalog import _available_dt, filter_videos, pick_for_year, score_video
from vanalysis.roster import channel_headers

logger = logging.getLogger(__name__)

HOLODEX_VIDEOS_URL = "https://holodex.net/api/v2/videos"
PAGE = 50
PAGE_SLEEP_S = 1.5
RATE_LIMIT_SLEEP_S = 90.0
DISK_GB_PER_PICK = 0.16

Fetcher = Callable[[str], list[dict]]


class RateLimitedError(RuntimeError):
    """A Holodex 429 for one channel listing (after the one pause)."""

    def __init__(self, channel_id: str):
        super().__init__(f"Holodex rate-limited the listing for channel {channel_id}")
        self.channel_id = channel_id


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _http_fetcher(api_key: str) -> Fetcher:
    headers = channel_headers(api_key)

    def fetch(url: str) -> list[dict]:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
        if not isinstance(payload, list):
            raise ValueError(f"expected a JSON array from {url}")
        return payload

    return fetch


def _page_url(channel_id: str, offset: int) -> str:
    params = urllib.parse.urlencode(
        {
            "channel_id": channel_id,
            "include": "mentions",
            "limit": PAGE,
            "offset": offset,
        }
    )
    return f"{HOLODEX_VIDEOS_URL}?{params}"


def cache_path(cache_dir: Path | str, channel_id: str) -> Path:
    return Path(cache_dir) / f"{channel_id}.json"


def _read_cache(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"cache file {path} is not a JSON list")
    return payload


def _write_cache(path: Path, rows: list[dict]) -> None:
    """Atomic-enough: write a temp file, then rename over the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def fetch_channel_videos(
    channel_id: str,
    *,
    fetcher: Fetcher | None = None,
    api_key: str | None = None,
    cache_dir: Path | str | None = None,
) -> list[dict]:
    """Return the channel's raw Holodex video listing (a list of rows).

    With ``cache_dir``: an existing cache file is used verbatim and the
    fetcher is NOT called; a missing listing is fetched paginated (until a
    page shorter than ``PAGE``) and written to the cache. The default
    fetcher is a urllib GET carrying ``X-APIKEY`` and the pinned
    ``User-Agent``. A 429 sleeps ``RATE_LIMIT_SLEEP_S`` once, then raises
    ``RateLimitedError``.
    """
    if cache_dir is not None:
        path = cache_path(cache_dir, channel_id)
        if path.is_file():
            return _read_cache(path)
    if fetcher is None:
        if api_key is None:
            raise ValueError("fetch_channel_videos needs a fetcher or an api_key")
        fetcher = _http_fetcher(api_key)
    rows: list[dict] = []
    offset = 0
    while True:
        try:
            page = fetcher(_page_url(channel_id, offset))
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                _sleep(RATE_LIMIT_SLEEP_S)
                raise RateLimitedError(channel_id) from exc
            raise
        rows.extend(row for row in page if isinstance(row, dict))
        if len(page) < PAGE:
            break
        offset += PAGE
        _sleep(PAGE_SLEEP_S)
    if cache_dir is not None:
        _write_cache(cache_path(cache_dir, channel_id), rows)
    return rows


def _plan_talent(talent: dict, videos: list[dict]) -> dict:
    """Plan one talent from its raw listing via the shared catalog code."""
    eligible = filter_videos(videos)
    years = sorted(
        {when.year for video in eligible if (when := _available_dt(video)) is not None}
    )
    picks: list[dict] = []
    for year in years:
        for video in pick_for_year(eligible, year, 4):
            when = _available_dt(video)  # non-None: pick_for_year required it
            picks.append(
                {
                    "video_id": video["id"],
                    "year": when.year,
                    "quarter": (when.month - 1) // 3 + 1,
                    "published_at": video.get("published_at"),
                    "available_at": video.get("available_at"),
                    "score": score_video(video),
                }
            )
    picks.sort(key=lambda p: (p["year"], p["quarter"]))  # stable: keeps pick_for_year order
    return {
        "talent_id": talent.get("id"),
        "name": talent.get("name") or "",
        "group": talent.get("group") or "",
        "years_covered": sorted({p["year"] for p in picks}),
        "picks": picks,
        "pick_count": len(picks),
    }


def build_plan(
    roster: list[dict],
    cache_dir: Path | str,
    *,
    fetcher: Fetcher | None = None,
    api_key: str | None = None,
) -> dict:
    """Build the plan document for every roster talent.

    Per talent: fetch-or-cache the listing, plan it via the shared catalog
    code. Any per-talent failure (rate limit, network, bad data) is logged
    as a warning and that talent is skipped — the run continues. Output
    talents are sorted by pick_count descending, then name.
    """
    talents: list[dict] = []
    for talent in roster:
        channel_id = talent.get("id")
        if not channel_id:
            logger.warning("roster row without id skipped: %s", talent.get("name"))
            continue
        try:
            videos = fetch_channel_videos(
                channel_id, fetcher=fetcher, api_key=api_key, cache_dir=cache_dir
            )
            record = _plan_talent(talent, videos)
        except Exception as exc:  # per-talent errors never abort the plan run
            logger.warning(
                "talent %s (%s): %s: %s",
                channel_id,
                talent.get("name"),
                type(exc).__name__,
                exc,
            )
            continue
        talents.append(record)
    talents.sort(key=lambda record: (-record["pick_count"], record["name"]))
    total_picks = sum(record["pick_count"] for record in talents)
    return {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "roster_count": len(roster),
        "talents_planned": len(talents),
        "total_picks": total_picks,
        "est_disk_gb": round(total_picks * DISK_GB_PER_PICK, 2),
        "talents": talents,
    }


def run_plan(
    roster_path: Path | str,
    cache_dir: Path | str,
    out_path: Path | str,
    *,
    api_key: str | None = None,
    fetcher: Fetcher | None = None,
) -> dict:
    """Load the roster document, build the plan, write it to ``out_path``."""
    document = json.loads(Path(roster_path).read_text(encoding="utf-8"))
    talents = document["talents"] if isinstance(document, dict) else document
    payload = build_plan(talents, cache_dir, fetcher=fetcher, api_key=api_key)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload
