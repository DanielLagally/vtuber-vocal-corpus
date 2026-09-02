"""Product tests for the per-talent v1 sampling plan (metadata only).

User-visible rules:

1. Per talent, picks come from the REAL catalog machinery: eligible =
   ``filter_videos`` over the channel listing, then ``pick_for_year`` with
   n=4 for every year present — 4 streams per year, one per quarter
   (highest score), filled from leftovers when a quarter is empty.
2. Junk never reaches picks: collabs (mentions), clips, singing, shorts,
   membersonly, Music_Cover, too-short streams, and karaoke/guest titles
   are all excluded by the shared catalog filter/score code.
3. Channel listings are cached per talent at
   ``<cache-dir>/<channel_id>.json`` (raw listing list, written via
   temp-file + rename). If the cache file exists it is used and the
   fetcher is NOT called; if it is missing the listing is fetched
   paginated (``channel_id`` + ``include=mentions`` + ``limit=50`` +
   ``offset``) and written to the cache. No leftover temp files.
4. A Holodex 429 sleeps the rate-limit pause once, then FAILS that one
   talent with a logged warning and the plan run CONTINUES (per-talent
   errors never abort the plan; a failed talent is simply absent).
5. Plan schema: ``{"built_at", "roster_count", "talents_planned",
   "total_picks", "est_disk_gb", "talents": [...]}``; talents sorted by
   pick_count descending then name; est_disk_gb = total_picks x 0.16.
6. A talent with ZERO eligible videos is still listed, with
   ``picks: []`` and ``pick_count: 0``.
7. The ``plan`` CLI subcommand loads ``--roster``, uses ``--cache-dir``
   and writes ``-o`` (defaults: data/catalog/roster.json,
   data/catalog/video_cache, data/catalog/expansion_plan.json).

Fixture rows are synthetic holodex_videos-style metadata only (no Cover
media, no network: every fetch goes through an injectable fake fetcher;
sleeps are monkeypatched).
"""

import json
import logging
import urllib.error
import urllib.parse
from pathlib import Path

import pytest

from vvc import expand
from vvc.catalog import score_video
from vvc.expand import RateLimitedError, build_plan, fetch_channel_videos

T1 = "UCtalentA0000001"
T2 = "UCtalentB0000002"


# ---------------------------------------------------------------- fixtures


def _video(
    video_id: str,
    *,
    channel: str = T1,
    year: int = 2024,
    month: int = 1,
    day: int = 15,
    duration: int = 7200,
    topic: str | None = "chatting",
    type: str = "stream",
    mentions: list | None = None,
    title: str = "Free chat — talking about games",
) -> dict:
    """A holodex_videos-style row (synthetic metadata only)."""
    stamp = f"{year:04d}-{month:02d}-{day:02d}T12:00:00.000Z"
    return {
        "id": video_id,
        "title": title,
        "type": type,
        "channel_id": channel,
        "published_at": stamp,
        "available_at": stamp,
        "duration": duration,
        "topic_id": topic,
        "mentions": mentions if mentions is not None else [],
    }


def _talent(tid: str, name: str, group: str = "0th Generation") -> dict:
    return {"id": tid, "name": name, "group": group, "type": "vtuber", "inactive": False}


class _PagedFetcher:
    """Fake fetcher: per-channel queue of pages; records every URL."""

    def __init__(self, pages_by_channel: dict[str, list[list[dict]]]):
        self.pages_by_channel = {k: list(v) for k, v in pages_by_channel.items()}
        self.urls: list[str] = []
        self.api_key: str | None = None

    def __call__(self, url: str) -> list[dict]:
        self.urls.append(url)
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        pages = self.pages_by_channel.get(query["channel_id"][0], [])
        if not pages:
            return []
        return pages.pop(0)


# ------------------------------------------- a. quarterly pick semantics


def test_picks_respect_quarterly_semantics_via_real_pick_for_year(tmp_path: Path) -> None:
    """Rule 1: 2024 has two clear quarters (Q1: A newest 70, B 70, D 50;
    Q3: C 70). pick_for_year must take the Q1 winner and the Q3 winner
    first, then top up to 4 from leftovers. The plan records year,
    quarter (from the month), raw date strings, and score per pick."""
    rows = [
        _video("chatJanQ1", year=2024, month=1, day=20),  # Q1, 70, older
        _video("chatFebQ1", year=2024, month=2, day=10),  # Q1, 70, newest
        _video("miscDq1", year=2024, month=3, day=5, topic=None),  # Q1, 50
        _video("chatCq3", year=2024, month=8, day=15, duration=3600),  # Q3, 70
    ]
    fetcher = _PagedFetcher({T1: [rows]})
    payload = build_plan([_talent(T1, "Alpha Talent")], tmp_path, fetcher=fetcher)

    assert len(payload["talents"]) == 1
    record = payload["talents"][0]
    assert record["talent_id"] == T1
    assert record["years_covered"] == [2024]
    assert record["pick_count"] == 4
    got = [(p["video_id"], p["quarter"]) for p in record["picks"]]
    assert got == [
        ("chatFebQ1", 1),
        ("chatJanQ1", 1),
        ("miscDq1", 1),
        ("chatCq3", 3),
    ]
    by_id = {p["video_id"]: p for p in record["picks"]}
    assert set(by_id) == {"chatJanQ1", "chatFebQ1", "miscDq1", "chatCq3"}
    assert set(by_id["chatFebQ1"]) == {
        "video_id",
        "year",
        "quarter",
        "published_at",
        "available_at",
        "score",
    }
    assert by_id["chatFebQ1"]["year"] == 2024
    assert by_id["chatFebQ1"]["score"] == pytest.approx(score_video(rows[1]))
    assert by_id["chatFebQ1"]["published_at"] == "2024-02-10T12:00:00.000Z"
    assert by_id["chatFebQ1"]["available_at"] == "2024-02-10T12:00:00.000Z"
    assert by_id["miscDq1"]["score"] == pytest.approx(50.0)


# ------------------------------------- b. junk rows never reach the picks


def test_collab_singing_shorts_rows_never_appear_in_picks(tmp_path: Path) -> None:
    """Rule 2: the shared catalog filter + score machinery must keep every
    junk flavor out of the plan's picks."""
    junk = [
        _video("cliprow", type="clip", duration=60),
        _video("singrow", topic="singing"),
        _video("collabrow", mentions=[{"id": "UCother", "name": "Other"}]),
        _video("shortstype", type="shorts", duration=60),
        _video("topicshorts", topic="shorts"),
        _video("coverrow", topic="Music_Cover"),
        _video("membersrow", topic="membersonly"),
        _video("shortrow", duration=600),  # < 15 min
        _video("karaokerow", title="karaoke night"),
    ]
    good = _video("goodrow", year=2023, month=5)
    fetcher = _PagedFetcher({T1: [[good, *junk]]})
    payload = build_plan([_talent(T1, "Alpha Talent")], tmp_path, fetcher=fetcher)

    record = payload["talents"][0]
    picked_ids = {p["video_id"] for p in record["picks"]}
    assert picked_ids == {"goodrow"}
    assert record["years_covered"] == [2023]
    assert record["pick_count"] == 1


# ------------------------------------------------------- c. cache behavior


def test_existing_cache_used_and_fetcher_not_called(tmp_path: Path) -> None:
    """Rule 3: an existing <cache_dir>/<channel_id>.json listing is used
    verbatim and the fetcher is never invoked."""
    cached_rows = [_video("fromcache", year=2022, month=4)]
    cache_file = tmp_path / f"{T1}.json"
    cache_file.write_text(json.dumps(cached_rows), encoding="utf-8")

    def forbidden(url: str) -> list[dict]:
        raise AssertionError(f"fetcher must not be called when cache exists: {url}")

    payload = build_plan([_talent(T1, "Alpha Talent")], tmp_path, fetcher=forbidden)
    record = payload["talents"][0]
    assert [p["video_id"] for p in record["picks"]] == ["fromcache"]
    assert record["pick_count"] == 1


def test_missing_cache_fetched_paginated_and_written_atomically(tmp_path: Path) -> None:
    """Rule 3: a missing cache listing is fetched (channel_id +
    include=mentions + limit + offset params), written to the cache as the
    RAW listing, and leaves no temp files behind."""
    page1 = [_video(f"vid{i:03d}", month=(i % 12) + 1, year=2021) for i in range(50)]
    page2 = [_video("lastvid", year=2021, month=12)]
    fetcher = _PagedFetcher({T1: [page1, page2]})

    payload = build_plan([_talent(T1, "Alpha Talent")], tmp_path, fetcher=fetcher)

    assert len(fetcher.urls) == 2, "full 50-row page must trigger page 2"
    queries = [urllib.parse.parse_qs(urllib.parse.urlparse(u).query) for u in fetcher.urls]
    assert all(q["channel_id"] == [T1] for q in queries)
    assert all(q["include"] == ["mentions"] for q in queries)
    assert all(q["limit"] == ["50"] for q in queries)
    assert [q["offset"][0] for q in queries] == ["0", "50"]
    assert all(u.startswith("https://holodex.net/api/v2/videos") for u in fetcher.urls)

    cache_file = tmp_path / f"{T1}.json"
    assert cache_file.is_file()
    assert json.loads(cache_file.read_text(encoding="utf-8")) == [*page1, *page2]
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"{T1}.json"], (
        "cache write must be temp+rename: no temp files may survive"
    )
    assert payload["talents"][0]["pick_count"] == 4


def test_fetch_channel_videos_direct_api_and_cache(tmp_path: Path) -> None:
    """Rule 3 at the function level: no cache_dir -> no cache files;
    with cache_dir the second call reuses the cache without fetching."""
    fetcher = _PagedFetcher({T1: [[_video("direct1")]]})

    rows = fetch_channel_videos(T1, fetcher=fetcher)
    assert [r["id"] for r in rows] == ["direct1"]
    assert not list(tmp_path.iterdir()), "no cache_dir -> no cache writes"

    rows = fetch_channel_videos(T1, fetcher=fetcher, cache_dir=tmp_path)
    assert len(fetcher.urls) == 2
    assert (tmp_path / f"{T1}.json").is_file()
    fetcher.urls.clear()
    rows_again = fetch_channel_videos(T1, fetcher=fetcher, cache_dir=tmp_path)
    assert rows_again == rows
    assert fetcher.urls == [], "second call must be served from the cache"


# ------------------------------------- d. 429: talent skipped, run continues


def test_rate_limited_talent_skipped_and_plan_continues(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    """Rule 4: a 429 sleeps the rate-limit pause ONCE, logs a warning for
    that talent, and the plan continues with the other talents."""
    sleeps: list[float] = []
    monkeypatch.setattr(expand, "_sleep", lambda s: sleeps.append(s))

    def rate_limited(url: str) -> list[dict]:
        raise urllib.error.HTTPError(url, 429, "Too Many Requests", None, None)

    good = _PagedFetcher({T2: [[_video("t2row", channel=T2, year=2023, month=6)]]})

    def flaky(url: str) -> list[dict]:
        if T1 in url:
            return rate_limited(url)
        return good(url)

    with caplog.at_level("WARNING"):
        payload = build_plan(
            [_talent(T1, "Alpha Talent"), _talent(T2, "Beta Talent")],
            tmp_path,
            fetcher=flaky,
        )

    assert sleeps.count(90.0) == 1, "429 must sleep the rate-limit pause exactly once"
    assert [t["talent_id"] for t in payload["talents"]] == [T2]
    assert any(T1 in rec.getMessage() for rec in caplog.records)
    assert any(rec.levelno == logging.WARNING for rec in caplog.records), "must be a warning"


def test_generic_talent_error_skipped_and_plan_continues(tmp_path: Path, caplog) -> None:
    """Rule 4: any per-talent failure (here: a fetcher bug) is logged and
    the plan run continues — it must never abort."""
    def broken(url: str) -> list[dict]:
        raise ValueError("boom")

    good = _PagedFetcher({T2: [[_video("t2row", channel=T2, year=2023, month=6)]]})

    def mixed(url: str) -> list[dict]:
        if T1 in url:
            return broken(url)
        return good(url)

    with caplog.at_level("WARNING"):
        payload = build_plan(
            [_talent(T1, "Alpha Talent"), _talent(T2, "Beta Talent")],
            tmp_path,
            fetcher=mixed,
        )
    assert [t["talent_id"] for t in payload["talents"]] == [T2]
    assert any("Alpha Talent" in rec.getMessage() for rec in caplog.records)


def test_rate_limited_error_type_carries_channel(tmp_path: Path, monkeypatch) -> None:
    """Rule 4: the rate-limit error is a distinct, channel-tagged failure."""
    monkeypatch.setattr(expand, "_sleep", lambda s: None)
    fetcher = _PagedFetcher({})

    def rate_limited(url: str) -> list[dict]:
        raise urllib.error.HTTPError(url, 429, "Too Many Requests", None, None)

    monkeypatch.setattr(expand, "_http_fetcher", lambda key: rate_limited)
    with pytest.raises(RateLimitedError) as excinfo:
        fetch_channel_videos(T1, api_key="k", cache_dir=tmp_path)
    assert excinfo.value.channel_id == T1


# ------------------------------------------------ e. schema + sort order


def test_plan_schema_fields_and_sort_order(tmp_path: Path) -> None:
    """Rule 5: exact top-level fields, correct aggregates, talents sorted
    by pick_count desc then name, est_disk_gb = picks x 0.16."""
    pages = {
        "UCbig00000000001": [
            [
                _video(f"big{i}", channel="UCbig00000000001", year=2022, month=2 * i + 1)
                for i in range(4)
            ]
        ],
        "UCalpha000000001": [[_video("arow", channel="UCalpha000000001", year=2023, month=5)]],
        "UCzeta0000000001": [[_video("zrow", channel="UCzeta0000000001", year=2023, month=6)]],
        "UCempty000000001": [
            [
                _video(
                    "collabonly",
                    channel="UCempty000000001",
                    year=2023,
                    month=7,
                    mentions=[{"id": "UCother", "name": "Other"}],
                )
            ]
        ],
    }
    roster = [
        _talent("UCzeta0000000001", "Zeta Talent"),
        _talent("UCalpha000000001", "Alpha Talent"),
        _talent("UCbig00000000001", "Big Talent"),
        _talent("UCempty000000001", "Empty Talent"),
    ]
    payload = build_plan(roster, tmp_path, fetcher=_PagedFetcher(pages))

    assert set(payload) == {
        "built_at",
        "roster_count",
        "talents_planned",
        "total_picks",
        "est_disk_gb",
        "talents",
    }
    assert payload["roster_count"] == 4
    assert payload["talents_planned"] == 4
    assert payload["total_picks"] == 6
    assert payload["est_disk_gb"] == pytest.approx(6 * 0.16)
    from datetime import datetime

    parsed = datetime.fromisoformat(payload["built_at"])
    assert parsed.tzinfo is not None
    assert [t["name"] for t in payload["talents"]] == [
        "Big Talent",  # 4 picks
        "Alpha Talent",  # 1 pick, name before Zeta
        "Zeta Talent",  # 1 pick
        "Empty Talent",  # 0 picks last
    ]


# ------------------------------------- f. zero-eligible talent still listed


def test_empty_eligible_talent_listed_with_zero_picks(tmp_path: Path) -> None:
    """Rule 6: a talent whose listing has no eligible video (only a collab
    here) is still in the plan with picks [] and pick_count 0."""
    collab = _video(
        "onlycollab", year=2024, month=3, mentions=[{"id": "UCx", "name": "X"}]
    )
    fetcher = _PagedFetcher({T1: [[collab]]})
    payload = build_plan([_talent(T1, "Alpha Talent", group="mekPark")], tmp_path, fetcher=fetcher)

    record = payload["talents"][0]
    assert record == {
        "talent_id": T1,
        "name": "Alpha Talent",
        "group": "mekPark",
        "years_covered": [],
        "picks": [],
        "pick_count": 0,
    }
    assert payload["talents_planned"] == 1
    assert payload["total_picks"] == 0


# --------------------------------------------------------- g. CLI wiring


def test_cli_plan_wiring(tmp_path: Path, monkeypatch, capsys) -> None:
    """Rule 7: `vvc plan` loads --roster, keys the default fetcher
    with the Holodex key, caches per talent under --cache-dir, and writes
    the plan document to -o."""
    import vvc.__main__ as cli

    roster_path = tmp_path / "roster.json"
    roster_path.write_text(
        json.dumps(
            {
                "fetched_at": "2026-09-01T00:00:00+00:00",
                "count": 2,
                "talents": [
                    _talent(T1, "Alpha Talent"),
                    _talent(T2, "Beta Talent"),
                ],
            }
        ),
        encoding="utf-8",
    )
    cache_dir = tmp_path / "video_cache"
    out_path = tmp_path / "expansion_plan.json"

    monkeypatch.setattr(cli, "_holodex_key", lambda: "test-key")
    fetcher = _PagedFetcher(
        {
            T1: [[_video("t1row", year=2024, month=4)]],
            T2: [[_video("t2row", channel=T2, year=2023, month=6)]],
        }
    )

    def factory(api_key: str):
        fetcher.api_key = api_key
        return fetcher

    monkeypatch.setattr(expand, "_http_fetcher", factory)

    cli.main(
        [
            "plan",
            "--roster",
            str(roster_path),
            "--cache-dir",
            str(cache_dir),
            "-o",
            str(out_path),
        ]
    )

    assert out_path.is_file()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["roster_count"] == 2
    assert payload["talents_planned"] == 2
    assert payload["total_picks"] == 2
    assert fetcher.api_key == "test-key"
    assert sorted(p.name for p in cache_dir.iterdir()) == [f"{T1}.json", f"{T2}.json"]
    assert "2/2" in capsys.readouterr().out
