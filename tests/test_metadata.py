"""Per-clip stream metadata, recovered from the cached Holodex listings.

v1 stored only a month per clip, so the corpus cannot answer basic questions
about itself: what kind of stream a measurement came from, exactly when it
aired, or how long it ran. Those are not decorations — stream type has to enter
an analysis as a covariate rather than an assumption, and a month-granularity
date cannot carry a regression on real time.

None of it needs the API. The cached channel listings already hold it for
essentially the whole corpus, so this is a lookup, not a re-fetch.

See reference/schema.md and reference/sampling.md.
"""

from __future__ import annotations

import json

import pytest

from vvc import metadata


@pytest.fixture
def cache_dir(tmp_path):
    videos = [
        {
            "id": "talkclip",
            "title": "zatsudan",
            "type": "stream",
            "topic_id": "talk",
            "published_at": "2021-03-05T15:01:09.000Z",
            "available_at": "2021-03-05T14:55:00.000Z",
            "duration": 3600,
        },
        {
            "id": "gameclip",
            "title": "minecraft",
            "type": "stream",
            "topic_id": "minecraft",
            "published_at": "2022-07-11T09:00:00.000Z",
            "available_at": "2022-07-11T09:00:00.000Z",
            "duration": 7200,
        },
        {
            "id": "watchclip",
            "title": "movie night",
            "type": "stream",
            "topic_id": "watchalong",
            "published_at": "2023-01-02T03:04:05.000Z",
            "available_at": "2023-01-02T03:04:05.000Z",
            "duration": 5400,
        },
    ]
    directory = tmp_path / "video_cache"
    directory.mkdir()
    (directory / "UCchannel.json").write_text(json.dumps(videos), encoding="utf-8")
    return directory


class TestLoadVideoCache:
    def test_indexes_every_cached_video_by_id(self, cache_dir):
        cache = metadata.load_video_cache(cache_dir)
        assert set(cache) == {"talkclip", "gameclip", "watchclip"}

    def test_an_empty_cache_directory_is_not_an_error(self, tmp_path):
        assert metadata.load_video_cache(tmp_path) == {}


class TestClipMetadata:
    def test_recovers_exact_date_topic_and_duration(self, cache_dir):
        cache = metadata.load_video_cache(cache_dir)
        row = metadata.clip_metadata("talkclip", cache)
        assert row["topic_id"] == "talk"
        assert row["duration_s"] == 3600
        assert row["available_at"].startswith("2021-03-05")

    def test_an_unknown_id_returns_none_rather_than_a_guess(self, cache_dir):
        """A fabricated date would silently corrupt a time regression."""
        cache = metadata.load_video_cache(cache_dir)
        assert metadata.clip_metadata("nosuchid", cache) is None

    def test_carries_the_reliability_band_the_pipeline_already_computes(self, cache_dir):
        """catalog.reliability_band has existed with zero call sites. Storing
        its verdict is what makes stream type usable downstream."""
        cache = metadata.load_video_cache(cache_dir)
        assert metadata.clip_metadata("watchclip", cache)["reliability_band"] == "low"
        assert metadata.clip_metadata("talkclip", cache)["reliability_band"] == "high"

    def test_distinguishes_talk_from_game_streams(self, cache_dir):
        """The corpus is not chatting-only; the record must say which it was."""
        cache = metadata.load_video_cache(cache_dir)
        assert metadata.clip_metadata("talkclip", cache)["is_talk"] is True
        assert metadata.clip_metadata("gameclip", cache)["is_talk"] is False


class TestEnrich:
    def test_adds_metadata_to_records_without_touching_measurements(self, cache_dir):
        cache = metadata.load_video_cache(cache_dir)
        records = [
            {"id": "talkclip", "month": "2021-03", "features": {"median_f0": 300.0}}
        ]
        enriched = metadata.enrich(records, cache)
        assert enriched[0]["features"] == {"median_f0": 300.0}
        assert enriched[0]["topic_id"] == "talk"

    def test_does_not_mutate_the_input_records(self, cache_dir):
        """Enrichment feeds a new corpus; it must never edit the frozen one."""
        cache = metadata.load_video_cache(cache_dir)
        records = [{"id": "talkclip", "month": "2021-03"}]
        metadata.enrich(records, cache)
        assert "topic_id" not in records[0]

    def test_unresolvable_records_are_marked_not_dropped(self, cache_dir):
        """A clip we lack metadata for is still a measurement."""
        cache = metadata.load_video_cache(cache_dir)
        enriched = metadata.enrich([{"id": "ghost", "month": "2020-01"}], cache)
        assert len(enriched) == 1
        assert enriched[0]["topic_id"] is None
        assert enriched[0]["metadata_resolved"] is False
