"""Product tests for the vanalysis catalog filter.

User-visible rules (streams only — chatting, not singing, not collabs):

1. Keep a Holodex-shaped video row only if ALL of:
   a. ``type == "stream"`` (clips are dropped),
   b. ``topic_id`` is not ``"singing"`` (a missing/None topic is still
      kept — chatting streams often carry no topic),
   c. ``mentions`` is missing, None, or an empty list (any non-empty
      mentions = collab, dropped).
2. Filter, don't rewrite: kept rows keep their field values untouched
   and still include at least id, channel_id, available_at, topic_id.
3. Empty list in -> empty list out.

pick_monthly (Luna monthly batch — one eligible chatting stream per
calendar month):

4. ONE eligible stream per calendar month. The bucket is the (year,
   month) of available_at — the same month number in different years
   are distinct buckets. Empty months are ABSENT from the result, never
   filled with a placeholder or a rejected row; result is ordered by
   month ascending.
5. Within a month the highest score_video wins; a score tie is broken
   by newest available_at (same sort as pick_for_year: -score,
   -timestamp).
6. Rejected rows are NEVER picked — not even when they are the only
   stream of their month: collab mentions; drop topics singing/shorts/
   Original_Song/Music_Cover/Teaser/membersonly; duration < 900 s
   (900 s exactly is eligible); title karaoke/歌枠/ゲスト/guest;
   type != stream.
7. Picked rows are returned field-for-field unchanged.
8. Empty list in -> empty list out.

Fixture: tests/fixtures/holodex_videos.json — Holodex-like metadata only
(one good chatting stream, one clip, one singing stream, one collab).
No Cover/hololive audio, no real Holodex/YouTube API calls.
"""

import json
from pathlib import Path

import pytest

from vanalysis import catalog

TESTS_DIR = Path(__file__).resolve().parent
HOLODEX_FIXTURE = TESTS_DIR / "fixtures" / "holodex_videos.json"

GOOD_CHAT_ID = "goodchat01"


# ---------------------------------------------------------------- fixtures


def _load_fixture_videos() -> list[dict]:
    with open(HOLODEX_FIXTURE, encoding="utf-8") as f:
        return json.load(f)


def _video(**overrides) -> dict:
    """A minimal known-good chatting stream row, overridable per test."""
    row = {
        "id": "inline0001",
        "type": "stream",
        "channel_id": "UCgoodchan0001",
        "available_at": "2026-08-05T12:00:00.000Z",
        "topic_id": "chatting",
        "mentions": [],
    }
    row.update(overrides)
    return row


def _ids(videos: list[dict]) -> list[str]:
    return [v["id"] for v in videos]


# ------------------------------------------------------------------- tests


def test_filter_videos_keeps_only_good_chatting_stream() -> None:
    """Rule 1: from the fixture, only the good chatting stream survives —
    the clip (type), the singing stream (topic), and the collab (mentions)
    are all dropped."""
    kept = catalog.filter_videos(_load_fixture_videos())
    assert _ids(kept) == [GOOD_CHAT_ID], f"expected only {GOOD_CHAT_ID!r}, got {_ids(kept)}"


def test_filter_videos_does_not_rewrite_kept_rows() -> None:
    """Rule 2: the kept row is the fixture row, field for field."""
    fixture_videos = _load_fixture_videos()
    good_row = next(v for v in fixture_videos if v["id"] == GOOD_CHAT_ID)
    kept = catalog.filter_videos(fixture_videos)
    assert kept[0] == good_row, "filter_videos must not rewrite fields"


def test_filter_videos_kept_rows_have_core_fields() -> None:
    """Rule 2: every kept row still carries id, channel_id, available_at,
    topic_id (available_at is the timestamp this project relies on)."""
    kept = catalog.filter_videos(_load_fixture_videos())
    assert kept, "the good chatting stream should survive the filter"
    for row in kept:
        for field in ("id", "channel_id", "available_at", "topic_id"):
            assert field in row, f"kept row {row.get('id')!r} lost {field!r}"


def test_filter_videos_empty_input() -> None:
    """Rule 3: empty list in -> empty list out."""
    assert catalog.filter_videos([]) == []


def test_filter_videos_keeps_null_topic() -> None:
    """Rule 1b: a None topic_id is not 'singing' — keep the row."""
    row = _video(topic_id=None)
    assert _ids(catalog.filter_videos([row])) == ["inline0001"]


def test_filter_videos_keeps_missing_topic_key() -> None:
    """Rule 1b: no topic_id key at all is not 'singing' — keep the row."""
    row = _video()
    del row["topic_id"]
    assert _ids(catalog.filter_videos([row])) == ["inline0001"]


def test_filter_videos_keeps_null_and_missing_mentions() -> None:
    """Rule 1c: mentions None, or no mentions key, is not a collab — keep."""
    row_null = _video(id="mentions-null", mentions=None)
    row_absent = _video(id="mentions-absent")
    del row_absent["mentions"]
    assert _ids(catalog.filter_videos([row_null, row_absent])) == [
        "mentions-null",
        "mentions-absent",
    ]


def test_filter_videos_drops_singing_even_without_mentions() -> None:
    """Rule 1b: topic_id 'singing' is dropped regardless of mentions."""
    row = _video(id="singing-no-mentions", topic_id="singing")
    assert catalog.filter_videos([row]) == []


def test_filter_videos_drops_any_nonempty_mentions() -> None:
    """Rule 1c: a single mention is already a collab — drop the row,
    even when type/topic would otherwise be fine."""
    row = _video(id="one-mention", topic_id=None, mentions=[{"id": "UCx"}])
    assert catalog.filter_videos([row]) == []


# ------------------------------------------------- pick_monthly helpers


def _eligible(**overrides) -> dict:
    """A known-eligible chatting stream (passes every reject rule), with a
    duration long enough to score — overridable per test."""
    row = _video(
        title="zatsudan!",
        duration=3600,  # 60 min: 30–180 min bonus band
    )
    row.update(overrides)
    return row


def _month(video: dict) -> str:
    """Calendar month bucket as "YYYY-MM" (ISO available_at prefix)."""
    return str(video["available_at"])[:7]


def _months(videos: list[dict]) -> list[str]:
    return [_month(v) for v in videos]


# ------------------------------------------------------- pick_monthly


def test_pick_monthly_one_per_month_highest_score_wins() -> None:
    """Rule 4+5: two eligible streams in one month -> exactly one picked,
    the one with the higher score_video (chatting 60 min beats untopic'd
    30 min)."""
    strong = _eligible(id="strong0700", available_at="2025-07-05T12:00:00.000Z")
    weak = _eligible(
        id="weak005000", available_at="2025-07-20T12:00:00.000Z",
        topic_id=None, duration=1800,
    )
    assert catalog.score_video(strong) > catalog.score_video(weak), (
        "precondition: the strong row must actually score higher"
    )
    picked = catalog.pick_monthly([weak, strong])
    assert _ids(picked) == ["strong0700"]


def test_pick_monthly_score_tie_prefers_newest_available_at() -> None:
    """Rule 5: equal scores in one month -> the newest available_at wins
    (same tiebreak as pick_for_year: -score, -timestamp)."""
    older = _eligible(id="oldermay10", available_at="2025-05-10T12:00:00.000Z")
    newer = _eligible(id="newermay28", available_at="2025-05-28T12:00:00.000Z")
    assert catalog.score_video(older) == catalog.score_video(newer), (
        "precondition: both rows must tie on score"
    )
    picked = catalog.pick_monthly([older, newer])
    assert _ids(picked) == ["newermay28"]


def test_pick_monthly_empty_month_absent_not_filled() -> None:
    """Rule 4: a month with no eligible stream is simply absent — no
    placeholder, no fill from another month; result ordered by month."""
    jan = _eligible(id="janstream1", available_at="2025-01-08T12:00:00.000Z")
    mar = _eligible(id="marstream1", available_at="2025-03-14T12:00:00.000Z")
    picked = catalog.pick_monthly([mar, jan])
    assert _months(picked) == ["2025-01", "2025-03"], (
        "February must be absent (gap), and months must be ascending"
    )
    assert _ids(picked) == ["janstream1", "marstream1"]


def test_pick_monthly_same_month_number_distinct_years() -> None:
    """Rule 4: May 2024 and May 2025 are different buckets — both picked,
    not collapsed into one 'May'."""
    v2024 = _eligible(id="may2024row0", available_at="2024-05-10T12:00:00.000Z")
    v2025 = _eligible(id="may2025row0", available_at="2025-05-28T12:00:00.000Z")
    picked = catalog.pick_monthly([v2025, v2024])
    assert _ids(picked) == ["may2024row0", "may2025row0"]
    assert _months(picked) == ["2024-05", "2025-05"]


@pytest.mark.parametrize(
    ("flavor", "overrides"),
    [
        ("collab_mentions", {"mentions": [{"id": "UCother0001"}]}),
        ("topic_singing", {"topic_id": "singing"}),
        ("topic_shorts", {"topic_id": "shorts"}),
        ("topic_original_song", {"topic_id": "Original_Song"}),
        ("topic_music_cover", {"topic_id": "Music_Cover"}),
        ("topic_teaser", {"topic_id": "Teaser"}),
        ("topic_membersonly", {"topic_id": "membersonly"}),
        ("too_short_899s", {"duration": 899}),
        ("title_karaoke", {"title": "karaoke night!!"}),
        ("title_kagowaku", {"title": "歌枠"}),
        ("title_guest_jp", {"title": "ゲストと雑談"}),
        ("title_guest_en", {"title": "guest drop"}),
        ("clip_not_stream", {"type": "clip"}),
    ],
)
def test_pick_monthly_rejected_only_stream_never_picked(
    flavor: str, overrides: dict
) -> None:
    """Rule 6: whatever the rejection flavor, a month whose ONLY stream is
    rejected contributes nothing — the picker must not fall back to it."""
    good = _eligible(id="goodjanuary1", available_at="2025-01-08T12:00:00.000Z")
    bad = _eligible(
        id=f"rejected_{flavor}", available_at="2025-03-14T12:00:00.000Z",
        **overrides,
    )
    assert catalog.score_video(bad) is None, (
        f"precondition: {flavor} row must be rejected by score_video"
    )
    picked = catalog.pick_monthly([bad, good])
    assert _ids(picked) == ["goodjanuary1"], (
        f"{flavor} must never be picked, even as the only March stream"
    )
    assert _months(picked) == ["2025-01"], "March (rejected-only) must be absent"


def test_pick_monthly_duration_900_exactly_is_eligible() -> None:
    """Rule 6 boundary: the reject rule is duration < 900 s — 900 s exactly
    is eligible and gets picked when it is the month's only stream."""
    row = _eligible(id="borderline9", duration=900, available_at="2025-02-02T12:00:00.000Z")
    picked = catalog.pick_monthly([row])
    assert _ids(picked) == ["borderline9"]


def test_pick_monthly_returns_rows_unchanged() -> None:
    """Rule 7: picked rows are the input rows, field for field."""
    videos = [
        _eligible(id="rowkeep0001", available_at="2025-04-01T12:00:00.000Z"),
        _eligible(id="rowlost0002", available_at="2025-04-20T12:00:00.000Z",
                  topic_id=None, duration=1800),
    ]
    picked = catalog.pick_monthly(videos)
    assert len(picked) == 1
    assert picked[0] == videos[0], "pick_monthly must not rewrite fields"


def test_pick_monthly_empty_input() -> None:
    """Rule 8: empty list in -> empty list out."""
    assert catalog.pick_monthly([]) == []
