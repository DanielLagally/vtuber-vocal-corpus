from __future__ import annotations

from datetime import datetime

_DROP_TOPICS = frozenset(
    {
        "singing",
        "shorts",
        "Original_Song",
        "Music_Cover",
        "Teaser",
        "membersonly",
    }
)
_TALK_TOPICS = frozenset({"talk", "chatting", "morning"})
_TITLE_DROP = ("karaoke", "歌枠", "ゲスト", "guest")
_MIN_DURATION_S = 900
_OFFICIAL_GROUPS = frozenset({"Official", "Misc"})


def filter_videos(videos: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for video in videos:
        if video.get("type") != "stream":
            continue
        if video.get("topic_id") == "singing":
            continue
        mentions = video.get("mentions")
        if mentions:
            continue
        kept.append(video)
    return kept


def is_hololive_talent(channel: dict) -> bool:
    if channel.get("type") != "vtuber" or channel.get("inactive"):
        return False
    group = channel.get("suborg") or channel.get("group") or ""
    if group in _OFFICIAL_GROUPS:
        return False
    if "HOLOSTARS" in group.upper():
        return False
    return True


def reject_reason(video: dict) -> str | None:
    if video.get("type") != "stream":
        return "not_stream"
    topic = video.get("topic_id")
    if topic in _DROP_TOPICS:
        return f"topic:{topic}"
    if video.get("mentions"):
        return "collab"
    duration = video.get("duration") or 0
    if duration < _MIN_DURATION_S:
        return "too_short"
    title = video.get("title") or ""
    lowered = title.lower()
    for needle in _TITLE_DROP:
        if needle in lowered or needle in title:
            if needle in ("ゲスト", "guest"):
                return "title_guest"
            return "title_singing"
    return None


def score_video(video: dict) -> float | None:
    if reject_reason(video) is not None:
        return None
    topic = video.get("topic_id")
    if topic in _TALK_TOPICS:
        score = 40.0
    elif topic == "watchalong":
        score = 0.0
    elif topic is None:
        score = 20.0
    else:
        score = 10.0
    minutes = (video.get("duration") or 0) / 60.0
    if 30 <= minutes <= 180:
        score += 30.0
    elif minutes > 180:
        score += 20.0
    else:
        score += 10.0
    return score


def reliability_band(video: dict) -> str | None:
    if reject_reason(video) is not None:
        return None
    topic = video.get("topic_id")
    minutes = (video.get("duration") or 0) / 60.0
    talk = topic in _TALK_TOPICS
    if topic == "watchalong":
        return "low"
    if talk and 30 <= minutes <= 180:
        return "high"
    if minutes < 30:
        return "low"
    return "mid"


def _available_dt(video: dict) -> datetime | None:
    raw = video.get("available_at") or video.get("published_at")
    if not raw:
        return None
    if isinstance(raw, datetime):
        return raw
    text = str(raw).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def pick_monthly(videos: list[dict]) -> list[dict]:
    by_month: dict[str, list[tuple[float, float, dict]]] = {}
    for video in videos:
        points = score_video(video)
        when = _available_dt(video)
        if points is None or when is None:
            continue
        key = f"{when.year:04d}-{when.month:02d}"
        by_month.setdefault(key, []).append((points, when.timestamp(), video))
    picked: list[dict] = []
    for key in sorted(by_month):
        bucket = sorted(by_month[key], key=lambda item: (-item[0], -item[1]))
        picked.append(bucket[0][2])
    return picked


def pick_for_year(videos: list[dict], year: int, n: int = 4) -> list[dict]:
    scored: list[tuple[float, dict, datetime]] = []
    for video in videos:
        points = score_video(video)
        when = _available_dt(video)
        if points is None or when is None or when.year != year:
            continue
        scored.append((points, video, when))
    by_quarter: dict[int, list[tuple[float, dict, datetime]]] = {1: [], 2: [], 3: [], 4: []}
    for item in scored:
        by_quarter[(item[2].month - 1) // 3 + 1].append(item)
    picked: list[dict] = []
    used: set[str] = set()
    for q in (1, 2, 3, 4):
        bucket = sorted(by_quarter[q], key=lambda it: (-it[0], -it[2].timestamp()))
        if not bucket:
            continue
        video = bucket[0][1]
        picked.append(video)
        used.add(video["id"])
        if len(picked) == n:
            return picked
    rest = sorted(scored, key=lambda it: (-it[0], -it[2].timestamp()))
    for points, video, when in rest:
        if video["id"] in used:
            continue
        picked.append(video)
        used.add(video["id"])
        if len(picked) == n:
            break
    return picked
