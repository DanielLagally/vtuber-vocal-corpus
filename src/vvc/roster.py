"""Hololive talent roster from the Holodex channels endpoint (metadata only).

Roster rule (D13, scout-verified): keep a channel as a talent iff
``type == "vtuber"`` ∧ ``inactive is False`` ∧ ``group`` not in
``{"Official", "Misc"}`` ∧ ``group`` does not start with ``"HOLOSTARS"``
— i.e. the active JP/EN/ID/DEV_IS talent gens. Fail-closed: rows missing
``id``/``group``/``inactive`` (or a non-``vtuber`` type) are dropped,
never guessed into the roster.

The endpoint's default sort is NOT page-stable, so ``fetch_channels``
pages the whole listing and deduplicates by channel ``id``. Every
request carries ``X-APIKEY`` and ``User-Agent: vvc/0.1`` —
Holodex returns 403 without the User-Agent.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from vvc.holodex import holodex_key

HOLODEX_CHANNELS_URL = "https://holodex.net/api/v2/channels"
PAGE = 50
USER_AGENT = "vvc/0.1"

# D13: channels out of the speech corpus (brand channels + male branch).
_DROP_GROUPS = {"Official", "Misc"}
_DROP_GROUP_PREFIX = "HOLOSTARS"

Fetcher = Callable[[str], list[dict]]


def channel_headers(api_key: str) -> dict[str, str]:
    """Headers every Holodex request must carry (403 without the UA)."""
    return {"X-APIKEY": api_key, "User-Agent": USER_AGENT}


def filter_talents(channels: list[dict]) -> list[dict]:
    """Apply the D13 roster rule; output sorted by (group, name).

    Kept rows are returned field-for-field unchanged. Fail-closed: a row
    missing ``id``/``group``/``inactive``, a null/empty ``group``, a
    ``inactive`` that is not exactly ``False``, or a non-``vtuber`` type
    is dropped.
    """
    kept: list[dict] = []
    for row in channels:
        if not isinstance(row, dict):
            continue
        if not row.get("id"):
            continue
        if row.get("type") != "vtuber":
            continue
        if row.get("inactive") is not False:
            continue
        group = row.get("group")
        if not isinstance(group, str) or not group:
            continue
        if group in _DROP_GROUPS or group.startswith(_DROP_GROUP_PREFIX):
            continue
        kept.append(row)
    return sorted(kept, key=lambda row: (row["group"], row.get("name") or ""))


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


def _page_url(org: str, offset: int) -> str:
    params = urllib.parse.urlencode(
        {"org": org, "type": "vtuber", "limit": PAGE, "offset": offset}
    )
    return f"{HOLODEX_CHANNELS_URL}?{params}"


def fetch_channels(
    fetcher: Fetcher | None = None,
    org: str = "Hololive",
    api_key: str | None = None,
) -> list[dict]:
    """Fetch every channel page for ``org`` and dedupe rows by ``id``.

    Paginates (``limit``/``offset``) until a page shorter than ``PAGE``
    (an empty page included). The default fetcher is a urllib GET with
    the required ``X-APIKEY``/``User-Agent`` headers. Rows without an
    ``id`` are skipped; a repeated ``id`` (unstable default sort) keeps
    its first occurrence only.
    """
    if fetcher is None:
        if api_key is None:
            api_key = holodex_key()
        fetcher = _http_fetcher(api_key)
    rows: list[dict] = []
    seen: set[str] = set()
    offset = 0
    while True:
        page = fetcher(_page_url(org, offset))
        for row in page:
            channel_id = row.get("id")
            if not channel_id or channel_id in seen:
                continue
            seen.add(channel_id)
            rows.append(row)
        if len(page) < PAGE:
            break
        offset += PAGE
    return rows


def write_roster(talents: list[dict], out_path: Path | str) -> dict:
    """Write the roster document:
    ``{"fetched_at": <ISO-8601 UTC>, "count": N, "talents": [...]}``."""
    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "count": len(talents),
        "talents": talents,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload
