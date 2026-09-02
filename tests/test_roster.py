"""Product tests for the hololive talent roster (Holodex channels, metadata only).

User-visible rules:

1. A Holodex channel row is a roster talent only if ALL of:
   a. ``type == "vtuber"`` (subbers/clippers are dropped),
   b. ``inactive is False`` (graduated/retired channels are dropped —
      e.g. a Gura-shaped row with ``inactive: true`` must not appear),
   c. ``group`` is present and not in ``{"Official", "Misc"}`` and does
      not start with ``"HOLOSTARS"`` (main/music/earth channels and the
      male branch are out of scope for the corpus),
   d. ``id`` is present.
   Fail-closed: a row missing ANY of these fields (``group``/``inactive``
   missing or null, missing id) is DROPPED, never guessed into the roster.
2. Filter, don't rewrite: kept rows keep their field values untouched.
3. ``fetch_channels`` pages the Holodex channels endpoint until a short
   page (page length < limit), passing ``org``, ``type=vtuber``,
   ``limit`` and ``offset`` query params, and DEDUPLICATES by ``id`` —
   the API default sort is not page-stable, so the same channel can
   appear on two pages and must end up in the result exactly once.
4. Requests carry BOTH required headers: ``X-APIKEY`` and
   ``User-Agent: vvc/0.1`` (Holodex returns 403 without a UA).
5. Output order is deterministic: talents are sorted by ``(group, name)``.
6. The saved roster file has the schema
   ``{"fetched_at": <ISO-8601>, "count": N, "talents": [rows...]}``
   with ``count == len(talents)``.
7. The ``roster`` CLI subcommand fetches (with the org from ``--org``,
   default ``Hololive``, and the Holodex key), filters, and writes the
   roster to ``-o/--out`` (default ``data/catalog/roster.json``).

Fixture: tests/fixtures/holodex_channels.json — synthetic Holodex-like
channel metadata only (no Cover/hololive media, no network calls in
tests; every network-touching path is exercised through injectable
fakes).
"""

import json
import urllib.parse
from datetime import datetime
from pathlib import Path

import pytest

from vvc import roster
from vvc.roster import filter_talents, fetch_channels, write_roster

TESTS_DIR = Path(__file__).resolve().parent
CHANNELS_FIXTURE = TESTS_DIR / "fixtures" / "holodex_channels.json"

SORA_ID = "UCSora0thGen01"
LUNA_ID = "UCLunaGen4Row1"
CALLI_ID = "UCCalliENMyth01"
RISU_ID = "UCIDGen1Talent1"


# ---------------------------------------------------------------- fixtures


def _load_fixture_channels() -> list[dict]:
    with open(CHANNELS_FIXTURE, encoding="utf-8") as f:
        return json.load(f)


def _talent(**overrides) -> dict:
    """A minimal known-good active vtuber row, overridable per test."""
    row = {
        "id": "UCgoodtalent01",
        "name": "Tokino Sora",
        "english_name": "Tokino Sora",
        "group": "0th Generation",
        "org": "Hololive",
        "type": "vtuber",
        "inactive": False,
        "video_count": 100,
        "subscriber_count": 1_000_000,
    }
    row.update(overrides)
    return row


# ------------------------------------------------------- filter_talents


def test_filter_keeps_active_jp_gen_row_unchanged() -> None:
    """Rule 1+2: an active JP-generation vtuber row passes the filter
    field-for-field, untouched."""
    row = _talent(id="UCjpGen00001", group="GEN 4", name="Himemori Luna")
    kept = filter_talents([row])
    assert kept == [row], "filter must keep the row unchanged"


def test_filter_from_fixture_keeps_exactly_the_active_talents() -> None:
    """Rules 1–2 over the fixture: only the active talent rows survive —
    the Official main channel, the Misc Holoearth row, both HOLOSTARS
    rows, the inactive Gura-shaped row, the subber row, and the rows
    missing group/inactive/id are all dropped."""
    kept = filter_talents(_load_fixture_channels())
    assert [row["id"] for row in kept] == [
        "UCAZKi0thGen02",
        SORA_ID,
        LUNA_ID,
        CALLI_ID,
        RISU_ID,
    ]


@pytest.mark.parametrize(
    ("flavor", "mutate"),
    [
        ("group_official", lambda r: r.update({"group": "Official"})),
        ("group_misc", lambda r: r.update({"group": "Misc"})),
        (
            "group_holostars",
            lambda r: r.update({"group": "HOLOSTARS"}),
        ),
        (
            "group_holostars_gen",
            lambda r: r.update({"group": "HOLOSTARS 3rd Gen (TriNero)"}),
        ),
        ("inactive_true", lambda r: r.update({"inactive": True})),
        ("inactive_null", lambda r: r.update({"inactive": None})),
        ("type_subber", lambda r: r.update({"type": "subber"})),
        ("type_missing", lambda r: r.pop("type")),
        ("group_missing", lambda r: r.pop("group")),
        ("group_null", lambda r: r.update({"group": None})),
        ("inactive_missing", lambda r: r.pop("inactive")),
        ("id_missing", lambda r: r.pop("id")),
    ],
)
def test_filter_fail_closed_drops(flavor: str, mutate) -> None:
    """Rule 1 (fail-closed): whatever the flavor, a row that fails any
    keep-condition — including rows MISSING the pinned fields — is
    dropped, never kept by default."""
    row = _talent()
    mutate(row)
    assert filter_talents([row]) == [], f"{flavor} row must be dropped"


def test_filter_empty_input() -> None:
    """Empty list in -> empty list out."""
    assert filter_talents([]) == []


def test_filter_output_sorted_by_group_then_name() -> None:
    """Rule 5: output is sorted by (group, name) — a stable, documented
    order, independent of input order (the API sort is not stable)."""
    luna = _talent(id="UCsortLuna0001", group="GEN 4", name="Himemori Luna")
    sora = _talent(id="UCsortSora0001", group="0th Generation", name="Tokino Sora")
    calli = _talent(
        id="UCsortCalli001", group="Holo English -Myth-", name="Mori Calliope"
    )
    azki = _talent(id="UCsortAZKi0001", group="0th Generation", name="AZKi")
    kept = filter_talents([luna, calli, sora, azki])
    assert [row["name"] for row in kept] == [
        "AZKi",
        "Tokino Sora",
        "Himemori Luna",
        "Mori Calliope",
    ]
    assert [row["group"] for row in kept] == [
        "0th Generation",
        "0th Generation",
        "GEN 4",
        "Holo English -Myth-",
    ]


# ------------------------------------------------------- fetch_channels


class _PagedFetcher:
    """Fake fetcher serving queued pages; records every requested URL."""

    def __init__(self, pages: list[list[dict]]):
        self.pages = list(pages)
        self.urls: list[str] = []

    def __call__(self, url: str) -> list[dict]:
        self.urls.append(url)
        if not self.pages:
            return []
        return self.pages.pop(0)


def _bulk_rows(prefix: str, n: int) -> list[dict]:
    return [
        _talent(id=f"{prefix}{i:03d}", name=f"Talent {prefix} {i:03d}")
        for i in range(n)
    ]


def test_fetch_channels_paginates_until_short_page_and_dedupes() -> None:
    """Rule 3: a full page (limit=50) triggers the next offset; a short
    page ends pagination. A duplicated id across pages survives exactly
    once, and query params carry org/type/limit/offset."""
    page1 = _bulk_rows("UCpage1tal", 50)
    duplicate = dict(page1[7])
    page2 = [duplicate, *_bulk_rows("UCpage2tal", 2)]
    fetcher = _PagedFetcher([page1, page2])

    rows = fetch_channels(fetcher=fetcher, api_key="test-key")

    assert len(fetcher.urls) == 2, "50-row page must trigger page 2, short page stops"
    assert len(rows) == 52, "duplicate id must be deduped (51 unique rows)"
    assert len({row["id"] for row in rows}) == 52
    queries = [urllib.parse.parse_qs(urllib.parse.urlparse(u).query) for u in fetcher.urls]
    assert all(q["org"] == ["Hololive"] for q in queries)
    assert all(q["type"] == ["vtuber"] for q in queries)
    assert all(q["limit"] == ["50"] for q in queries)
    assert [q["offset"][0] for q in queries] == ["0", "50"]
    assert all(u.startswith("https://holodex.net/api/v2/channels") for u in fetcher.urls)


def test_fetch_channels_empty_first_page() -> None:
    """Rule 3: an empty first page ends pagination with no rows."""
    fetcher = _PagedFetcher([[]])
    assert fetch_channels(fetcher=fetcher, api_key="test-key") == []
    assert len(fetcher.urls) == 1


def test_fetch_channels_single_page_needs_no_second_request() -> None:
    """Rule 3: a page shorter than the limit stops after one request."""
    fetcher = _PagedFetcher([_bulk_rows("UCsinglepag", 3)])
    rows = fetch_channels(fetcher=fetcher, api_key="test-key")
    assert len(fetcher.urls) == 1
    assert len(rows) == 3


def test_channel_headers_require_key_and_user_agent() -> None:
    """Rule 4: every request must carry X-APIKEY and the pinned
    User-Agent — Holodex returns 403 without a User-Agent."""
    headers = roster.channel_headers("test-key")
    assert headers["X-APIKEY"] == "test-key"
    assert headers["User-Agent"] == "vvc/0.1"


def test_fetch_channels_dedupes_within_a_single_page() -> None:
    """Rule 3: dedupe also applies within one page (same id twice)."""
    row = _talent(id="UCtwiceOnPage1")
    fetcher = _PagedFetcher([[row, dict(row)]])
    rows = fetch_channels(fetcher=fetcher, api_key="test-key")
    assert [r["id"] for r in rows] == ["UCtwiceOnPage1"]


# --------------------------------------------------------- write_roster


def test_write_roster_schema(tmp_path: Path) -> None:
    """Rule 6: the saved roster is {"fetched_at": ISO-8601, "count": N,
    "talents": [...]} with count == len(talents) and the rows unchanged."""
    talents = filter_talents(
        [
            _talent(id=SORA_ID, group="0th Generation", name="Tokino Sora"),
            _talent(
                id="UCdropMe000001", type="subber"
            ),
            _talent(id=CALLI_ID, group="Holo English -Myth-", name="Mori Calliope"),
        ]
    )
    out = tmp_path / "nested" / "roster.json"
    write_roster(talents, out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(data) == {"fetched_at", "count", "talents"}
    assert data["count"] == 2
    assert data["talents"] == talents
    parsed = datetime.fromisoformat(data["fetched_at"])
    assert parsed.tzinfo is not None, "fetched_at must be a timezone-aware ISO-8601 stamp"


def test_write_roster_empty(tmp_path: Path) -> None:
    """Rule 6: an empty roster is still a valid schema document."""
    out = tmp_path / "roster.json"
    write_roster([], out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data == {"count": 0, "talents": []} or (
        data["count"] == 0 and data["talents"] == []
    )
    datetime.fromisoformat(data["fetched_at"])


# ------------------------------------------------------------ CLI wiring


def test_cli_roster_writes_default_path(tmp_path: Path, monkeypatch, capsys) -> None:
    """Rule 7: `vvc roster` fetches with --org (default Hololive)
    and the Holodex key, then writes the filtered roster to the default
    out path data/catalog/roster.json, printing the talent count."""
    import vvc.__main__ as cli

    monkeypatch.setattr(cli, "_holodex_key", lambda: "test-key")
    captured: dict = {}

    def fake_fetch(**kwargs):
        captured.update(kwargs)
        return filter_talents(
            [
                _talent(id=SORA_ID, group="0th Generation", name="Tokino Sora"),
                _talent(id="UCsubberDrop1", type="subber"),
            ]
        )

    monkeypatch.setattr(cli, "fetch_channels", fake_fetch)
    monkeypatch.chdir(tmp_path)
    cli.main(["roster"])
    out = tmp_path / "data" / "catalog" / "roster.json"
    assert out.is_file(), "default --out must be data/catalog/roster.json"
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["count"] == 1
    assert data["talents"][0]["id"] == SORA_ID
    assert captured["org"] == "Hololive"
    assert captured["api_key"] == "test-key"
    assert "1" in capsys.readouterr().out


def test_cli_roster_respects_out_and_org(tmp_path: Path, monkeypatch) -> None:
    """Rule 7: -o overrides the output path, --org is passed through."""
    import vvc.__main__ as cli

    monkeypatch.setattr(cli, "_holodex_key", lambda: "test-key")
    captured: dict = {}

    def fake_fetch(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(cli, "fetch_channels", fake_fetch)
    out = tmp_path / "roster.json"
    cli.main(["roster", "-o", str(out), "--org", "Hololive"])
    assert out.is_file()
    assert json.loads(out.read_text(encoding="utf-8"))["count"] == 0
    assert captured["org"] == "Hololive"


def test_cli_roster_missing_key_exits() -> None:
    """Rule 7: no Holodex key -> exit code 2, no fetch attempted."""
    import vvc.__main__ as cli

    def boom(**kwargs):
        raise AssertionError("fetch_channels must not run without a key")

    monkey = pytest.MonkeyPatch()
    monkey.setattr(cli, "_holodex_key", lambda: _missing_key())
    monkey.setattr(cli, "fetch_channels", boom)
    with pytest.raises(SystemExit) as exc:
        cli.main(["roster"])
    assert exc.value.code == 2
    monkey.undo()


def _missing_key() -> str:
    import sys

    print("HOLODEX_API_KEY is not set", file=sys.stderr)
    sys.exit(2)
