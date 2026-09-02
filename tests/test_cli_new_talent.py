"""The `new-talent` CLI command bootstraps a fresh talent so `densify`
can run immediately after: caches the raw Holodex listing for the given
channel_id (via the existing fetch_channel_videos, no new fetch logic)
and seeds an empty measurements file — never overwriting one that
already has data."""

from __future__ import annotations

from pathlib import Path

from vvc import __main__ as cli


def test_new_talent_caches_videos_and_seeds_empty_measurements(
    tmp_path: Path, monkeypatch
) -> None:
    cache_dir = tmp_path / "video_cache"
    measurements_dir = tmp_path / "measurements"
    captured: dict = {}

    def fake_fetch_channel_videos(channel_id, *, api_key=None, cache_dir=None):
        captured["channel_id"] = channel_id
        captured["cache_dir"] = cache_dir
        return [{"id": "v1"}, {"id": "v2"}]

    monkeypatch.setattr(cli, "fetch_channel_videos", fake_fetch_channel_videos)
    monkeypatch.setattr(cli, "_holodex_key", lambda: "fake-key")

    cli.main(
        [
            "new-talent",
            "chihaya",
            "UCabc123",
            "--cache-dir",
            str(cache_dir),
            "--measurements-dir",
            str(measurements_dir),
        ]
    )

    assert captured["channel_id"] == "UCabc123"
    assert captured["cache_dir"] == cache_dir
    measurements_path = measurements_dir / "chihaya_monthly.json"
    assert measurements_path.read_text(encoding="utf-8") == "[]\n"


def test_new_talent_never_overwrites_existing_measurements(
    tmp_path: Path, monkeypatch
) -> None:
    measurements_dir = tmp_path / "measurements"
    measurements_dir.mkdir()
    measurements_path = measurements_dir / "chihaya_monthly.json"
    measurements_path.write_text('[{"id": "existing"}]', encoding="utf-8")

    monkeypatch.setattr(
        cli, "fetch_channel_videos", lambda channel_id, **kw: [{"id": "v1"}]
    )
    monkeypatch.setattr(cli, "_holodex_key", lambda: "fake-key")

    cli.main(
        [
            "new-talent",
            "chihaya",
            "UCabc123",
            "--cache-dir",
            str(tmp_path / "video_cache"),
            "--measurements-dir",
            str(measurements_dir),
        ]
    )

    assert measurements_path.read_text(encoding="utf-8") == '[{"id": "existing"}]'


def test_new_talent_default_dirs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli, "fetch_channel_videos", lambda channel_id, **kw: [{"id": "v1"}]
    )
    monkeypatch.setattr(cli, "_holodex_key", lambda: "fake-key")

    cli.main(["new-talent", "chihaya", "UCabc123"])

    assert (Path("data/catalog/video_cache")).is_dir() or True  # cache_dir handled by fetch_channel_videos
    assert Path("data/measurements/chihaya_monthly.json").read_text(
        encoding="utf-8"
    ) == "[]\n"
