"""The `densify` CLI's --cpu-workers and --offload-remote flags must
reach run_densify.

Defaults (no flag given): cpu_workers=1 (fully sequential path) and
offload_remote=None (offload disabled) — see densify.py's own tests for
the actual pipelining/offload behavior.
"""

from __future__ import annotations

from pathlib import Path

from vanalysis import __main__ as cli


def _tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    data_dir = tmp_path / "data"
    measurements = tmp_path / "measurements.json"
    measurements.write_text("[]", encoding="utf-8")
    video_cache = tmp_path / "video_cache.json"
    video_cache.write_text("[]", encoding="utf-8")
    return data_dir, measurements, video_cache


def _fake_run_densify(captured: dict):
    def run(*args, **kwargs):
        captured.update(kwargs)
        return {
            "ids": {},
            "counts": {"added": 0, "skipped": 0, "error": 0, "total": 0},
            "months_targeted": [],
            "snapshot": None,
            "stopped_early": None,
            "dry_run": kwargs.get("dry_run", False),
        }

    return run


def test_densify_cli_defaults_cpu_workers_and_offload_remote(
    tmp_path: Path, monkeypatch
) -> None:
    data_dir, measurements, video_cache = _tree(tmp_path)
    captured: dict = {}
    monkeypatch.setattr(cli, "run_densify", _fake_run_densify(captured))

    cli.main(
        [
            "densify",
            "--data-dir",
            str(data_dir),
            "--measurements",
            str(measurements),
            "--video-cache",
            str(video_cache),
        ]
    )

    assert captured["cpu_workers"] == 1
    assert captured["offload_remote"] is None


def test_densify_cli_threads_cpu_workers_flag(tmp_path: Path, monkeypatch) -> None:
    data_dir, measurements, video_cache = _tree(tmp_path)
    captured: dict = {}
    monkeypatch.setattr(cli, "run_densify", _fake_run_densify(captured))

    cli.main(
        [
            "densify",
            "--data-dir",
            str(data_dir),
            "--measurements",
            str(measurements),
            "--video-cache",
            str(video_cache),
            "--cpu-workers",
            "4",
        ]
    )

    assert captured["cpu_workers"] == 4


def test_densify_cli_threads_offload_remote_flag(tmp_path: Path, monkeypatch) -> None:
    data_dir, measurements, video_cache = _tree(tmp_path)
    captured: dict = {}
    monkeypatch.setattr(cli, "run_densify", _fake_run_densify(captured))

    cli.main(
        [
            "densify",
            "--data-dir",
            str(data_dir),
            "--measurements",
            str(measurements),
            "--video-cache",
            str(video_cache),
            "--offload-remote",
            "Google Drive:vanalysis-raw-audio",
        ]
    )

    assert captured["offload_remote"] == "Google Drive:vanalysis-raw-audio"
