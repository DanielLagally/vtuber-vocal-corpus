"""Product tests for densify.py (PLAN "how we can improve further" — the
more-data lever: bring every month up to target_n clips using the
already-cached raw catalog, no new Holodex calls).

User-visible rules (synthetic wavs + fake fetch/isolate runners ONLY;
never Cover/hololive audio, never GPU, never real network):

1. ``candidate_ids(records, cached_videos, target_n)`` returns, per
   month whose current record count is below target_n, the next
   highest-scored eligible videos (score desc, newest first) NOT
   already present as a record — capped at exactly what's needed to
   reach target_n. A month already at or above target_n is absent. A
   month with no further eligible video (nothing left to add) is
   absent, never padded.
2. ``run_densify`` fetches, windows, isolates, and Praat-measures each
   candidate, appending ONE new record per success with
   tracker="praat_ac"; existing records are never modified.
3. A pre-densify snapshot is written once, before the first append.
4. dry_run=True computes without writing anything.
5. A per-id fetch failure is skipped (never fabricated) and the batch
   continues with the remaining candidates.
"""

from __future__ import annotations

import array
import json
import math
import shutil
import sys
import wave
from pathlib import Path

import pytest

from vanalysis import densify
from vanalysis import fetch as densify_fetch
from vanalysis.fetch import audio_path
from vanalysis.isolate import DEFAULT_MODEL_FILENAME, vocals_path

SR = 8_000
MODEL_CKPT = DEFAULT_MODEL_FILENAME


def _video(video_id: str, month: str, *, score_boost: float = 0.0) -> dict:
    day = "05" if score_boost == 0.0 else "10"
    return {
        "id": video_id,
        "type": "stream",
        "topic_id": "chatting",
        "duration": 3600 + score_boost,  # nudges score_video's tie order via distinct id/day only
        "available_at": f"{month}-{day}T00:00:00.000Z",
        "published_at": f"{month}-{day}T00:00:00.000Z",
        "title": "zatsudan",
    }


def test_candidate_ids_returns_missing_video_for_undersized_month() -> None:
    records = [{"id": "haveid00001", "month": "2024-05"}]
    cached = [_video("haveid00001", "2024-05"), _video("newid000001", "2024-05")]
    buckets = densify.candidate_ids(records, cached, target_n=2)
    assert list(buckets) == ["2024-05"]
    assert [v["id"] for v in buckets["2024-05"]] == ["newid000001"]


def test_candidate_ids_skips_month_already_at_target() -> None:
    records = [
        {"id": "haveid00001", "month": "2024-06"},
        {"id": "haveid00002", "month": "2024-06"},
    ]
    cached = [
        _video("haveid00001", "2024-06"),
        _video("haveid00002", "2024-06"),
        _video("newid000002", "2024-06"),
    ]
    buckets = densify.candidate_ids(records, cached, target_n=2)
    assert buckets == {}


def test_candidate_ids_absent_when_no_further_eligible_video() -> None:
    records = [{"id": "onlyid00001", "month": "2024-07"}]
    cached = [_video("onlyid00001", "2024-07")]
    buckets = densify.candidate_ids(records, cached, target_n=3)
    assert buckets == {}


# ---------------------------------------------------------------- fixtures


def _samples_to_bytes(samples: list[int]) -> bytes:
    buf = array.array("h", samples)
    if sys.byteorder == "big":
        buf.byteswap()
    return buf.tobytes()


def _sine(freq_hz: float, seconds: float, amp: float = 0.6) -> list[int]:
    peak = amp * 32767.0
    n = int(seconds * SR)
    fade = int(0.005 * SR)
    out = []
    for i in range(n):
        env = min(1.0, i / fade, (n - 1 - i) / fade)
        out.append(int(round(peak * env * math.sin(2.0 * math.pi * freq_hz * i / SR))))
    return out


def _write_wav(path: Path, samples: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(_samples_to_bytes(samples))


@pytest.fixture(scope="module")
def source_wav() -> Path:
    """A 200 s fake 15-min-stand-in: silence [0,20), a clean 220 Hz tone
    island [20,110), silence [110,200) — the only all-voiced 90 s grid
    window is [20,110)."""
    path = Path(__file__).resolve().parent.parent / "fixtures" / "densify_source.wav"
    if not path.is_file():
        _write_wav(path, [0] * (20 * SR) + _sine(220.0, 90.0) + [0] * (90 * SR))
    return path


class _FakeFetch:
    """Stand-in for yt-dlp: writes a copy of source_wav to the -o
    destination, recording every call. One id can be configured to fail
    ordinarily (simulating a yt-dlp non-zero exit), and/or a different
    id to trigger the bot-check signal."""

    def __init__(
        self,
        source: Path,
        fail_for: str | None = None,
        bot_check_for: str | None = None,
    ) -> None:
        self.source = source
        self.fail_for = fail_for
        self.bot_check_for = bot_check_for
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str]) -> None:
        self.calls.append(list(argv))
        video_id = argv[-1].rsplit("=", 1)[-1]
        if self.bot_check_for is not None and video_id == self.bot_check_for:
            raise densify_fetch.BotCheckDetected(
                video_id, "Sign in to confirm you’re not a bot."
            )
        if self.fail_for is not None and video_id == self.fail_for:
            import subprocess

            raise subprocess.CalledProcessError(1, argv)
        dest = Path(argv[argv.index("-o") + 1]).with_suffix(".wav")
        shutil.copyfile(self.source, dest)


class _FakeIsolate:
    """Stand-in for the audio-separator runner: writes the stem file
    isolate's naming produces (a copy of its input, padded past 1 MB)."""

    def __call__(self, argv: list[str]) -> None:
        src = Path(argv[-1])
        out_dir = Path(argv[argv.index("--output_dir") + 1])
        model = argv[argv.index("--model_filename") + 1]
        dest = vocals_path(src, out_dir, model_filename=model)
        with wave.open(str(src), "rb") as w:
            params = w.getparams()
            frames = w.readframes(w.getnframes())
        repeat = max(1, 1_000_001 // max(1, len(frames)) + 1)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(dest), "wb") as w:
            w.setparams(params)
            w.writeframes(frames * repeat)


def _tree(tmp_path: Path, records: list[dict], cached_videos: list[dict]) -> dict:
    data_dir = tmp_path / "data"
    (data_dir / "audio").mkdir(parents=True)
    (data_dir / "windows").mkdir(parents=True)
    measurements_dir = data_dir / "measurements"
    measurements_dir.mkdir(parents=True)
    measurements_path = measurements_dir / "luna_monthly.json"
    measurements_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    video_cache_path = tmp_path / "video_cache.json"
    video_cache_path.write_text(json.dumps(cached_videos), encoding="utf-8")
    return {
        "data_dir": data_dir,
        "measurements_path": measurements_path,
        "video_cache_path": video_cache_path,
        "windows_path": data_dir / "windows" / "windows.json",
        "stems_dir": data_dir / "stems_fast",
    }


def test_run_densify_appends_praat_record_for_new_id(
    tmp_path: Path, source_wav: Path
) -> None:
    records = [
        {
            "id": "existid001",
            "month": "2024-08",
            "score": 70.0,
            "window": {"start_s": 20.0, "end_s": 110.0},
            "features": {"median_f0": 300.0, "f0_iqr": 20.0, "voiced_fraction": 0.7},
            "qc": {"pass": True, "reason": None},
            "model": MODEL_CKPT,
            "tracker": "praat_ac",
        }
    ]
    cached = [_video("existid001", "2024-08"), _video("newid000003", "2024-08")]
    tree = _tree(tmp_path, records, cached)
    fetch = _FakeFetch(source_wav)
    isolate = _FakeIsolate()

    summary = densify.run_densify(
        tree["data_dir"],
        measurements_path=tree["measurements_path"],
        video_cache_path=tree["video_cache_path"],
        windows_path=tree["windows_path"],
        stems_dir=tree["stems_dir"],
        model_filename=MODEL_CKPT,
        fetch_runner=fetch,
        isolate_runner=isolate,
    )

    assert summary["counts"]["added"] == 1
    assert summary["ids"]["newid000003"] == "added"
    out = json.loads(tree["measurements_path"].read_text(encoding="utf-8"))
    assert len(out) == 2
    added = next(r for r in out if r["id"] == "newid000003")
    assert added["month"] == "2024-08"
    assert added["tracker"] == "praat_ac"
    assert math.isfinite(added["features"]["median_f0"])
    assert abs(added["features"]["median_f0"] - 220.0) < 5.0
    # the existing record is byte-identical
    assert out[0] == records[0]
    snapshot = tree["measurements_path"].with_name("luna_monthly_pre_densify.json")
    assert json.loads(snapshot.read_text(encoding="utf-8")) == records


def test_run_densify_dry_run_writes_nothing(tmp_path: Path, source_wav: Path) -> None:
    records = [{"id": "existid002", "month": "2024-09"}]
    cached = [_video("existid002", "2024-09"), _video("newid000004", "2024-09")]
    tree = _tree(tmp_path, records, cached)
    original_bytes = tree["measurements_path"].read_text(encoding="utf-8")

    summary = densify.run_densify(
        tree["data_dir"],
        measurements_path=tree["measurements_path"],
        video_cache_path=tree["video_cache_path"],
        windows_path=tree["windows_path"],
        stems_dir=tree["stems_dir"],
        model_filename=MODEL_CKPT,
        fetch_runner=_FakeFetch(source_wav),
        isolate_runner=_FakeIsolate(),
        dry_run=True,
    )

    assert summary["dry_run"] is True
    assert tree["measurements_path"].read_text(encoding="utf-8") == original_bytes
    assert not tree["measurements_path"].with_name(
        "luna_monthly_pre_densify.json"
    ).exists()


def test_run_densify_fetch_failure_is_skipped_batch_continues(
    tmp_path: Path, source_wav: Path
) -> None:
    records: list[dict] = []
    cached = [_video("failid00001", "2024-10"), _video("okid0000001", "2024-11")]
    tree = _tree(tmp_path, records, cached)
    fetch = _FakeFetch(source_wav, fail_for="failid00001")

    summary = densify.run_densify(
        tree["data_dir"],
        measurements_path=tree["measurements_path"],
        video_cache_path=tree["video_cache_path"],
        windows_path=tree["windows_path"],
        stems_dir=tree["stems_dir"],
        model_filename=MODEL_CKPT,
        fetch_runner=fetch,
        isolate_runner=_FakeIsolate(),
    )

    assert summary["ids"]["failid00001"] == "skipped_fetch_failed"
    assert summary["ids"]["okid0000001"] == "added"
    assert summary["counts"]["added"] == 1


def test_run_densify_stops_immediately_on_bot_check(
    tmp_path: Path, source_wav: Path
) -> None:
    """A YouTube bot-check is a session/IP-level signal, not a per-video
    gap (PLAN "how we can improve further" / the Lamy 2026-09 incident):
    unlike an ordinary fetch failure, densify must NOT continue to the
    next candidate after one — continuing only sends more suspicious
    traffic. The id that triggered it gets a distinct outcome and
    ``stopped_early`` names it; no later candidate is ever attempted."""
    records: list[dict] = []
    cached = [
        _video("okidbefore1", "2024-10"),
        _video("botcheckid1", "2024-11"),
        _video("nevertried1", "2024-12"),
    ]
    tree = _tree(tmp_path, records, cached)
    fetch_runner = _FakeFetch(source_wav, bot_check_for="botcheckid1")

    summary = densify.run_densify(
        tree["data_dir"],
        measurements_path=tree["measurements_path"],
        video_cache_path=tree["video_cache_path"],
        windows_path=tree["windows_path"],
        stems_dir=tree["stems_dir"],
        model_filename=MODEL_CKPT,
        fetch_runner=fetch_runner,
        isolate_runner=_FakeIsolate(),
    )

    assert summary["ids"]["okidbefore1"] == "added"
    assert summary["ids"]["botcheckid1"] == "skipped_bot_check"
    assert "nevertried1" not in summary["ids"], (
        "no candidate after the bot-check may ever be attempted"
    )
    assert summary["stopped_early"] == "botcheckid1"
    attempted_ids = {c[-1].rsplit("=", 1)[-1] for c in fetch_runner.calls}
    assert "nevertried1" not in attempted_ids


def test_run_densify_does_not_refetch_when_already_windowed_and_sliced(
    tmp_path: Path, source_wav: Path
) -> None:
    """The raw wav is only needed to PRODUCE the window/slice — once a
    recorded window key AND the raw90 slice already exist (e.g. the raw
    wav was deliberately deleted after processing, or a crash-resume),
    densify must NOT re-fetch it. Isolation still proceeds from the
    existing slice."""
    records: list[dict] = []
    cached = [_video("nofetchid1", "2024-10")]
    tree = _tree(tmp_path, records, cached)

    # Pre-seed the window key + slice, but there is NO raw audio file at all.
    tree["windows_path"].parent.mkdir(parents=True, exist_ok=True)
    tree["windows_path"].write_text(
        json.dumps({"nofetchid1_raw90": {"start_s": 0.0, "end_s": 1.0}}),
        encoding="utf-8",
    )
    slice_path = tree["data_dir"] / "windows" / "nofetchid1_raw90.wav"
    shutil.copyfile(source_wav, slice_path)
    assert not (tree["data_dir"] / "audio" / "nofetchid1.wav").exists()

    fetch_runner = _FakeFetch(source_wav)  # would materialize the raw wav if called
    summary = densify.run_densify(
        tree["data_dir"],
        measurements_path=tree["measurements_path"],
        video_cache_path=tree["video_cache_path"],
        windows_path=tree["windows_path"],
        stems_dir=tree["stems_dir"],
        model_filename=MODEL_CKPT,
        fetch_runner=fetch_runner,
        isolate_runner=_FakeIsolate(),
    )

    assert fetch_runner.calls == [], "raw wav must never be fetched when not needed"
    assert summary["ids"]["nofetchid1"] == "added"
    assert not (tree["data_dir"] / "audio" / "nofetchid1.wav").exists()
