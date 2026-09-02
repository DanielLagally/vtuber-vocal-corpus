"""Product tests for the stem-hunt rescue batch (STATE Run 2 R1; PLAN
"goal shift" lever 3).

User-visible rules (synthetic wavs + a fake isolate runner ONLY — tones;
never Cover/hololive audio, never GPU, never downloads):

1. ``rescue.run_rescue(ids, data_dir, *, measurements_path,
   windows_path, stems_dir, model_filename=DEFAULT_MODEL_FILENAME,
   model_file_dir="data/models", isolate_runner=None, out_path=None,
   dry_run=False, log=None)`` returns a summary dict: ``{"ids": {id:
   outcome}, "counts": {replaced, kept_fail, skipped, error, total},
   "windows_added": [keys], "snapshot": path|None, "dry_run": bool}``.
   A per-id error NEVER aborts the batch (outcome "error", warn, next
   id still processed).
2. Happy path: for a QC-failing record the FULL 15-min wav
   ``data/audio/<id>.wav`` is isolated into ``stems_dir/<id>_(vocals)_
   <model_filename stem>.wav`` (exactly ``isolate.vocals_path``'s
   naming, ``--model_file_dir data/models`` forwarded by default), the
   90 s window is hunted ON THE STEM (``best_speech_window``, BGM gone
   — the raw hunt's music lock-on is the bug being fixed), the STEM is
   sliced to ``data/windows/<id>_stem90.wav`` (measurement input), and
   if QC passes the record is REPLACED in place keeping
   ``id``/``month``/``score``/``model`` from the old record, with
   ``window`` = the stem90 window, ``features`` from the stem90 slice
   and ``qc`` from the verdict.
3. If the stem hunt still fails QC the existing record is untouched
   (byte-identical measurements file, no snapshot); the stem, the
   windows.json entry and the stem90 slice still exist (``kept_fail``).
4. Snapshot: BEFORE the first replacement of a run the measurements
   file is copied to ``<measurements_stem>_pre_rescue.json`` in the
   same directory. The snapshot holds the pre-run file bytes; an
   existing snapshot is NEVER overwritten (idempotent re-runs).
5. A missing raw wav (and no ready stem), an id without a measurement
   record, an already-QC-passing record, or a hunt ValueError is
   skipped with a warning and the batch continues (``skipped_*``).
6. If the isolate runner raises, the id is an ``error``: warn and
   continue, and there is NO partial record (and no stem90 window).
7. Crash-resume: when the ``<id>_stem90`` windows key, a ready full
   stem (> 1 MB) and the stem90 slice already exist, the runner is NOT
   called; the recorded window is reused, the slice re-measured, and a
   still-failing record is replaced. ``windows_added`` stays empty.
8. ``dry_run=True`` computes everything possible (with a ready stem +
   recorded stem90 window + existing slice: the WOULD-BE outcome) but
   writes NOTHING: no snapshot, no windows.json rewrite, no slice, no
   stem, no measurements rewrite. Steps that would have to write
   (missing full stem / missing-or-stale slice) become
   ``skipped_dry_run``.
9. Default id selection (ids=None, or the CLI without --ids) is EXACTLY
   the records whose ``qc.pass`` is false — nan/IQR/high fail reasons
   all included, file order — and never a passing id; a passing id
   passed explicitly is ``skipped_qc_pass`` and untouched.
10. windows.json is APPEND/merge-only: the 16 legacy array entries of
    the real index survive as arrays, pre-existing keys (first-window
    and ``_raw90`` entries) are never rewritten, and the file is only
    rewritten when a new key was actually added.
11. CLI ``python -m vvc rescue`` wires the same function with the
    retry flag family, ``--model-file-dir`` defaulting to
    ``"data/models"`` (unlike retry's None), ``--ids-file`` accepting
    dash-leading ids unioned after ``--ids``, and prints the summary as
    JSON.

Fixtures are tiny synthetic wavs (8 kHz mono 16-bit, tones) synthesized
here into <repo>/fixtures/ — deterministic, no third-party test deps.
The fake runner copies its input (the FULL raw wav) to the stem path
isolate's naming produces, padding past 1 MB so resume semantics are
exercised.
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

from vvc import rescue
from vvc.__main__ import main
from vvc.isolate import vocals_path
from vvc.windows import slice_wav

SR = 8_000
ID = "rescueAaa01"
GOOD = "rescueBbb02"
PASS_ID = "rescueCcc03"
MODEL_CKPT = "bs_roformer_vocals_resurrection_unwa.ckpt"
STEM_SUFFIX = "_(vocals)_bs_roformer_vocals_resurrection_unwa.wav"
FIRST_WIN = {"start_s": 0.0, "end_s": 90.0}
STEM_WIN = {"start_s": 90.0, "end_s": 180.0}
STEM90_KEY = f"{ID}_stem90"
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"

# ---------------------------------------------------------------- synthesis


def _samples_to_bytes(samples: list[int]) -> bytes:
    buf = array.array("h", samples)
    if sys.byteorder == "big":
        buf.byteswap()
    return buf.tobytes()


def _sine(freq_hz: float, seconds: float, amp: float = 0.6) -> list[int]:
    peak = amp * 32767.0
    n = int(seconds * SR)
    fade = int(0.005 * SR)  # 5 ms fade to avoid clicks
    out = []
    for i in range(n):
        env = min(1.0, i / fade, (n - 1 - i) / fade)
        out.append(int(round(peak * env * math.sin(2.0 * math.pi * freq_hz * i / SR))))
    return out


def _write_wav(path: Path, samples: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit PCM
        w.setframerate(SR)
        w.writeframes(_samples_to_bytes(samples))


@pytest.fixture(scope="module")
def island_wav() -> Path:
    """A 270 s fake full stem: silence [0,90), a clean 220 Hz tone
    island [90,180), silence [180,270). Hunted ON this audio, the only
    all-voiced 90 s grid window is [90,180) — a tight-IQR tone (QC
    pass). 8 kHz keeps the fixture small; features.py resamples to
    16 kHz itself."""
    path = FIXTURES_DIR / "rescue_island.wav"
    if not path.is_file():
        silence = [0] * (90 * SR)
        _write_wav(path, silence + _sine(220.0, 90.0) + silence)
    return path


@pytest.fixture(scope="module")
def junk_wav() -> Path:
    """A 270 s fake full stem of alternating 200/450 Hz half-second
    tones — every 90 s window has f0_iqr ~ 250 (QC fail), so the stem
    hunt cannot find a passing window."""
    path = FIXTURES_DIR / "rescue_junk.wav"
    if not path.is_file():
        half: list[int] = _sine(200.0, 0.5) + _sine(450.0, 0.5)
        _write_wav(path, half * 180)
    return path


# ---------------------------------------------------------------- helpers


class _FakeIsolate:
    """Stand-in for the audio-separator runner: writes the stem file
    isolate's naming produces (a copy of its input — the FULL raw wav —
    padded past 1 MB) and records every call."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str]) -> None:
        self.calls.append(list(argv))
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


class _ExplodingIsolate:
    """Fails (like a crashing audio-separator) for one id only; other ids
    isolate fine."""

    def __init__(self, fail_for: str = ID) -> None:
        self.fail_for = fail_for

    def __call__(self, argv: list[str]) -> None:
        if Path(argv[-1]).stem.startswith(self.fail_for):
            raise RuntimeError("isolate boom")
        _FakeIsolate()(argv)


def _record(video_id: str = ID, **overrides) -> dict:
    """A stored QC-failing (f0_iqr) measurement record, first window only."""
    rec = {
        "id": video_id,
        "month": "2024-07",
        "score": 63.5,
        "window": dict(FIRST_WIN),
        "features": {"median_f0": 250.0, "f0_iqr": 250.0, "voiced_fraction": 0.9},
        "qc": {"pass": False, "reason": "f0_iqr"},
        "model": MODEL_CKPT,
    }
    rec.update(overrides)
    return rec


def _tree(
    tmp_path: Path,
    wavs: dict[str, Path],
    records: list[dict],
    *,
    windows_extra: dict | None = None,
    stems_for: tuple[str, ...] = (),
    slices_for: tuple[str, ...] = (),
) -> dict:
    """A fake data tree: data/audio/<id>.wav, windows.json (two legacy
    array entries + each record's first window), the measurements file,
    optional pre-made full stems and optional pre-made stem90 slices."""
    data_dir = tmp_path / "data"
    (data_dir / "audio").mkdir(parents=True)
    (data_dir / "windows").mkdir(parents=True)
    measurements_dir = data_dir / "measurements"
    measurements_dir.mkdir(parents=True)
    stems_dir = data_dir / "stems"
    for vid, wav in wavs.items():
        shutil.copyfile(wav, data_dir / "audio" / f"{vid}.wav")
    windows: dict = {"legacy01": [5.0, 25.0], "legacy02": [30.0, 50.0]}
    for rec in records:
        windows[rec["id"]] = dict(FIRST_WIN)
    windows.update(windows_extra or {})
    windows_path = data_dir / "windows" / "windows.json"
    windows_path.write_text(json.dumps(windows, indent=2) + "\n", encoding="utf-8")
    measurements_path = measurements_dir / "luna_monthly.json"
    measurements_path.write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    stems_dir.mkdir(parents=True, exist_ok=True)
    for vid in stems_for:
        # a ready >1 MB full stem: a copy of the raw wav (270 s @ 8 kHz
        # mono 16-bit = 4.32 MB) named like the isolate runner writes it
        shutil.copyfile(wavs[vid], stems_dir / f"{vid}{STEM_SUFFIX}")
    for vid in slices_for:
        slice_wav(
            data_dir / "audio" / f"{vid}.wav",
            data_dir / "windows" / f"{vid}_stem90.wav",
            STEM_WIN["start_s"],
            STEM_WIN["end_s"],
        )
    return {
        "data_dir": data_dir,
        "measurements_path": measurements_path,
        "windows_path": windows_path,
        "stems_dir": stems_dir,
    }


def _run_rescue(tree: dict, ids, *, runner=None, **kwargs) -> dict:
    return rescue.run_rescue(
        ids,
        tree["data_dir"],
        measurements_path=tree["measurements_path"],
        windows_path=tree["windows_path"],
        stems_dir=tree["stems_dir"],
        model_filename=MODEL_CKPT,
        isolate_runner=runner,
        **kwargs,
    )


def _records(tree: dict) -> list[dict]:
    return json.loads(tree["measurements_path"].read_text(encoding="utf-8"))


def _windows(tree: dict) -> dict:
    return json.loads(tree["windows_path"].read_text(encoding="utf-8"))


def _snapshot_path(tree: dict) -> Path:
    return tree["measurements_path"].with_name(
        tree["measurements_path"].stem + "_pre_rescue.json"
    )


def _stem90_slice(tree: dict) -> Path:
    return tree["data_dir"] / "windows" / f"{ID}_stem90.wav"


# ------------------------------------------------------------------- tests


def test_rescue_happy_path_isolates_full_wav_and_replaces(
    tmp_path: Path, island_wav: Path
) -> None:
    """Rules 1+2+4: a failing record + a clean stem island -> the FULL
    raw wav is isolated (runner input = data/audio/<id>.wav, output =
    stems_dir, --model_file_dir data/models forwarded by default), the
    window is hunted on the stem, the stem is sliced to <id>_stem90.wav,
    the record is replaced in place (month/score/model preserved,
    window = stem90 window, features measured from the stem90 slice, qc
    pass), and the snapshot holds the PRE-run measurements bytes."""
    tree = _tree(tmp_path, {ID: island_wav}, [_record()])
    fake = _FakeIsolate()
    pre_measurements = tree["measurements_path"].read_bytes()
    pre_windows = tree["windows_path"].read_bytes()

    summary = _run_rescue(tree, [ID], runner=fake)

    assert summary["ids"] == {ID: "replaced"}
    assert summary["counts"] == {
        "replaced": 1,
        "kept_fail": 0,
        "skipped": 0,
        "error": 0,
        "total": 1,
    }
    assert summary["dry_run"] is False
    assert summary["windows_added"] == [STEM90_KEY]
    assert summary["snapshot"] == str(_snapshot_path(tree))
    assert fake.calls, "the isolate runner must have been called once"
    argv = fake.calls[0]
    assert argv[-1] == str(tree["data_dir"] / "audio" / f"{ID}.wav"), (
        "rescue must isolate the FULL raw wav, not a 90 s slice"
    )
    assert argv[argv.index("--output_dir") + 1] == str(tree["stems_dir"])
    assert argv[argv.index("--model_file_dir") + 1] == "data/models", (
        "run_rescue must forward model_file_dir='data/models' by default"
    )
    full_stem = tree["stems_dir"] / f"{ID}{STEM_SUFFIX}"
    assert full_stem.is_file(), (
        f"the full stem must be created at vocals_path's naming, "
        f"expected {full_stem.name}"
    )

    records = _records(tree)
    assert len(records) == 1
    rec = records[0]
    assert rec["id"] == ID
    assert rec["month"] == "2024-07", "month must be preserved from the old record"
    assert rec["score"] == 63.5, "score must be preserved from the old record"
    assert rec["model"] == MODEL_CKPT, "model provenance must be preserved"
    assert rec["window"] == STEM_WIN, "window metadata must be the stem90 window"
    feats = rec["features"]
    assert set(feats) == {"median_f0", "f0_iqr", "voiced_fraction"}
    assert abs(feats["median_f0"] - 220.0) < 5.0, (
        f"features must be measured from the stem90 slice, got {feats}"
    )
    assert math.isfinite(feats["f0_iqr"])
    assert rec["qc"] == {"pass": True, "reason": None}

    snapshot = _snapshot_path(tree)
    assert snapshot.is_file(), "a pre-rescue snapshot must be written"
    assert snapshot.name == "luna_monthly_pre_rescue.json"
    assert snapshot.read_bytes() == pre_measurements, (
        "the snapshot must hold the PRE-run measurements bytes (taken "
        "before the first replacement)"
    )

    windows = _windows(tree)
    assert windows[STEM90_KEY] == STEM_WIN
    assert windows["legacy01"] == [5.0, 25.0] and isinstance(windows["legacy01"], list)
    assert windows["legacy02"] == [30.0, 50.0] and isinstance(windows["legacy02"], list)
    assert windows[ID] == FIRST_WIN
    pre_w = json.loads(pre_windows)
    post_w = dict(windows)
    del post_w[STEM90_KEY]
    assert pre_w == post_w, "no pre-existing windows key may be rewritten"


def test_rescue_stem_hunt_qc_fail_keeps_record(
    tmp_path: Path, junk_wav: Path
) -> None:
    """Rule 3: a stem whose every 90 s window fails IQR -> kept_fail: the
    record is byte-identical, but the stem, the windows.json entry and
    the stem90 slice still exist (the rescue work is durable, only the
    record stays the old fail). No snapshot without a replacement."""
    tree = _tree(tmp_path, {ID: junk_wav}, [_record()])
    fake = _FakeIsolate()
    pre_measurements = tree["measurements_path"].read_bytes()

    summary = _run_rescue(tree, [ID], runner=fake)

    assert summary["ids"] == {ID: "kept_fail"}
    assert summary["counts"]["kept_fail"] == 1
    assert summary["counts"]["replaced"] == 0
    assert _records(tree) == [_record()], (
        "a QC-failing stem hunt must leave the existing record untouched"
    )
    assert tree["measurements_path"].read_bytes() == pre_measurements, (
        "no measurements rewrite may happen when nothing was replaced"
    )
    win = _windows(tree)[STEM90_KEY]
    assert isinstance(win, dict) and set(win) == {"start_s", "end_s"}, (
        "the stem90 window entry must still be recorded"
    )
    assert (tree["stems_dir"] / f"{ID}{STEM_SUFFIX}").is_file(), (
        "the isolated full stem must still exist"
    )
    assert _stem90_slice(tree).is_file(), "the stem90 slice must still exist"
    assert _snapshot_path(tree).is_file() is False, (
        "no snapshot without a replacement"
    )


def test_rescue_missing_raw_wav_skips_and_continues(
    tmp_path: Path, island_wav: Path
) -> None:
    """Rule 5: a missing data/audio/<id>.wav (and no ready stem) -> warn +
    skipped_missing_wav + continue; the NEXT id is still processed
    (replaced)."""
    tree = _tree(tmp_path, {GOOD: island_wav}, [_record(ID), _record(GOOD)])
    fake = _FakeIsolate()

    with pytest.warns(UserWarning, match=ID):
        summary = _run_rescue(tree, [ID, GOOD], runner=fake)

    assert summary["ids"][ID] == "skipped_missing_wav"
    assert summary["ids"][GOOD] == "replaced"
    assert summary["counts"] == {
        "replaced": 1,
        "kept_fail": 0,
        "skipped": 1,
        "error": 0,
        "total": 2,
    }
    records = _records(tree)
    assert records[0] == _record(ID), "the gap record must stay untouched"
    assert records[1]["id"] == GOOD


def test_rescue_isolate_error_skips_and_continues(
    tmp_path: Path, island_wav: Path
) -> None:
    """Rule 6: a raising isolate runner -> warn + error + continue, and NO
    partial record (and no stem90 window): the failing record is
    untouched while the next id is still replaced."""
    tree = _tree(tmp_path, {ID: island_wav, GOOD: island_wav},
                 [_record(ID), _record(GOOD)])

    with pytest.warns(UserWarning, match=ID):
        summary = _run_rescue(tree, [ID, GOOD], runner=_ExplodingIsolate())

    assert summary["ids"][ID] == "error"
    assert summary["ids"][GOOD] == "replaced"
    assert summary["counts"]["error"] == 1
    records = _records(tree)
    assert records[0] == _record(ID), "no partial record for the errored id"
    assert records[1]["qc"]["pass"] is True
    assert STEM90_KEY not in _windows(tree), (
        "a failed isolation must not leave a stem90 window entry"
    )
    assert not (tree["stems_dir"] / f"{ID}{STEM_SUFFIX}").exists()


def test_rescue_dry_run_touches_nothing(tmp_path: Path, island_wav: Path) -> None:
    """Rule 8: dry_run computes everything possible but the whole
    filesystem is byte-identical afterwards. With nothing prepared the id
    is skipped_dry_run (isolation would have to write); with a ready
    stem + recorded stem90 window + existing slice the WOULD-BE outcome
    is measured and reported as ``replaced`` — still with zero writes."""
    # fresh tree: isolation would have to run -> skipped, nothing written
    fresh = _tree(tmp_path / "fresh", {ID: island_wav}, [_record()])
    pre_fresh = {
        "measurements": fresh["measurements_path"].read_bytes(),
        "windows": fresh["windows_path"].read_bytes(),
    }

    with pytest.warns(UserWarning, match=ID):
        summary = _run_rescue(fresh, [ID], runner=_FakeIsolate(), dry_run=True)

    assert summary["dry_run"] is True
    assert summary["ids"] == {ID: "skipped_dry_run"}
    assert summary["snapshot"] is None
    assert fresh["measurements_path"].read_bytes() == pre_fresh["measurements"]
    assert fresh["windows_path"].read_bytes() == pre_fresh["windows"]
    assert not _snapshot_path(fresh).exists(), "dry-run writes no snapshot"
    assert not (fresh["stems_dir"] / f"{ID}{STEM_SUFFIX}").exists(), (
        "dry-run must never isolate"
    )
    assert not _stem90_slice(fresh).exists(), "dry-run writes no slice"

    # ready tree: stem + recorded window + slice -> full forecast, no writes
    ready = _tree(
        tmp_path / "ready",
        {ID: island_wav},
        [_record()],
        windows_extra={STEM90_KEY: STEM_WIN},
        stems_for=(ID,),
        slices_for=(ID,),
    )
    pre_ready = {
        "measurements": ready["measurements_path"].read_bytes(),
        "windows": ready["windows_path"].read_bytes(),
        "stem": (ready["stems_dir"] / f"{ID}{STEM_SUFFIX}").read_bytes(),
        "slice": _stem90_slice(ready).read_bytes(),
    }
    fake = _FakeIsolate()

    summary = _run_rescue(ready, [ID], runner=fake, dry_run=True)

    assert summary["dry_run"] is True
    assert summary["ids"] == {ID: "replaced"}, (
        "dry-run reports the would-be outcome for an already-measurable id"
    )
    assert summary["snapshot"] is None
    assert fake.calls == [], "dry-run must never isolate"
    assert ready["measurements_path"].read_bytes() == pre_ready["measurements"]
    assert ready["windows_path"].read_bytes() == pre_ready["windows"]
    assert (ready["stems_dir"] / f"{ID}{STEM_SUFFIX}").read_bytes() == (
        pre_ready["stem"]
    )
    assert _stem90_slice(ready).read_bytes() == pre_ready["slice"]
    assert not _snapshot_path(ready).exists()


def test_rescue_resume_reuses_window_stem_and_slice(
    tmp_path: Path, island_wav: Path
) -> None:
    """Rule 7: existing <id>_stem90 windows key + ready full stem (>1 MB) +
    existing stem90 slice -> the runner is NOT called; the recorded
    window is reused, the slice re-measured and the still-failing record
    is replaced."""
    tree = _tree(
        tmp_path,
        {ID: island_wav},
        [_record()],
        windows_extra={STEM90_KEY: STEM_WIN},
        stems_for=(ID,),
        slices_for=(ID,),
    )
    fake = _FakeIsolate()

    summary = _run_rescue(tree, [ID], runner=fake)

    assert summary["ids"] == {ID: "replaced"}
    assert fake.calls == [], "a ready stem must skip isolation entirely"
    rec = _records(tree)[0]
    assert rec["window"] == STEM_WIN, "the recorded stem90 window is reused"
    assert abs(rec["features"]["median_f0"] - 220.0) < 5.0, (
        "the stem90 slice must be re-measured"
    )
    assert rec["qc"] == {"pass": True, "reason": None}
    assert summary["windows_added"] == [], "no new window was added"
    assert _snapshot_path(tree).is_file()


def test_rescue_existing_snapshot_not_overwritten(
    tmp_path: Path, island_wav: Path
) -> None:
    """Rule 4 (idempotent): an existing pre-rescue snapshot is never
    overwritten, even when this run replaces records."""
    tree = _tree(tmp_path, {ID: island_wav}, [_record()])
    sentinel = b'{"sentinel": true}\n'
    _snapshot_path(tree).write_bytes(sentinel)

    summary = _run_rescue(tree, [ID], runner=_FakeIsolate())

    assert summary["ids"] == {ID: "replaced"}
    assert _snapshot_path(tree).read_bytes() == sentinel, (
        "the existing snapshot must survive untouched"
    )


def test_rescue_failing_ids_are_exactly_the_qc_fails() -> None:
    """Rule 9 (unit): the default id selection is exactly the
    qc.pass==false records — nan (f0_missing), iqr (f0_iqr) and high
    (f0_high) fails all included, passing records excluded, file order
    kept (pinned record dicts, not audio)."""
    records = [
        _record("nan0000001", features={
            "median_f0": float("nan"), "f0_iqr": float("nan"),
            "voiced_fraction": 0.0,
        }, qc={"pass": False, "reason": "f0_missing"}),
        _record("pass000001", qc={"pass": True, "reason": None}),
        _record("iqr0000001", features={
            "median_f0": 250.0, "f0_iqr": 260.0, "voiced_fraction": 0.9,
        }, qc={"pass": False, "reason": "f0_iqr"}),
        _record("high000001", features={
            "median_f0": 615.0, "f0_iqr": 30.0, "voiced_fraction": 0.9,
        }, qc={"pass": False, "reason": "f0_high"}),
    ]

    assert rescue.failing_ids(records) == [
        "nan0000001", "iqr0000001", "high000001"
    ]


def test_rescue_default_selection_runs_qc_fails_only(
    tmp_path: Path, island_wav: Path
) -> None:
    """Rule 9 (through the batch): ids=None selects exactly the failing
    records; the passing one is not probed at all (no wav needed for it)."""
    records = [
        _record(ID),
        _record(PASS_ID, qc={"pass": True, "reason": None}),
        _record(GOOD),
    ]
    tree = _tree(tmp_path, {ID: island_wav, GOOD: island_wav}, records)
    fake = _FakeIsolate()

    summary = _run_rescue(tree, None, runner=fake)

    assert set(summary["ids"]) == {ID, GOOD}, (
        "the default selection must exclude the qc-passing record"
    )
    assert summary["counts"]["total"] == 2


def test_rescue_qc_passing_ids_are_untouched(
    tmp_path: Path, island_wav: Path
) -> None:
    """Rule 9 (explicit): a QC-passing id handed to the batch explicitly is
    skipped_qc_pass and its record is untouched — rescue exists for fails."""
    tree = _tree(
        tmp_path,
        {ID: island_wav},
        [_record(ID), _record(PASS_ID, qc={"pass": True, "reason": None})],
    )
    fake = _FakeIsolate()

    with pytest.warns(UserWarning, match=PASS_ID):
        summary = _run_rescue(tree, [ID, PASS_ID], runner=fake)

    assert summary["ids"][PASS_ID] == "skipped_qc_pass"
    assert summary["ids"][ID] == "replaced"
    assert _records(tree)[1] == _record(PASS_ID, qc={"pass": True, "reason": None}), (
        "a passing record must never be rewritten"
    )


def test_rescue_windows_merge_only_and_rewrite_only_on_add(
    tmp_path: Path, junk_wav: Path
) -> None:
    """Rule 10: windows.json is merge-only — legacy array entries survive
    as arrays, pre-existing keys (first window and a <id>_raw90 entry
    from the earlier raw-hunt era) are never rewritten — and the file is
    only rewritten when a new key was added (a second run that reuses
    the recorded stem90 window leaves the bytes identical)."""
    raw90_key = f"{ID}_raw90"
    tree = _tree(
        tmp_path,
        {ID: junk_wav},
        [_record()],
        windows_extra={raw90_key: dict(FIRST_WIN)},
    )
    fake = _FakeIsolate()

    summary = _run_rescue(tree, [ID], runner=fake)
    assert summary["ids"] == {ID: "kept_fail"}
    assert summary["windows_added"] == [STEM90_KEY]
    after_first = tree["windows_path"].read_bytes()

    windows = _windows(tree)
    assert windows["legacy01"] == [5.0, 25.0] and isinstance(windows["legacy01"], list)
    assert windows["legacy02"] == [30.0, 50.0] and isinstance(windows["legacy02"], list)
    assert windows[ID] == FIRST_WIN
    assert windows[raw90_key] == FIRST_WIN, "pre-existing keys are never rewritten"
    assert isinstance(windows[STEM90_KEY], dict)

    # second run: the recorded stem90 window is reused -> no rewrite at all
    summary2 = _run_rescue(tree, [ID], runner=_FakeIsolate())
    assert summary2["ids"] == {ID: "kept_fail"}
    assert summary2["windows_added"] == [], "nothing new was added"
    assert tree["windows_path"].read_bytes() == after_first, (
        "windows.json must not be rewritten when nothing was added"
    )


# ---------------------------------------------------------------- CLI tests


def test_cli_rescue_ids_file_dash_leading_id_reaches_run_rescue(
    tmp_path: Path, monkeypatch
) -> None:
    """Rule 11: `rescue --ids-file` accepts dash-leading ids and unions
    them after --ids exactly like retry does."""
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("-DwvlhziHBI\n\nCUx63C9SkW8\n", encoding="utf-8")
    seen: dict = {}

    def fake_run_rescue(ids, data_dir, **kwargs):
        seen["ids"] = ids
        seen["kwargs"] = kwargs
        return {"ids": {}, "counts": {"total": 0}}

    monkeypatch.setattr(rescue, "run_rescue", fake_run_rescue)
    import vvc.__main__ as cli

    monkeypatch.setattr(cli, "run_rescue", fake_run_rescue)
    cli.main([
        "rescue",
        "--measurements", str(tmp_path / "m.json"),
        "--windows", str(tmp_path / "w.json"),
        "--data-dir", str(tmp_path / "data"),
        "--stems-dir", str(tmp_path / "stems"),
        "--ids", "-AbcdEfGh123",
        "--ids-file", str(ids_file),
    ])

    assert seen["ids"] == ["-AbcdEfGh123", "-DwvlhziHBI", "CUx63C9SkW8"]


def test_cli_rescue_model_file_dir_default_is_data_models(
    tmp_path: Path, monkeypatch
) -> None:
    """Rule 11: the rescue CLI's --model-file-dir default is
    "data/models" (different from retry's None) and reaches run_rescue."""
    seen: dict = {}

    def fake_run_rescue(ids, data_dir, **kwargs):
        seen["kwargs"] = kwargs
        return {"ids": {}, "counts": {"total": 0}}

    import vvc.__main__ as cli

    monkeypatch.setattr(cli, "run_rescue", fake_run_rescue)
    cli.main([
        "rescue",
        "--measurements", str(tmp_path / "m.json"),
        "--windows", str(tmp_path / "w.json"),
        "--data-dir", str(tmp_path / "data"),
        "--stems-dir", str(tmp_path / "stems"),
    ])

    assert seen["kwargs"]["model_file_dir"] == "data/models"


def test_cli_rescue_dry_run_prints_summary_and_writes_nothing(
    tmp_path: Path, island_wav: Path, capsys
) -> None:
    """Rule 11 (end to end): `python -m vvc rescue --dry-run` prints
    the summary JSON and writes nothing."""
    tree = _tree(
        tmp_path,
        {ID: island_wav},
        [_record()],
        windows_extra={STEM90_KEY: STEM_WIN},
        stems_for=(ID,),
        slices_for=(ID,),
    )
    pre_measurements = tree["measurements_path"].read_bytes()
    pre_windows = tree["windows_path"].read_bytes()

    main([
        "rescue",
        "--measurements", str(tree["measurements_path"]),
        "--windows", str(tree["windows_path"]),
        "--data-dir", str(tree["data_dir"]),
        "--stems-dir", str(tree["stems_dir"]),
        "--model-filename", MODEL_CKPT,
        "--dry-run",
    ])

    out = capsys.readouterr().out
    summary = json.loads(out)
    assert summary["ids"] == {ID: "replaced"}
    assert summary["dry_run"] is True
    assert tree["measurements_path"].read_bytes() == pre_measurements
    assert tree["windows_path"].read_bytes() == pre_windows
    assert not _snapshot_path(tree).exists()
