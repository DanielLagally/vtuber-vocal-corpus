"""Product tests for the 2nd-window retry batch (PLAN next-step 1; A4).

User-visible rules (synthetic wavs + a fake isolate runner ONLY — tones;
never Cover/hololive audio, never GPU, never downloads):

1. ``retry.run_retry(ids, data_dir, *, measurements_path, windows_path,
   stems_dir, model_filename=DEFAULT_MODEL_FILENAME, isolate_runner=None,
   out_path=None, dry_run=False, model_file_dir=None, log=None)`` returns
   a summary dict: ``{"ids": {id: outcome}, "counts": {replaced,
   kept_fail, skipped, error, total}, "windows_added": [keys],
   "snapshot": path|None, "dry_run": bool}``. A per-id error NEVER aborts
   the batch (outcome "error", warn, next id still processed).
2. Happy path: for a QC-failing record, the 2nd speech window is hunted
   on ``data/audio/<id>.wav`` (never overlapping the record's first
   window), recorded in windows.json under key ``<id>_raw90b`` as
   ``{"start_s": .., "end_s": ..}`` — APPEND/merge only: pre-existing
   keys, including the 16 legacy array entries of the real index, are
   never rewritten. Then ``data/windows/<id>_raw90b.wav`` is sliced, the
   stem is isolated via the injectable runner, the WHOLE stem is
   measured, and if QC passes the record is REPLACED in place keeping
   ``id``/``month``/``score``/``model`` from the old record, with
   ``window`` = the 2nd window, ``features`` from the stem and ``qc``
   from the verdict (D3 replace-in-place).
3. If the 2nd window fails QC the existing record is untouched (the
   first-window fail stays); the windows.json entry and the stem still
   exist (``kept_fail``).
4. Snapshot (D3): BEFORE the first replacement of a run the measurements
   file is copied to ``<measurements_stem>_pre_retry_snapshot.json`` in
   the same directory. The snapshot holds the pre-run file bytes; an
   existing snapshot is NEVER overwritten (idempotent re-runs).
5. A missing raw wav, an id without a measurement record, an
   already-QC-passing record, or a wav with no fitting 2nd window is
   skipped with a warning and the batch continues (``skipped_*``).
6. If the isolate runner raises, the id is an ``error``: warn and
   continue, and the measurements file has NO partial record.
7. Crash-resume: when the ``<id>_raw90b`` windows key and a ready stem
   (> 1 MB, the isolate CLI's skip-existing threshold) already exist,
   the runner is NOT called; the recorded window is reused, the stem is
   re-measured, and a still-failing record is replaced.
8. ``dry_run=True`` computes everything possible (window hunt; whole
   measurement of an already-ready stem -> the WOULD-BE outcome in the
   summary) but writes NOTHING: no snapshot, no windows.json rewrite, no
   slice, no stem, no measurements rewrite.
9. Default id selection (ids=None, or the CLI without --ids) is EXACTLY
   the records whose ``qc.pass`` is false — nan/IQR/high fail reasons
   all included, file order — and never a passing id; a passing id
   passed explicitly is ``skipped_qc_pass`` and untouched.
10. Crash-resume writes: the measurements file is rewritten after each
    successful replacement; windows.json is rewritten only when a new
    window was added.
11. CLI ``python -m vvc retry`` wires the same function with
    defaults --measurements data/measurements/luna_monthly.json,
    --windows data/windows/windows.json, --data-dir data,
    --stems-dir data/stems_fast, --model-filename DEFAULT_MODEL_FILENAME,
    --ids (optional), --dry-run, and prints the summary as JSON.
12. ``model_file_dir`` (audio-separator's ``--model_file_dir``, CLI
    ``--model-file-dir``): when passed, it is forwarded so the model ckpt
    cache lands in that dir (in-tree ``data/models``); when None it is
    NOT forwarded at all — the isolate call is unchanged by default.

Fixtures are tiny synthetic wavs (8 kHz mono 16-bit, tones) synthesized
here into <repo>/fixtures/ — deterministic, no third-party test deps.
The fake runner copies the input wav to the stem path isolate's naming
produces (padding past 1 MB so resume semantics are exercised).
"""

import array
import json
import math
import shutil
import sys
import wave
from pathlib import Path

import pytest

from vvc import retry
from vvc.__main__ import main
from vvc.isolate import vocals_path

SR = 8_000
ID = "retryAaa01"
GOOD = "retryBbb02"
PASS_ID = "retryCcc03"
MODEL_CKPT = "bs_roformer_vocals_resurrection_unwa.ckpt"
STEM_SUFFIX = "_(vocals)_bs_roformer_vocals_resurrection_unwa.wav"
FIRST_WIN = {"start_s": 0.0, "end_s": 90.0}
SECOND_WIN = {"start_s": 90.0, "end_s": 180.0}
RAW90B_KEY = f"{ID}_raw90b"
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
def clean2nd_wav() -> Path:
    """180 s: first window [0,90) then a clean 220 Hz tail — the only
    non-overlapping 90 s candidate [90,180) is a tight-IQR tone (QC
    pass). 8 kHz keeps the fixture small; features.py resamples to
    16 kHz itself."""
    path = FIXTURES_DIR / "retry_clean2nd.wav"
    if not path.is_file():
        _write_wav(path, _sine(220.0, 180.0))
    return path


@pytest.fixture(scope="module")
def junk2nd_wav() -> Path:
    """180 s: first window [0,90) clean, tail [90,180) alternating
    200/450 Hz half-second tones — any 90 s candidate there has
    f0_iqr ~ 250 (QC fail)."""
    path = FIXTURES_DIR / "retry_junk2nd.wav"
    if not path.is_file():
        tail: list[int] = []
        for _ in range(90):
            tail += _sine(200.0, 0.5) + _sine(450.0, 0.5)
        _write_wav(path, _sine(220.0, 90.0) + tail)
    return path


# ---------------------------------------------------------------- helpers


class _FakeIsolate:
    """Stand-in for the audio-separator runner: writes the stem file
    isolate's naming produces (input wav, repeated past 1 MB) and
    records every call."""

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
) -> dict:
    """A fake data tree: data/audio/<id>.wav, windows.json (two legacy
    array entries + each record's first window), the measurements file,
    and optional pre-made raw90b stems."""
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
        # a ready >1 MB raw90b stem: a copy of the raw wav (180 s @ 8 kHz
        # mono 16-bit = 2.88 MB) named like the isolate runner writes it
        shutil.copyfile(
            wavs[vid], stems_dir / f"{vid}_raw90b{STEM_SUFFIX}"
        )
    return {
        "data_dir": data_dir,
        "measurements_path": measurements_path,
        "windows_path": windows_path,
        "stems_dir": stems_dir,
    }


def _run_retry(tree: dict, ids, *, runner=None, **kwargs) -> dict:
    return retry.run_retry(
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
        tree["measurements_path"].stem + "_pre_retry_snapshot.json"
    )


# ------------------------------------------------------------------- tests


def test_retry_happy_path_replaces_record_and_snapshots(
    tmp_path: Path, clean2nd_wav: Path
) -> None:
    """Rules 1+2+4: a failing record + a clean 2nd window -> the record is
    replaced in place (month/score/model preserved, window = 2nd window,
    features measured from the raw90b stem, qc pass), the snapshot holds
    the PRE-run measurements bytes, and windows.json gained only the
    <id>_raw90b key while the legacy array entries survive unchanged."""
    tree = _tree(tmp_path, {ID: clean2nd_wav}, [_record()])
    fake = _FakeIsolate()
    pre_measurements = tree["measurements_path"].read_bytes()
    pre_windows = tree["windows_path"].read_bytes()

    summary = _run_retry(tree, [ID], runner=fake)

    assert summary["ids"] == {ID: "replaced"}
    assert summary["counts"] == {
        "replaced": 1,
        "kept_fail": 0,
        "skipped": 0,
        "error": 0,
        "total": 1,
    }
    assert summary["dry_run"] is False
    assert fake.calls, "the isolate runner must have been called once"

    records = _records(tree)
    assert len(records) == 1
    rec = records[0]
    assert rec["id"] == ID
    assert rec["month"] == "2024-07", "month must be preserved from the old record"
    assert rec["score"] == 63.5, "score must be preserved from the old record"
    assert rec["model"] == MODEL_CKPT, "model provenance must be preserved"
    assert rec["window"] == SECOND_WIN, "window metadata must be the 2nd window"
    feats = rec["features"]
    assert set(feats) == {"median_f0", "f0_iqr", "voiced_fraction"}
    assert abs(feats["median_f0"] - 220.0) < 5.0, (
        f"features must be measured from the raw90b stem, got {feats}"
    )
    assert math.isfinite(feats["f0_iqr"])
    assert rec["qc"] == {"pass": True, "reason": None}

    snapshot = _snapshot_path(tree)
    assert snapshot.is_file(), "a pre-retry snapshot must be written"
    assert snapshot.read_bytes() == pre_measurements, (
        "the snapshot must hold the PRE-run measurements bytes (taken "
        "before the first replacement)"
    )
    assert snapshot.name == "luna_monthly_pre_retry_snapshot.json"

    windows = _windows(tree)
    assert windows[RAW90B_KEY] == SECOND_WIN
    assert windows["legacy01"] == [5.0, 25.0] and isinstance(windows["legacy01"], list)
    assert windows["legacy02"] == [30.0, 50.0] and isinstance(windows["legacy02"], list)
    assert windows[ID] == FIRST_WIN
    pre_w = json.loads(pre_windows)
    post_w = dict(windows)
    del post_w[RAW90B_KEY]
    assert pre_w == post_w, "no pre-existing windows key may be rewritten"


def test_retry_second_window_qc_fail_keeps_record(
    tmp_path: Path, junk2nd_wav: Path
) -> None:
    """Rule 3: a junk 2nd window (f0_iqr ~ 250) -> kept_fail: the record is
    byte-identical, but the windows.json entry and the stem still exist
    (the retry work is durable, only the record stays the first fail)."""
    tree = _tree(tmp_path, {ID: junk2nd_wav}, [_record()])
    fake = _FakeIsolate()
    pre_measurements = tree["measurements_path"].read_bytes()

    summary = _run_retry(tree, [ID], runner=fake)

    assert summary["ids"] == {ID: "kept_fail"}
    assert summary["counts"]["kept_fail"] == 1
    assert summary["counts"]["replaced"] == 0
    assert _records(tree) == [_record()], (
        "a QC-failing 2nd window must leave the existing record untouched"
    )
    assert tree["measurements_path"].read_bytes() == pre_measurements, (
        "no measurements rewrite may happen when nothing was replaced"
    )
    assert _windows(tree)[RAW90B_KEY] == SECOND_WIN, (
        "the 2nd window entry must still be recorded"
    )
    stem = tree["stems_dir"] / f"{ID}_raw90b{STEM_SUFFIX}"
    assert stem.is_file(), "the isolated stem must still exist"
    assert _snapshot_path(tree).is_file() is False, (
        "no snapshot without a replacement"
    )


def test_retry_missing_raw_wav_skips_and_continues(
    tmp_path: Path, clean2nd_wav: Path
) -> None:
    """Rule 5: a missing data/audio/<id>.wav -> warn + skipped_missing_wav
    + continue; the NEXT id is still processed (replaced)."""
    tree = _tree(tmp_path, {GOOD: clean2nd_wav}, [_record(ID), _record(GOOD)])
    fake = _FakeIsolate()

    with pytest.warns(UserWarning, match=ID):
        summary = _run_retry(tree, [ID, GOOD], runner=fake)

    assert summary["ids"][ID] == "skipped_missing_wav"
    assert summary["ids"][GOOD] == "replaced"
    assert summary["counts"] == {
        "replaced": 1,
        "kept_fail": 0,
        "skipped": 1,
        "error": 0,
        "total": 2,
    }
    assert _records(tree)[1]["id"] == GOOD


def test_retry_isolate_error_skips_and_continues(
    tmp_path: Path, clean2nd_wav: Path
) -> None:
    """Rule 6: a raising isolate runner -> warn + error + continue, and NO
    partial record: the failing record is untouched while the next id is
    still replaced."""
    tree = _tree(tmp_path, {ID: clean2nd_wav, GOOD: clean2nd_wav},
                 [_record(ID), _record(GOOD)])

    with pytest.warns(UserWarning, match=ID):
        summary = _run_retry(tree, [ID, GOOD], runner=_ExplodingIsolate())

    assert summary["ids"][ID] == "error"
    assert summary["ids"][GOOD] == "replaced"
    assert summary["counts"]["error"] == 1
    records = _records(tree)
    assert records[0] == _record(ID), "no partial record for the errored id"
    assert records[1]["qc"]["pass"] is True


def test_retry_dry_run_touches_nothing(tmp_path: Path, clean2nd_wav: Path) -> None:
    """Rule 8: dry_run computes everything possible (here: the stem is
    already ready, so the WOULD-BE outcome is measurable) but the whole
    filesystem is byte-identical afterwards — no snapshot, no windows.json
    rewrite, no slice, no stem write, no measurements rewrite."""
    tree = _tree(
        tmp_path,
        {ID: clean2nd_wav},
        [_record()],
        stems_for=(ID,),
    )
    fake = _FakeIsolate()
    pre = {
        "measurements": tree["measurements_path"].read_bytes(),
        "windows": tree["windows_path"].read_bytes(),
        "stem": (tree["stems_dir"] / f"{ID}_raw90b{STEM_SUFFIX}").read_bytes(),
    }

    summary = _run_retry(tree, [ID], runner=fake, dry_run=True)

    assert summary["dry_run"] is True
    assert summary["ids"] == {ID: "replaced"}, (
        "dry-run reports the would-be outcome for an already-measurable stem"
    )
    assert summary["snapshot"] is None
    assert fake.calls == [], "dry-run must never isolate"
    assert tree["measurements_path"].read_bytes() == pre["measurements"]
    assert tree["windows_path"].read_bytes() == pre["windows"]
    assert (tree["stems_dir"] / f"{ID}_raw90b{STEM_SUFFIX}").read_bytes() == (
        pre["stem"]
    )
    assert not _snapshot_path(tree).exists(), "dry-run writes no snapshot"
    assert not (tree["data_dir"] / "windows" / f"{ID}_raw90b.wav").exists(), (
        "dry-run writes no slice"
    )


def test_retry_resume_reuses_window_and_stem_without_runner(
    tmp_path: Path, clean2nd_wav: Path
) -> None:
    """Rule 7: existing <id>_raw90b windows key + ready stem (>1 MB) ->
    the runner is NOT called; the recorded window is reused, the stem is
    re-measured and the still-failing record is replaced."""
    tree = _tree(
        tmp_path,
        {ID: clean2nd_wav},
        [_record()],
        windows_extra={RAW90B_KEY: SECOND_WIN},
        stems_for=(ID,),
    )
    fake = _FakeIsolate()

    summary = _run_retry(tree, [ID], runner=fake)

    assert summary["ids"] == {ID: "replaced"}
    assert fake.calls == [], "a ready stem must skip isolation entirely"
    rec = _records(tree)[0]
    assert rec["window"] == SECOND_WIN, "the recorded 2nd window is reused"
    assert abs(rec["features"]["median_f0"] - 220.0) < 5.0, (
        "the stem must be re-measured"
    )
    assert rec["qc"] == {"pass": True, "reason": None}
    assert summary["windows_added"] == [], "no new window was added"


def test_retry_existing_snapshot_not_overwritten(
    tmp_path: Path, clean2nd_wav: Path
) -> None:
    """Rule 4 (idempotent): an existing pre-retry snapshot is never
    overwritten, even when this run replaces records."""
    tree = _tree(tmp_path, {ID: clean2nd_wav}, [_record()])
    sentinel = b'{"sentinel": true}\n'
    _snapshot_path(tree).write_bytes(sentinel)

    summary = _run_retry(tree, [ID], runner=_FakeIsolate())

    assert summary["ids"] == {ID: "replaced"}
    assert _snapshot_path(tree).read_bytes() == sentinel, (
        "the existing snapshot must survive untouched"
    )


def test_default_ids_are_exactly_the_qc_fails() -> None:
    """Rule 9: the default id selection is exactly the qc.pass==false
    records — nan (f0_missing), iqr (f0_iqr) and high (f0_high) fails all
    included, passing records excluded, file order kept. Pinned feature
    dicts: the tracker cannot literally produce >=600 Hz medians from
    synthetic tones, so the stored records are pinned, not the audio."""
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

    assert retry.failing_ids(records) == ["nan0000001", "iqr0000001", "high000001"]


def test_default_run_retry_selects_failing_ids(
    tmp_path: Path, clean2nd_wav: Path
) -> None:
    """Rule 9 (through the batch): ids=None selects exactly the failing
    records; the passing one is not probed at all (no wav needed for it)."""
    records = [
        _record(ID),
        _record(PASS_ID, qc={"pass": True, "reason": None}),
        _record(GOOD),
    ]
    tree = _tree(tmp_path, {ID: clean2nd_wav, GOOD: clean2nd_wav}, records)
    fake = _FakeIsolate()

    summary = _run_retry(tree, None, runner=fake)

    assert set(summary["ids"]) == {ID, GOOD}, (
        "the default selection must exclude the qc-passing record"
    )
    assert summary["counts"]["total"] == 2


def test_qc_passing_ids_are_untouched(tmp_path: Path, clean2nd_wav: Path) -> None:
    """Rule 9 (explicit): a QC-passing id handed to the batch explicitly is
    skipped_qc_pass and its record is untouched — retry exists for fails."""
    tree = _tree(
        tmp_path,
        {ID: clean2nd_wav},
        [_record(ID), _record(PASS_ID, qc={"pass": True, "reason": None})],
    )
    fake = _FakeIsolate()

    with pytest.warns(UserWarning, match=PASS_ID):
        summary = _run_retry(tree, [ID, PASS_ID], runner=fake)

    assert summary["ids"][PASS_ID] == "skipped_qc_pass"
    assert summary["ids"][ID] == "replaced"
    assert _records(tree)[1] == _record(PASS_ID, qc={"pass": True, "reason": None}), (
        "a passing record must never be rewritten"
    )


# ---------------------------------------------------------------- CLI tests


def test_cli_retry_dry_run_prints_summary_and_writes_nothing(
    tmp_path: Path, clean2nd_wav: Path, capsys
) -> None:
    """Rule 11 (dry-run): `python -m vvc retry --dry-run` prints the
    summary JSON and writes nothing."""
    tree = _tree(tmp_path, {ID: clean2nd_wav}, [_record()], stems_for=(ID,))
    pre_measurements = tree["measurements_path"].read_bytes()
    pre_windows = tree["windows_path"].read_bytes()

    main([
        "retry",
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


def test_cli_retry_replaces_and_prints_summary(
    tmp_path: Path, clean2nd_wav: Path, capsys, monkeypatch
) -> None:
    """Rule 11: `python -m vvc retry` (no --ids -> the qc-fail
    defaults) replaces the record, writes the snapshot and windows key,
    and prints the summary JSON."""
    tree = _tree(tmp_path, {ID: clean2nd_wav}, [_record()])

    def fake_isolate(src, out_dir, *, model_filename=None, runner=None):
        dest = vocals_path(src, out_dir, model_filename=model_filename)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        return dest

    monkeypatch.setattr(retry, "isolate_vocals", fake_isolate)

    main([
        "retry",
        "--measurements", str(tree["measurements_path"]),
        "--windows", str(tree["windows_path"]),
        "--data-dir", str(tree["data_dir"]),
        "--stems-dir", str(tree["stems_dir"]),
        "--model-filename", MODEL_CKPT,
    ])

    summary = json.loads(capsys.readouterr().out)
    assert summary["ids"] == {ID: "replaced"}
    assert summary["counts"]["total"] == 1, "default selection: only the failing id"
    rec = _records(tree)[0]
    assert rec["qc"]["pass"] is True and rec["window"] == SECOND_WIN
    assert _snapshot_path(tree).is_file()
    assert _windows(tree)[RAW90B_KEY] == SECOND_WIN


# ---------------------------------------------------- model_file_dir option


def test_retry_model_file_dir_passthrough(tmp_path: Path, clean2nd_wav: Path) -> None:
    """Rule 12: run_retry(model_file_dir=...) forwards it into the isolate
    argv as --model_file_dir <dir> (in-tree ckpt cache)."""
    tree = _tree(tmp_path, {ID: clean2nd_wav}, [_record()])
    fake = _FakeIsolate()
    models_dir = str(tmp_path / "models")

    summary = _run_retry(tree, [ID], runner=fake, model_file_dir=models_dir)

    assert summary["ids"] == {ID: "replaced"}
    assert fake.calls, "the isolate runner must have been called"
    argv = fake.calls[0]
    assert "--model_file_dir" in argv, f"--model_file_dir missing from argv {argv!r}"
    assert argv[argv.index("--model_file_dir") + 1] == models_dir


def test_retry_default_run_does_not_forward_model_file_dir(
    tmp_path: Path, clean2nd_wav: Path
) -> None:
    """Rule 12: without model_file_dir the isolate argv carries no
    --model_file_dir at all — the default isolate call is unchanged."""
    tree = _tree(tmp_path, {ID: clean2nd_wav}, [_record()])
    fake = _FakeIsolate()

    summary = _run_retry(tree, [ID], runner=fake)

    assert summary["ids"] == {ID: "replaced"}
    assert fake.calls, "the isolate runner must have been called"
    assert not any("model_file_dir" in part for part in fake.calls[0]), (
        f"default argv must not carry --model_file_dir: {fake.calls[0]!r}"
    )


def test_cli_retry_model_file_dir_flag_passthrough(
    tmp_path: Path, clean2nd_wav: Path, capsys, monkeypatch
) -> None:
    """Rule 12 (CLI): `retry --model-file-dir <dir>` forwards the value
    into run_retry -> isolate_vocals."""
    tree = _tree(tmp_path, {ID: clean2nd_wav}, [_record()])
    models_dir = str(tmp_path / "models")
    seen: dict[str, object] = {}

    def fake_isolate(src, out_dir, *, model_filename=None, runner=None,
                     model_file_dir=None):
        seen["model_file_dir"] = model_file_dir
        dest = vocals_path(src, out_dir, model_filename=model_filename)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        return dest

    monkeypatch.setattr(retry, "isolate_vocals", fake_isolate)

    main([
        "retry",
        "--measurements", str(tree["measurements_path"]),
        "--windows", str(tree["windows_path"]),
        "--data-dir", str(tree["data_dir"]),
        "--stems-dir", str(tree["stems_dir"]),
        "--model-filename", MODEL_CKPT,
        "--model-file-dir", models_dir,
    ])

    summary = json.loads(capsys.readouterr().out)
    assert summary["ids"] == {ID: "replaced"}
    assert seen["model_file_dir"] == models_dir, (
        "the CLI --model-file-dir value must reach isolate_vocals"
    )
