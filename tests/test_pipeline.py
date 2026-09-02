"""End-to-end pipeline integration test (STATE A7).

User-visible rule: the REAL library functions chain into a working
pipeline on synthetic tone/noise audio with a FAKE isolate runner (no
GPU, no network, no Cover/hololive audio, no downloads):

best_speech_window -> slice_wav -> isolate_vocals(runner=fake-copy) ->
measure.stem_features -> qc.qc_verdict -> retry.run_retry (fake runner,
2nd window on a two-island file) -> series.f0_quarterly +
series.write_plots + series.write_quarterly_plots.

Pinned behavior:
1. The first window lands on the continuous "melody island" (highest
   voiced fraction) and its stem FAILS QC (f0_iqr ~ 250 — alternating
   200/450 Hz tones read as melody junk, the exact QC case retry exists
   for). The measurement record is sane: a median in the tone range,
   the QC verdict matching the shared rule.
2. run_retry hunts a 2nd window that does NOT overlap the first and
   replaces the record ONLY because the 2nd window passes QC: the
   gated 300 Hz "speech island" wins the non-overlapping hunt at
   (210.0, 300.0) on the 15 s grid, the replaced record measures
   median ~ 300 Hz (the island's tone), qc pass, and the pre-retry
   snapshot holds the pre-run bytes. (The replace-only-if-pass flip
   side is covered by tests/test_retry.py kept_fail cases.)
3. The quarterly series reflects the FINAL measurement set: the
   replaced record's quarter aggregates the NEW median (not the
   original failing one), a fabricated QC-failing record in another
   quarter stays in the all-view but is a gap in the qc-view, and all
   four plot PNGs exist (monthly f0 + IQR, quarterly all + qc).

Fixture: 300 s @ 8 kHz mono 16-bit wav synthesized here into
<repo>/fixtures/ — deterministic (seeded RNG), stdlib only, no
third-party test deps: 0-20 s white noise, silence, 60-150 s melody
island (alternating 200/450 Hz half-second tones, continuous -> voiced
fraction 1.0, exactly one full 90 s grid window), silence, 210-290 s
speech island (300 Hz tone gated 0.5 s on / 0.5 s off -> voiced
fraction ~0.44), silence.
"""

import array
import json
import math
import random
import shutil
import sys
import wave
from pathlib import Path

import pytest

from vvc import measure, series, windows
from vvc.isolate import (
    DEFAULT_MODEL_FILENAME,
    isolate_vocals,
    vocals_path,
)
from vvc.qc import qc_verdict
from vvc.retry import run_retry

SR = 8_000
ID = "pipelineAa1"
MONTH = "2024-07"  # 2024-Q3 — mirrors a real gap quarter in Luna's data
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"

SECOND_WIN = {"start_s": 210.0, "end_s": 300.0}

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


def _noise(seconds: float, amp: int, rng: random.Random) -> list[int]:
    return [rng.randint(-amp, amp) for _ in range(int(seconds * SR))]


def _write_wav(path: Path, samples: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit PCM
        w.setframerate(SR)
        w.writeframes(_samples_to_bytes(samples))


@pytest.fixture(scope="module")
def two_island_wav() -> Path:
    """300 s two-island wav (see module docstring): the melody island
    wins the FIRST window hunt but fails QC; the gated speech island
    loses the first hunt yet wins the non-overlapping 2nd hunt and
    passes QC — so the retry replaces the record exactly because the
    2nd window passes."""
    path = FIXTURES_DIR / "pipeline_two_island.wav"
    if not path.is_file():
        rng = random.Random(1234)
        melody: list[int] = []
        for _ in range(90):  # 60-150 s: exactly one full 90 s grid window
            melody += _sine(200.0, 0.5) + _sine(450.0, 0.5)
        speech: list[int] = []
        for _ in range(80):  # 210-290 s: 0.5 s on / 0.5 s off
            speech += _sine(300.0, 0.5) + [0] * (SR // 2)
        samples = (
            _noise(20.0, 400, rng)  # 0-20 s: unvoiced white noise
            + [0] * (40 * SR)  # 20-60 s: silence
            + melody
            + [0] * (60 * SR)  # 150-210 s: silence
            + speech
            + [0] * (10 * SR)  # 290-300 s: silence
        )
        _write_wav(path, samples)
    return path


# ---------------------------------------------------------------- helpers


def _fake_isolate(argv: list[str]) -> None:
    """Stand-in for audio-separator: copies the input to the stem path
    isolate's naming produces (real naming math, no GPU/network)."""
    src = Path(argv[-1])
    out_dir = Path(argv[argv.index("--output_dir") + 1])
    model = argv[argv.index("--model_filename") + 1]
    dest = vocals_path(src, out_dir, model_filename=model)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)


# ------------------------------------------------------------------- tests


def test_pipeline_window_slice_isolate_measure_retry_aggregate_plots(
    tmp_path: Path, two_island_wav: Path
) -> None:
    """The full chain (module docstring rules 1-3): window -> slice ->
    isolate(fake) -> measure -> QC -> retry(fake) -> aggregate -> plots."""
    data_dir = tmp_path / "data"
    audio_dir = data_dir / "audio"
    wins_dir = data_dir / "windows"
    stems_dir = data_dir / "stems"
    measurements_dir = data_dir / "measurements"
    audio_dir.mkdir(parents=True)
    wins_dir.mkdir()
    measurements_dir.mkdir()
    shutil.copyfile(two_island_wav, audio_dir / f"{ID}.wav")
    raw_wav = audio_dir / f"{ID}.wav"
    windows_path = wins_dir / "windows.json"
    windows_path.write_text(json.dumps({}) + "\n", encoding="utf-8")
    measurements_path = measurements_dir / "luna_monthly.json"
    out_dir = tmp_path / "plots"

    # 1+2+3. first window, slice, isolate (fake runner — no GPU/network).
    # The melody island spans exactly one full grid window (60-150), the
    # only candidate with voiced fraction 1.0 — the winner is pinned.
    start_s, end_s = windows.best_speech_window(raw_wav)
    assert (start_s, end_s) == (60.0, 150.0), (
        f"the first window must be the full melody island (60-150 s), "
        f"got {(start_s, end_s)}"
    )
    slice_path = wins_dir / f"{ID}_raw90.wav"
    windows.slice_wav(raw_wav, slice_path, start_s, end_s)
    stem = isolate_vocals(
        slice_path,
        stems_dir,
        model_filename=DEFAULT_MODEL_FILENAME,
        runner=_fake_isolate,
    )
    assert stem.is_file()

    # 4+5. measure the WHOLE stem, apply the shared QC rule.
    features = measure.stem_features(stem)
    qc_pass, qc_reason = qc_verdict(features)
    assert qc_pass is False and qc_reason == "f0_iqr", (
        f"the melody island must fail QC via IQR, got {features}"
    )
    assert features["f0_iqr"] >= 200.0
    assert 190.0 <= features["median_f0"] <= 460.0, (
        f"median must sit inside the alternating 200/450 Hz tone span "
        f"(boundary frames included), got {features}"
    )

    # The measurement record, shaped exactly like run_monthly persists it.
    record = {
        "id": ID,
        "month": MONTH,
        "score": 63.5,
        "window": {"start_s": start_s, "end_s": end_s},
        "features": features,
        "qc": {"pass": qc_pass, "reason": qc_reason},
        "model": DEFAULT_MODEL_FILENAME,
    }
    records = [record]
    measurements_path.write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    pre_retry_bytes = measurements_path.read_bytes()

    # 6. retry on a 2nd window (fake runner): the gated 300 Hz speech
    # island is the best NON-overlapping window and PASSES QC, so and
    # only so the record is replaced.
    summary = run_retry(
        [ID],
        data_dir,
        measurements_path=measurements_path,
        windows_path=windows_path,
        stems_dir=stems_dir,
        model_filename=DEFAULT_MODEL_FILENAME,
        isolate_runner=_fake_isolate,
    )
    assert summary["ids"] == {ID: "replaced"}
    assert summary["counts"]["replaced"] == 1
    assert summary["windows_added"] == [f"{ID}_raw90b"]

    replaced = json.loads(measurements_path.read_text(encoding="utf-8"))
    assert len(replaced) == 1
    rec = replaced[0]
    assert rec["id"] == ID and rec["month"] == MONTH
    assert rec["score"] == 63.5 and rec["model"] == DEFAULT_MODEL_FILENAME
    assert rec["window"] == SECOND_WIN, (
        f"the 2nd window must be the gated 300 Hz island at (210, 300), "
        f"got {rec['window']}"
    )
    assert abs(rec["features"]["median_f0"] - 300.0) < 5.0, (
        f"the replaced record must measure the speech island's 300 Hz "
        f"tone, got {rec['features']}"
    )
    assert rec["qc"] == {"pass": True, "reason": None}
    snapshot = measurements_path.with_name(
        measurements_path.stem + "_pre_retry_snapshot.json"
    )
    assert snapshot.is_file()
    assert snapshot.read_bytes() == pre_retry_bytes, (
        "the snapshot must hold the pre-retry measurements bytes"
    )
    stored_windows = json.loads(windows_path.read_text(encoding="utf-8"))
    assert stored_windows[f"{ID}_raw90b"] == SECOND_WIN

    # 7. the quarterly series reflects the FINAL measurement set, then
    # all four plot PNGs. The extra 2024-10 record is a fabricated
    # synthetic dict (QC-failing) so the all-vs-qc gap is visible.
    records = json.loads(measurements_path.read_text(encoding="utf-8"))
    records.append(
        {
            "id": "fabricated01",
            "month": "2024-10",
            "score": 63.5,
            "window": {"start_s": 0.0, "end_s": 90.0},
            "features": {
                "median_f0": 310.0,
                "f0_iqr": 260.0,
                "voiced_fraction": 0.6,
            },
            "qc": {"pass": False, "reason": "f0_iqr"},
            "model": DEFAULT_MODEL_FILENAME,
        }
    )
    quarterly_all = series.f0_quarterly(records)
    assert [(p["quarter"], p["n"]) for p in quarterly_all] == [
        ("2024-Q3", 1),
        ("2024-Q4", 1),
    ], "the final set: the replaced 2024-Q3 record + the failing 2024-Q4 one"
    assert abs(quarterly_all[0]["mean"] - rec["features"]["median_f0"]) < 1e-9, (
        "the 2024-Q3 point must aggregate the REPLACED median (the final "
        "measurement set), not the original failing first-window one"
    )
    assert quarterly_all[0]["min"] == quarterly_all[0]["max"] == quarterly_all[0]["mean"]
    quarterly_qc = series.f0_quarterly(records, qc=True)
    assert [(p["quarter"], p["n"]) for p in quarterly_qc] == [("2024-Q3", 1)], (
        "the QC-failing 2024-10 record must be a gap in the qc view"
    )

    series.write_plots(records, out_dir)
    series.write_quarterly_plots(records, out_dir)
    assert {p.name for p in out_dir.glob("*.png")} == {
        "f0_monthly.png",
        "f0_iqr_monthly.png",
        "f0_quarterly_all.png",
        "f0_quarterly_qc.png",
    }
