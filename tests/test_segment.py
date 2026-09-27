"""Measuring a chosen stretch of one stream as a one-off data point.

The ordinary corpus takes one 90 s window per stream, chosen by a speech score.
A one-off (a talent speaking another language on one stream, say) instead
covers a stretch the operator names, so every 90 s of it is measured: that
shows how much the voice varies inside the stretch rather than trusting one
window. Synthetic tones only — never Cover audio.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from vvc import segment

SR = 16_000


def _tone(path: Path, seconds: float) -> Path:
    t = np.arange(int(seconds * SR)) / SR
    sf.write(str(path), (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32), SR, subtype="PCM_16")
    return path


@pytest.fixture
def video():
    return {
        "id": "vidA",
        "topic_id": "talk",
        "duration": 5400,
        "available_at": "2026-09-26T12:00:00.000Z",
    }


def test_a_15_minute_section_becomes_ten_consecutive_90s_windows(tmp_path):
    audio = _tone(tmp_path / "vidA.wav", 900.0)
    slices = segment.slice_section(
        audio, tmp_path / "windows", "vidA", section_start_s=600.0, window_s=90.0
    )
    assert len(slices) == 10
    ids = [s["id"] for s in slices]
    assert len(set(ids)) == 10
    assert ids[0] == "vidA@600" and ids[-1] == "vidA@1410"
    assert slices[1]["window"] == {"start_s": 90.0, "end_s": 180.0}
    for s in slices:
        assert s["path"].name == f"{s['id']}_raw90.wav"
        info = sf.info(str(s["path"]))
        assert info.duration == pytest.approx(90.0, abs=1e-3)


def test_a_trailing_partial_window_is_dropped_not_padded(tmp_path):
    audio = _tone(tmp_path / "vidA.wav", 200.0)
    slices = segment.slice_section(
        audio, tmp_path / "windows", "vidA", section_start_s=0.0, window_s=90.0
    )
    assert [s["id"] for s in slices] == ["vidA@0", "vidA@90"]


def test_records_carry_the_source_streams_month_and_the_one_off_marker(tmp_path, video):
    audio = _tone(tmp_path / "vidA.wav", 180.0)
    slices = segment.slice_section(
        audio, tmp_path / "windows", "vidA", section_start_s=600.0, window_s=90.0
    )
    records = segment.v1_records(
        slices,
        video,
        section=(600.0, 780.0),
        measure=lambda path: {"median_f0": 220.0, "f0_iqr": 5.0, "voiced_fraction": 0.9},
        model="m.ckpt",
        language="en",
    )
    assert [r["id"] for r in records] == ["vidA@600", "vidA@690"]
    assert all(r["language"] == "en" for r in records)
    for r in records:
        assert r["month"] == "2026-09"
        assert r["one_off"] is True
        assert r["section"] == {"start_s": 600.0, "end_s": 780.0}
        assert r["model"] == "m.ckpt"
        assert "pass" in r["qc"]


def test_a_stream_with_no_air_date_is_refused_rather_than_given_an_invented_month(
    tmp_path, video
):
    audio = _tone(tmp_path / "vidA.wav", 90.0)
    slices = segment.slice_section(
        audio, tmp_path / "windows", "vidA", section_start_s=0.0, window_s=90.0
    )
    undated = {k: v for k, v in video.items() if k != "available_at"}
    with pytest.raises(ValueError):
        segment.v1_records(
            slices, undated, section=(0.0, 90.0), measure=lambda p: {}, model="m"
        )


def _fakes(tmp_path, seconds=180.0):
    calls = {"fetch": [], "isolate": []}

    def fetcher(video_id, data_dir, *, section):
        calls["fetch"].append((video_id, section))
        return _tone(Path(data_dir) / "audio" / f"{video_id}.wav", seconds)

    def isolator(src, out_dir):
        calls["isolate"].append(Path(src).name)
        out = Path(out_dir) / f"{Path(src).stem}_(vocals)_m.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(Path(src).read_bytes())
        return out

    return calls, fetcher, isolator


def test_run_segment_writes_a_one_off_measurement_file(tmp_path, video):
    (tmp_path / "audio").mkdir()
    calls, fetcher, isolator = _fakes(tmp_path)
    out = segment.run_segment(
        "vidA",
        talent="vidA_en",
        section=(600.0, 780.0),
        video=video,
        data_dir=tmp_path,
        measurements_dir=tmp_path / "measurements",
        fetch=fetcher,
        isolate=isolator,
        measure=lambda path: {"median_f0": 220.0, "f0_iqr": 5.0, "voiced_fraction": 0.9},
        model="m.ckpt",
    )
    import json

    records = json.loads(out.read_text())
    assert out.name == "vidA_en_monthly.json"
    assert [r["id"] for r in records] == ["vidA@600", "vidA@690"]
    assert calls["fetch"] == [("vidA", (600.0, 780.0))]
    assert calls["isolate"] == ["vidA@600_raw90.wav", "vidA@690_raw90.wav"]
    # The stretch's audio is kept under its own name, so it can never be
    # mistaken for the stream's ordinary 15:00-30:00 sample.
    assert not (tmp_path / "audio" / "vidA.wav").exists()
    assert (tmp_path / "audio" / "vidA@600-780.wav").is_file()


def test_run_segment_refuses_to_overwrite_an_ordinary_talent_file(tmp_path, video):
    measurements = tmp_path / "measurements"
    measurements.mkdir()
    (measurements / "kyoko_monthly.json").write_text('[{"id": "x", "month": "2026-09"}]')
    (tmp_path / "audio").mkdir()
    _, fetcher, isolator = _fakes(tmp_path)
    with pytest.raises(FileExistsError):
        segment.run_segment(
            "vidA",
            talent="kyoko",
            section=(600.0, 780.0),
            video=video,
            data_dir=tmp_path,
            measurements_dir=measurements,
            fetch=fetcher,
            isolate=isolator,
            measure=lambda p: {},
            model="m",
        )
