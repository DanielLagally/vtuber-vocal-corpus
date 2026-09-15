"""Building the v2 corpus from local audio, without disturbing v1.

v1's measurement files are frozen: they still feed the published site and they
are the baseline every v2 comparison is made against. v2 is a new corpus in a
new place, built by re-measuring the audio that is still on disk and replaying
each record's stored window so the two are a like-for-like comparison rather
than a fresh sample.

Roughly a quarter of measured clips no longer have local audio. Those are
marked and excluded from corrected analyses — never silently mixed in with
re-measured records, and never dropped, because a clip we cannot re-measure is
still a clip the corpus sampled.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vvc import corpus


@pytest.fixture
def v1_records():
    return [
        {
            "id": "clip1",
            "month": "2021-03",
            "score": 70.0,
            "window": {"start_s": 15.0, "end_s": 105.0},
            "features": {"median_f0": 300.0},
            "qc": {"pass": True, "reason": None},
            "model": "old_model.ckpt",
            "tracker": "praat_ac",
        },
        {
            "id": "gone",
            "month": "2021-04",
            "score": 40.0,
            "window": {"start_s": 15.0, "end_s": 105.0},
            "features": {"median_f0": 280.0},
            "qc": {"pass": True, "reason": None},
            "model": "old_model.ckpt",
            "tracker": "praat_ac",
        },
    ]


@pytest.fixture
def cache():
    return {
        "clip1": {
            "id": "clip1",
            "title": "zatsudan",
            "type": "stream",
            "topic_id": "talk",
            "published_at": "2021-03-05T15:01:09.000Z",
            "available_at": "2021-03-05T14:55:00.000Z",
            "duration": 3600,
        }
    }


def _fake_extract(path, config):
    return {
        "median_f0": 311.0,
        "f0_q25": 280.0,
        "f0_q50": 311.0,
        "f0_q75": 340.0,
        "f0_iqr_semitones": 3.4,
        "voiced_fraction": 0.55,
        "voiced_frames": 2000,
        "total_frames": 4000,
        "fraction_below_160hz": 0.01,
        "brightness_hz": 2400.0,
        "f1_hz": 700.0,
        "formant_voiced_frames": 1800,
        "formant_total_frames": 3000,
    }


def _build(v1_records, cache, **kwargs):
    kwargs.setdefault("resolve_audio", lambda vid: Path(f"/fake/{vid}.wav") if vid == "clip1" else None)
    kwargs.setdefault("extract", _fake_extract)
    kwargs.setdefault("background_ratio", lambda vid: -30.0)
    return corpus.build_records(v1_records, cache=cache, release="test-release", **kwargs)


class TestBuild:
    def test_remeasures_clips_whose_audio_is_local(self, v1_records, cache):
        built = _build(v1_records, cache)
        row = next(r for r in built if r["id"] == "clip1")
        assert row["features"]["median_f0"] == 311.0
        assert row["legacy"] is False

    def test_marks_clips_whose_audio_is_gone_instead_of_dropping_them(self, v1_records, cache):
        built = _build(v1_records, cache)
        row = next(r for r in built if r["id"] == "gone")
        assert row["legacy"] is True
        assert row["legacy_reason"] == "audio_unavailable"

    def test_a_legacy_record_keeps_v1_features_rather_than_inventing_any(
        self, v1_records, cache
    ):
        built = _build(v1_records, cache)
        row = next(r for r in built if r["id"] == "gone")
        assert row["features"]["median_f0"] == 280.0
        assert row["tracker"] == "praat_ac"

    def test_every_record_survives_the_build(self, v1_records, cache):
        assert len(_build(v1_records, cache)) == len(v1_records)


class TestProvenance:
    def test_records_which_audio_file_was_measured(self, v1_records, cache):
        built = _build(v1_records, cache)
        row = next(r for r in built if r["id"] == "clip1")
        assert row["source_audio"].endswith("clip1.wav")

    def test_records_the_release_identifier(self, v1_records, cache):
        built = _build(v1_records, cache)
        assert all(r["corpus_release"] == "test-release" for r in built)

    def test_records_when_it_was_measured(self, v1_records, cache):
        built = _build(v1_records, cache)
        row = next(r for r in built if r["id"] == "clip1")
        assert row["measured_at"]

    def test_records_the_background_ratio_and_the_separation_decision(
        self, v1_records, cache
    ):
        built = _build(v1_records, cache)
        row = next(r for r in built if r["id"] == "clip1")
        assert row["background_ratio_db"] == -30.0
        assert row["separation_applied"] is True  # v1 separated unconditionally

    def test_preserves_the_v1_window_so_the_comparison_is_like_for_like(
        self, v1_records, cache
    ):
        built = _build(v1_records, cache)
        row = next(r for r in built if r["id"] == "clip1")
        assert row["window"] == {"start_s": 15.0, "end_s": 105.0}


class TestMetadata:
    def test_attaches_stream_type_and_exact_date(self, v1_records, cache):
        built = _build(v1_records, cache)
        row = next(r for r in built if r["id"] == "clip1")
        assert row["topic_id"] == "talk"
        assert row["available_at"].startswith("2021-03-05")

    def test_an_unresolvable_clip_is_marked_not_guessed(self, v1_records, cache):
        built = _build(v1_records, cache)
        row = next(r for r in built if r["id"] == "gone")
        assert row["metadata_resolved"] is False
        assert row["topic_id"] is None


class TestQualityGate:
    def test_applies_the_v2_gate_to_remeasured_records(self, v1_records, cache):
        built = _build(v1_records, cache)
        row = next(r for r in built if r["id"] == "clip1")
        assert row["qc"]["pass"] is True

    def test_a_contaminated_clip_fails(self, v1_records, cache):
        def contaminated(path, config):
            row = _fake_extract(path, config)
            row.update({"median_f0": 130.0, "fraction_below_160hz": 0.7})
            return row

        built = _build(v1_records, cache, extract=contaminated)
        row = next(r for r in built if r["id"] == "clip1")
        assert row["qc"]["pass"] is False
        assert row["qc"]["reason"] == "contaminated"


class TestExtractManyReportsFailures:
    """A build once reported success while measuring nothing at all.

    Every extraction had failed — the GPU was full — and each failure was
    swallowed individually, so the run produced an empty corpus, called every
    clip "legacy", and exited zero. Individually tolerating a bad clip is
    right; silently tolerating *all* of them is not.
    """

    def test_reports_how_many_clips_failed(self, tmp_path):
        def boom(path, config):
            raise RuntimeError("no GPU memory")

        result = corpus.extract_many([tmp_path / "a.wav"], extract=boom)
        assert result.failed == 1
        assert result.features == {}

    def test_reports_successes_alongside_failures(self, tmp_path):
        def half(path, config):
            if path.name == "bad.wav":
                raise RuntimeError("nope")
            return {"median_f0": 300.0}

        result = corpus.extract_many(
            [tmp_path / "good.wav", tmp_path / "bad.wav"], extract=half
        )
        assert result.succeeded == 1
        assert result.failed == 1

    def test_a_total_failure_is_distinguishable_from_having_no_work(self, tmp_path):
        def boom(path, config):
            raise RuntimeError("no GPU memory")

        total = corpus.extract_many([tmp_path / "a.wav"], extract=boom)
        empty = corpus.extract_many([], extract=boom)
        assert total.all_failed is True
        assert empty.all_failed is False


class TestIsolation:
    def test_does_not_mutate_the_v1_records(self, v1_records, cache):
        before = json.dumps(v1_records, sort_keys=True)
        _build(v1_records, cache)
        assert json.dumps(v1_records, sort_keys=True) == before

    def test_writing_targets_a_separate_directory(self, tmp_path, v1_records, cache):
        out = tmp_path / "v2"
        corpus.write_talent(out, "alpha", _build(v1_records, cache))
        assert (out / "alpha.json").is_file()
        assert not (out / "alpha_monthly.json").exists()
