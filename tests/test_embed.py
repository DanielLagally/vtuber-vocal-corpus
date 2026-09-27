"""Speaker embeddings: one voice fingerprint per clip, and one per talent.

An embedding is a vector a speaker-recognition network produces for a clip,
arranged so that clips of the same voice land close together. That makes it
the strongest available measure of "these two sound alike" — and also the
exact input a zero-shot voice-cloning system conditions on. So embeddings are
computed and kept locally (gitignored); only similarity scores derived from
them are ever published.

These tests use a fake encoder: no model download, no Cover audio.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import soundfile as sf

from vvc import embed

SR = 16_000


class FakeEncoder:
    """Deterministic stand-in: a vector from the clip's spectrum."""

    name = "fake-encoder"
    dim = 8

    def __init__(self):
        self.calls = 0

    def encode(self, wave: np.ndarray, sr: int) -> np.ndarray:
        self.calls += 1
        spectrum = np.abs(np.fft.rfft(wave, 64))[: self.dim]
        return spectrum * 3.0 + 1.0  # deliberately not unit length


def _tone(path, f0, seconds=2.0):
    t = np.arange(int(seconds * SR)) / SR
    sf.write(str(path), (0.3 * np.sin(2 * np.pi * f0 * t)).astype(np.float32), SR)
    return str(path)


def _rec(vid, audio, *, passed=True, legacy=False):
    return {
        "id": vid,
        "month": "2024-01",
        "source_audio": audio,
        "qc": {"pass": passed, "reason": None if passed else "x"},
        "features": {},
        "legacy": legacy,
    }


def test_clip_embeddings_are_unit_length_and_deterministic(tmp_path):
    audio = _tone(tmp_path / "a.wav", 220.0)
    encoder = FakeEncoder()
    a = embed.embed_clip(audio, encoder)
    b = embed.embed_clip(audio, encoder)
    assert a.shape == (encoder.dim,)
    assert np.linalg.norm(a) == pytest.approx(1.0)
    assert np.array_equal(a, b)


def test_embedding_a_talent_skips_clips_already_embedded(tmp_path):
    encoder = FakeEncoder()
    records = [_rec("a", _tone(tmp_path / "a.wav", 220.0)), _rec("b", _tone(tmp_path / "b.wav", 330.0))]
    first = embed.embed_records(records, encoder)
    assert encoder.calls == 2
    second = embed.embed_records(records, encoder, existing=first)
    assert encoder.calls == 2
    assert set(second.ids) == {"a", "b"}


def test_a_clip_with_no_audio_is_a_gap_not_a_zero_vector(tmp_path):
    encoder = FakeEncoder()
    records = [_rec("a", _tone(tmp_path / "a.wav", 220.0)), _rec("gone", str(tmp_path / "gone.wav"))]
    result = embed.embed_records(records, encoder)
    assert list(result.ids) == ["a"]


def test_embeddings_round_trip_through_disk(tmp_path):
    encoder = FakeEncoder()
    records = [_rec("a", _tone(tmp_path / "a.wav", 220.0))]
    result = embed.embed_records(records, encoder)
    path = embed.write_embeddings(tmp_path / "emb", "alpha", result)
    loaded = embed.load_embeddings(path)
    assert list(loaded.ids) == ["a"]
    assert loaded.model == "fake-encoder"
    assert np.allclose(loaded.vectors, result.vectors)


def test_centroid_uses_only_qc_passing_non_legacy_clips(tmp_path):
    encoder = FakeEncoder()
    records = [
        _rec("good", _tone(tmp_path / "g.wav", 220.0)),
        _rec("failed", _tone(tmp_path / "f.wav", 900.0), passed=False),
        _rec("old", _tone(tmp_path / "o.wav", 1500.0), legacy=True),
    ]
    result = embed.embed_records(records, encoder)
    centroid = embed.talent_centroid(records, result)
    only_good = result.vectors[list(result.ids).index("good")]
    assert np.allclose(centroid, only_good)
    assert np.linalg.norm(centroid) == pytest.approx(1.0)


def test_a_talent_with_no_usable_clip_has_no_centroid(tmp_path):
    encoder = FakeEncoder()
    records = [_rec("failed", _tone(tmp_path / "f.wav", 220.0), passed=False)]
    result = embed.embed_records(records, encoder)
    assert embed.talent_centroid(records, result) is None


def test_cosine_similarity_is_one_for_identical_voices():
    v = np.array([1.0, 2.0, 3.0])
    assert embed.cosine(v, v) == pytest.approx(1.0)
    assert math.isclose(embed.cosine(v, -v), -1.0)
