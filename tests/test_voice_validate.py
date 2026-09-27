"""Are we measuring voice identity? Checks that need no human judgement.

- **Speaker identification**: hold one clip out, and ask which talent's
  typical profile it lands nearest. If a feature set captures who is speaking,
  this is right far more often than chance.
- **Between-talent share**: how much of a metric's clip-to-clip variation is
  about *who* is speaking rather than which clip it was.
- **Agreement**: do the measured-metric route and the embedding route rank
  talent pairs alike?
- **Stability**: do a talent's nearest neighbours survive splitting their clips
  into two independent halves?

Synthetic clusters with known structure only.
"""

from __future__ import annotations

import numpy as np
import pytest

from vvc import voice_validate


def _clusters(n_talents=6, n=20, spread=0.1, dim=4, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 1, (n_talents, dim))
    X = np.vstack([c + rng.normal(0, spread, (n, dim)) for c in centres])
    labels = np.repeat([f"t{i}" for i in range(n_talents)], n)
    return X, labels


class TestSpeakerId:
    def test_tight_clusters_are_identified_every_time(self):
        X, y = _clusters(spread=0.05)
        assert voice_validate.speaker_id_accuracy(X, y) == pytest.approx(1.0)

    def test_shuffled_labels_are_near_chance(self):
        X, y = _clusters(spread=0.05)
        rng = np.random.default_rng(1)
        acc = voice_validate.speaker_id_accuracy(X, rng.permutation(y))
        assert acc < 0.35

    def test_held_out_clip_does_not_vote_for_its_own_centroid(self):
        # Two clips per talent, far apart: leaving one out must make the
        # other decide, so a leaky implementation would score perfectly.
        X = np.array([[0.0], [10.0], [0.1], [10.1]])
        y = np.array(["a", "a", "b", "b"])
        assert voice_validate.speaker_id_accuracy(X, y) == pytest.approx(0.0)

    def test_cosine_mode_uses_direction_not_length(self):
        X = np.array([[1.0, 0.0], [5.0, 0.1], [0.0, 1.0], [0.1, 4.0]])
        y = np.array(["a", "a", "b", "b"])
        assert voice_validate.speaker_id_accuracy(X, y, cosine=True) == pytest.approx(1.0)


def test_between_talent_share_is_one_when_clips_never_vary():
    values = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    labels = np.array(["a", "a", "b", "b", "c", "c"])
    assert voice_validate.between_talent_share(values, labels) == pytest.approx(1.0)


def test_between_talent_share_is_zero_when_talents_do_not_differ():
    values = np.array([1.0, 2.0, 1.0, 2.0])
    labels = np.array(["a", "a", "b", "b"])
    assert voice_validate.between_talent_share(values, labels) == pytest.approx(0.0)


def test_rank_agreement_of_identical_orderings_is_one():
    a = {("x", "y"): 1.0, ("x", "z"): 2.0, ("y", "z"): 3.0}
    b = {("x", "y"): 10.0, ("x", "z"): 20.0, ("y", "z"): 30.0}
    assert voice_validate.rank_agreement(a, b) == pytest.approx(1.0)


def test_top_k_overlap():
    a = {"x": ["p", "q", "r"], "y": ["p", "q", "r"]}
    b = {"x": ["p", "q", "s"], "y": ["s", "t", "u"]}
    assert voice_validate.top_k_overlap(a, b, k=3) == pytest.approx((2 / 3 + 0) / 2)


def test_split_half_neighbours_are_stable_for_tight_clusters():
    X, y = _clusters(spread=0.05, n=20)
    assert voice_validate.split_half_stability(X, y, k=2, seed=0) > 0.9


def test_split_half_neighbours_are_unstable_for_pure_noise():
    X, y = _clusters(spread=5.0, n=6)
    assert voice_validate.split_half_stability(X, y, k=2, seed=0) < 0.7


def test_time_split_compares_early_against_late_clips():
    """Split by time, a talent whose recording changed mid-career changes
    neighbours; split at random, the change is averaged into both halves."""
    rng = np.random.default_rng(0)
    # t0 moves from near t1 (early) to near t2 (late); t1/t2/t3 stay put.
    centres = {"t1": [0.0, 0.0], "t2": [10.0, 0.0], "t3": [0.0, 10.0]}
    X, y, order = [], [], []
    for t, c in centres.items():
        for i in range(10):
            X.append(np.array(c) + rng.normal(0, 0.1, 2)); y.append(t); order.append(i)
    for i in range(10):
        c = centres["t1"] if i < 5 else centres["t2"]
        X.append(np.array(c) + [0.5, 0.5] + rng.normal(0, 0.1, 2)); y.append("t0"); order.append(i)
    X, y, order = np.array(X), np.array(y), np.array(order)
    by_time = voice_validate.split_half_stability(X, y, k=1, order=order)
    assert by_time < 1.0


def _corpus(n_talents=5, n=12, seed=0):
    """Records + embeddings where both routes see the same cluster structure."""
    from vvc.embed import Embeddings

    rng = np.random.default_rng(seed)
    records, embeddings = {}, {}
    for t in range(n_talents):
        name = f"t{t}"
        centre = rng.normal(0, 3, 3)
        rows, ids, vecs = [], [], []
        for i in range(n):
            f = centre + rng.normal(0, 0.2, 3)
            rows.append(
                {
                    "id": f"{name}-{i}",
                    "month": f"2024-{1 + i % 12:02d}",
                    "legacy": False,
                    "qc": {"pass": True, "reason": None},
                    "features": {"median_f0": 200 * 2 ** (f[0] / 12), "cpp_db": f[1], "h1h2_db": f[2]},
                }
            )
            ids.append(f"{name}-{i}")
            vecs.append(np.concatenate([f, [1.0]]))
        records[name] = rows
        embeddings[name] = Embeddings(tuple(ids), np.array(vecs), "fake")
    return records, embeddings


def test_report_scores_both_routes_on_the_same_clips():
    records, embeddings = _corpus()
    report = voice_validate.build_report(
        records,
        embeddings,
        feature_sets={"toy": ("median_f0", "cpp_db", "h1h2_db")},
    )
    assert report["n_talents"] == 5
    assert report["chance"] == pytest.approx(0.2)
    assert report["speaker_id"]["toy"] > 0.9
    assert report["speaker_id"]["toy (whitened)"] > 0.9
    assert report["speaker_id"]["embeddings"] > 0.9
    assert report["agreement"]["spearman"] > 0.5
    assert set(report["between_talent_share"]) >= {"median_f0", "cpp_db", "h1h2_db"}


def test_report_leaves_out_one_offs_and_clips_without_embeddings():
    records, embeddings = _corpus()
    records["t0"] = [{**r, "one_off": True} for r in records["t0"]]
    records["t1"].append(
        {**records["t1"][0], "id": "t1-noemb"}
    )
    report = voice_validate.build_report(
        records, embeddings, feature_sets={"toy": ("median_f0", "cpp_db", "h1h2_db")}
    )
    assert report["n_talents"] == 4
    assert report["n_clips"] == 4 * 12


def test_a_feature_set_most_clips_lack_is_reported_unavailable():
    records, embeddings = _corpus()
    report = voice_validate.build_report(
        records, embeddings, feature_sets={"future": ("median_f0", "not_measured_yet")}
    )
    assert report["speaker_id"]["future"] is None


def test_invariance_compares_the_shift_with_the_noise_floor():
    pairs = [
        ({"cpp_db": 20.0, "h1h2_db": 5.0}, {"cpp_db": 24.0, "h1h2_db": 5.1}),
        ({"cpp_db": 18.0, "h1h2_db": 6.0}, {"cpp_db": 22.0, "h1h2_db": 5.9}),
        ({"cpp_db": 21.0, "h1h2_db": float("nan")}, {"cpp_db": 25.0, "h1h2_db": 4.0}),
    ]
    floor = {"cpp_db": 2.0, "h1h2_db": 1.0}
    out = voice_validate.invariance(pairs, floor, metrics=("cpp_db", "h1h2_db"))
    assert out["n_clips"] == 3
    assert out["metrics"]["cpp_db"]["median_abs_shift"] == pytest.approx(4.0)
    assert out["metrics"]["cpp_db"]["shift_over_floor"] == pytest.approx(2.0)
    # A missing side is left out of that metric, not treated as a shift.
    assert out["metrics"]["h1h2_db"]["n"] == 2
    assert out["metrics"]["h1h2_db"]["median_abs_shift"] == pytest.approx(0.1)
    # Direction matters for a systematic bias (separation raising CPP).
    assert out["metrics"]["cpp_db"]["median_signed_shift"] == pytest.approx(4.0)


def test_report_includes_future_stream_recognition():
    records, embeddings = _corpus()
    report = voice_validate.build_report(
        records, embeddings, feature_sets={"toy": ("median_f0", "cpp_db", "h1h2_db")}
    )
    assert report["recognition"]["accuracy"] > 0.9
    assert report["recognition"]["n_test"] > 0
