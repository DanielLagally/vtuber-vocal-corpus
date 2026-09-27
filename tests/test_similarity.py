"""Which talents sound most alike, from measurements alone.

Two independent routes, kept separate so each can check the other:

- **Measured metrics.** Talents are compared on their typical values, with
  each difference measured against how much that metric varies between one
  talent's own clips (the pooled within-talent covariance). So a metric that
  jumps around from clip to clip counts for little, a stable one counts for a
  lot, and several metrics reading the same quality cannot vote several times.
- **Embeddings.** Cosine similarity of talent centroids.

Scores are reported as an absolute voice match (0% a typical unrelated pair,
100% as alike as a talent is to themselves) and as "closer than X% of all
talent pairs". Nobody hand-picks which pairs "should" match.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vvc import similarity


def _clips(talent_means: dict[str, dict[str, float]], noise: dict[str, float], n=30, seed=0):
    rng = np.random.default_rng(seed)
    clips = []
    for talent, means in talent_means.items():
        for i in range(n):
            clips.append(
                (talent, {m: v + rng.normal(0, noise[m]) for m, v in means.items()})
            )
    return clips


class TestMetricDistance:
    def test_a_difference_in_a_noisy_metric_counts_for_less(self):
        noise = {"stable": 1.0, "noisy": 10.0}
        means = {
            "A": {"stable": 0.0, "noisy": 0.0},
            "B": {"stable": 5.0, "noisy": 0.0},  # differs on the stable metric
            "C": {"stable": 0.0, "noisy": 5.0},  # same size gap on the noisy one
        }
        model = similarity.fit_metric_space(_clips(means, noise), metrics=("stable", "noisy"))
        typicals = {t: m for t, m in means.items()}
        d_ab = model.distance(typicals["A"], typicals["B"])
        d_ac = model.distance(typicals["A"], typicals["C"])
        assert d_ac < d_ab / 3

    def test_duplicated_metrics_do_not_vote_twice(self):
        noise = {"x": 1.0, "y": 1.0}
        means = {"A": {"x": 0.0, "y": 0.0}, "B": {"x": 3.0, "y": 0.0}}
        clips = _clips(means, noise)
        base = similarity.fit_metric_space(clips, metrics=("x", "y"))
        # A near-copy of x: same quality, read twice.
        rng = np.random.default_rng(5)
        dup_clips = [(t, {**f, "x2": f["x"] + rng.normal(0, 0.05)}) for t, f in clips]
        dup = similarity.fit_metric_space(dup_clips, metrics=("x", "y", "x2"))
        a = {"x": 0.0, "y": 0.0, "x2": 0.0}
        b = {"x": 3.0, "y": 0.0, "x2": 3.0}
        assert dup.distance(a, b) == pytest.approx(base.distance(a, b), rel=0.25)

    def test_identical_talents_are_at_distance_zero(self):
        noise = {"x": 1.0}
        model = similarity.fit_metric_space(_clips({"A": {"x": 1.0}}, noise), metrics=("x",))
        assert model.distance({"x": 2.0}, {"x": 2.0}) == pytest.approx(0.0)

    def test_a_metric_missing_on_either_side_is_skipped_not_zeroed(self):
        noise = {"x": 1.0, "y": 1.0}
        means = {"A": {"x": 0.0, "y": 0.0}, "B": {"x": 2.0, "y": 9.0}}
        model = similarity.fit_metric_space(_clips(means, noise), metrics=("x", "y"))
        only_x = model.distance({"x": 0.0, "y": math.nan}, {"x": 2.0, "y": 9.0})
        x_model = similarity.fit_metric_space(_clips(means, noise), metrics=("x",))
        assert only_x == pytest.approx(x_model.distance({"x": 0.0}, {"x": 2.0}), rel=0.05)

    def test_nothing_in_common_is_no_distance(self):
        model = similarity.fit_metric_space(
            _clips({"A": {"x": 0.0, "y": 0.0}}, {"x": 1.0, "y": 1.0}), metrics=("x", "y")
        )
        assert math.isnan(model.distance({"x": 1.0, "y": math.nan}, {"x": math.nan, "y": 1.0}))

    def test_closest_metrics_explain_a_match(self):
        noise = {"x": 1.0, "y": 1.0, "z": 1.0}
        means = {"A": {"x": 0.0, "y": 0.0, "z": 0.0}}
        model = similarity.fit_metric_space(_clips(means, noise), metrics=("x", "y", "z"))
        close = model.closest_metrics({"x": 0.0, "y": 4.0, "z": 0.1}, {"x": 0.0, "y": 0.0, "z": 0.0}, k=2)
        assert close == ["x", "z"]


class TestNeighbours:
    def _distances(self):
        return {
            ("A", "B"): 1.0, ("A", "C"): 5.0, ("A", "D"): 3.0,
            ("B", "C"): 4.0, ("B", "D"): 2.0, ("C", "D"): 6.0,
        }

    def test_neighbours_are_ranked_closest_first_and_exclude_self(self):
        dist = self._distances()
        out = similarity.neighbours(["A", "B", "C", "D"], lambda a, b: dist[tuple(sorted((a, b)))], k=3)
        assert [n["name"] for n in out["A"]] == ["B", "D", "C"]

    def test_score_is_the_share_of_pairs_further_apart(self):
        dist = self._distances()
        out = similarity.neighbours(["A", "B", "C", "D"], lambda a, b: dist[tuple(sorted((a, b)))], k=3)
        # A-B is the closest of 6 pairs: 5 of 6 pairs are further apart.
        assert out["A"][0]["closer_than_pct"] == pytest.approx(100 * 5 / 6)

    def test_a_one_off_is_listed_and_tagged_but_does_not_shape_the_scores(self):
        """A one-off appears in other talents' lists, tagged, so a match is
        visible from both sides; it stays out of the pair distribution, so
        nobody else's scores move because of it."""
        dist = self._distances()
        score = lambda a, b: dist[tuple(sorted((a, b)))]
        with_b = similarity.neighbours(["A", "B", "C", "D"], score, k=3, one_off={"B"})
        a_row = next(n for n in with_b["A"] if n["name"] == "B")
        assert a_row["one_off"] is True
        assert [n["name"] for n in with_b["B"]][0] == "A"
        without_b = similarity.neighbours(["A", "C", "D"], score, k=3)
        for row in without_b["A"]:
            same = next(n for n in with_b["A"] if n["name"] == row["name"])
            assert same["closer_than_pct"] == row["closer_than_pct"]
            assert same["one_off"] is False

    def test_similarity_can_be_higher_is_closer(self):
        sims = {("A", "B"): 0.9, ("A", "C"): 0.1, ("B", "C"): 0.5}
        out = similarity.neighbours(
            ["A", "B", "C"], lambda a, b: sims[tuple(sorted((a, b)))], k=2, higher_is_closer=True
        )
        assert [n["name"] for n in out["A"]] == ["B", "C"]

    def test_an_undefined_distance_is_left_out(self):
        out = similarity.neighbours(
            ["A", "B", "C"], lambda a, b: math.nan if "C" in (a, b) else 1.0, k=2
        )
        assert [n["name"] for n in out["A"]] == ["B"]
        assert out["C"] == []


def test_a_match_is_explained_by_unusual_closeness_not_by_a_noisy_metric():
    """A metric that wanders a lot within one talent makes every pair look
    close in within-talent units. The explanation should name metrics where
    this pair is closer than talents usually are."""
    noise = {"noisy": 10.0, "stable": 1.0}
    means = {"A": {"noisy": 0.0, "stable": 0.0}}
    space = similarity.fit_metric_space(_clips(means, noise), metrics=("noisy", "stable"))
    corpus = [
        {"noisy": 0.0, "stable": 0.0},
        {"noisy": 1.0, "stable": 10.0},
        {"noisy": 2.0, "stable": 20.0},
        {"noisy": 3.0, "stable": 30.0},
    ]
    spread = similarity.between_spread(space, corpus)
    a, b = {"noisy": 0.0, "stable": 0.0}, {"noisy": 2.0, "stable": 1.0}
    # In within-talent units the noisy metric looks closest (0.2 vs 1.0) ...
    assert space.closest_metrics(a, b, k=1) == ["noisy"]
    # ... but against how talents differ, the stable one is the real match.
    assert space.closest_metrics(a, b, k=1, spread=spread) == ["stable"]


def test_explanations_can_be_limited_to_audible_metrics():
    noise = {"x": 1.0, "y": 1.0, "z": 1.0}
    space = similarity.fit_metric_space(
        _clips({"A": {"x": 0.0, "y": 0.0, "z": 0.0}}, noise), metrics=("x", "y", "z")
    )
    a, b = {"x": 0.0, "y": 1.0, "z": 2.0}, {"x": 0.0, "y": 0.0, "z": 0.0}
    assert space.closest_metrics(a, b, k=2, among=("y", "z")) == ["y", "z"]



class TestVoiceMatch:
    """An absolute score: 0% is a typical unrelated pair, 100% is as alike as a
    talent is to themselves across their own clips. Unlike a rank, it does not
    saturate: the top matches of every talent are not all near 100%."""

    def test_similarity_scale(self):
        assert similarity.match_percent(0.0, zero=0.0, full=1.0) == pytest.approx(0.0)
        assert similarity.match_percent(1.0, zero=0.0, full=1.0) == pytest.approx(100.0)
        assert similarity.match_percent(0.45, zero=-0.05, full=0.95) == pytest.approx(50.0)

    def test_distance_scale_runs_the_other_way(self):
        # Smaller distance is closer: zero anchors at the typical pair distance.
        assert similarity.match_percent(4.0, zero=4.0, full=1.0) == pytest.approx(0.0)
        assert similarity.match_percent(2.5, zero=4.0, full=1.0) == pytest.approx(50.0)

    def test_it_is_clipped_to_the_scale(self):
        assert similarity.match_percent(-0.3, zero=0.0, full=1.0) == 0.0
        assert similarity.match_percent(1.2, zero=0.0, full=1.0) == 100.0

    def test_an_undefined_scale_gives_no_score(self):
        assert math.isnan(similarity.match_percent(0.5, zero=1.0, full=1.0))

    def test_neighbours_carry_the_match_when_given_a_scale(self):
        sims = {("A", "B"): 0.9, ("A", "C"): 0.1, ("B", "C"): 0.5}
        out = similarity.neighbours(
            ["A", "B", "C"],
            lambda a, b: sims[tuple(sorted((a, b)))],
            k=2,
            higher_is_closer=True,
            scale=(0.0, 1.0),
        )
        assert out["A"][0]["match_pct"] == pytest.approx(90.0)
