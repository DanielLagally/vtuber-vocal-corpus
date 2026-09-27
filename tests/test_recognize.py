"""Recognising which talent is speaking, from an embedding.

A linear classifier on top of the general speaker embeddings — the first,
cheapest form of a recognition model. Accuracy is measured on a time split:
each talent's later clips are held out, so the number means "recognises their
future streams", not "recognises the same recording era again". The model
stays local (data/models/), like the embeddings. Synthetic clusters only.
"""

from __future__ import annotations

import numpy as np
import pytest

from vvc import recognize


def _data(n_talents=4, n=20, spread=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X, y, months = [], [], []
    for t in range(n_talents):
        centre = rng.normal(0, 1, 6)
        for i in range(n):
            X.append(centre + rng.normal(0, spread, 6))
            y.append(f"t{t}")
            months.append(f"20{20 + i // 12}-{1 + i % 12:02d}")
    return np.array(X), np.array(y), np.array(months)


def test_time_split_holds_out_each_talents_latest_clips():
    X, y, months = _data()
    train, test = recognize.time_split(y, months, test_share=0.25)
    assert not set(train) & set(test)
    for t in set(y.tolist()):
        tr = [months[i] for i in train if y[i] == t]
        te = [months[i] for i in test if y[i] == t]
        assert te and max(tr) <= min(te)


def test_separable_voices_are_recognised_on_held_out_future_clips():
    X, y, months = _data(spread=0.1)
    result = recognize.evaluate(X, y, months)
    assert result["accuracy"] > 0.95
    assert result["chance"] == pytest.approx(0.25)


def test_model_round_trips_and_predicts(tmp_path):
    X, y, months = _data()
    model = recognize.train(X, y)
    path = recognize.save(model, tmp_path / "rec.npz")
    loaded = recognize.load(path)
    assert list(loaded.predict(X[:3])) == list(model.predict(X[:3]))
    assert set(loaded.labels) == set(y.tolist())
