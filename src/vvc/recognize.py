"""Recognising which talent is speaking, from a speaker embedding.

The first, cheapest form of a recognition model: a linear classifier (a
logistic-regression "probe") on top of the frozen general speaker embeddings
from ``embed.py``. It needs no retraining of the network, trains in seconds,
and is hard to overfit — which matters with a few hundred clips per talent.
Fine-tuning the network itself is a later step, and only worth taking if this
leaves accuracy on the table.

Accuracy is measured on a **time split**: each talent's latest clips are held
out, so the number means "recognises their future streams", not "recognises
the same recording era again" — a random split would let a microphone
fingerprint pass for a voice.

Diarization (who is speaking when, in a collab) builds on the same pieces:
embed short segments, then classify each.

Like the embeddings, the model stays local (``data/models/``) and is never
published: a talent classifier is also a scoring function for a cloning
attempt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_MODEL_PATH = Path("data/models/talent-recognizer.npz")


def _unit_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms > 0, norms, 1.0)


@dataclass(frozen=True)
class Recognizer:
    labels: tuple[str, ...]
    weights: np.ndarray  # (n_labels, dim)
    bias: np.ndarray  # (n_labels,)

    def scores(self, X: np.ndarray) -> np.ndarray:
        return _unit_rows(np.asarray(X, dtype=float)) @ self.weights.T + self.bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array(self.labels)[np.argmax(self.scores(X), axis=1)]


def train(X: np.ndarray, y: np.ndarray, *, c: float = 10.0) -> Recognizer:
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=c, max_iter=2000)
    clf.fit(_unit_rows(np.asarray(X, dtype=float)), np.asarray(y))
    weights, bias = clf.coef_, clf.intercept_
    if weights.shape[0] == 1:  # binary case: sklearn stores one row
        weights = np.vstack([-weights[0], weights[0]])
        bias = np.array([-bias[0], bias[0]])
    return Recognizer(tuple(str(c) for c in clf.classes_), weights, bias)


def time_split(
    labels: np.ndarray, order: np.ndarray, *, test_share: float = 0.2
) -> tuple[list[int], list[int]]:
    """Per talent, the latest ``test_share`` of clips (by ``order``) are test.

    A talent with a single clip contributes it to training only.
    """
    labels, order = np.asarray(labels), np.asarray(order)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for talent in sorted(set(labels.tolist())):
        idx = np.flatnonzero(labels == talent)
        idx = idx[np.argsort(order[idx], kind="stable")]
        n_test = int(round(test_share * idx.size)) if idx.size > 1 else 0
        n_test = max(1, n_test) if idx.size > 1 else 0
        train_idx.extend(idx[: idx.size - n_test].tolist())
        test_idx.extend(idx[idx.size - n_test :].tolist())
    return train_idx, test_idx


def evaluate(X: np.ndarray, y: np.ndarray, order: np.ndarray, *, test_share: float = 0.2) -> dict:
    """Train on each talent's earlier clips, score on their later ones."""
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    train_idx, test_idx = time_split(y, order, test_share=test_share)
    model = train(X[train_idx], y[train_idx])
    predicted = model.predict(X[test_idx])
    talents = sorted(set(y.tolist()))
    per_talent = {
        t: float(np.mean(predicted[y[test_idx] == t] == t))
        for t in talents
        if np.any(y[test_idx] == t)
    }
    return {
        "accuracy": float(np.mean(predicted == y[test_idx])),
        "chance": 1.0 / len(talents),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "per_talent": per_talent,
    }


def save(model: Recognizer, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, labels=np.array(model.labels, dtype=str), weights=model.weights, bias=model.bias)
    return path


def load(path: Path | str) -> Recognizer:
    with np.load(path, allow_pickle=False) as data:
        return Recognizer(
            tuple(str(x) for x in data["labels"]), np.array(data["weights"]), np.array(data["bias"])
        )
