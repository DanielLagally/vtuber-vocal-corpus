"""Are we measuring voice identity? Checks that need no human judgement.

Every check here is decided by the measurements themselves, so nobody's idea
of who "should" sound alike leaks into the answer:

- **Speaker identification** — hold one clip out and ask which talent's
  typical profile it lands nearest. A feature set that captures who is
  speaking gets this right far more often than chance; one that captures the
  microphone or the game being played does not. This is the headline number:
  adding a metric that measures voice raises it.
- **Between-talent share** — per metric, the fraction of clip-to-clip variance
  explained by which talent is speaking (eta squared).
- **Agreement** — rank correlation between the measured-metric route and the
  embedding route over every talent pair, and how many of each talent's
  top neighbours the two share.
- **Stability** — split each talent's clips into two random halves; do the
  halves name the same neighbours? A neighbour list that does not replicate on
  independent data is noise, whatever it looks like.

See reference/statistics.md.
"""

from __future__ import annotations

import numpy as np


def _unit_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms > 0, norms, 1.0)


def speaker_id_accuracy(X: np.ndarray, labels: np.ndarray, *, cosine: bool = False) -> float:
    """Leave-one-out nearest-centroid accuracy.

    The held-out clip is removed from its own talent's centroid before
    comparing — otherwise it votes for itself. A talent left with no other clip
    cannot be predicted for that clip. ``cosine`` compares directions (for
    embeddings); otherwise Euclidean, so the caller chooses the scaling.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    if cosine:
        X = _unit_rows(X)
    talents = sorted(set(labels.tolist()))
    index = {t: i for i, t in enumerate(talents)}
    owner = np.array([index[t] for t in labels])
    sums = np.zeros((len(talents), X.shape[1]))
    np.add.at(sums, owner, X)
    counts = np.bincount(owner, minlength=len(talents)).astype(float)
    hits = 0
    for i in range(X.shape[0]):
        own = owner[i]
        cent_sums = sums.copy()
        cent_counts = counts.copy()
        cent_sums[own] -= X[i]
        cent_counts[own] -= 1
        valid = cent_counts > 0
        centroids = cent_sums[valid] / cent_counts[valid, None]
        ids = np.flatnonzero(valid)
        if cosine:
            scores = _unit_rows(centroids) @ X[i]
            predicted = ids[int(np.argmax(scores))]
        else:
            predicted = ids[int(np.argmin(np.sum((centroids - X[i]) ** 2, axis=1)))]
        hits += int(predicted == own)
    return hits / X.shape[0] if X.shape[0] else float("nan")


def between_talent_share(values: np.ndarray, labels: np.ndarray) -> float:
    """Eta squared: between-talent sum of squares over total."""
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    keep = np.isfinite(values)
    values, labels = values[keep], labels[keep]
    if values.size < 2:
        return float("nan")
    grand = values.mean()
    total = float(np.sum((values - grand) ** 2))
    if total == 0:
        return float("nan")
    between = sum(
        float(np.sum(labels == t)) * (float(values[labels == t].mean()) - grand) ** 2
        for t in set(labels.tolist())
    )
    return between / total


def rank_agreement(a: dict, b: dict) -> float:
    """Spearman correlation between two scorings of the same pairs."""
    from scipy.stats import spearmanr

    shared = [key for key in a if key in b]
    if len(shared) < 3:
        return float("nan")
    rho = spearmanr([a[k] for k in shared], [b[k] for k in shared]).statistic
    return float(rho)


def top_k_overlap(a: dict[str, list[str]], b: dict[str, list[str]], *, k: int) -> float:
    """Mean share of each talent's top-k neighbours that both lists name."""
    shared = [name for name in a if name in b]
    if not shared:
        return float("nan")
    return float(np.mean([len(set(a[n][:k]) & set(b[n][:k])) / k for n in shared]))


def _centroid_neighbours(X: np.ndarray, labels: np.ndarray, k: int, cosine: bool):
    talents = sorted(set(labels.tolist()))
    C = np.vstack([X[labels == t].mean(axis=0) for t in talents])
    if cosine:
        C = _unit_rows(C)
        D = -(C @ C.T)
    else:
        D = np.sum((C[:, None, :] - C[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(D, np.inf)
    return {t: [talents[j] for j in np.argsort(D[i])[:k]] for i, t in enumerate(talents)}


def split_half_stability(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    k: int = 5,
    seed: int = 0,
    cosine: bool = False,
    order: np.ndarray | None = None,
) -> float:
    """Top-k neighbour overlap between two halves of every talent's clips.

    Halves are random, or — given ``order`` (e.g. month) — each talent's
    earlier clips against their later ones, which asks whether neighbours
    survive a change of recording era rather than only resampling. Talents
    with fewer than two clips cannot be split and are left out.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    if cosine:
        X = _unit_rows(X)
    rng = np.random.default_rng(seed)
    half_a, half_b = [], []
    for t in sorted(set(labels.tolist())):
        idx = np.flatnonzero(labels == t)
        if idx.size < 2:
            continue
        if order is None:
            idx = rng.permutation(idx)
        else:
            idx = idx[np.argsort(np.asarray(order)[idx], kind="stable")]
        half_a.extend(idx[: idx.size // 2])
        half_b.extend(idx[idx.size // 2 :])
    a = _centroid_neighbours(X[half_a], labels[half_a], k, cosine)
    b = _centroid_neighbours(X[half_b], labels[half_b], k, cosine)
    return top_k_overlap(a, b, k=k)


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------

#: Feature sets compared in the report. The first is the set the site had
#: before the voice-source work — the baseline every addition is judged by.
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "baseline (pitch, periodicity, brightness, F1-F4)": (
        "median_f0", "f0_iqr_semitones", "dynamism_semitones", "jitter_local",
        "shimmer_local", "hnr_db", "brightness_hz", "f1_hz", "f2_hz", "f3_hz", "f4_hz",
    ),
    "voice metrics (with source, timbre, tempo)": None,  # filled from similarity
}

#: A feature set is scored only if this share of usable clips carries all of
#: its metrics; otherwise it is reported unavailable rather than scored on an
#: unrepresentative remainder.
MIN_COVERAGE = 0.9
STABILITY_SEEDS = 5
NEIGHBOURS_K = 5


def _feature_sets() -> dict[str, tuple[str, ...]]:
    from .similarity import VOICE_METRICS

    return {
        name: (metrics if metrics is not None else VOICE_METRICS)
        for name, metrics in FEATURE_SETS.items()
    }


def _matrix(rows: list[dict], metrics: tuple[str, ...]) -> np.ndarray:
    from .similarity import voice_value

    return np.array([[voice_value(m, r["features"].get(m)) for m in metrics] for r in rows])


def _whiten(X: np.ndarray, labels: np.ndarray, metrics: tuple[str, ...]) -> np.ndarray:
    from .similarity import fit_metric_space

    # X is already on the distance scale (voice_value applied). Key its columns
    # by position so fit_metric_space, which applies voice_value by metric
    # name, cannot scale them a second time.
    keys = tuple(f"m{i}" for i in range(len(metrics)))
    clips = [(labels[i], dict(zip(keys, X[i]))) for i in range(X.shape[0])]
    space = fit_metric_space(clips, metrics=keys)
    chol = np.linalg.cholesky(space.covariance)
    return np.linalg.solve(chol, X.T).T


def _stability(X, labels, months, *, cosine: bool) -> dict:
    return {
        "random_halves": float(
            np.mean(
                [
                    split_half_stability(X, labels, k=NEIGHBOURS_K, seed=s, cosine=cosine)
                    for s in range(STABILITY_SEEDS)
                ]
            )
        ),
        "early_vs_late": split_half_stability(
            X, labels, k=NEIGHBOURS_K, cosine=cosine, order=months
        ),
    }


def build_report(
    records_by_talent: dict[str, list[dict]],
    embeddings_by_talent: dict,
    *,
    feature_sets: dict[str, tuple[str, ...]] | None = None,
) -> dict:
    """Every measurement-only check, on one common set of clips.

    Clips count when usable (re-measured, QC-passing), not part of a one-off,
    and embedded — so every route is scored on exactly the same clips.
    """
    from .embed import usable

    feature_sets = feature_sets if feature_sets is not None else _feature_sets()
    rows, labels, vectors = [], [], []
    for talent, records in sorted(records_by_talent.items()):
        emb = embeddings_by_talent.get(talent)
        if emb is None:
            continue
        for record in records:
            if record.get("one_off") or not usable(record):
                continue
            vector = emb.get(str(record.get("id", "")))
            if vector is None:
                continue
            rows.append(record)
            labels.append(talent)
            vectors.append(vector)

    available: dict[str, tuple[str, ...]] = {}
    for name, metrics in feature_sets.items():
        X = _matrix(rows, metrics) if rows else np.zeros((0, len(metrics)))
        coverage = float(np.mean(np.all(np.isfinite(X), axis=1))) if rows else 0.0
        if coverage >= MIN_COVERAGE:
            available[name] = metrics

    keep = np.ones(len(rows), bool)
    for metrics in available.values():
        keep &= np.all(np.isfinite(_matrix(rows, metrics)), axis=1)
    rows = [r for r, k in zip(rows, keep) if k]
    y = np.array([t for t, k in zip(labels, keep) if k])
    E = np.array([v for v, k in zip(vectors, keep) if k])
    months = np.array([str(r.get("month") or "") for r in rows])
    talents = sorted(set(y.tolist()))

    speaker_id: dict[str, float | None] = {name: None for name in feature_sets}
    whitened: dict[str, np.ndarray] = {}
    for name, metrics in available.items():
        X = _matrix(rows, metrics)
        Z = (X - X.mean(axis=0)) / np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
        speaker_id[name] = speaker_id_accuracy(Z, y)
        W = _whiten(X, y, metrics)
        whitened[name] = W
        speaker_id[f"{name} (whitened)"] = speaker_id_accuracy(W, y)
    speaker_id["embeddings"] = speaker_id_accuracy(E, y, cosine=True) if len(E) else None

    all_metrics = sorted({m for metrics in feature_sets.values() for m in metrics})
    shares = {}
    for metric in all_metrics:
        values = _matrix(rows, (metric,))[:, 0] if rows else np.zeros(0)
        share = between_talent_share(values, y) if values.size else float("nan")
        if share == share:
            shares[metric] = share

    report: dict = {
        "n_clips": len(rows),
        "n_talents": len(talents),
        "chance": 1.0 / len(talents) if talents else float("nan"),
        "speaker_id": speaker_id,
        "between_talent_share": dict(sorted(shares.items(), key=lambda kv: -kv[1])),
        "feature_sets": {name: list(m) for name, m in feature_sets.items()},
    }
    if len(E) and len(talents) > 1:
        from .recognize import evaluate

        result = evaluate(E, y, months)
        report["recognition"] = {k: v for k, v in result.items() if k != "per_talent"}
        report["recognition"]["weakest"] = sorted(result["per_talent"].items(), key=lambda kv: kv[1])[:5]
    if not whitened or not len(E):
        return report

    # The richest available set stands for the measured route.
    route = list(whitened)[-1]
    W = whitened[route]
    Cm = np.vstack([W[y == t].mean(axis=0) for t in talents])
    Ce = _unit_rows(np.vstack([_unit_rows(E[y == t]).mean(axis=0) for t in talents]))
    metric_dist, embed_dist = {}, {}
    for i, a in enumerate(talents):
        for j in range(i + 1, len(talents)):
            b = talents[j]
            metric_dist[(a, b)] = float(np.linalg.norm(Cm[i] - Cm[j]))
            embed_dist[(a, b)] = float(1.0 - Ce[i] @ Ce[j])
    k = min(NEIGHBOURS_K, len(talents) - 1)
    near_m = _centroid_neighbours(W, y, k, cosine=False)
    near_e = _centroid_neighbours(E, y, k, cosine=True)
    report["measured_route"] = route
    report["agreement"] = {
        "spearman": rank_agreement(metric_dist, embed_dist),
        f"top{k}_overlap": top_k_overlap(near_m, near_e, k=k),
    }
    report["stability"] = {
        "measured": _stability(W, y, months, cosine=False),
        "embeddings": _stability(E, y, months, cosine=True),
    }
    report["neighbours"] = {"measured": near_m, "embeddings": near_e}
    return report


def render_markdown(report: dict) -> str:
    """The report as a readable page (numbers recomputed on every run)."""

    def pct(x):
        return "—" if x is None or x != x else f"{100 * x:.1f}%"

    lines = [
        "# Voice-identity validation",
        "",
        f"{report['n_clips']} clips, {report['n_talents']} talents "
        f"(chance {pct(report['chance'])}). Every route is scored on the same clips.",
        "",
        "## Speaker identification (leave-one-out, nearest talent centroid)",
        "",
        "| Feature set | Accuracy |",
        "|---|---|",
    ]
    for name, acc in report["speaker_id"].items():
        lines.append(f"| {name} | {pct(acc) if acc is not None else 'unavailable'} |")
    if "recognition" in report:
        rec = report["recognition"]
        lines += [
            "",
            "## Recognition of future streams (linear probe on embeddings)",
            "",
            f"Trained on each talent's earlier clips, scored on their latest "
            f"{rec['n_test']}: **{pct(rec['accuracy'])}** (chance {pct(rec['chance'])}). "
            "Hardest talents: "
            + ", ".join(f"{t} {pct(a)}" for t, a in rec["weakest"])
            + ".",
        ]
    lines += ["", "## Between-talent share of clip variance (eta²)", "", "| Metric | Share |", "|---|---|"]
    for metric, share in report["between_talent_share"].items():
        lines.append(f"| {metric} | {share:.2f} |")
    if "agreement" in report:
        lines += [
            "",
            f"## Agreement: measured route ({report['measured_route']}, whitened) vs embeddings",
            "",
        ]
        for key, value in report["agreement"].items():
            lines.append(f"- {key}: {value:.2f}")
        lines += ["", "## Neighbour stability (top-5 overlap between halves)", ""]
        for route, values in report["stability"].items():
            lines.append(
                f"- {route}: random halves {values['random_halves']:.2f}, "
                f"early vs late {values['early_vs_late']:.2f}"
            )
    if "invariance" in report:
        lines += [
            "",
            "## Invariance to separation (same clip, raw window vs separated stem)",
            "",
            "| Metric | Median shift | Noise floor | Shift / floor |",
            "|---|---|---|---|",
        ]
        for metric, row in report["invariance"]["metrics"].items():
            ratio = row.get("shift_over_floor")
            lines.append(
                f"| {metric} | {row['median_abs_shift']:.3g} | "
                f"{row['noise_floor'] if row['noise_floor'] is None else format(row['noise_floor'], '.3g')} | "
                f"{'—' if ratio is None else format(ratio, '.2f')} |"
            )
    return "\n".join(lines) + "\n"


def invariance(
    pairs: list[tuple[dict, dict]],
    noise_floor: dict[str, float | None],
    *,
    metrics: tuple[str, ...],
) -> dict:
    """How far separation moves each metric, against its noise floor.

    ``pairs`` are (measured directly, measured on the separated stem) for the
    same clips. A shift well under the floor means the metric survives
    separation; a shift at or above it means a separated and a direct clip are
    not comparable on that metric. The signed shift shows a systematic bias
    (a denoiser raising periodicity measures by construction).
    """
    out: dict = {"n_clips": len(pairs), "metrics": {}}
    for metric in metrics:
        diffs = []
        for direct, separated in pairs:
            a, b = direct.get(metric), separated.get(metric)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a == a and b == b:
                diffs.append(float(b) - float(a))
        if not diffs:
            continue
        shift = float(np.median(np.abs(diffs)))
        floor = noise_floor.get(metric)
        out["metrics"][metric] = {
            "n": len(diffs),
            "median_abs_shift": shift,
            "median_signed_shift": float(np.median(diffs)),
            "noise_floor": floor,
            "shift_over_floor": (shift / floor) if floor else None,
        }
    return out
