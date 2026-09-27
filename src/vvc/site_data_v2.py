"""Aggregate export for the v2 site (``docs/``, replacing the placeholder
that pointed at ``docs/v1/``).

Pure data, same spirit as v1's ``site_data.py``: JSON-serializable Python
values only, so the frontend never re-derives a statistic and this module is
testable without a browser. It deliberately does not reuse v1's aggregation
(``series.py``'s mean-at-month/quarter, ``median_low``-at-year) — see
``series_v2.py`` for why a v2 site needs its own month-first, ordinary-median,
bootstrap-CI aggregation.

Metric families mirror reference/statistics.md's "Presenting fewer things
than there are columns" section exactly, and each family's ``robust`` flag
mirrors reference/measurement.md's per-feature validity table. Getting these
two documents and this module out of sync would let the site claim more than
the methodology supports, so a family/metric added here should be added there
first.

Talent branch/generation resolution (roster matching, name aliases, the
graduated-talent lookup, multi-generation overrides) is NOT duplicated here —
it is imported from ``site_data`` and reused as-is. That table is hand-curated
real-world fact, and two independently-maintained copies would drift exactly
the way reference/qc.md warns a stored-vs-recomputed verdict drifts.

Talent colors are assigned client-side (``docs/app.js``), not here — see that
file for why.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import series_v2, similarity
from .quality import verdict_any
from .reliability import corpus_noise_floor, noise_floor
from .site_data import _branch_for_group, _generation_order, _talent_groups_and_branch

Loader = Callable[[str], list[dict]]

#: reference/statistics.md's families, in the order that document lists
#: them. ``robust`` is None for the context/quality family: those metrics are
#: recording/selection diagnostics, not claims about a voice, so the
#: robust/experimental badge (which is about measuring the voice faithfully)
#: does not apply to them the same way.
FAMILIES: tuple[dict, ...] = (
    {
        "key": "pitch_level",
        "label": "Pitch level",
        "robust": True,
        "note": "Typical speaking pitch. The most reliable measure here.",
        "metrics": ("median_f0",),
    },
    {
        "key": "pitch_movement",
        "label": "Pitch movement",
        "robust": True,
        "note": "How widely and how quickly pitch moves, in semitones.",
        "metrics": ("f0_iqr_semitones", "dynamism_semitones"),
    },
    {
        "key": "voice_source",
        "label": "Voice source & timbre",
        "robust": False,
        "note": (
            "Airy or pressed, soft or bright. Affected by microphone and EQ; "
            "CPP also reads higher on vocal-separated clips."
        ),
        "metrics": (
            "h1h2_db",
            "cpp_db",
            "harmonic_tilt_db_per_octave",
            "alpha_ratio_db",
            "hammarberg_db",
        ),
    },
    {
        "key": "periodicity_noise",
        "label": "Periodicity & noise",
        "robust": False,
        "note": (
            "Designed for sustained vowels rather than conversation, and "
            "strongly affected by vocal separation."
        ),
        "metrics": ("jitter_local", "shimmer_local", "hnr_db"),
    },
    {
        "key": "spectrum_resonance",
        "label": "Spectrum & resonance",
        "robust": False,
        "note": (
            "Brightness and vocal-tract resonances. Strongly affected by "
            "recording setup; formant dispersion is the most stable of these."
        ),
        "metrics": (
            "brightness_hz",
            "formant_dispersion_hz",
            "f1_hz",
            "f2_hz",
            "f3_hz",
            "f4_hz",
        ),
    },
    {
        "key": "tempo",
        "label": "Tempo",
        "robust": False,
        "note": "Syllables per second of voiced speech.",
        "metrics": ("speaking_rate_syl_per_s",),
    },
    {
        "key": "context_quality",
        "label": "Recording context",
        "robust": None,
        "note": "Properties of the recording rather than the voice.",
        "metrics": ("voiced_fraction", "loudness_dynamics_db", "background_ratio_db"),
    },
)

ALL_METRICS: tuple[str, ...] = tuple(
    metric for family in FAMILIES for metric in family["metrics"]
)


def _default_loader(path: str) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _v2_path_for(v1_path: str, v2_dir: Path) -> str:
    stem = Path(v1_path).stem
    if stem.endswith("_monthly"):
        stem = stem[: -len("_monthly")]
    return str(v2_dir / f"{stem}.json")


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value


def _rank_percentiles(values: dict[str, float]) -> dict[str, float]:
    """0-100 rank percentile (0 = lowest, 100 = highest). Every name tied at
    the same value lands at the midpoint 50.0 instead of an arbitrary
    tie-break order."""
    names = sorted(values)
    if max(values.values()) == min(values.values()):
        return {name: 50.0 for name in names}
    ranked = sorted(names, key=lambda name: values[name])
    n = len(names)
    return {name: 100.0 * rank / (n - 1) for rank, name in enumerate(ranked)}


def _is_one_off(records: list[dict]) -> bool:
    """A talent entry built only from a named stretch of one stream (see
    ``segment.py``). It gets its own card, but it is one sample, and several
    windows of one stream would masquerade as independent clips, so it stays
    out of every corpus-wide statistic: percentile ranks and the noise floor."""
    return bool(records) and all(r.get("one_off") for r in records)


def _placement_percentile(value: float, ranked: dict[str, float]) -> float:
    """Where ``value`` would fall among already-ranked talents' values, 0-100,
    ties counting half — without joining (and so shifting) their ranking."""
    values = list(ranked.values())
    below = sum(1 for v in values if v < value)
    tied = sum(1 for v in values if v == value)
    return 100.0 * (below + 0.5 * tied) / len(values)


#: How many closest voices each talent's profile lists, per route.
NEIGHBOURS_K = 8
#: A voice metric joins the measured route only when this share of usable clips
#: carries it — a metric only a few clips have would decide distances from an
#: unrepresentative corner of the corpus.
NEIGHBOUR_METRIC_COVERAGE = 0.9


def _usable(record: dict) -> bool:
    return not record.get("legacy") and verdict_any(record.get("features") or {})[0]


#: Axes of the comparison radar: voice metrics a listener hears, in the order
#: they are drawn. Jitter, shimmer and HNR stay off it — calibrated for sustained
#: vowels, and the weakest separators of talents — though the measured route
#: still weighs them.
RADAR_METRICS: tuple[str, ...] = (
    "median_f0",
    "f0_iqr_semitones",
    "dynamism_semitones",
    "speaking_rate_syl_per_s",
    "h1h2_db",
    "cpp_db",
    "harmonic_tilt_db_per_octave",
    "alpha_ratio_db",
    "hammarberg_db",
    "brightness_hz",
    "formant_dispersion_hz",
)


def _fit_space(
    talents_records: dict[str, list[dict]], one_off: dict[str, bool]
) -> similarity.MetricSpace | None:
    """The measured route's metric space, over the voice metrics enough clips
    carry (see NEIGHBOUR_METRIC_COVERAGE). ``None`` when nothing qualifies."""
    clips = [
        (name, record.get("features") or {})
        for name, records in talents_records.items()
        if not one_off[name]
        for record in records
        if _usable(record)
    ]
    if not clips:
        return None
    metrics = [
        metric
        for metric in similarity.VOICE_METRICS
        if np.mean([_finite(features.get(metric)) for _, features in clips])
        >= NEIGHBOUR_METRIC_COVERAGE
    ]
    if not metrics:
        return None
    try:
        return similarity.fit_metric_space(clips, metrics=metrics)
    except ValueError:
        return None


def _noise_units(
    space: similarity.MetricSpace,
    typicals: dict[str, dict[str, float]],
    one_off: dict[str, bool],
) -> dict[str, dict[str, float]]:
    """Each talent's distance from the corpus median, per metric, in units of
    within-talent variation (the square root of the measured route's own
    covariance diagonal). One-offs are placed but do not move the median."""
    out: dict[str, dict[str, float]] = {name: {} for name in typicals}
    scales = np.sqrt(np.diag(space.covariance))
    for index, metric in enumerate(space.metrics):
        scaled = {
            name: similarity.voice_value(metric, values.get(metric))
            for name, values in typicals.items()
        }
        ranked = [v for name, v in scaled.items() if v == v and not one_off[name]]
        if not ranked or scales[index] <= 0:
            continue
        centre = float(np.median(ranked))
        for name, value in scaled.items():
            if value == value:
                out[name][metric] = (value - centre) / float(scales[index])
    return out


def _measured_neighbours(
    space: similarity.MetricSpace | None,
    talents_records: dict[str, list[dict]],
    typicals: dict[str, dict[str, float]],
    one_off: dict[str, bool],
) -> dict[str, list[dict]]:
    """Closest voices by the measured route (similarity.py), explained by the
    metrics on which each pair sits closest."""
    if space is None:
        return {}
    names = [name for name in talents_records if typicals[name]]
    spread = similarity.between_spread(
        space, [typicals[n] for n in names if not one_off[n]]
    )
    lists = similarity.neighbours(
        names,
        lambda a, b: space.distance(typicals[a], typicals[b]),
        k=NEIGHBOURS_K,
        one_off={name for name in names if one_off[name]},
    )
    return {
        name: [
            {
                "name": row["name"],
                "distance": row["value"],
                "closer_than_pct": row["closer_than_pct"],
                "closest_metrics": space.closest_metrics(
                    typicals[name], typicals[row["name"]], k=3, spread=spread, among=RADAR_METRICS
                ),
            }
            for row in rows
        ]
        for name, rows in lists.items()
    }


def language_group(branch: str | None) -> str | None:
    """The language a branch streams in, for embedding compensation. DEV_IS
    is Japanese-speaking; "Graduated" is a status, not a language."""
    if branch in ("JP", "DEV_IS"):
        return "JP"
    if branch in ("EN", "ID"):
        return branch
    return None


def declared_language(records: list[dict]) -> str | None:
    """A one-off may declare what it was spoken in (``segment --language``)."""
    languages = {str(r["language"]).upper() for r in records if r.get("language")}
    return languages.pop() if len(languages) == 1 else None


def _voice_neighbours(
    talents_records: dict[str, list[dict]],
    slugs: dict[str, str],
    embeddings: dict,
    one_off: dict[str, bool],
    space: similarity.MetricSpace | None = None,
    typicals: dict[str, dict[str, float]] | None = None,
    language_of: dict[str, str | None] | None = None,
) -> dict[str, list[dict]]:
    """Closest voices by embedding (embed.py). Only the similarity of talent
    centroids leaves this function — never a vector. The embedding cannot say
    why two voices match, so each row borrows the measured route's closest
    metrics for that pair as the explanation.

    A speaker model hears language as well as voice — left alone, a
    Japanese-speaking talent's closest voices are all Japanese speakers. So
    each language group's mean centroid (over ranked talents) is subtracted
    before comparing; what remains is the voice. A talent with no known
    language is compared uncompensated."""
    from .embed import cosine, talent_centroid

    centroids = {}
    for name, records in talents_records.items():
        emb = embeddings.get(slugs[name])
        if emb is None:
            continue
        centroid = talent_centroid(records, emb)
        if centroid is not None:
            centroids[name] = centroid
    language_of = language_of or {}
    group_means = {}
    for language in {language_of.get(n) for n in centroids} - {None}:
        members = [
            centroids[n] for n in centroids if language_of.get(n) == language and not one_off[n]
        ]
        if members:
            group_means[language] = np.mean(members, axis=0)
    centroids = {
        n: c - group_means[language_of[n]] if language_of.get(n) in group_means else c
        for n, c in centroids.items()
    }
    lists = similarity.neighbours(
        list(centroids),
        lambda a, b: cosine(centroids[a], centroids[b]),
        k=NEIGHBOURS_K,
        one_off={name for name in centroids if one_off[name]},
        higher_is_closer=True,
    )
    spread = (
        similarity.between_spread(space, [typicals[n] for n in typicals if not one_off[n]])
        if space is not None and typicals is not None
        else None
    )

    def reasons(a: str, b: str) -> list[str]:
        if space is None or typicals is None:
            return []
        return space.closest_metrics(typicals[a], typicals[b], k=3, spread=spread, among=RADAR_METRICS)

    return {
        name: [
            {
                "name": r["name"],
                "similarity": r["value"],
                "closer_than_pct": r["closer_than_pct"],
                "closest_metrics": reasons(name, r["name"]),
            }
            for r in rows
        ]
        for name, rows in lists.items()
    }


def _legacy_fraction(records: list[dict]) -> float:
    if not records:
        return 0.0
    legacy = sum(1 for r in records if r.get("legacy"))
    return legacy / len(records)


def _reliability_band_counts(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        band = record.get("reliability_band") or "unknown"
        counts[band] = counts.get(band, 0) + 1
    return counts


def build_site_data_v2(
    registry: dict[str, str],
    *,
    v2_dir: Path | str = Path("data/measurements/v2"),
    roster: list[dict] | None = None,
    v2_loader: Loader | None = None,
    v1_loader: Loader | None = None,
    embeddings: dict | None = None,
    language_of: dict[str, str | None] | None = None,
) -> dict:
    """Build the full v2 export.

    ``registry`` is talents.json's ``{v1_measurements_path: display_name}``
    shape, reused so the v2 site names talents identically to v1's. Each
    entry's v2 file is derived by stripping v1's ``_monthly`` suffix from the
    stem (the convention ``corpus.write_talent`` uses). A talent whose v2 file
    does not exist at all (``build-v2`` never ran for them) falls back to
    their v1 records directly, rather than vanishing from the site.

    ``embeddings`` (talent slug -> ``embed.Embeddings``, local only) adds the
    embedding route to each talent's closest voices. Only names and scores
    derived from them are exported. ``language_of`` (display name -> language
    group) overrides the language inferred from each talent's branch, for the
    embedding route's language compensation.
    """
    v2_dir = Path(v2_dir)
    load_v2 = v2_loader if v2_loader is not None else _default_loader
    load_v1 = v1_loader if v1_loader is not None else _default_loader

    roster_by_english_name = {
        entry["english_name"]: entry["group"]
        for entry in (roster or [])
        if entry.get("english_name") and entry.get("group")
    }

    talents_records: dict[str, list[dict]] = {}
    legacy_fallback: dict[str, bool] = {}
    slugs: dict[str, str] = {}
    for v1_path, name in registry.items():
        slugs[name] = Path(v1_path).stem.removesuffix("_monthly")
        v2_path = _v2_path_for(v1_path, v2_dir)
        try:
            records = load_v2(v2_path)
            legacy_fallback[name] = False
        except FileNotFoundError:
            records = load_v1(v1_path)
            legacy_fallback[name] = True
        talents_records[name] = records

    # Per-metric typical value for every talent, needed before percentiles
    # can be ranked (a percentile compares across the whole included set).
    typicals: dict[str, dict[str, float]] = {name: {} for name in talents_records}
    summaries: dict[str, dict[str, dict]] = {name: {} for name in talents_records}
    for name, records in talents_records.items():
        for metric in ALL_METRICS:
            summary = series_v2.typical_and_trend(records, metric)
            summaries[name][metric] = summary
            if _finite(summary["typical"]):
                typicals[name][metric] = summary["typical"]

    one_off = {name: _is_one_off(records) for name, records in talents_records.items()}
    percentiles: dict[str, dict[str, float]] = {name: {} for name in talents_records}
    for metric in ALL_METRICS:
        by_talent = {
            name: values[metric]
            for name, values in typicals.items()
            if metric in values and not one_off[name]
        }
        if len(by_talent) < 2:
            continue
        ranked = _rank_percentiles(by_talent)
        for name, pct in ranked.items():
            percentiles[name][metric] = pct
        for name, values in typicals.items():
            if one_off[name] and metric in values:
                percentiles[name][metric] = _placement_percentile(values[metric], by_talent)

    languages: dict[str, str | None] = {}
    for name, records in talents_records.items():
        declared = declared_language(records)
        if declared:
            languages[name] = declared
        elif roster is not None:
            groups, _ = _talent_groups_and_branch(name, roster_by_english_name)
            languages[name] = language_group(_branch_for_group(groups[0]))
    languages.update(language_of or {})
    space = _fit_space(talents_records, one_off)
    measured_near = _measured_neighbours(space, talents_records, typicals, one_off)
    voice_near = (
        _voice_neighbours(
            talents_records, slugs, embeddings, one_off, space, typicals, languages
        )
        if embeddings
        else {}
    )
    units = _noise_units(space, typicals, one_off) if space is not None else {}

    talents: dict[str, dict] = {}
    groups_present: set[str] = set()
    for name, records in talents_records.items():
        if roster is None:
            groups, branch = ["Unknown"], "Unknown"
        else:
            groups, branch = _talent_groups_and_branch(name, roster_by_english_name)
        groups_present.update(groups)

        metrics: dict[str, dict] = {}
        for metric in ALL_METRICS:
            summary = summaries[name][metric]
            entry = {
                "typical": summary["typical"],
                "ci_low": summary["ci_low"],
                "ci_high": summary["ci_high"],
                "trend": summary["trend"],
                "noise_floor": noise_floor(records, feature_keys=(metric,))[metric],
                "monthly": series_v2.monthly_series(records, metric),
                "quarterly": series_v2.quarterly_series(records, metric),
                "yearly": series_v2.yearly_series(records, metric),
            }
            if metric in percentiles[name]:
                entry["percentile"] = percentiles[name][metric]
            if metric in units.get(name, {}):
                entry["noise_units"] = units[name][metric]
            metrics[metric] = entry

        months = sorted({r.get("month") for r in records if r.get("month")})
        n_pass = sum(1 for r in records if verdict_any(r.get("features") or {})[0])
        talents[name] = {
            "group": groups,
            "branch": branch,
            "legacy_fallback": legacy_fallback[name],
            "one_off": one_off[name],
            "legacy_fraction": _legacy_fraction(records),
            "n_clips": len(records),
            "n_pass": n_pass,
            "months_covered": len(months),
            "first_month": months[0] if months else None,
            "last_month": months[-1] if months else None,
            "reliability_band": _reliability_band_counts(records),
            "metrics": metrics,
            "neighbours": {
                "measured": measured_near.get(name, []),
                **({"voice": voice_near.get(name, [])} if embeddings else {}),
            },
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "families": [dict(family) for family in FAMILIES],
        "corpus_noise_floor": corpus_noise_floor(
            {name: records for name, records in talents_records.items() if not one_off[name]},
            feature_keys=ALL_METRICS,
        ),
        "generation_order": _generation_order(groups_present),
        "radar_metrics": [
            m for m in RADAR_METRICS if space is not None and m in space.metrics
        ],
        "talents": talents,
    }


def write_site_data_v2(
    registry: dict[str, str],
    out_path: Path | str,
    *,
    v2_dir: Path | str = Path("data/measurements/v2"),
    roster: list[dict] | None = None,
    v2_loader: Loader | None = None,
    v1_loader: Loader | None = None,
    embeddings: dict | None = None,
) -> dict:
    """Writes ``window.SITE_DATA_V2 = {...};`` — a plain script the page
    loads directly, matching v1's convention (see site_data.write_site_data
    for why: a static, no-server ``file://`` page cannot ``fetch()`` a
    separate JSON file past CORS)."""
    payload = build_site_data_v2(
        registry,
        v2_dir=v2_dir,
        roster=roster,
        v2_loader=v2_loader,
        v1_loader=v1_loader,
        embeddings=embeddings,
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2, ensure_ascii=False)
    out.write_text(f"window.SITE_DATA_V2 = {body};\n", encoding="utf-8")
    return payload
