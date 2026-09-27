"""The Highlights page: a handful of plain-language awards drawn from the same
numbers the rest of the site shows.

Pure data, built from ``site_data_v2``'s per-talent export, so the page only
filters and slices rankings and never computes a statistic. Every scale gets
an award at each end, each with its own positive name, and every eligible
talent gets a signature (the award they rank highest in), so nobody reads as
the bottom of a list.

Only talents with enough data take part: a one-off, a talent still on v1
records, or one with too few clips or months would win on noise. Ineligible
talents are left out before ranking, so they move nobody's place.
"""

from __future__ import annotations

import math

import numpy as np

from . import similarity

#: A talent needs this many usable clips, spread over this many months, to place.
MIN_CLIPS = 20
MIN_MONTHS = 6
#: A pitch journey compares the median of a talent's first and latest this-many
#: measured months, so it needs twice that many, and counts only a change of at
#: least this many semitones.
JOURNEY_WINDOW_MONTHS = 12
JOURNEY_MIN_SEMITONES = 1.0
#: A generation needs this many eligible members for its members to be compared.
GROUP_MIN_MEMBERS = 3
#: Voice-twin pairs exported (the page may filter them by branch).
TWIN_PAIRS = 30
#: Groups that are not a generation.
NOT_GENERATIONS = frozenset({"Graduated", "Unknown"})
#: Entries measured as one voice that are really two people.
DUOS = frozenset({"FUWAMOCO"})

SECTIONS: tuple[dict, ...] = (
    {"key": "pitch", "label": "Pitch"},
    {"key": "pace", "label": "Pace"},
    {"key": "tone", "label": "Tone"},
    {"key": "expression", "label": "Expression"},
    {"key": "identity", "label": "Voice identity"},
    {"key": "groups", "label": "Generations"},
    {"key": "years", "label": "Through the years"},
)

#: Awards read straight off one metric's typical value. ``higher`` says which
#: end of the scale wins; ``delta`` how a place compares with the typical
#: talent: in semitones, as a percentage, or in the metric's own unit.
METRIC_AWARDS: tuple[dict, ...] = (
    {"key": "highest_voice", "section": "pitch", "metric": "median_f0", "higher": True,
     "delta": "semitones", "title": "Highest voice",
     "blurb": "The highest typical speaking pitch."},
    {"key": "deepest_voice", "section": "pitch", "metric": "median_f0", "higher": False,
     "delta": "semitones", "title": "Deepest voice",
     "blurb": "The lowest typical speaking pitch."},
    {"key": "most_melodic", "section": "pitch", "metric": "f0_iqr_semitones", "higher": True,
     "delta": "unit", "title": "Most melodic speaker",
     "blurb": "The widest range of pitch in everyday speech."},
    {"key": "most_composed", "section": "pitch", "metric": "f0_iqr_semitones", "higher": False,
     "delta": "unit", "title": "Most composed delivery",
     "blurb": "Speech that stays close to one comfortable pitch."},
    {"key": "most_animated", "section": "pitch", "metric": "dynamism_semitones", "higher": True,
     "delta": "unit", "title": "Most animated voice",
     "blurb": "Pitch that moves the most from moment to moment."},
    {"key": "smoothest_delivery", "section": "pitch", "metric": "dynamism_semitones",
     "higher": False, "delta": "unit", "title": "Smoothest delivery",
     "blurb": "Pitch that glides rather than jumps."},
    {"key": "fastest_talker", "section": "pace", "metric": "speaking_rate_syl_per_s",
     "higher": True, "delta": "percent", "title": "Fastest talker",
     "blurb": "The most syllables per second."},
    {"key": "most_unhurried", "section": "pace", "metric": "speaking_rate_syl_per_s",
     "higher": False, "delta": "percent", "title": "Most unhurried storyteller",
     "blurb": "Takes their time with every word."},
    {"key": "nonstop_talker", "section": "pace", "metric": "voiced_fraction", "higher": True,
     "delta": "percent", "title": "Nonstop talker",
     "blurb": "The largest share of each clip spent speaking."},
    {"key": "softest_voice", "section": "tone", "metric": "h1h2_db", "higher": True,
     "delta": "unit", "title": "Softest, airiest voice",
     "blurb": "The most breath in the voice: gentle and soft."},
    {"key": "most_punchy", "section": "tone", "metric": "h1h2_db", "higher": False,
     "delta": "unit", "title": "Most punchy voice",
     "blurb": "A firm, pressed voice that cuts through."},
    {"key": "most_sparkling", "section": "tone", "metric": "brightness_hz", "higher": True,
     "delta": "percent", "title": "Most sparkling voice",
     "blurb": "The most energy in the high frequencies."},
    {"key": "warmest_voice", "section": "tone", "metric": "brightness_hz", "higher": False,
     "delta": "percent", "title": "Warmest voice",
     "blurb": "The most energy in the low, rounded frequencies."},
    {"key": "crystal_clear", "section": "tone", "metric": "cpp_db", "higher": True,
     "delta": "unit", "title": "Crystal-clear voice",
     "blurb": "The most clearly defined voice."},
    {"key": "biggest_swings", "section": "expression", "metric": "loudness_dynamics_db",
     "higher": True, "delta": "unit", "title": "Biggest volume swings",
     "blurb": "From a whisper to a shout: the widest loudness range."},
    {"key": "most_even_keeled", "section": "expression", "metric": "loudness_dynamics_db",
     "higher": False, "delta": "unit", "title": "Most even-keeled",
     "blurb": "The steadiest volume from moment to moment."},
)

#: Awards computed from more than one number. Same fields, minus ``metric``.
OTHER_AWARDS: tuple[dict, ...] = (
    {"key": "one_of_a_kind", "section": "identity", "title": "One of a kind",
     "blurb": "Even the closest other voice is a long way off."},
    {"key": "voice_twins", "section": "identity", "title": "Voice twins",
     "blurb": "Pairs of talents whose voices sound most alike."},
    {"key": "most_harmonious_generation", "section": "groups",
     "title": "Most harmonious generation",
     "blurb": "Members whose voices sit closest together."},
    {"key": "most_varied_generation", "section": "groups", "title": "Most varied generation",
     "blurb": "Members who each bring a very different voice."},
    {"key": "pitch_journey", "section": "years", "title": "Biggest pitch journey",
     "blurb": "The largest change in speaking pitch from the first year to the latest."},
    {"key": "longest_record", "section": "years", "title": "Longest voice record",
     "blurb": "The most months of measured speech."},
)


def eligible(talent: dict) -> bool:
    return (
        not talent.get("one_off")
        and not talent.get("legacy_fallback")
        and talent.get("n_pass", 0) >= MIN_CLIPS
        and talent.get("months_covered", 0) >= MIN_MONTHS
    )


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _row(name: str, value: float, delta: float | None = None, **extra) -> dict:
    return {"name": name, "value": value, "delta": delta, "too_close": False, **extra}


def _delta(value: float, typical: float, kind: str) -> float | None:
    if kind == "semitones":
        return 12 * math.log2(value / typical) if value > 0 and typical > 0 else None
    if kind == "percent":
        return 100 * (value / typical - 1) if typical else None
    return value - typical


def _overlaps(a: dict, b: dict) -> bool:
    lows, highs = (a.get("ci_low"), b.get("ci_low")), (a.get("ci_high"), b.get("ci_high"))
    if not all(_finite(v) for v in (*lows, *highs)):
        return False
    return max(lows) <= min(highs)


def _metric_award(spec: dict, talents: dict[str, dict]) -> dict:
    metric = spec["metric"]
    entries = {
        name: t["metrics"][metric]
        for name, t in talents.items()
        if _finite(t["metrics"].get(metric, {}).get("typical"))
    }
    order = sorted(
        entries,
        key=lambda n: (-entries[n]["typical"] if spec["higher"] else entries[n]["typical"], n),
    )
    typical = float(np.median([e["typical"] for e in entries.values()])) if entries else math.nan
    ranking = []
    for i, name in enumerate(order):
        value = entries[name]["typical"]
        row = _row(name, value, _delta(value, typical, spec["delta"]))
        row["too_close"] = i > 0 and _overlaps(entries[name], entries[order[i - 1]])
        ranking.append(row)
    return {
        **{k: spec[k] for k in ("key", "section", "title", "blurb", "metric")},
        "delta_unit": spec["delta"],
        "ranking": ranking,
    }


def _signatures(awards: list[dict], robust: dict[str, bool | None]) -> dict[str, dict]:
    """Each talent's best placing among the metric awards: the smallest share
    of the field ahead of them. Ties go to robust measures, then to the
    award listed first."""
    best: dict[str, tuple] = {}
    for index, award in enumerate(awards):
        of = len(award["ranking"])
        for rank, row in enumerate(award["ranking"], start=1):
            key = (rank / of, robust.get(award["metric"]) is not True, index)
            if row["name"] not in best or key < best[row["name"]][0]:
                best[row["name"]] = (key, {"award": award["key"], "rank": rank, "of": of})
    return {name: entry for name, (_, entry) in best.items()}


def _journey(talents: dict[str, dict]) -> list[dict]:
    rows = []
    window = JOURNEY_WINDOW_MONTHS
    for name, t in talents.items():
        months = t["metrics"].get("median_f0", {}).get("monthly") or []
        values = [m["median"] for m in months if _finite(m.get("median"))]
        if len(values) < 2 * window:
            continue
        start, end = float(np.median(values[:window])), float(np.median(values[-window:]))
        change = 12 * math.log2(end / start)
        if abs(change) < JOURNEY_MIN_SEMITONES:
            continue
        rows.append(
            _row(name, change, from_hz=start, to_hz=end,
                 since=months[0]["period"], until=months[-1]["period"])
        )
    return sorted(rows, key=lambda r: (-abs(r["value"]), r["name"]))


def _identity_awards(talents: dict[str, dict], identity: dict) -> dict[str, list[dict]]:
    zero, full = identity["scale"]
    pairs = {
        (a, b): cos for (a, b), cos in identity["pairs"].items() if a in talents and b in talents
    }

    def pct(cos: float) -> float:
        return similarity.match_percent(cos, zero=zero, full=full)

    twins = sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0]))[:TWIN_PAIRS]
    out = {
        "voice_twins": [
            _row(f"{a} & {b}", pct(cos), members=[a, b]) for (a, b), cos in twins
        ]
    }

    closest: dict[str, tuple[float, str]] = {}
    for (a, b), cos in pairs.items():
        for me, other in ((a, b), (b, a)):
            if me not in closest or cos > closest[me][0]:
                closest[me] = (cos, other)
    out["one_of_a_kind"] = [
        _row(name, pct(cos), closest=other)
        for name, (cos, other) in sorted(closest.items(), key=lambda kv: (kv[1][0], kv[0]))
    ]

    groups: dict[str, list[str]] = {}
    for name, t in talents.items():
        for group in t.get("group") or []:
            if group not in NOT_GENERATIONS and name in closest:
                groups.setdefault(group, []).append(name)
    means = {}
    for group, members in groups.items():
        if len(members) < GROUP_MIN_MEMBERS:
            continue
        sims = [
            pairs.get((a, b), pairs.get((b, a)))
            for i, a in enumerate(members)
            for b in members[i + 1 :]
        ]
        sims = [s for s in sims if s is not None]
        if sims:
            means[group] = (float(np.mean(sims)), len(members))
    # A generation scores the share of all talent pairs less alike than its
    # average member pair: most generations sit below a typical pair, where
    # the absolute match would pin them all at 0%.
    everyone = np.sort(np.array(list(pairs.values())))

    def pairs_below(cos: float) -> float:
        return 100.0 * np.searchsorted(everyone, cos, side="left") / len(everyone)

    harmonious = sorted(means, key=lambda g: (-means[g][0], g))
    out["most_harmonious_generation"] = [
        _row(g, pairs_below(means[g][0]), members=sorted(groups[g])) for g in harmonious
    ]
    out["most_varied_generation"] = list(reversed(out["most_harmonious_generation"]))
    return out


def build_highlights(
    talents: dict[str, dict],
    families: list[dict] | tuple[dict, ...],
    voice_identity: dict | None = None,
) -> dict:
    """``talents`` is site_data_v2's per-talent export; ``voice_identity`` the
    embedding route's raw similarities (``_voice_neighbours``), when local
    embeddings exist. Awards with nothing to rank are left out."""
    robust = {m: f["robust"] for f in families for m in f["metrics"]}
    field = {name: t for name, t in talents.items() if eligible(t)}

    metric_awards = [_metric_award(spec, field) for spec in METRIC_AWARDS]
    metric_awards = [a for a in metric_awards if a["ranking"]]

    rankings: dict[str, list[dict]] = {}
    if voice_identity is not None:
        rankings.update(_identity_awards(field, voice_identity))
    rankings["pitch_journey"] = _journey(field)
    rankings["longest_record"] = sorted(
        (_row(n, t["months_covered"], since=t.get("first_month")) for n, t in field.items()),
        key=lambda r: (-r["value"], r["name"]),
    )
    other_awards = [
        {**spec, "metric": None, "delta_unit": None, "ranking": rankings[spec["key"]]}
        for spec in OTHER_AWARDS
        if rankings.get(spec["key"])
    ]

    return {
        "min_clips": MIN_CLIPS,
        "min_months": MIN_MONTHS,
        "sections": [dict(s) for s in SECTIONS],
        "awards": metric_awards + other_awards,
        "signatures": _signatures(metric_awards, robust),
        "duos": sorted(DUOS & set(field)),
    }
