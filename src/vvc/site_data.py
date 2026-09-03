"""Aggregate export for the interactive comparison site (``docs/``, the
directory GitHub Pages serves).

Pure data — no matplotlib/plotly here, only JSON-serializable Python
values built from vvc.series's existing aggregation functions, so
this module is testable without a plotting dependency and the frontend
never needs to re-derive any statistics itself.

The cute/mature percentile is the first real implementation of the rule
PLAN.md has described since the start ("PLAN L36" / "What we measure"):
a scatter of F0 vs brightness plus a percentile from equal-weight
z-scores of F0, brightness, and dynamism vs the corpus. Concretely: for
every talent with at least one QC-pass clip, take the plain mean of that
talent's QC-pass median_f0 / brightness_hz / dynamism_semitones; z-score
each axis across the included talents (population stats — the included
talents ARE the comparison set, not a sample of a larger one); average
the three z-scores equally; rank that combined z among the included
talents into a 0-100 percentile (0 = lowest, 100 = highest). A talent
missing QC-pass data is omitted rather than plotted at a fabricated
(0, 0), and never influences the corpus stats used to score others. With
zero variance on an axis (every included talent tied), that axis
contributes a z-score of 0 rather than dividing by zero; if there is no
discriminating signal on ANY axis for every included talent, all of them
land at the midpoint (50.0) instead of an arbitrary tie-break order.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from .series import f0_quarterly, f0_series, f0_yearly

Loader = Callable[[str], list[dict]]

# The 7 numeric feature keys plotted as yearly series today (matches
# __main__.py's _EXTRA_FEATURE_PLOTS plus median_f0 itself).
YEARLY_FEATURE_KEYS = (
    "median_f0",
    "brightness_hz",
    "dynamism_semitones",
    "jitter_local",
    "shimmer_local",
    "hnr_db",
    "loudness_dynamics_db",
    "f1_hz",
    "f2_hz",
    "f3_hz",
    "f4_hz",
)

_CUTE_MATURE_AXES = ("median_f0", "brightness_hz", "dynamism_semitones")

# talents.json display names that don't equal roster.json's english_name
# for that person (DEV_IS members registered under short given names).
_ROSTER_NAME_ALIASES = {
    "Niko": "Koganei Niko",
    "Su": "Mizumiya Su",
    "Chihaya": "Rindo Chihaya",
    "Hajime": "Todoroki Hajime",
    "Kanade": "Otonose Kanade",
    "Raden": "Juufuutei Raden",
    "Riona": "Isaki Riona",
    "Ririka": "Ichijou Ririka",
    "Vivi": "Kikirara Vivi",
}


# Graduated talents are entirely absent from roster.json (Holodex's
# roster fetch is active-only), but their real generation is still
# knowable — queried once directly against each talent's own Holodex
# channel (not the roster list) and hardcoded here, so "Graduated"
# describes their branch, not their era.
_KNOWN_GRADUATED_GROUPS = {
    "Hiodoshi Ao": "DEV_IS ReGLOSS",
    "Kiryu Coco": "4th Generation (holoForce)",
}

# Talents holding more than one generation membership — Holodex's own
# per-channel "group" field is single-valued, so this can't come from
# the roster at all; it's a deliberate, small, hand-maintained override.
# Shirakami Fubuki debuted in 1st Generation and later joined GAMERS
# (formed from members across generations, not a numbered cohort itself).
_MULTI_GROUP_TALENTS = {
    "Shirakami Fubuki": ["1st Generation", "GAMERS"],
}

# Real-world debut chronology (earliest member's Holodex published_at
# per group, resolved once directly against the API on 2026-09-03 — see
# PLAN.md). Alphabetical order badly scrambles this (e.g. "DEV_IS
# ReGLOSS" (2023) would sort before "English -Myth-" (2020)), so
# generation_order in the export uses this reference table instead of a
# plain string sort.
_GROUP_CHRONOLOGICAL_ORDER = (
    "0th Generation",
    "1st Generation",
    "2nd Generation",
    "GAMERS",
    "3rd Generation (Fantasy)",
    "4th Generation (holoForce)",
    "Indonesia 1st Gen (AREA 15)",
    "5th Generation (holoFive)",
    "English -Myth-",
    "Indonesia 2nd Gen (holoro)",
    "English -Promise-",
    "6th Generation -holoX-",
    "Indonesia 3rd Gen (holoh3ro)",
    "English -Advent-",
    "DEV_IS ReGLOSS",
    "English -Justice-",
    "DEV_IS FLOW GLOW",
    "mekPark",
)


def _branch_for_group(group: str) -> str:
    if group.startswith("English"):
        return "EN"
    if group.startswith("Indonesia"):
        return "ID"
    if group.startswith("DEV_IS"):
        return "DEV_IS"
    if group == "Graduated":
        return "Graduated"
    return "JP"


def _talent_groups_and_branch(
    display_name: str, roster_by_english_name: dict[str, str]
) -> tuple[list[str], str]:
    if display_name in _MULTI_GROUP_TALENTS:
        groups = _MULTI_GROUP_TALENTS[display_name]
        return groups, _branch_for_group(groups[0])
    lookup_name = _ROSTER_NAME_ALIASES.get(display_name, display_name)
    group = roster_by_english_name.get(lookup_name)
    if group is not None:
        return [group], _branch_for_group(group)
    graduated_group = _KNOWN_GRADUATED_GROUPS.get(display_name)
    if graduated_group is not None:
        return [graduated_group], "Graduated"
    return ["Graduated"], "Graduated"


def _generation_order(groups_present: set[str]) -> list[str]:
    known = [g for g in _GROUP_CHRONOLOGICAL_ORDER if g in groups_present]
    unknown = sorted(groups_present - set(_GROUP_CHRONOLOGICAL_ORDER))
    return known + unknown


def _default_loader(path: str) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _qc_pass_values(entries: list[dict], feature_key: str) -> list[float]:
    values: list[float] = []
    for entry in entries:
        if not (entry.get("qc") or {}).get("pass"):
            continue
        value = (entry.get("features") or {}).get(feature_key)
        if isinstance(value, (int, float)) and value == value:  # exclude NaN
            values.append(float(value))
    return values


def _present_yearly_keys(all_entries: list[list[dict]]) -> list[str]:
    present: set[str] = set()
    for entries in all_entries:
        for entry in entries:
            present.update((entry.get("features") or {}).keys())
    return [key for key in YEARLY_FEATURE_KEYS if key in present]


def _rank_percentiles(scores: dict[str, float]) -> dict[str, float]:
    """0-100 rank percentile (0 = lowest, 100 = highest) from a mapping of
    name -> comparable score (a z-score, a combined z-score, or any other
    already-computed axis). Every name tied at the same score (no
    discriminating signal at all) lands at the midpoint 50.0 instead of an
    arbitrary tie-break order."""
    names = sorted(scores)
    if max(scores.values()) == min(scores.values()):
        return {name: 50.0 for name in names}
    ranked = sorted(names, key=lambda name: scores[name])
    n = len(names)
    return {name: 100.0 * rank / (n - 1) for rank, name in enumerate(ranked)}


def _zscore_axis(names: list[str], means: dict[str, float]) -> dict[str, float]:
    """Population z-score (the included names ARE the comparison set, not
    a sample of a larger one) of each name's mean on one axis. Zero
    corpus-wide variance (every included name tied) gives z=0 for all
    rather than dividing by zero."""
    axis_values = [means[name] for name in names]
    mean = statistics.mean(axis_values)
    sd = statistics.pstdev(axis_values)
    return {
        name: (0.0 if sd == 0.0 else (means[name] - mean) / sd) for name in names
    }


def _axis_means(
    talents_entries: dict[str, list[dict]], feature_key: str
) -> dict[str, float]:
    """Plain mean of each talent's QC-pass values on one axis — no
    z-score, no ranking, no comparison-talent requirement. Exported
    as-is as "raw_means"."""
    return {
        name: statistics.mean(values)
        for name, entries in talents_entries.items()
        if (values := _qc_pass_values(entries, feature_key))
    }


def _single_axis_percentiles(means: dict[str, float]) -> dict[str, float]:
    """Rank percentile on ONE feature axis independently — unlike
    cute_mature's combined-then-ranked score, each yearly metric gets its
    own separate ranking. With fewer than 2 contributing talents the
    axis has nothing to compare against and is empty."""
    if len(means) < 2:
        return {}
    names = sorted(means)
    return _rank_percentiles(_zscore_axis(names, means))


def _cute_mature(talents_entries: dict[str, list[dict]]) -> dict[str, dict]:
    means: dict[str, dict[str, float]] = {}
    for name, entries in talents_entries.items():
        axis_means = {}
        ok = True
        for axis in _CUTE_MATURE_AXES:
            values = _qc_pass_values(entries, axis)
            if not values:
                ok = False
                break
            axis_means[axis] = statistics.mean(values)
        if ok:
            means[name] = axis_means

    if len(means) < 2:
        return {}

    names = sorted(means)
    z_by_axis = {
        axis: _zscore_axis(names, {name: means[name][axis] for name in names})
        for axis in _CUTE_MATURE_AXES
    }

    combined = {
        name: sum(z_by_axis[axis][name] for axis in _CUTE_MATURE_AXES) / len(_CUTE_MATURE_AXES)
        for name in names
    }
    percentiles = _rank_percentiles(combined)

    return {
        name: {
            "f0_mean": means[name]["median_f0"],
            "brightness_mean": means[name]["brightness_hz"],
            "dynamism_mean": means[name]["dynamism_semitones"],
            "percentile": percentiles[name],
        }
        for name in names
    }


def build_site_data(
    registry: dict[str, str],
    *,
    roster: list[dict] | None = None,
    loader: Loader | None = None,
) -> dict:
    """Build the full export: per-talent series (from series.py, unchanged)
    plus the cute/mature percentile scatter. ``registry`` is the same
    ``{measurements_path: display_name}`` shape as talents.json. ``roster``
    is the ``talents`` list from roster.json (each dict needs at least
    ``english_name``/``group``); when given, every talent entry gets
    ``group``/``branch`` (see module rules 7-9 in the test file); when
    omitted, every talent gets group=branch='Unknown' rather than a
    missing key."""
    load = loader if loader is not None else _default_loader

    roster_by_english_name = {
        entry["english_name"]: entry["group"]
        for entry in (roster or [])
        if entry.get("english_name") and entry.get("group")
    }

    talents_entries: dict[str, list[dict]] = {
        display_name: load(path) for path, display_name in registry.items()
    }
    yearly_keys = _present_yearly_keys(list(talents_entries.values()))
    axis_means = {key: _axis_means(talents_entries, key) for key in yearly_keys}
    axis_percentiles = {
        key: _single_axis_percentiles(axis_means[key]) for key in yearly_keys
    }

    talents: dict[str, dict] = {}
    groups_present: set[str] = set()
    for name, entries in talents_entries.items():
        qc_pass = sum(1 for e in entries if (e.get("qc") or {}).get("pass"))
        if roster is None:
            groups, branch = ["Unknown"], "Unknown"
        else:
            groups, branch = _talent_groups_and_branch(name, roster_by_english_name)
        groups_present.update(groups)
        talents[name] = {
            "monthly": {
                key: f0_series(entries, qc=True, feature_key=key)
                for key in yearly_keys
            },
            "quarterly": {
                key: f0_quarterly(entries, qc=True, feature_key=key)
                for key in yearly_keys
            },
            "yearly": {
                key: f0_yearly(entries, qc=True, feature_key=key)
                for key in yearly_keys
            },
            "qc_summary": {"qc_pass": qc_pass, "total": len(entries)},
            "group": groups,
            "branch": branch,
            "percentiles": {
                key: axis_percentiles[key][name]
                for key in yearly_keys
                if name in axis_percentiles[key]
            },
            "raw_means": {
                key: axis_means[key][name]
                for key in yearly_keys
                if name in axis_means[key]
            },
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "talents": talents,
        "cute_mature": _cute_mature(talents_entries),
        "generation_order": _generation_order(groups_present),
    }


def write_site_data(
    registry: dict[str, str],
    out_path: Path | str,
    *,
    roster: list[dict] | None = None,
    loader: Loader | None = None,
) -> dict:
    """Writes ``window.SITE_DATA = {...};`` — a plain <script src=...>
    the page loads directly, NOT bare JSON fetched at runtime. A
    ``fetch("data.json")`` from a page opened via ``file://`` (the whole
    point of a static, no-server site) is blocked by CORS in every major
    browser; a script-tag assignment has no such restriction."""
    payload = build_site_data(registry, roster=roster, loader=loader)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2, ensure_ascii=False)
    out.write_text(f"window.SITE_DATA = {body};\n", encoding="utf-8")
    return payload
