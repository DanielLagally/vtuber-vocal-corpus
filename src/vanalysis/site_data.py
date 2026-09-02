"""Aggregate export for the interactive comparison site (``docs/``, the
directory GitHub Pages serves).

Pure data — no matplotlib/plotly here, only JSON-serializable Python
values built from vanalysis.series's existing aggregation functions, so
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
)

_CUTE_MATURE_AXES = ("median_f0", "brightness_hz", "dynamism_semitones")


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
    z_by_axis: dict[str, dict[str, float]] = {}
    for axis in _CUTE_MATURE_AXES:
        axis_values = [means[name][axis] for name in names]
        mean = statistics.mean(axis_values)
        sd = statistics.pstdev(axis_values)
        z_by_axis[axis] = {
            name: (0.0 if sd == 0.0 else (means[name][axis] - mean) / sd)
            for name in names
        }

    combined = {
        name: sum(z_by_axis[axis][name] for axis in _CUTE_MATURE_AXES) / len(_CUTE_MATURE_AXES)
        for name in names
    }

    n = len(names)
    if max(combined.values()) == min(combined.values()):
        percentiles = {name: 50.0 for name in names}
    else:
        ranked = sorted(names, key=lambda name: combined[name])
        percentiles = {
            name: 100.0 * rank / (n - 1) for rank, name in enumerate(ranked)
        }

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
    registry: dict[str, str], *, loader: Loader | None = None
) -> dict:
    """Build the full export: per-talent series (from series.py, unchanged)
    plus the cute/mature percentile scatter. ``registry`` is the same
    ``{measurements_path: display_name}`` shape as talents.json."""
    load = loader if loader is not None else _default_loader

    talents_entries: dict[str, list[dict]] = {
        display_name: load(path) for path, display_name in registry.items()
    }
    yearly_keys = _present_yearly_keys(list(talents_entries.values()))

    talents: dict[str, dict] = {}
    for name, entries in talents_entries.items():
        qc_pass = sum(1 for e in entries if (e.get("qc") or {}).get("pass"))
        talents[name] = {
            "monthly_f0_all": f0_series(entries),
            "monthly_f0_qc": f0_series(entries, qc=True),
            "quarterly_f0": f0_quarterly(entries, qc=True),
            "yearly": {
                key: f0_yearly(entries, qc=True, feature_key=key)
                for key in yearly_keys
            },
            "qc_summary": {"qc_pass": qc_pass, "total": len(entries)},
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "talents": talents,
        "cute_mature": _cute_mature(talents_entries),
    }


def write_site_data(
    registry: dict[str, str], out_path: Path | str, *, loader: Loader | None = None
) -> dict:
    """Writes ``window.SITE_DATA = {...};`` — a plain <script src=...>
    the page loads directly, NOT bare JSON fetched at runtime. A
    ``fetch("data.json")`` from a page opened via ``file://`` (the whole
    point of a static, no-server site) is blocked by CORS in every major
    browser; a script-tag assignment has no such restriction."""
    payload = build_site_data(registry, loader=loader)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2, ensure_ascii=False)
    out.write_text(f"window.SITE_DATA = {body};\n", encoding="utf-8")
    return payload
