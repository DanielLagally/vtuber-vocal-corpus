from __future__ import annotations

import math
import statistics
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .qc import qc_verdict


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


# Plain-language plot text (CLAUDE.md: plots are a permanent record — they
# should stand on their own for a reader who has never seen this repo).
_WHAT_IS_F0 = (
    "Median F0 = typical pitch of voiced speech in a ~90 s clip of a public\n"
    "chatting stream. Higher = higher-pitched voice."
)
_QC_LABEL_ALL = "all clips"
_QC_LABEL_PASS = "QC-pass only"
_QC_FOOTER = (
    "QC excludes clips with too little voice, unstable pitch (BGM/tracker\n"
    "error), or an implausible reading. Gaps are missing/failed data, never 0 Hz."
)


def _title(talent: str | None, subject: str) -> str:
    return f"{talent} — {subject}" if talent else subject


def _add_caption(fig: plt.Figure, *lines: str) -> None:
    # supxlabel (not fig.text) so constrained_layout reserves space for it —
    # a raw fig.text at a fixed y overlaps rotated x-tick labels instead.
    fig.supxlabel("\n".join(lines), fontsize="x-small", color="dimgray", ha="center")


def new_run_dir(base: Path | str, label: str | None = None) -> Path:
    """A fresh, never-reused directory under ``base/runs/`` (CLAUDE.md:
    plots are a permanent record — every plot run gets its own directory,
    none are overwritten). Name is a ``%Y%m%dT%H%M%S`` timestamp, plus
    ``-<label>`` when given. On the rare same-second collision (two runs,
    same label, same second) a ``-2``, ``-3``, ... suffix is appended
    rather than reusing the existing directory."""
    base = Path(base)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    stem = f"{stamp}-{label}" if label else stamp
    runs = base / "runs"
    candidate = runs / stem
    n = 1
    while candidate.exists():
        n += 1
        candidate = runs / f"{stem}-{n}"
    candidate.mkdir(parents=True)
    return candidate


def _cap_entries_per_month(
    entries: list[dict], max_per_month: int | None
) -> list[dict]:
    """Non-destructive "view this data as if only max_per_month clips
    had ever been fetched that calendar month": per month, keeps only
    the ``max_per_month`` highest-``score`` entries (ties broken by
    whichever comes first — same spirit as densify's own ranking).
    ``entries`` itself is never mutated, so calling again with a higher
    N (or None) on the same list always restores exactly what capping
    removed — reversible in both directions, nothing is ever discarded
    from the underlying measurements file by this function. ``None``
    (the default everywhere) is a no-op passthrough."""
    if max_per_month is None:
        return entries
    by_month: dict[object, list[dict]] = {}
    for entry in entries:
        by_month.setdefault(entry.get("month"), []).append(entry)
    kept: list[dict] = []
    for month_entries in by_month.values():
        ranked = sorted(
            month_entries, key=lambda e: -(e.get("score") or 0.0)
        )
        kept.extend(ranked[:max_per_month])
    return kept


def f0_series(
    entries: list[dict],
    qc: bool = False,
    *,
    feature_key: str = "median_f0",
    max_per_month: int | None = None,
) -> list[tuple[str, float]]:
    """Monthly series, ONE point per calendar month (STATE R3: multi-clip
    months): the plain float mean (no rounding) of that month's finite
    ``feature_key`` values (default ``median_f0``, but any numeric
    feature key works — brightness_hz, dynamism_semitones, etc., same
    generalization as ``f0_yearly``). With ``qc=True`` the shared QC rule
    filters the entries FIRST, then the survivors are averaged — a month
    with no surviving value is a gap in the QC series (it can still be
    present in the all-clip series). A single-record month is unchanged
    (mean of one is the value itself). Sorted by month.

    ``max_per_month`` (default None = every record counts) caps each
    month at its N highest-``score`` entries BEFORE any other
    filtering — see ``_cap_entries_per_month``."""
    entries = _cap_entries_per_month(entries, max_per_month)
    buckets: dict[str, list[float]] = {}
    for entry in entries:
        features = entry.get("features") or {}
        if qc:
            qc_pass, _ = qc_verdict(features)
            if not qc_pass:
                continue
        value = features.get(feature_key)
        if not _finite(value):
            continue
        buckets.setdefault(entry["month"], []).append(float(value))
    points = [
        (month, sum(values) / len(values)) for month, values in buckets.items()
    ]
    points.sort(key=lambda point: point[0])
    return points


def iqr_series(
    entries: list[dict], *, max_per_month: int | None = None
) -> list[tuple[str, float]]:
    """Monthly F0-IQR series, ONE point per calendar month (STATE R3:
    multi-clip months): the plain float mean (no rounding) of that
    month's finite ``f0_iqr`` values. Non-finite IQRs contribute
    nothing; a single-record month is unchanged (mean of one). Sorted
    by month. ``max_per_month``: see ``f0_series``."""
    entries = _cap_entries_per_month(entries, max_per_month)
    buckets: dict[str, list[float]] = {}
    for entry in entries:
        iqr = (entry.get("features") or {}).get("f0_iqr")
        if not _finite(iqr):
            continue
        buckets.setdefault(entry["month"], []).append(float(iqr))
    points = [
        (month, sum(values) / len(values)) for month, values in buckets.items()
    ]
    points.sort(key=lambda point: point[0])
    return points


def _quarter(month: object) -> str | None:
    """"YYYY-MM" -> "YYYY-Qn" (PLAN L36 quarter point). Anything that is
    not a parseable month (None, malformed) cannot be placed on the
    timeline and yields None."""
    if not isinstance(month, str):
        return None
    year, dash, mm = month.partition("-")
    if not dash or not year.isdigit() or not mm.isdigit():
        return None
    m = int(mm)
    if not 1 <= m <= 12:
        return None
    return f"{year}-Q{(m - 1) // 3 + 1}"


def f0_quarterly(
    entries: list[dict],
    qc: bool = False,
    *,
    feature_key: str = "median_f0",
    max_per_month: int | None = None,
) -> list[dict]:
    """Per calendar quarter (PLAN L36): the MEAN of that quarter's clip
    ``feature_key`` values (default ``median_f0``, but any numeric
    feature key works, same generalization as ``f0_yearly``/``f0_series``),
    plus min and max of the same values, and n, sorted by quarter.

    A clip with a non-finite value is excluded ENTIRELY from its
    quarterly point (it cannot plot — it contributes to nothing, not
    even n); a quarter left without plottable clips is simply absent (a
    gap). With ``qc=True`` the shared QC rule filters the entries BEFORE
    aggregating, so nan-median, IQR >= 200, and median >= 600 clips all
    drop first (the QC rule is always about F0/IQR/voiced-fraction,
    regardless of which feature is being aggregated). A n=1 quarter
    carries min == max == the single value (the plot decides band vs
    bare point — the aggregation stays pure).

    ``max_per_month`` caps EACH CALENDAR MONTH (not quarter) at its N
    highest-score clips before aggregating — see ``f0_series``.
    """
    entries = _cap_entries_per_month(entries, max_per_month)
    buckets: dict[str, list[float]] = {}
    for entry in entries:
        features = entry.get("features") or {}
        if qc:
            qc_pass, _ = qc_verdict(features)
            if not qc_pass:
                continue
        value = features.get(feature_key)
        if not _finite(value):
            continue
        quarter = _quarter(entry.get("month"))
        if quarter is None:
            continue
        buckets.setdefault(quarter, []).append(float(value))
    return [
        {
            "quarter": quarter,
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "n": len(values),
        }
        for quarter, values in sorted(buckets.items())
    ]


def _year(month: object) -> str | None:
    """Validated "YYYY" from a "YYYY-MM" month string (PLAN L36 year
    point). Anything that is not a parseable month (None, malformed)
    cannot be placed on the timeline and yields None."""
    if not isinstance(month, str):
        return None
    year, dash, mm = month.partition("-")
    if not dash or not year.isdigit() or not mm.isdigit():
        return None
    if not 1 <= int(mm) <= 12:
        return None
    return year


def f0_yearly(
    entries: list[dict],
    qc: bool = False,
    *,
    feature_key: str = "median_f0",
    max_per_month: int | None = None,
) -> list[dict]:
    """Per calendar year (PLAN L36 year rule): the MEDIAN of that year's
    clip ``feature_key`` values (clip-level — not month-means first, the
    clip is the sample; default ``median_f0``, but any numeric feature
    key works the same way — brightness_hz, dynamism_semitones, etc.),
    plus min and max of the same values (the between-clip spread), and
    n, sorted by year.

    A clip with a non-finite value is excluded ENTIRELY from its year
    point (it cannot plot — it contributes to nothing, not even n); a
    year left without plottable clips is simply absent (a gap). With
    ``qc=True`` the shared QC rule filters the entries BEFORE
    aggregating (the QC rule itself is always about median_f0/f0_iqr/
    voiced_fraction, regardless of which feature is being aggregated —
    it's a signal-trustworthiness gate, not specific to F0). A n=1 year
    carries min == max == the single value.

    ``max_per_month`` caps EACH CALENDAR MONTH at its N highest-score
    clips before aggregating — see ``f0_series``.

    The median is the lower median for an even clip count (median_low):
    an even-count year reports an actually-observed clip median, not an
    interpolated midpoint — pinned by rule 10 ([300, 615] -> 300.0).
    """
    entries = _cap_entries_per_month(entries, max_per_month)
    buckets: dict[str, list[float]] = {}
    for entry in entries:
        features = entry.get("features") or {}
        if qc:
            qc_pass, _ = qc_verdict(features)
            if not qc_pass:
                continue
        median = features.get(feature_key)
        if not _finite(median):
            continue
        year = _year(entry.get("month"))
        if year is None:
            continue
        buckets.setdefault(year, []).append(float(median))
    return [
        {
            "year": year,
            "median": statistics.median_low(values),
            "min": min(values),
            "max": max(values),
            "n": len(values),
        }
        for year, values in sorted(buckets.items())
    ]


def _plot_points(ax: plt.Axes, points: list[tuple[str, float]], label: str) -> None:
    months = [month for month, _ in points]
    values = [value for _, value in points]
    ax.plot(months, values, marker="o", label=label, markersize=4)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)


def _plot_quarterly(ax: plt.Axes, points: list[dict], label: str) -> None:
    """Quarter points as means, with a min–max band ONLY where n >= 2 —
    a n=1 quarter is anecdotal (PLAN L36) and renders as a bare point.
    Empty quarters are absent from the series and so leave a natural
    gap on the x axis. Each point is direct-labeled with its exact Hz
    value; horizontal gridlines make reading values off the axis
    easier without a label on every single line."""
    quarters = [p["quarter"] for p in points]
    xs = list(range(len(points)))
    means = [p["mean"] for p in points]
    ax.plot(xs, means, marker="o", label=label, markersize=4)
    for x, mean in zip(xs, means):
        ax.annotate(
            f"{mean:.0f}", (x, mean), xytext=(0, 6), textcoords="offset points",
            ha="center", fontsize="xx-small", alpha=0.85,
        )
    band_x = [x for x, p in zip(xs, points) if p["n"] >= 2]
    if band_x:
        ax.fill_between(
            band_x,
            [points[x]["min"] for x in band_x],
            [points[x]["max"] for x in band_x],
            alpha=0.2,
            linewidth=0,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(quarters, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)


def write_plots(entries: list[dict], out_dir: Path, *, talent: str | None = None) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_points = f0_series(entries)
    qc_points = f0_series(entries, qc=True)
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    _plot_points(ax, all_points, _QC_LABEL_ALL)
    _plot_points(ax, qc_points, _QC_LABEL_PASS)
    fig.suptitle(_title(talent, "Median Pitch (F0) by Month"), fontweight="bold")
    ax.set_title(_WHAT_IS_F0, fontsize="small", color="dimgray")
    ax.set_ylabel("Median F0 (Hz)")
    ax.tick_params(axis="x", rotation=45)
    if all_points or qc_points:
        ax.legend()
    _add_caption(fig, _QC_FOOTER)
    fig.savefig(out_dir / "f0_monthly.png")
    plt.close(fig)

    iqr_points = iqr_series(entries)
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    _plot_points(ax, iqr_points, _QC_LABEL_ALL)
    fig.suptitle(_title(talent, "Pitch Variability (F0 IQR) by Month"), fontweight="bold")
    ax.set_title(
        "Spread of pitch within one clip — tight = one steady voice,\n"
        "wide = background music, mixed audio, or tracker error.",
        fontsize="small",
        color="dimgray",
    )
    ax.set_ylabel("F0 IQR (Hz) — lower is more consistent")
    ax.tick_params(axis="x", rotation=45)
    if iqr_points:
        ax.legend()
    _add_caption(
        fig,
        "IQR >= 200 Hz fails QC (see the F0 plot). Shown here for every clip,\n"
        "pass or fail, so cleaning is visible.",
    )
    fig.savefig(out_dir / "f0_iqr_monthly.png")
    plt.close(fig)


def write_quarterly_plots(
    entries: list[dict], out_dir: Path, *, talent: str | None = None
) -> None:
    """The two quarterly PNGs (PLAN L36 quarter point): mean of clip
    medians + min–max band where n >= 2 (n=1 anecdotal, bare point),
    empty quarters as gaps. ``*_all`` shows QC-failing clips (cleaning
    visible), ``*_qc`` shows QC passes only (fails are gaps)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = (
        ("f0_quarterly_all.png", f0_quarterly(entries), _QC_LABEL_ALL),
        ("f0_quarterly_qc.png", f0_quarterly(entries, qc=True), _QC_LABEL_PASS),
    )
    for name, points, label in variants:
        fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
        _plot_quarterly(ax, points, label)
        fig.suptitle(_title(talent, "Median Pitch (F0) by Quarter"), fontweight="bold")
        ax.set_title(_WHAT_IS_F0, fontsize="small", color="dimgray")
        ax.set_ylabel("Median F0 (Hz)")
        if points:
            ax.legend()
        _add_caption(
            fig,
            "Line = mean of that quarter's clip medians. Shaded band ="
            " between-clip min–max\nwhere a quarter has 2+ clips (1 clip ="
            " a bare point, anecdotal).",
            _QC_FOOTER,
        )
        fig.savefig(out_dir / name)
        plt.close(fig)


def _year_month_sets(entries: list[dict]) -> tuple[dict[str, set], dict[str, set]]:
    """Per-year month sets computed from the raw entries: months holding
    at least one record, and months holding at least one QC-passing
    record. Entries with an unplaceable month contribute to neither."""
    with_records: dict[str, set] = {}
    with_pass: dict[str, set] = {}
    for entry in entries:
        year = _year(entry.get("month"))
        if year is None:
            continue
        with_records.setdefault(year, set()).add(entry["month"])
        qc_pass, _ = qc_verdict(entry.get("features") or {})
        if qc_pass:
            with_pass.setdefault(year, set()).add(entry["month"])
    return with_records, with_pass


def write_yearly_plot(
    entries: list[dict], out_dir: Path, *, talent: str | None = None
) -> None:
    """The single yearly PNG (PLAN L36 year rule), the QC view: a
    two-panel figure over the QC-pass years. TOP = per-year median of
    clip medians with the between-clip min–max spread as vlines (a n=1
    year renders as a bare point). BOTTOM = bars of QC-pass clip count
    with "n=X" labels and a per-year QC-pass-months / months-with-records
    annotation. Empty years are gaps; gaps and n=1 years do not crash."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    points = f0_yearly(entries, qc=True)
    with_records, with_pass = _year_month_sets(entries)
    years = [p["year"] for p in points]
    xs = list(range(len(points)))
    fig, (ax, axn) = plt.subplots(
        2, 1, sharex=True, figsize=(max(7.5, 1.1 * max(1, len(points))), 7.2),
        constrained_layout=True,
    )
    fig.suptitle(_title(talent, "Median Pitch (F0) by Year"), fontweight="bold")
    ax.set_title(_WHAT_IS_F0, fontsize="small", color="dimgray")
    medians = [p["median"] for p in points]
    ax.plot(xs, medians, marker="o", label=_QC_LABEL_PASS)
    spread_x = [x for x, p in zip(xs, points) if p["n"] >= 2]
    if spread_x:
        ax.vlines(
            spread_x,
            [points[x]["min"] for x in spread_x],
            [points[x]["max"] for x in spread_x],
            alpha=0.6,
            linewidth=2,
        )
    for x, p in zip(xs, points):
        text = (
            f"{p['median']:.0f} Hz\n[{p['min']:.0f}–{p['max']:.0f}]"
            if p["n"] >= 2
            else f"{p['median']:.0f} Hz"
        )
        ax.annotate(
            text, (x, p["max"] if p["n"] >= 2 else p["median"]),
            xytext=(0, 8), textcoords="offset points",
            ha="center", va="bottom", fontsize="x-small",
        )
    ax.set_ylabel("Median F0 (Hz)")
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)
    if points:
        ax.set_ylim(top=(max(p["max"] for p in points)) * 1.12)
        ax.legend()
    ns = [p["n"] for p in points]
    axn.bar(xs, ns, color="tab:blue")
    if points:
        head = max(ns) * 1.5 + 1
        axn.set_ylim(0, head)
        for x, n in zip(xs, ns):
            axn.annotate(
                f"n={n}", (x, n), xytext=(0, 2), textcoords="offset points",
                ha="center", va="bottom", fontsize="x-small",
            )
        for x, year in zip(xs, years):
            n_rec = len(with_records.get(year, ()))
            n_pass = len(with_pass.get(year, ()))
            axn.annotate(
                f"{n_pass}/{n_rec} mo covered", (x, head * 0.97), ha="center",
                va="top", fontsize="x-small", alpha=0.8,
            )
    axn.set_ylabel("QC-pass clips (n)")
    axn.set_xticks(xs)
    axn.set_xticklabels(years, rotation=45, ha="right")
    _add_caption(
        fig,
        "Point = median of that year's QC-pass clip medians; bar = between-clip\n"
        'spread (2+ clips only). Bottom: QC-pass clip count and months covered\n'
        "per year \u2014 the most recent year is usually partial.",
        _QC_FOOTER,
    )
    fig.savefig(out_dir / "f0_yearly.png")
    plt.close(fig)


def write_multi_talent_plot(talents: dict[str, list[dict]], out_dir: Path) -> None:
    """Cross-talent comparison, QC-pass only: one line per talent on a
    shared quarterly and yearly x-axis (the union of every talent's
    quarters/years). A talent's missing quarter/year is a genuine gap
    (NaN) — the line breaks there rather than connecting across it.
    Talent order is alphabetical, so color assignment is stable across
    runs regardless of dict insertion order."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(talents)

    per_talent_q = {
        name: {p["quarter"]: p["mean"] for p in f0_quarterly(talents[name], qc=True)}
        for name in names
    }
    all_quarters = sorted({q for d in per_talent_q.values() for q in d})
    xs = list(range(len(all_quarters)))
    fig, ax = plt.subplots(
        figsize=(max(9.0, 0.32 * len(all_quarters)), 5.5), constrained_layout=True
    )
    for name in names:
        ys = [per_talent_q[name].get(q, math.nan) for q in all_quarters]
        ax.plot(xs, ys, marker="o", label=name, markersize=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(all_quarters, rotation=45, ha="right")
    ax.set_ylabel("Median F0 (Hz)")
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.suptitle("Median Pitch (F0) by Quarter — Talent Comparison", fontweight="bold")
    ax.set_title(_WHAT_IS_F0, fontsize="small", color="dimgray")
    if all_quarters:
        ax.legend()
    _add_caption(
        fig,
        "QC-pass clips only, one line per talent — a line breaks where that\n"
        "talent has no QC-pass data that quarter (never interpolated).",
        _QC_FOOTER,
    )
    fig.savefig(out_dir / "f0_quarterly_multi.png")
    plt.close(fig)

    write_feature_multi_talent_yearly_plot(
        talents, out_dir,
        feature_key="median_f0", filename="f0_yearly_multi.png",
        subject="Median Pitch (F0) by Year — Talent Comparison",
        subtitle=_WHAT_IS_F0, unit_label="Median F0 (Hz)",
    )


def write_feature_multi_talent_yearly_plot(
    talents: dict[str, list[dict]],
    out_dir: Path,
    *,
    feature_key: str,
    filename: str,
    subject: str,
    subtitle: str,
    unit_label: str,
    caveat: str | None = None,
) -> None:
    """Cross-talent yearly comparison for any numeric feature (the
    generalization of write_multi_talent_plot's yearly panel — used
    for median_f0 there, and for brightness/dynamism/voice-quality/
    loudness by the CLI when a corpus actually has those keys). QC-pass
    only; a talent's missing year for this feature is a genuine gap
    (NaN), never interpolated across."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(talents)
    per_talent = {
        name: {
            p["year"]: p["median"]
            for p in f0_yearly(talents[name], qc=True, feature_key=feature_key)
        }
        for name in names
    }
    all_years = sorted({y for d in per_talent.values() for y in d})
    xs = list(range(len(all_years)))
    fig, ax = plt.subplots(
        figsize=(max(7.5, 1.1 * max(1, len(all_years))), 5.5), constrained_layout=True
    )
    for name in names:
        ys = [per_talent[name].get(y, math.nan) for y in all_years]
        ax.plot(xs, ys, marker="o", label=name, markersize=6)
        for x, y in zip(xs, ys):
            if not math.isnan(y):
                ax.annotate(
                    f"{y:.2f}", (x, y), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize="x-small",
                )
    ax.set_xticks(xs)
    ax.set_xticklabels(all_years, rotation=45, ha="right")
    ax.set_ylabel(unit_label)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.suptitle(subject, fontweight="bold")
    ax.set_title(subtitle, fontsize="small", color="dimgray")
    if all_years:
        all_vals = [y for d in per_talent.values() for y in d.values()]
        top, bottom = max(all_vals), min(all_vals)
        headroom = (top - bottom) * 0.15 or top * 0.05
        ax.set_ylim(bottom - headroom * 0.3, top + headroom)
        ax.legend()
    caption_lines = [
        "QC-pass clips only (median of that year's clip values), one line\n"
        "per talent — a gap means that talent has no QC-pass data that year."
    ]
    if caveat:
        caption_lines.append(caveat)
    caption_lines.append(_QC_FOOTER)
    _add_caption(fig, *caption_lines)
    fig.savefig(out_dir / filename)
    plt.close(fig)


def write_feature_yearly_plot(
    entries: list[dict],
    out_dir: Path,
    *,
    feature_key: str,
    filename: str,
    subject: str,
    subtitle: str,
    unit_label: str,
    talent: str | None = None,
    caveat: str | None = None,
) -> None:
    """A yearly plot for any numeric feature (brightness, dynamism,
    jitter/shimmer/HNR, loudness dynamics, ...) — same shape as
    write_yearly_plot's top panel (median + between-clip min–max) plus
    a QC-pass sample-size bar, generalized via feature_key. QC-pass
    still gates on the shared median_f0/f0_iqr/voiced_fraction rule
    (a trustworthiness filter, not specific to F0) so a feature computed
    from a QC-failing clip never appears here."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    points = f0_yearly(entries, qc=True, feature_key=feature_key)
    years = [p["year"] for p in points]
    xs = list(range(len(points)))
    fig, (ax, axn) = plt.subplots(
        2, 1, sharex=True, figsize=(max(7.5, 1.1 * max(1, len(points))), 7.2),
        constrained_layout=True, height_ratios=(3, 1),
    )
    fig.suptitle(_title(talent, subject), fontweight="bold")
    ax.set_title(subtitle, fontsize="small", color="dimgray")
    medians = [p["median"] for p in points]
    ax.plot(xs, medians, marker="o", label=_QC_LABEL_PASS)
    spread_x = [x for x, p in zip(xs, points) if p["n"] >= 2]
    if spread_x:
        ax.vlines(
            spread_x,
            [points[x]["min"] for x in spread_x],
            [points[x]["max"] for x in spread_x],
            alpha=0.6, linewidth=2,
        )
    for x, p in zip(xs, points):
        text = (
            f"{p['median']:.2f}\n[{p['min']:.2f}–{p['max']:.2f}]"
            if p["n"] >= 2 else f"{p['median']:.2f}"
        )
        ax.annotate(
            text, (x, p["max"] if p["n"] >= 2 else p["median"]),
            xytext=(0, 8), textcoords="offset points",
            ha="center", va="bottom", fontsize="x-small",
        )
    ax.set_ylabel(unit_label)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)
    if points:
        top = max(p["max"] for p in points)
        bottom = min(p["min"] for p in points)
        headroom = (top - bottom) * 0.15 or top * 0.05
        ax.set_ylim(bottom - headroom * 0.3, top + headroom)
        ax.legend()
    ns = [p["n"] for p in points]
    axn.bar(xs, ns, color="tab:blue")
    if points:
        head = max(ns) * 1.4 + 1
        axn.set_ylim(0, head)
        for x, n in zip(xs, ns):
            axn.annotate(
                f"n={n}", (x, n), xytext=(0, 2), textcoords="offset points",
                ha="center", va="bottom", fontsize="x-small",
            )
    axn.set_ylabel("QC-pass clips (n)")
    axn.set_xticks(xs)
    axn.set_xticklabels(years, rotation=45, ha="right")
    caption_lines = [
        f"Point = median of that year's QC-pass clip {feature_key} values; bar =\n"
        "between-clip min–max (2+ clips only). Bottom: QC-pass sample size per year."
    ]
    if caveat:
        caption_lines.append(caveat)
    caption_lines.append(_QC_FOOTER)
    _add_caption(fig, *caption_lines)
    fig.savefig(out_dir / filename)
    plt.close(fig)
