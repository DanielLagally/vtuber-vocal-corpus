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


def f0_series(entries: list[dict], qc: bool = False) -> list[tuple[str, float]]:
    """Monthly median-F0 series, ONE point per calendar month (STATE R3:
    multi-clip months): the plain float mean (no rounding) of that
    month's finite ``median_f0`` values. With ``qc=True`` the shared QC
    rule filters the entries FIRST, then the survivors are averaged —
    a month with no surviving value is a gap in the QC series (it can
    still be present in the all-clip series). A single-record month is
    unchanged (mean of one is the value itself). Sorted by month."""
    buckets: dict[str, list[float]] = {}
    for entry in entries:
        features = entry.get("features") or {}
        if qc:
            qc_pass, _ = qc_verdict(features)
            if not qc_pass:
                continue
        median = features.get("median_f0")
        if not _finite(median):
            continue
        buckets.setdefault(entry["month"], []).append(float(median))
    points = [
        (month, sum(values) / len(values)) for month, values in buckets.items()
    ]
    points.sort(key=lambda point: point[0])
    return points


def iqr_series(entries: list[dict]) -> list[tuple[str, float]]:
    """Monthly F0-IQR series, ONE point per calendar month (STATE R3:
    multi-clip months): the plain float mean (no rounding) of that
    month's finite ``f0_iqr`` values. Non-finite IQRs contribute
    nothing; a single-record month is unchanged (mean of one). Sorted
    by month."""
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


def f0_quarterly(entries: list[dict], qc: bool = False) -> list[dict]:
    """Per calendar quarter (PLAN L36): the MEAN of that quarter's clip
    ``median_f0`` values, plus min and max of the same values, and n,
    sorted by quarter.

    A clip with a non-finite median is excluded ENTIRELY from its
    quarterly point (it cannot plot — it contributes to nothing, not
    even n); a quarter left without plottable clips is simply absent (a
    gap). With ``qc=True`` the shared QC rule filters the entries BEFORE
    aggregating, so nan-median, IQR >= 200, and median >= 600 clips all
    drop first. A n=1 quarter carries min == max == the single value
    (the plot decides band vs bare point — the aggregation stays pure).
    """
    buckets: dict[str, list[float]] = {}
    for entry in entries:
        features = entry.get("features") or {}
        if qc:
            qc_pass, _ = qc_verdict(features)
            if not qc_pass:
                continue
        median = features.get("median_f0")
        if not _finite(median):
            continue
        quarter = _quarter(entry.get("month"))
        if quarter is None:
            continue
        buckets.setdefault(quarter, []).append(float(median))
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


def f0_yearly(entries: list[dict], qc: bool = False) -> list[dict]:
    """Per calendar year (PLAN L36 year rule): the MEDIAN of that year's
    clip ``median_f0`` values (clip-level — not month-means first, the
    clip is the sample), plus min and max of the same values (the
    between-clip spread), and n, sorted by year.

    A clip with a non-finite median is excluded ENTIRELY from its year
    point (it cannot plot — it contributes to nothing, not even n); a
    year left without plottable clips is simply absent (a gap). With
    ``qc=True`` the shared QC rule filters the entries BEFORE
    aggregating, so nan-median, IQR >= 200, and median >= 600 clips all
    drop first. A n=1 year carries min == max == the single value.

    The median is the lower median for an even clip count (median_low):
    an even-count year reports an actually-observed clip median, not an
    interpolated midpoint — pinned by rule 10 ([300, 615] -> 300.0).
    """
    buckets: dict[str, list[float]] = {}
    for entry in entries:
        features = entry.get("features") or {}
        if qc:
            qc_pass, _ = qc_verdict(features)
            if not qc_pass:
                continue
        median = features.get("median_f0")
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
    ax.plot(months, values, marker="o", label=label)


def _plot_quarterly(ax: plt.Axes, points: list[dict], label: str) -> None:
    """Quarter points as means, with a min–max band ONLY where n >= 2 —
    a n=1 quarter is anecdotal (PLAN L36) and renders as a bare point.
    Empty quarters are absent from the series and so leave a natural
    gap on the x axis."""
    quarters = [p["quarter"] for p in points]
    xs = list(range(len(points)))
    ax.plot(xs, [p["mean"] for p in points], marker="o", label=label)
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


def write_plots(entries: list[dict], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_points = f0_series(entries)
    qc_points = f0_series(entries, qc=True)
    fig, ax = plt.subplots()
    _plot_points(ax, all_points, "all")
    _plot_points(ax, qc_points, "qc (finite f0, f0 < 600, iqr < 200)")
    ax.set_ylabel("median F0 (Hz)")
    if all_points or qc_points:
        ax.legend()
    fig.savefig(out_dir / "f0_monthly.png")
    plt.close(fig)
    iqr_points = iqr_series(entries)
    fig, ax = plt.subplots()
    _plot_points(ax, iqr_points, "all")
    ax.set_ylabel("F0 IQR (Hz)")
    if iqr_points:
        ax.legend()
    fig.savefig(out_dir / "f0_iqr_monthly.png")
    plt.close(fig)


def write_quarterly_plots(entries: list[dict], out_dir: Path) -> None:
    """The two quarterly PNGs (PLAN L36 quarter point): mean of clip
    medians + min–max band where n >= 2 (n=1 anecdotal, bare point),
    empty quarters as gaps. ``*_all`` shows QC-failing clips (cleaning
    visible), ``*_qc`` shows QC passes only (fails are gaps)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = (
        (
            "f0_quarterly_all.png",
            f0_quarterly(entries),
            "all (mean of clip medians; min-max band where n >= 2; n=1 anecdotal)",
        ),
        (
            "f0_quarterly_qc.png",
            f0_quarterly(entries, qc=True),
            "qc (finite f0, f0 < 600, iqr < 200; min-max band where n >= 2;"
            " n=1 anecdotal)",
        ),
    )
    for name, points, label in variants:
        fig, ax = plt.subplots()
        _plot_quarterly(ax, points, label)
        ax.set_ylabel("median F0 (Hz)")
        if points:
            ax.legend()
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


def write_yearly_plot(entries: list[dict], out_dir: Path) -> None:
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
        2, 1, sharex=True, figsize=(max(6.4, 1.1 * max(1, len(points))), 6.4),
        constrained_layout=True,
    )
    ax.plot(
        xs,
        [p["median"] for p in points],
        marker="o",
        label="qc (finite f0, f0 < 600, iqr < 200)",
    )
    spread_x = [x for x, p in zip(xs, points) if p["n"] >= 2]
    if spread_x:
        ax.vlines(
            spread_x,
            [points[x]["min"] for x in spread_x],
            [points[x]["max"] for x in spread_x],
            alpha=0.6,
            linewidth=2,
        )
    ax.set_ylabel("median F0 (Hz)")
    if points:
        ax.legend()
    ns = [p["n"] for p in points]
    axn.bar(xs, ns)
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
                f"{n_pass}/{n_rec} mo", (x, head * 0.97), ha="center",
                va="top", fontsize="x-small", alpha=0.8,
            )
    axn.set_ylabel("QC-pass clips")
    axn.set_xticks(xs)
    axn.set_xticklabels(years, rotation=45, ha="right")
    fig.supxlabel(
        "median of QC-pass clip medians per year; bar = between-clip"
        " min\u2013max; current year partial",
        fontsize="small",
    )
    fig.savefig(out_dir / "f0_yearly.png")
    plt.close(fig)
