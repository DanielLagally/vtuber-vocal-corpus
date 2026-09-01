from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .qc import qc_verdict


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def f0_series(entries: list[dict], qc: bool = False) -> list[tuple[str, float]]:
    points: list[tuple[str, float]] = []
    for entry in entries:
        features = entry.get("features") or {}
        if qc:
            qc_pass, _ = qc_verdict(features)
            if not qc_pass:
                continue
        median = features.get("median_f0")
        if not _finite(median):
            continue
        points.append((entry["month"], float(median)))
    points.sort(key=lambda point: point[0])
    return points


def iqr_series(entries: list[dict]) -> list[tuple[str, float]]:
    points: list[tuple[str, float]] = []
    for entry in entries:
        iqr = (entry.get("features") or {}).get("f0_iqr")
        if not _finite(iqr):
            continue
        points.append((entry["month"], float(iqr)))
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
