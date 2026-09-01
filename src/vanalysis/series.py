from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

_IQR_QC_MAX = 200.0


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def f0_series(entries: list[dict], qc: bool = False) -> list[tuple[str, float]]:
    points: list[tuple[str, float]] = []
    for entry in entries:
        features = entry.get("features") or {}
        if qc:
            iqr = features.get("f0_iqr")
            if not _finite(iqr) or iqr >= _IQR_QC_MAX:
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


def _plot_points(ax: plt.Axes, points: list[tuple[str, float]], label: str) -> None:
    months = [month for month, _ in points]
    values = [value for _, value in points]
    ax.plot(months, values, marker="o", label=label)


def write_plots(entries: list[dict], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_points = f0_series(entries)
    qc_points = f0_series(entries, qc=True)
    fig, ax = plt.subplots()
    _plot_points(ax, all_points, "all")
    _plot_points(ax, qc_points, "qc (f0_iqr < 200)")
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
