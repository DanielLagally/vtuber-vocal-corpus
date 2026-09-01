"""CLI ``plot``'s auto-registry / auto-comparison behavior.

User-visible rules:

1. ``plot --measurements PATH --talent NAME --registry R`` records
   ``{str(PATH): NAME}`` into the registry file R (creating it if
   absent).
2. With FEWER than 2 talents registered, no comparison plots are
   written (a "comparison" of one talent against itself is pointless
   clutter) — only that talent's own single-talent plots exist.
3. Once a SECOND talent is registered (by any prior or current `plot`
   run), a subsequent `plot` run for either talent regenerates the
   cross-talent comparison plots (f0_yearly_multi.png at minimum) into
   its OWN run directory, alongside its own single-talent plots.
4. ``--no-compare`` skips both the registry update and the comparison
   regeneration entirely.
5. A registered talent whose measurements file has since disappeared
   is skipped (warning, not a crash) when building the comparison —
   the run still completes using whichever registered talents remain.
"""

from __future__ import annotations

import json
from pathlib import Path

from vanalysis import __main__ as cli

MODEL = "bs_roformer_vocals_resurrection_unwa.ckpt"


def _record(vid: str, month: str, median_f0: float) -> dict:
    return {
        "id": vid,
        "month": month,
        "score": 70.0,
        "window": {"start_s": 0.0, "end_s": 90.0},
        "features": {"median_f0": median_f0, "f0_iqr": 45.0, "voiced_fraction": 0.6},
        "qc": {"pass": True, "reason": None},
        "model": MODEL,
    }


def _write_measurements(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records) + "\n", encoding="utf-8")


def _plot_argv(measurements: Path, talent: str, out_dir: Path, registry: Path, *extra: str) -> list[str]:
    return [
        "plot",
        "--measurements", str(measurements),
        "--talent", talent,
        "--out-dir", str(out_dir),
        "--registry", str(registry),
        *extra,
    ]


def test_plot_registers_first_talent_no_comparison_yet(tmp_path: Path) -> None:
    """Rule 1 + 2: a single registered talent gets its own plots and a
    registry entry, but no comparison plot (nothing to compare against)."""
    measurements = tmp_path / "a_monthly.json"
    _write_measurements(measurements, [_record("a0000001", "2024-01", 300.0)])
    out_dir = tmp_path / "plots"
    registry = tmp_path / "talents.json"

    cli.main(_plot_argv(measurements, "Talent A", out_dir, registry))

    assert json.loads(registry.read_text()) == {str(measurements): "Talent A"}
    [run_dir] = list((out_dir / "runs").iterdir())
    assert (run_dir / "f0_yearly.png").is_file()
    assert not (run_dir / "f0_yearly_multi.png").exists()


def test_plot_second_talent_triggers_comparison_in_both_runs(tmp_path: Path) -> None:
    """Rule 3: once a 2nd talent is registered, comparison plots appear
    in that run's own directory."""
    measurements_a = tmp_path / "a_monthly.json"
    measurements_b = tmp_path / "b_monthly.json"
    _write_measurements(measurements_a, [_record("a0000001", "2024-01", 300.0)])
    _write_measurements(measurements_b, [_record("b0000001", "2024-01", 250.0)])
    out_dir = tmp_path / "plots"
    registry = tmp_path / "talents.json"

    cli.main(_plot_argv(measurements_a, "Talent A", out_dir, registry))
    cli.main(_plot_argv(measurements_b, "Talent B", out_dir, registry))

    assert json.loads(registry.read_text()) == {
        str(measurements_a): "Talent A", str(measurements_b): "Talent B",
    }
    run_dirs = sorted((out_dir / "runs").iterdir())
    assert len(run_dirs) == 2
    # the SECOND run (Talent B's) must carry the comparison plot
    assert (run_dirs[1] / "f0_yearly_multi.png").is_file()


def test_plot_no_compare_skips_registry_and_comparison(tmp_path: Path) -> None:
    """Rule 4: --no-compare touches neither the registry nor comparison
    plots, even with a talent name given."""
    measurements = tmp_path / "a_monthly.json"
    _write_measurements(measurements, [_record("a0000001", "2024-01", 300.0)])
    out_dir = tmp_path / "plots"
    registry = tmp_path / "talents.json"

    cli.main(_plot_argv(measurements, "Talent A", out_dir, registry, "--no-compare"))

    assert not registry.exists()


def test_plot_skips_missing_registered_talent_without_crashing(tmp_path: Path) -> None:
    """Rule 5: a registered talent whose file vanished is skipped, not
    fatal — the comparison still runs on the remaining registered
    talents."""
    measurements_a = tmp_path / "a_monthly.json"
    measurements_b = tmp_path / "b_monthly.json"
    measurements_c = tmp_path / "c_monthly.json"
    _write_measurements(measurements_a, [_record("a0000001", "2024-01", 300.0)])
    _write_measurements(measurements_b, [_record("b0000001", "2024-01", 250.0)])
    out_dir = tmp_path / "plots"
    registry = tmp_path / "talents.json"

    cli.main(_plot_argv(measurements_a, "Talent A", out_dir, registry))
    cli.main(_plot_argv(measurements_b, "Talent B", out_dir, registry))
    measurements_b.unlink()  # Talent B's file is now gone, but still registered

    _write_measurements(measurements_c, [_record("c0000001", "2024-01", 275.0)])
    cli.main(_plot_argv(measurements_c, "Talent C", out_dir, registry))  # must not raise

    run_dirs = sorted((out_dir / "runs").iterdir())
    assert (run_dirs[-1] / "f0_yearly_multi.png").is_file()
