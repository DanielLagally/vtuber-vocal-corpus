"""`vvc reliability` reports the corpus noise floor and discordance flags.

The library is only useful if something calls it. `catalog.reliability_band`
is the cautionary precedent in this repo: a correct classifier with zero call
sites, so the knowledge it produced never reached a single published number.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vvc import __main__ as cli


def _write_corpus(root: Path) -> Path:
    measurements = root / "measurements"
    measurements.mkdir(parents=True)

    def rec(vid: str, month: str, f0: float) -> dict:
        return {
            "id": vid,
            "month": month,
            "features": {
                "median_f0": f0,
                "f0_iqr": 100.0,
                "voiced_fraction": 0.5,
            },
            "qc": {"pass": True, "reason": None},
        }

    # One clean month, one month holding an octave-discordant pair.
    (measurements / "alpha_monthly.json").write_text(
        json.dumps(
            [
                rec("a1", "2020-01", 300.0),
                rec("a2", "2020-01", 310.0),
                rec("a3", "2020-02", 300.0),
                rec("a4", "2020-02", 150.0),
            ]
        ),
        encoding="utf-8",
    )
    return measurements


def test_reliability_cli_writes_noise_floor_and_flags(tmp_path: Path, capsys) -> None:
    measurements = _write_corpus(tmp_path)
    out = tmp_path / "reliability.json"

    cli.main(["reliability", "--measurements-dir", str(measurements), "--out", str(out)])

    payload = json.loads(out.read_text(encoding="utf-8"))

    floor = payload["noise_floor"]["median_f0"]
    assert floor["n_pairs"] == 2
    assert floor["median_abs_diff"] > 0

    flags = payload["flags"]
    assert len(flags) == 1
    assert flags[0]["talent"] == "alpha"
    assert flags[0]["month"] == "2020-02"
    assert flags[0]["semitones"] > 6.0

    assert "median_f0" in capsys.readouterr().out


def test_reliability_cli_never_pairs_clips_from_different_talents(tmp_path: Path) -> None:
    """Two people in the same month are not a replicate of anything."""
    measurements = tmp_path / "measurements"
    measurements.mkdir(parents=True)

    def rec(vid: str, f0: float) -> dict:
        return {
            "id": vid,
            "month": "2020-01",
            "features": {"median_f0": f0, "f0_iqr": 100.0, "voiced_fraction": 0.5},
            "qc": {"pass": True, "reason": None},
        }

    (measurements / "alpha_monthly.json").write_text(json.dumps([rec("a", 300.0)]))
    (measurements / "beta_monthly.json").write_text(json.dumps([rec("b", 150.0)]))

    out = tmp_path / "reliability.json"
    cli.main(["reliability", "--measurements-dir", str(measurements), "--out", str(out)])

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["noise_floor"]["median_f0"]["n_pairs"] == 0
    assert payload["flags"] == []


def test_reliability_cli_reads_the_v2_layout_too(tmp_path: Path) -> None:
    """v1 files are <talent>_monthly.json, v2 files are <talent>.json. Globbing
    only the v1 pattern reported an empty corpus instead of failing."""
    measurements = tmp_path / "v2"
    measurements.mkdir(parents=True)

    def rec(vid: str, f0: float) -> dict:
        return {
            "id": vid,
            "month": "2020-01",
            "features": {"median_f0": f0, "f0_iqr": 100.0, "voiced_fraction": 0.5},
            "qc": {"pass": True, "reason": None},
        }

    (measurements / "alpha.json").write_text(json.dumps([rec("a", 300.0), rec("b", 312.0)]))
    out = tmp_path / "r.json"
    cli.main(["reliability", "--measurements-dir", str(measurements), "--out", str(out)])

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["noise_floor"]["median_f0"]["n_pairs"] == 1


def test_reliability_cli_errors_on_an_empty_directory(tmp_path: Path) -> None:
    """Silently reporting a zero noise floor would read as a perfect instrument."""
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(SystemExit):
        cli.main(
            ["reliability", "--measurements-dir", str(empty), "--out", str(tmp_path / "r.json")]
        )


def test_reliability_cli_is_read_only(tmp_path: Path) -> None:
    """An audit tool that edits the corpus it audits is not an audit tool."""
    measurements = _write_corpus(tmp_path)
    target = measurements / "alpha_monthly.json"
    before = target.read_text(encoding="utf-8")

    cli.main(
        [
            "reliability",
            "--measurements-dir",
            str(measurements),
            "--out",
            str(tmp_path / "r.json"),
        ]
    )

    assert target.read_text(encoding="utf-8") == before
