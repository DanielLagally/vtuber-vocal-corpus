"""`vvc site-data-v2` writes docs/data.js from the talent registry and
data/measurements/v2/, the same script-tag convention v1's `site-data`
uses (see tests/test_site_data_v2.py for the data-shape rules this wires
up)."""

from __future__ import annotations

import json
from pathlib import Path

from vvc import __main__ as cli


def _v2_rec(vid, month, f0):
    return {
        "id": vid,
        "month": month,
        "features": {
            "median_f0": f0,
            "f0_iqr_semitones": 4.0,
            "voiced_fraction": 0.5,
            "fraction_below_160hz": 0.01,
        },
        "qc": {"pass": True, "reason": None},
        "legacy": False,
        "reliability_band": "high",
    }


def test_site_data_v2_cli_writes_a_window_assignment(tmp_path: Path) -> None:
    measurements = tmp_path / "measurements"
    v2_dir = measurements / "v2"
    v2_dir.mkdir(parents=True)
    (measurements / "alpha_monthly.json").write_text("[]", encoding="utf-8")
    (v2_dir / "alpha.json").write_text(
        json.dumps([_v2_rec("a1", "2020-01", 300.0), _v2_rec("a2", "2020-02", 310.0)]),
        encoding="utf-8",
    )
    registry = tmp_path / "talents.json"
    registry.write_text(
        json.dumps({str(measurements / "alpha_monthly.json"): "Alpha Talent"}),
        encoding="utf-8",
    )
    out = tmp_path / "data.js"

    cli.main(
        [
            "site-data-v2",
            "--registry",
            str(registry),
            "--v2-dir",
            str(v2_dir),
            "--roster",
            str(tmp_path / "nonexistent-roster.json"),
            "-o",
            str(out),
        ]
    )

    text = out.read_text(encoding="utf-8")
    assert text.startswith("window.SITE_DATA_V2 = ")
    payload = json.loads(text[len("window.SITE_DATA_V2 = ") :].rstrip().rstrip(";"))
    assert payload["talents"]["Alpha Talent"]["n_clips"] == 2
    assert payload["talents"]["Alpha Talent"]["legacy_fallback"] is False
    assert [f["key"] for f in payload["families"]][0] == "pitch_level"
