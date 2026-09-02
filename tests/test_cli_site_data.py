"""The `site-data` CLI command must load the talent registry and write
vanalysis.site_data's export to --out (default docs/data.js)."""

from __future__ import annotations

from pathlib import Path

from vanalysis import __main__ as cli


def test_site_data_cli_writes_export_from_registry(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "talents.json"
    registry_path.write_text('{"a.json": "Talent A"}', encoding="utf-8")
    out_path = tmp_path / "site" / "data.json"
    captured: dict = {}

    def fake_write_site_data(registry, out, **kwargs):
        captured["registry"] = registry
        captured["out"] = out
        return {"talents": {"Talent A": {}}, "cute_mature": {}, "generated_at": "x"}

    monkeypatch.setattr(cli, "write_site_data", fake_write_site_data)

    cli.main(
        [
            "site-data",
            "--registry",
            str(registry_path),
            "--out",
            str(out_path),
        ]
    )

    assert captured["registry"] == {"a.json": "Talent A"}
    assert captured["out"] == out_path


def test_site_data_cli_default_paths(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    Path("data/measurements").mkdir(parents=True)
    Path("data/measurements/talents.json").write_text("{}", encoding="utf-8")
    captured: dict = {}

    def fake_write_site_data(registry, out, **kwargs):
        captured["out"] = out
        return {"talents": {}, "cute_mature": {}, "generated_at": "x"}

    monkeypatch.setattr(cli, "write_site_data", fake_write_site_data)

    cli.main(["site-data"])

    assert captured["out"] == Path("docs/data.js")
