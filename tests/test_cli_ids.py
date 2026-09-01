"""CLI ``--ids-file`` behavior (user-visible rules).

``retry --ids-file`` must accept dash-leading video ids (argparse rejects
them as positionals/unrecognized options) and must union file ids with
``--ids`` exactly the way fetch/window/isolate union positional ids with
``--ids-file``: positional ids first, then file lines in order, blank lines
skipped. An explicitly given but empty id set is a clean argparse error —
never a silent fall-through to the default. Omitting both ``--ids`` and
``--ids-file`` keeps the documented default (every record whose qc.pass is
false, passed through as ``None``).

The window command's ``--ids-file`` path is pinned here too: a file with a
dash-leading id produces the ``<id>_raw90.wav`` slice and a ``windows.json``
entry keyed by that id.
"""

from __future__ import annotations

import array
import json
import wave
from pathlib import Path

import pytest

from vanalysis import __main__ as cli


def _write_ids(tmp_path: Path, lines: list[str]) -> Path:
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ids_file


def _run_retry_spy(monkeypatch: pytest.MonkeyPatch) -> dict:
    seen: dict = {}

    def fake_run_retry(ids, data_dir, **kwargs):
        seen["ids"] = ids
        return {"ids": {}, "counts": {"total": 0}}

    monkeypatch.setattr(cli, "run_retry", fake_run_retry)
    return seen


def _retry_argv(tmp_path: Path, extra: list[str]) -> list[str]:
    return [
        "retry",
        "--measurements", str(tmp_path / "m.json"),
        "--windows", str(tmp_path / "w.json"),
        "--data-dir", str(tmp_path / "data"),
        "--stems-dir", str(tmp_path / "stems"),
        *extra,
    ]


def test_retry_ids_file_accepts_dash_leading_ids(tmp_path, monkeypatch):
    ids_file = _write_ids(tmp_path, ["-DwvlhziHBI", "", "YcNa8qRKnLE"])
    seen = _run_retry_spy(monkeypatch)
    cli.main(_retry_argv(tmp_path, ["--ids-file", str(ids_file)]))
    assert seen["ids"] == ["-DwvlhziHBI", "YcNa8qRKnLE"]


def test_retry_ids_file_unions_with_ids_flag_in_order(tmp_path, monkeypatch):
    ids_file = _write_ids(tmp_path, ["", "CUx63C9SkW8"])
    seen = _run_retry_spy(monkeypatch)
    cli.main(
        _retry_argv(tmp_path, ["--ids", "-DwvlhziHBI", "--ids-file", str(ids_file)])
    )
    assert seen["ids"] == ["-DwvlhziHBI", "CUx63C9SkW8"]


def test_retry_without_ids_or_file_passes_none_default(tmp_path, monkeypatch):
    seen = _run_retry_spy(monkeypatch)
    cli.main(_retry_argv(tmp_path, []))
    assert seen["ids"] is None


def test_retry_empty_ids_file_is_a_clean_error(tmp_path):
    ids_file = _write_ids(tmp_path, ["", "   "])
    with pytest.raises(SystemExit) as excinfo:
        cli.main(_retry_argv(tmp_path, ["--ids-file", str(ids_file)]))
    assert excinfo.value.code == 2


def test_window_ids_file_handles_dash_leading_id_end_to_end(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    src = data_dir / "audio" / "-DwvlhziHBI.wav"
    src.parent.mkdir(parents=True)
    with wave.open(str(src), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(array.array("h", [0] * 16000).tobytes())
    ids_file = _write_ids(tmp_path, ["-DwvlhziHBI"])

    monkeypatch.setattr(
        cli, "best_speech_window", lambda src, window_s=90.0, hop_s=15.0: (0.0, 1.0)
    )
    sliced: list[tuple[str, str]] = []

    def fake_slice(src_path, dest, start_s, end_s):
        sliced.append((Path(src_path).name, Path(dest).name))

    monkeypatch.setattr(cli, "slice_wav", fake_slice)

    cli.main(["window", "--data-dir", str(data_dir), "--ids-file", str(ids_file)])

    assert sliced == [("-DwvlhziHBI.wav", "-DwvlhziHBI_raw90.wav")]
    index = json.loads(
        (data_dir / "windows" / "windows.json").read_text(encoding="utf-8")
    )
    assert index["-DwvlhziHBI"] == {"start_s": 0.0, "end_s": 1.0}
