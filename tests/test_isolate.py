"""Product tests for vocal stem isolation.

User-visible rules (fake runner only — pytest must never invoke the real
audio-separator CLI, never download models, never touch Cover/hololive
audio; everything lives in tmp_path):

1. ``vocals_path(src, out_dir)`` is exactly
   ``out_dir / f"{Path(src).stem}_(Vocals)_preset_vocal_balanced.wav"``
   (audio-separator naming, lunalearn ``--ensemble_preset vocal_balanced``).
2. ``isolate_vocals(src, out_dir, *, model_filename=None, runner=None)``
   (legacy preset path, model_filename=None):
   a. creates ``out_dir`` if it does not exist yet,
   b. invokes ``runner`` with a list of argv whose first element is
      ``"audio-separator"`` and that includes the source path (as str),
      ``--output_format=wav``, and the vocal_balanced preset,
   c. returns exactly ``vocals_path(src, out_dir)`` — never a path outside
      ``out_dir`` — and that path exists after the call.
3. The default runner would subprocess the real CLI; these tests always
   pass a fake runner so pytest never runs the CLI and never downloads
   models. The fake only materializes an empty wav at the contracted path
   (isolate_vocals may create it itself, or leave it to the runner —
   either way the returned path must exist afterwards).
4. Exact signatures: ``vocals_path(src, out_dir, *, model_filename=None)``;
   ``isolate_vocals(src, out_dir, *, model_filename=None,
   model_file_dir=None, runner=None)``.
   Both accept Path | str for src, and the stem derives from the source
   filename.
5. Single-model path: ``model_filename="bs_roformer_vocals_resurrection_unwa.ckpt"``
   swaps the ensemble preset for ``--model_filename <ckpt>`` in the argv
   (NO ``--ensemble_preset`` anywhere) and the output name becomes
   ``<stem>_(vocals)_bs_roformer_vocals_resurrection_unwa.wav`` — ckpt
   stripped, lowercase "(vocals)" — exactly the data/stems_fast naming.
   The legacy ``model_filename=None`` path keeps rule 1/2b unchanged.
6. Both paths still create ``out_dir`` and keep ``runner`` keyword-only.
7. ``model_file_dir`` (audio-separator 0.47 ``--model_file_dir``): when
   passed, the argv carries ``--model_file_dir <dir>`` so the model ckpt
   cache lands in an in-tree dir instead of audio-separator's default
   ``/tmp/audio-separator-models/``; when not passed (None) the argv has
   NO ``--model_file_dir`` at all — default behavior unchanged. It
   influences only the argv, never the output naming (``vocals_path`` is
   unaffected).
"""

import inspect
from pathlib import Path

import pytest

from vanalysis import isolate

SRC_NAME = "chat_stream.wav"
EXPECTED_NAME = "chat_stream_(Vocals)_preset_vocal_balanced.wav"
MODEL_CKPT = "bs_roformer_vocals_resurrection_unwa.ckpt"
EXPECTED_FAST_NAME = "chat_stream_(vocals)_bs_roformer_vocals_resurrection_unwa.wav"


# ---------------------------------------------------------------- helpers


def _make_fake_runner(calls: list[list[str]], wav_path: Path):
    """An audio-separator stand-in: records argv, materializes an empty wav
    at the expected output path if the implementation did not already
    create it."""

    def runner(argv: list[str]) -> None:
        calls.append(list(argv))
        if not wav_path.exists():
            wav_path.write_bytes(b"")

    return runner


def _make_src(tmp_path: Path) -> Path:
    """A placeholder source file in tmp_path; nothing ever reads its
    contents, the argv contract only needs the path."""
    src = tmp_path / SRC_NAME
    src.write_bytes(b"")
    return src


def _assert_under_out_dir(result: Path, out_dir: Path) -> None:
    result_real = result.resolve()
    out_real = out_dir.resolve()
    assert result_real.is_relative_to(out_real), (
        f"{result} is outside out_dir {out_dir}"
    )


def _argv_option_value(argv: list[str], option: str) -> str | None:
    """The value of ``option`` in argv, either "--opt value" or
    "--opt=value" style; None when the option is absent."""
    for i, part in enumerate(argv):
        if part == option and i + 1 < len(argv):
            return argv[i + 1]
        if part.startswith(f"{option}="):
            return part.split("=", 1)[1]
    return None


# ------------------------------------------------------------------- tests


def test_vocals_path_layout(tmp_path: Path) -> None:
    """Rule 1: vocals_path is out_dir/<stem>_(Vocals)_preset_vocal_balanced.wav."""
    src = _make_src(tmp_path)
    assert isolate.vocals_path(src, tmp_path / "stems") == (
        tmp_path / "stems" / EXPECTED_NAME
    )


def test_vocals_path_stem_derives_from_source_filename(tmp_path: Path) -> None:
    """Rule 1: the stem comes from the source filename, not the directory —
    a src nested in subdirectories still maps onto out_dir directly."""
    src = tmp_path / "audio" / SRC_NAME
    out_dir = tmp_path / "stems"
    assert isolate.vocals_path(src, out_dir) == out_dir / EXPECTED_NAME


def test_vocals_path_accepts_str_src(tmp_path: Path) -> None:
    """Rule 4: vocals_path accepts Path | str for src."""
    src = _make_src(tmp_path)
    assert isolate.vocals_path(str(src), tmp_path / "stems") == (
        isolate.vocals_path(src, tmp_path / "stems")
    )


def test_isolate_vocals_calls_runner_with_audio_separator_argv(tmp_path: Path) -> None:
    """Rule 2b: the runner gets an argv list led by "audio-separator" that
    includes the source path (as str), --output_format=wav, and the
    vocal_balanced preset."""
    src = _make_src(tmp_path)
    out_dir = tmp_path / "stems"
    calls: list[list[str]] = []
    runner = _make_fake_runner(calls, isolate.vocals_path(src, out_dir))

    isolate.isolate_vocals(src, out_dir, runner=runner)

    assert calls, "isolate_vocals must invoke the runner"
    assert len(calls) == 1, f"runner should be invoked exactly once, got {len(calls)}"
    argv = calls[0]
    assert isinstance(argv, list), "runner must receive a list of argv, not a string"
    assert argv[0] == "audio-separator", f"argv[0] must be 'audio-separator', got {argv!r}"
    assert str(src) in argv, f"source path {str(src)!r} (as str) missing from argv {argv!r}"
    assert "--output_format=wav" in argv, f"--output_format=wav missing from argv {argv!r}"
    assert any("vocal_balanced" in part for part in argv), (
        f"vocal_balanced preset missing from argv {argv!r}"
    )


def test_isolate_vocals_creates_out_dir_and_returns_vocals_path(tmp_path: Path) -> None:
    """Rule 2a + 2c: out_dir is created (even if it did not exist), and the
    returned path is exactly vocals_path(...) and exists after the call."""
    src = _make_src(tmp_path)
    out_dir = tmp_path / "stems"
    assert not out_dir.exists(), "precondition: out_dir starts absent"
    calls: list[list[str]] = []
    runner = _make_fake_runner(calls, isolate.vocals_path(src, out_dir))

    result = isolate.isolate_vocals(src, out_dir, runner=runner)

    expected = out_dir / EXPECTED_NAME
    assert out_dir.is_dir(), "isolate_vocals must create out_dir"
    assert result == expected, f"expected {expected}, got {result}"
    assert result == isolate.vocals_path(src, out_dir)
    assert result.exists(), "the returned wav path must exist after isolate_vocals"


def test_isolate_vocals_result_stays_under_out_dir(tmp_path: Path) -> None:
    """Rule 2c: the result path never escapes out_dir (no repo-root or
    cwd writes)."""
    src = _make_src(tmp_path)
    out_dir = tmp_path / "somewhere" / "vanalysis-stems"
    calls: list[list[str]] = []
    runner = _make_fake_runner(calls, isolate.vocals_path(src, out_dir))

    result = isolate.isolate_vocals(src, out_dir, runner=runner)

    assert result.parent == out_dir, f"expected {result} directly in {out_dir}"
    _assert_under_out_dir(result, out_dir)


def test_isolate_vocals_accepts_str_src(tmp_path: Path) -> None:
    """Rule 4: isolate_vocals accepts Path | str for src."""
    src = _make_src(tmp_path)
    out_dir = tmp_path / "stems"
    calls: list[list[str]] = []
    runner = _make_fake_runner(calls, isolate.vocals_path(src, out_dir))

    result = isolate.isolate_vocals(str(src), out_dir, runner=runner)

    assert result == isolate.vocals_path(src, out_dir)
    assert result.exists()
    assert str(src) in calls[0], f"source path (as str) missing from argv {calls[0]!r}"


def test_isolate_vocals_runner_is_keyword_only(tmp_path: Path) -> None:
    """Rule 4: runner must not be passable positionally."""
    src = _make_src(tmp_path)
    out_dir = tmp_path / "stems"
    calls: list[list[str]] = []
    runner = _make_fake_runner(calls, isolate.vocals_path(src, out_dir))

    with pytest.raises(TypeError):
        isolate.isolate_vocals(src, out_dir, runner)  # type: ignore[misc]


def test_signatures_match_contract() -> None:
    """Rules 4+6: exact signatures —
    vocals_path(src, out_dir, *, model_filename=None);
    isolate_vocals(src, out_dir, *, model_filename=None,
    model_file_dir=None, runner=None)."""
    vp = inspect.signature(isolate.vocals_path)
    assert list(vp.parameters) == ["src", "out_dir", "model_filename"]
    assert all(
        vp.parameters[name].default is inspect.Parameter.empty
        for name in ("src", "out_dir")
    ), "src and out_dir must be required positional parameters"
    mf = vp.parameters["model_filename"]
    assert mf.kind is inspect.Parameter.KEYWORD_ONLY, (
        "vocals_path model_filename must be keyword-only"
    )
    assert mf.default is None, "vocals_path model_filename must default to None"

    iv = inspect.signature(isolate.isolate_vocals)
    assert list(iv.parameters) == [
        "src", "out_dir", "model_filename", "model_file_dir", "runner",
    ]
    assert all(
        iv.parameters[name].default is inspect.Parameter.empty
        for name in ("src", "out_dir")
    ), "src and out_dir must be required positional parameters"
    for name in ("model_filename", "model_file_dir", "runner"):
        param = iv.parameters[name]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"isolate_vocals {name} must be keyword-only"
        )
        assert param.default is None, f"isolate_vocals {name} must default to None"


# --------------------------------------------------- single-model path


def test_vocals_path_single_model_strips_ckpt_and_lowercases_vocals(
    tmp_path: Path,
) -> None:
    """Rule 5: with the single-model ckpt the output name is
    <stem>_(vocals)_bs_roformer_vocals_resurrection_unwa.wav — ckpt
    stripped, lowercase "(vocals)" — the data/stems_fast naming."""
    src = _make_src(tmp_path)
    result = isolate.vocals_path(
        src, tmp_path / "stems_fast", model_filename=MODEL_CKPT
    )
    assert result == tmp_path / "stems_fast" / EXPECTED_FAST_NAME, (
        f"expected the stems_fast name, got {result}"
    )


def test_isolate_vocals_single_model_argv_uses_model_filename_not_preset(
    tmp_path: Path,
) -> None:
    """Rule 5: the single-model argv carries
    --model_filename bs_roformer_vocals_resurrection_unwa.ckpt and NO
    --ensemble_preset anywhere; the runner still gets "audio-separator"
    plus the source path."""
    src = _make_src(tmp_path)
    out_dir = tmp_path / "stems_fast"
    calls: list[list[str]] = []
    runner = _make_fake_runner(
        calls, isolate.vocals_path(src, out_dir, model_filename=MODEL_CKPT)
    )

    isolate.isolate_vocals(src, out_dir, model_filename=MODEL_CKPT, runner=runner)

    assert calls, "isolate_vocals must invoke the runner"
    assert len(calls) == 1, f"runner should be invoked exactly once, got {len(calls)}"
    argv = calls[0]
    assert argv[0] == "audio-separator", f"argv[0] must be 'audio-separator', got {argv!r}"
    assert str(src) in argv, f"source path {str(src)!r} (as str) missing from argv {argv!r}"
    assert _argv_option_value(argv, "--model_filename") == MODEL_CKPT, (
        f"--model_filename {MODEL_CKPT!r} missing from argv {argv!r}"
    )
    assert not any("ensemble_preset" in part for part in argv), (
        f"single-model path must not carry --ensemble_preset, got {argv!r}"
    )


def test_isolate_vocals_single_model_returns_fast_stem_and_creates_out_dir(
    tmp_path: Path,
) -> None:
    """Rules 5+6: the single-model call creates out_dir and returns exactly
    out_dir/<stem>_(vocals)_<model>.wav, which exists afterwards."""
    src = _make_src(tmp_path)
    out_dir = tmp_path / "stems_fast"
    assert not out_dir.exists(), "precondition: out_dir starts absent"
    calls: list[list[str]] = []
    runner = _make_fake_runner(
        calls, isolate.vocals_path(src, out_dir, model_filename=MODEL_CKPT)
    )

    result = isolate.isolate_vocals(src, out_dir, model_filename=MODEL_CKPT, runner=runner)

    expected = out_dir / EXPECTED_FAST_NAME
    assert out_dir.is_dir(), "isolate_vocals must create out_dir (single-model too)"
    assert result == expected, f"expected {expected}, got {result}"
    assert result == isolate.vocals_path(src, out_dir, model_filename=MODEL_CKPT)
    assert result.exists(), "the returned wav path must exist after isolate_vocals"


# ------------------------------------------------------ model_file_dir option


def test_isolate_vocals_model_file_dir_lands_in_argv_when_passed(
    tmp_path: Path,
) -> None:
    """Rule 7: with model_file_dir set, the argv carries
    --model_file_dir <dir> (audio-separator 0.47 flag) so the RoFormer
    ckpt cache is pinned to an in-tree dir."""
    src = _make_src(tmp_path)
    out_dir = tmp_path / "stems_fast"
    models_dir = str(tmp_path / "models")
    calls: list[list[str]] = []
    runner = _make_fake_runner(
        calls, isolate.vocals_path(src, out_dir, model_filename=MODEL_CKPT)
    )

    isolate.isolate_vocals(
        src, out_dir,
        model_filename=MODEL_CKPT, model_file_dir=models_dir, runner=runner,
    )

    argv = calls[0]
    assert _argv_option_value(argv, "--model_file_dir") == models_dir, (
        f"--model_file_dir {models_dir!r} missing from argv {argv!r}"
    )


def test_isolate_vocals_model_file_dir_works_on_both_model_paths(
    tmp_path: Path,
) -> None:
    """Rule 7: the legacy preset path (model_filename=None) takes
    --model_file_dir too."""
    src = _make_src(tmp_path)
    out_dir = tmp_path / "stems"
    models_dir = str(tmp_path / "models")
    calls: list[list[str]] = []
    runner = _make_fake_runner(calls, isolate.vocals_path(src, out_dir))

    isolate.isolate_vocals(src, out_dir, model_file_dir=models_dir, runner=runner)

    argv = calls[0]
    assert _argv_option_value(argv, "--model_file_dir") == models_dir
    assert any("vocal_balanced" in part for part in argv), (
        "the preset path must keep --ensemble_preset when model_file_dir is set"
    )


def test_isolate_vocals_default_argv_has_no_model_file_dir(tmp_path: Path) -> None:
    """Rule 7: default behavior unchanged — without model_file_dir the
    argv carries NO --model_file_dir at all (audio-separator uses its own
    default model dir), on both the single-model and the preset path."""
    src = _make_src(tmp_path)
    calls: list[list[str]] = []

    runner = _make_fake_runner(
        calls, isolate.vocals_path(src, tmp_path / "s1", model_filename=MODEL_CKPT)
    )
    isolate.isolate_vocals(
        src, tmp_path / "s1", model_filename=MODEL_CKPT, runner=runner
    )
    assert not any("model_file_dir" in part for part in calls[0]), (
        f"default single-model argv must not carry --model_file_dir: {calls[0]!r}"
    )

    runner2 = _make_fake_runner(
        calls, isolate.vocals_path(src, tmp_path / "s2")
    )
    isolate.isolate_vocals(src, tmp_path / "s2", runner=runner2)
    assert not any("model_file_dir" in part for part in calls[-1]), (
        f"default preset argv must not carry --model_file_dir: {calls[-1]!r}"
    )


def test_isolate_vocals_model_file_dir_never_changes_output_name(
    tmp_path: Path,
) -> None:
    """Rule 7: the model cache dir influences only the argv — the output
    name is still vocals_path(...) with the same model stem."""
    src = _make_src(tmp_path)
    out_dir = tmp_path / "stems_fast"
    calls: list[list[str]] = []
    runner = _make_fake_runner(
        calls, isolate.vocals_path(src, out_dir, model_filename=MODEL_CKPT)
    )

    result = isolate.isolate_vocals(
        src, out_dir,
        model_filename=MODEL_CKPT, model_file_dir=str(tmp_path / "models"),
        runner=runner,
    )

    expected = out_dir / EXPECTED_FAST_NAME
    assert result == expected, f"expected {expected}, got {result}"
    assert result.exists()
