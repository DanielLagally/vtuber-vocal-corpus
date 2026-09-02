from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from pathlib import Path

_DEFAULT_PRESET = "vocal_balanced"
DEFAULT_MODEL_FILENAME = "bs_roformer_vocals_resurrection_unwa.ckpt"

_REPEATED_UNDERSCORE_RE = re.compile(r"_+")


def _sanitize_stem(name: str) -> str:
    """Mirrors audio-separator's own CommonSeparator.sanitize_filename
    (collapse repeated "_", then strip leading/trailing "_. ") so our
    predicted output path matches what it actually writes — a video id
    can legitimately start with "_", and audio-separator strips that
    before naming its output file."""
    sanitized = _REPEATED_UNDERSCORE_RE.sub("_", name)
    return sanitized.strip("_. ")


def _model_base(model_filename: str) -> str:
    return Path(model_filename).stem


def _stem_name(src: Path | str, model_filename: str | None) -> str:
    stem = _sanitize_stem(Path(src).stem)
    if model_filename is None:
        return f"{stem}_(Vocals)_preset_{_DEFAULT_PRESET}.wav"
    return f"{stem}_(vocals)_{_model_base(model_filename)}.wav"


def vocals_path(
    src: Path | str, out_dir: Path, *, model_filename: str | None = None
) -> Path:
    return Path(out_dir) / _stem_name(src, model_filename)


def _default_runner(argv: list[str]) -> object:
    return subprocess.run(argv, check=True)


def isolate_vocals(
    src: Path | str,
    out_dir: Path,
    *,
    model_filename: str | None = None,
    model_file_dir: str | None = None,
    runner: Callable[[list[str]], object] | None = None,
) -> Path:
    src_path = Path(src)
    dest_dir = Path(out_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    run = runner if runner is not None else _default_runner
    argv = [
        "audio-separator",
        "--output_format=wav",
        "--output_dir",
        str(dest_dir),
    ]
    if model_filename is None:
        argv.extend(["--ensemble_preset", _DEFAULT_PRESET])
    else:
        argv.extend(["--model_filename", model_filename])
    if model_file_dir is not None:
        argv.extend(["--model_file_dir", str(model_file_dir)])
    argv.append(str(src_path))
    run(argv)
    return vocals_path(src_path, dest_dir, model_filename=model_filename)
