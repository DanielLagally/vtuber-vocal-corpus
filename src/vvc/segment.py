"""A named stretch of one stream, measured as a one-off data point.

The corpus proper takes one 90 s window per stream, picked by a speech score,
so that each month's clips are comparable. A one-off answers a different
question — how does this talent sound *here* (another language, a special
stream) — so the operator names the stretch and every consecutive 90 s of it
is measured. Several windows show how much the voice varies inside the stretch
instead of trusting whichever single window a score happened to pick.

Each window is a clip id ``<video_id>@<start_s>`` (seconds into the stream), so
the stream's metadata still resolves (``metadata.base_video_id``). Records are
marked ``one_off`` and carry the stretch as ``section``; the site keeps them out
of corpus-wide statistics, because windows of one stream are not independent
samples the way clips of different streams are.
"""

from __future__ import annotations

import wave
from collections.abc import Callable
from pathlib import Path

from .catalog import _available_dt, score_video
from .metadata import SEGMENT_SEPARATOR
from .qc import qc_verdict
from .windows import slice_wav


def clip_id(video_id: str, stream_start_s: float) -> str:
    return f"{video_id}{SEGMENT_SEPARATOR}{int(round(stream_start_s))}"


def _duration_s(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def slice_section(
    audio_path: Path | str,
    windows_dir: Path | str,
    video_id: str,
    *,
    section_start_s: float,
    window_s: float = 90.0,
) -> list[dict]:
    """Cut consecutive ``window_s`` windows from ``audio_path``, which holds
    the stretch starting at ``section_start_s`` into the stream.

    A trailing remainder shorter than a window is dropped, never padded: a
    shorter clip would not be comparable, and padding would add silence the
    stream never contained. ``window`` offsets are relative to the fetched
    audio, as every other record's are; the clip id carries stream time.
    """
    audio_path = Path(audio_path)
    windows_dir = Path(windows_dir)
    duration = _duration_s(audio_path)
    slices: list[dict] = []
    offset = 0.0
    # A hair of tolerance so a stretch that is an exact multiple of the
    # window is not denied its last window by sample rounding.
    while offset + window_s <= duration + 1e-6:
        cid = clip_id(video_id, section_start_s + offset)
        dest = windows_dir / f"{cid}_raw90.wav"
        slice_wav(audio_path, dest, offset, offset + window_s)
        slices.append(
            {
                "id": cid,
                "path": dest,
                "window": {"start_s": offset, "end_s": offset + window_s},
            }
        )
        offset += window_s
    return slices


def v1_records(
    slices: list[dict],
    video: dict,
    *,
    section: tuple[float, float],
    measure: Callable[[Path], dict],
    model: str,
    language: str | None = None,
) -> list[dict]:
    """v1-shaped measurement records for the slices, ready for ``build-v2``.

    ``language`` records what the stretch is spoken in when it differs from
    the talent's usual language (e.g. ``"en"``), so embedding comparisons can
    compensate for language rather than mistake it for voice.

    ``measure`` takes the audio actually measured (normally the vocals stem
    of the window). The month comes from the stream's own air date; a stream
    without one is refused rather than given a month nobody observed.
    """
    when = _available_dt(video)
    if when is None:
        raise ValueError(f"stream {video.get('id')!r} has no air date; refusing to guess a month")
    month = when.strftime("%Y-%m")
    records: list[dict] = []
    for item in slices:
        features = measure(item["path"])
        passed, reason = qc_verdict(features)
        records.append(
            {
                "id": item["id"],
                "month": month,
                "score": score_video(video),
                "window": dict(item["window"]),
                "section": {"start_s": float(section[0]), "end_s": float(section[1])},
                "one_off": True,
                **({"language": language} if language else {}),
                "features": features,
                "qc": {"pass": passed, "reason": reason},
                "model": model,
            }
        )
    return records


def section_audio_path(video_id: str, data_dir: Path | str, section: tuple[float, float]) -> Path:
    """Where a stretch's audio lives. Named by the stretch, so it can never be
    mistaken for the stream's ordinary ``data/audio/<id>.wav`` sample."""
    start, end = (int(round(v)) for v in section)
    return Path(data_dir) / "audio" / f"{clip_id(video_id, start)}-{end}.wav"


def run_segment(
    video_id: str,
    *,
    talent: str,
    section: tuple[float, float],
    video: dict,
    data_dir: Path | str,
    measurements_dir: Path | str,
    fetch: Callable[..., Path],
    isolate: Callable[[Path, Path], Path],
    measure: Callable[[Path], dict],
    model: str,
    window_s: float = 90.0,
    language: str | None = None,
) -> Path:
    """Fetch the stretch, cut it into windows, separate and measure each, and
    write ``<talent>_monthly.json`` for ``build-v2`` to re-measure.

    Refuses to replace a measurement file that holds anything but one-off
    records: a slug collision with an ordinary talent would otherwise wipe
    that talent's corpus.
    """
    import json

    data_dir = Path(data_dir)
    out = Path(measurements_dir) / f"{talent}_monthly.json"
    if out.is_file():
        existing = json.loads(out.read_text(encoding="utf-8"))
        if any(not r.get("one_off") for r in existing):
            raise FileExistsError(f"{out} holds ordinary records; pick another slug")

    audio = section_audio_path(video_id, data_dir, section)
    if not audio.is_file():
        fetched = Path(fetch(video_id, data_dir, section=section))
        audio.parent.mkdir(parents=True, exist_ok=True)
        fetched.replace(audio)

    slices = slice_section(
        audio, data_dir / "windows", video_id, section_start_s=section[0], window_s=window_s
    )
    stems: dict[Path, Path] = {}
    for item in slices:
        stems[item["path"]] = Path(isolate(item["path"], data_dir / "stems_fast"))

    records = v1_records(
        slices,
        video,
        section=section,
        measure=lambda window: measure(stems[window]),
        model=model,
        language=language,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return out
