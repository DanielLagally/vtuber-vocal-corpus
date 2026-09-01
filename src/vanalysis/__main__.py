from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from vanalysis.catalog import filter_videos, pick_monthly
from vanalysis.fetch import audio_path, fetch_audio_many
from vanalysis.isolate import DEFAULT_MODEL_FILENAME, isolate_vocals, vocals_path
from vanalysis.measure import run_monthly
from vanalysis.series import write_plots
from vanalysis.windows import best_speech_window, slice_wav

_HOLODEX_VIDEOS = "https://holodex.net/api/v2/videos"
_PAGE = 50


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        os.environ.setdefault(name.strip(), value.strip())


def _holodex_key() -> str:
    _load_dotenv()
    key = os.environ.get("HOLODEX_API_KEY")
    if not key:
        print("HOLODEX_API_KEY is not set", file=sys.stderr)
        sys.exit(2)
    return key


def _list_holodex(api_key: str, channel: str | None = None) -> list[dict]:
    import requests

    rows: list[dict] = []
    offset = 0
    while True:
        params = {
            "org": "Hololive",
            "type": "stream",
            "status": "past",
            "limit": _PAGE,
            "offset": offset,
        }
        if channel is not None:
            params["channel_id"] = channel
            params["include"] = "mentions"
        response = requests.get(
            _HOLODEX_VIDEOS,
            headers={"X-APIKEY": api_key, "User-Agent": "vanalysis/0.1"},
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        page = response.json()
        if not page:
            break
        rows.extend(page)
        if len(page) < _PAGE:
            break
        offset += _PAGE
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="vanalysis")
    sub = parser.add_subparsers(dest="cmd", required=True)

    cat = sub.add_parser("catalog", help="write filtered Holodex stream ids")
    cat.add_argument("-o", "--out", type=Path, default=Path("catalog.json"))
    cat.add_argument("--channel", default=None)
    cat.add_argument("--monthly", action="store_true")

    get = sub.add_parser("fetch", help="download video ids to data/audio/<id>.wav")
    get.add_argument("video_ids", nargs="*")
    get.add_argument("--data-dir", type=Path, default=Path("data"))
    get.add_argument("--ids-file", type=Path, default=None)
    get.add_argument("--cookies", type=Path, default=None)

    iso = sub.add_parser("isolate", help="vocal-isolate wavs already in data/audio")
    iso.add_argument("video_ids", nargs="*")
    iso.add_argument("--data-dir", type=Path, default=Path("data"))
    iso.add_argument("--ids-file", type=Path, default=None)
    iso.add_argument("--out-dir", type=Path, default=None)
    iso.add_argument("--model-filename", default=None)
    iso.add_argument("--windowed", action="store_true")

    win = sub.add_parser("window", help="pick the 90 s speech window and slice it")
    win.add_argument("video_ids", nargs="*")
    win.add_argument("--data-dir", type=Path, default=Path("data"))
    win.add_argument("--ids-file", type=Path, default=None)
    win.add_argument("--window-s", type=float, default=90.0)
    win.add_argument("--hop-s", type=float, default=15.0)

    mea = sub.add_parser("measure", help="measure picks on their stems")
    mea.add_argument("--picks", type=Path, required=True)
    mea.add_argument("--windows", type=Path, default=Path("data/windows/windows.json"))
    mea.add_argument("--stems-dir", type=Path, default=Path("data/stems_fast"))
    mea.add_argument("--model-filename", default=DEFAULT_MODEL_FILENAME)
    mea.add_argument("--legacy", action="store_true")
    mea.add_argument("-o", "--out", type=Path, default=Path("data/measurements.json"))

    plo = sub.add_parser("plot", help="write monthly F0/IQR plots")
    plo.add_argument("--measurements", type=Path, required=True)
    plo.add_argument("--out-dir", type=Path, default=Path("data/plots"))

    args = parser.parse_args(argv)
    if args.cmd == "catalog":
        kept = filter_videos(_list_holodex(_holodex_key(), channel=args.channel))
        if args.monthly:
            kept = pick_monthly(kept)
        args.out.write_text(json.dumps(kept, indent=2) + "\n", encoding="utf-8")
        print(f"{len(kept)} rows -> {args.out}")
        return
    if args.cmd == "measure":
        picks = json.loads(args.picks.read_text(encoding="utf-8"))
        model_filename = None if args.legacy else args.model_filename
        entries = run_monthly(
            picks, args.windows, args.stems_dir, args.out, model_filename=model_filename
        )
        print(f"{len(entries)} entries -> {args.out}")
        return
    if args.cmd == "plot":
        entries = json.loads(args.measurements.read_text(encoding="utf-8"))
        write_plots(entries, args.out_dir)
        print(f"plots -> {args.out_dir}")
        return
    ids = list(args.video_ids)
    if args.ids_file is not None:
        ids.extend(
            line.strip()
            for line in args.ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not ids:
        parser.error("give at least one video id or --ids-file")
    if args.cmd == "window":
        windows_dir = args.data_dir / "windows"
        windows_dir.mkdir(parents=True, exist_ok=True)
        index_path = windows_dir / "windows.json"
        index = (
            json.loads(index_path.read_text(encoding="utf-8"))
            if index_path.is_file()
            else {}
        )
        for video_id in ids:
            src = audio_path(video_id, args.data_dir)
            if not src.is_file():
                print(f"missing {src}", file=sys.stderr)
                continue
            start_s, end_s = best_speech_window(
                src, window_s=args.window_s, hop_s=args.hop_s
            )
            dest = windows_dir / f"{video_id}_raw90.wav"
            slice_wav(src, dest, start_s, end_s)
            index[video_id] = {"start_s": start_s, "end_s": end_s}
            print(f"{video_id} {start_s:.1f}-{end_s:.1f} -> {dest}")
        index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        return
    if args.cmd == "isolate":
        if args.windowed:
            out_dir = (
                args.out_dir if args.out_dir is not None else args.data_dir / "stems_fast"
            )
        else:
            out_dir = (
                args.out_dir if args.out_dir is not None else args.data_dir / "stems"
            )
        for video_id in ids:
            if args.windowed:
                src_path = args.data_dir / "windows" / f"{video_id}_raw90.wav"
            else:
                src_path = audio_path(video_id, args.data_dir)
            dest = vocals_path(src_path, out_dir, model_filename=args.model_filename)
            if dest.is_file() and dest.stat().st_size > 1_000_000:
                print(f"skip {dest}")
                continue
            if not src_path.is_file():
                print(f"missing {src_path}", file=sys.stderr)
                continue
            print(isolate_vocals(src_path, out_dir, model_filename=args.model_filename))
        return
    cookies = args.cookies
    if cookies is None:
        fallback = args.data_dir / "youtube.cookies.txt"
        if fallback.is_file():
            cookies = fallback
    todo = []
    for video_id in ids:
        dest = audio_path(video_id, args.data_dir)
        if dest.is_file() and dest.stat().st_size > 1_000_000:
            print(f"skip {dest}")
            continue
        todo.append(video_id)
    for video_id, dest in fetch_audio_many(
        todo, args.data_dir, cookies=cookies
    ).items():
        if dest is not None:
            print(dest)


if __name__ == "__main__":
    main()
