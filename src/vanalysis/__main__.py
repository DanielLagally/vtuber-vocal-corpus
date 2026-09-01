from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vanalysis.catalog import filter_videos, pick_monthly
from vanalysis.densify import run_densify
from vanalysis.diagnose import run_diagnose
from vanalysis.remeasure import run_remeasure
from vanalysis.expand import run_plan
from vanalysis.fetch import BotCheckDetected, audio_path, fetch_audio_many
from vanalysis.holodex import holodex_key as _holodex_key
from vanalysis.holodex import load_dotenv as _load_dotenv  # noqa: F401 (compat alias)
from vanalysis.isolate import DEFAULT_MODEL_FILENAME, isolate_vocals, vocals_path
from vanalysis.measure import run_monthly
from vanalysis.rescue import run_rescue
from vanalysis.retry import run_retry
from vanalysis.roster import fetch_channels, filter_talents, write_roster
from vanalysis.series import (
    new_run_dir,
    write_feature_yearly_plot,
    write_multi_talent_plot,
    write_plots,
    write_quarterly_plots,
    write_yearly_plot,
)

# (feature_key, filename, subject, subtitle, unit_label, caveat|None) - each
# plotted only when at least one measurement record actually has the key
# (older corpora / pre-backfill records won't).
_EXTRA_FEATURE_PLOTS = (
    (
        "brightness_hz", "brightness_yearly.png", "Brightness by Year",
        "Spectral centroid - a brighter/more forward vs. darker/warmer voice.\n"
        "Mic/EQ and leftover BGM affect this too; trust the relative shape.",
        "Brightness (Hz)", None,
    ),
    (
        "dynamism_semitones", "dynamism_yearly.png", "Pitch Dynamism by Year",
        "Mean semitone change between consecutive voiced frames - how much\n"
        "the pitch actually moves, not just its static spread (F0 IQR).",
        "Dynamism (semitones)", None,
    ),
    (
        "loudness_dynamics_db", "loudness_dynamics_yearly.png",
        "Loudness Dynamics by Year",
        "Spread of frame loudness (RMS in dB) within a clip - animated\n"
        "volume swings vs. a flat, even delivery.",
        "Loudness spread (dB)", None,
    ),
    (
        "jitter_local", "jitter_yearly.png", "Jitter by Year",
        "Cycle-to-cycle pitch-period timing irregularity.",
        "Jitter (local, fraction)",
        "CAVEAT: calibrated for a sustained vowel, not conversational speech,\n"
        "and sensitive to residual vocal-isolation artifact - trust the\n"
        "relative shape within this pipeline, not the absolute number.",
    ),
    (
        "shimmer_local", "shimmer_yearly.png", "Shimmer by Year",
        "Cycle-to-cycle amplitude irregularity.",
        "Shimmer (local, fraction)",
        "CAVEAT: calibrated for a sustained vowel, not conversational speech,\n"
        "and sensitive to residual vocal-isolation artifact - trust the\n"
        "relative shape within this pipeline, not the absolute number.",
    ),
    (
        "hnr_db", "hnr_yearly.png", "Harmonics-to-Noise Ratio by Year",
        "Higher = clearer/more tonal voice; lower = breathier/noisier.",
        "HNR (dB)",
        "CAVEAT: calibrated for a sustained vowel, not conversational speech,\n"
        "and sensitive to residual vocal-isolation artifact - trust the\n"
        "relative shape within this pipeline, not the absolute number.",
    ),
)
from vanalysis.windows import best_speech_window, slice_wav

_HOLODEX_VIDEOS = "https://holodex.net/api/v2/videos"
_PAGE = 50


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


def _dash_ids_argv(argv: list[str] | None, ret: argparse.ArgumentParser) -> list[str]:
    """Normalize "--ids -<id>" to "--ids=<id>" so dash-leading ids survive parsing."""
    if argv is None:
        argv = sys.argv[1:]
    options = set(ret._option_string_actions)
    out: list[str] = []
    i = 0
    while i < len(argv):
        nxt = argv[i + 1] if i + 1 < len(argv) else None
        if (
            argv[i] == "--ids"
            and nxt is not None
            and nxt.startswith("-")
            and not nxt.startswith("--")
            and nxt not in options
        ):
            out.append("--ids=" + nxt)
            i += 2
        else:
            out.append(argv[i])
            i += 1
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="vanalysis")
    sub = parser.add_subparsers(dest="cmd", required=True)

    cat = sub.add_parser("catalog", help="write filtered Holodex stream ids")
    cat.add_argument("-o", "--out", type=Path, default=Path("catalog.json"))
    cat.add_argument("--channel", default=None)
    cat.add_argument("--monthly", action="store_true")

    ros = sub.add_parser(
        "roster",
        help="fetch the Holodex channel roster and write the active talent set",
    )
    ros.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("data/catalog/roster.json"),
    )
    ros.add_argument("--org", default="Hololive")

    plan = sub.add_parser(
        "plan",
        help="per-talent v1 sampling plan from cached Holodex listings (metadata only)",
    )
    plan.add_argument(
        "--roster", type=Path, default=Path("data/catalog/roster.json")
    )
    plan.add_argument(
        "--cache-dir", type=Path, default=Path("data/catalog/video_cache")
    )
    plan.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("data/catalog/expansion_plan.json"),
    )

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

    plo = sub.add_parser(
        "plot",
        help=(
            "write monthly/quarterly/yearly F0/IQR plots into a fresh "
            "data/plots/runs/<run>/ directory (never overwrites a prior run)"
        ),
    )
    plo.add_argument("--measurements", type=Path, required=True)
    plo.add_argument("--out-dir", type=Path, default=Path("data/plots"))
    plo.add_argument(
        "--label",
        default=None,
        help="optional suffix for the run directory name (e.g. luna-monthly)",
    )
    plo.add_argument(
        "--talent",
        default=None,
        help='display name for plot titles (e.g. "Himemori Luna")',
    )

    cmp = sub.add_parser(
        "plot-compare",
        help=(
            "cross-talent quarterly/yearly F0 comparison (QC-pass only) "
            "into a fresh data/plots/runs/<run>/ directory"
        ),
    )
    cmp.add_argument(
        "--talent",
        nargs=2,
        action="append",
        metavar=("NAME", "MEASUREMENTS"),
        required=True,
        dest="talents",
        help='repeatable: --talent "Display Name" path/to/measurements.json',
    )
    cmp.add_argument("--out-dir", type=Path, default=Path("data/plots"))
    cmp.add_argument(
        "--label",
        default=None,
        help="optional suffix for the run directory name",
    )

    ret = sub.add_parser(
        "retry",
        help=(
            "re-measure QC-failing months on a 2nd 90 s window "
            "(<id>_raw90b) and replace passing ones"
        ),
    )
    ret.add_argument(
        "--measurements",
        type=Path,
        default=Path("data/measurements/luna_monthly.json"),
    )
    ret.add_argument(
        "--windows", type=Path, default=Path("data/windows/windows.json")
    )
    ret.add_argument("--data-dir", type=Path, default=Path("data"))
    ret.add_argument("--stems-dir", type=Path, default=Path("data/stems_fast"))
    ret.add_argument("--model-filename", default=DEFAULT_MODEL_FILENAME)
    ret.add_argument(
        "--model-file-dir",
        default=None,
        help=(
            "optional audio-separator --model_file_dir (model ckpt cache "
            "dir, e.g. data/models); default: audio-separator's own"
        ),
    )
    ret.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help="ids to retry (default: every record whose qc.pass is false)",
    )
    ret.add_argument(
        "--ids-file",
        type=Path,
        default=None,
        help=(
            "file with one video id per line, appended after --ids "
            "(accepts dash-leading ids like -DwvlhziHBI)"
        ),
    )
    ret.add_argument(
        "--dry-run",
        action="store_true",
        help="compute everything possible but write nothing",
    )

    resc = sub.add_parser(
        "rescue",
        help=(
            "stem-hunt rescue: isolate the FULL wav, hunt the 90 s window "
            "on the stem (<id>_stem90) and replace passing months"
        ),
    )
    resc.add_argument(
        "--measurements",
        type=Path,
        default=Path("data/measurements/luna_monthly.json"),
    )
    resc.add_argument(
        "--windows", type=Path, default=Path("data/windows/windows.json")
    )
    resc.add_argument("--data-dir", type=Path, default=Path("data"))
    resc.add_argument("--stems-dir", type=Path, default=Path("data/stems_fast"))
    resc.add_argument("--model-filename", default=DEFAULT_MODEL_FILENAME)
    resc.add_argument(
        "--model-file-dir",
        default="data/models",
        help=(
            "audio-separator --model_file_dir (model ckpt cache dir); "
            "default: data/models"
        ),
    )
    resc.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help="ids to rescue (default: every record whose qc.pass is false)",
    )
    resc.add_argument(
        "--ids-file",
        type=Path,
        default=None,
        help=(
            "file with one video id per line, appended after --ids "
            "(accepts dash-leading ids like -DwvlhziHBI)"
        ),
    )
    resc.add_argument(
        "--dry-run",
        action="store_true",
        help="compute everything possible but write nothing",
    )

    prm = sub.add_parser(
        "remeasure-praat",
        help=(
            "re-measure EVERY record with Praat instead of numpy ACF, on "
            "the same audio each record already used; snapshots first"
        ),
    )
    prm.add_argument(
        "--measurements",
        type=Path,
        default=Path("data/measurements/luna_monthly.json"),
    )
    prm.add_argument(
        "--windows", type=Path, default=Path("data/windows/windows.json")
    )
    prm.add_argument("--data-dir", type=Path, default=Path("data"))
    prm.add_argument("--stems-dir", type=Path, default=Path("data/stems_fast"))
    prm.add_argument("--model-filename", default=DEFAULT_MODEL_FILENAME)
    prm.add_argument("-o", "--out", type=Path, default=None)

    dsf = sub.add_parser(
        "densify",
        help=(
            "bring every month below --target-n records up to target-n "
            "using the cached raw catalog (fetch+window+isolate+Praat "
            "measure); only appends, never touches existing records"
        ),
    )
    dsf.add_argument(
        "--measurements",
        type=Path,
        default=Path("data/measurements/luna_monthly.json"),
    )
    dsf.add_argument(
        "--video-cache",
        type=Path,
        default=Path("data/catalog/video_cache/UCa9Y57gfeY0Zro_noHRVrnw.json"),
        help="cached raw Holodex listing for the channel (no new API calls)",
    )
    dsf.add_argument(
        "--windows", type=Path, default=Path("data/windows/windows.json")
    )
    dsf.add_argument("--data-dir", type=Path, default=Path("data"))
    dsf.add_argument("--stems-dir", type=Path, default=Path("data/stems_fast"))
    dsf.add_argument("--target-n", type=int, default=2)
    dsf.add_argument("--model-filename", default=DEFAULT_MODEL_FILENAME)
    dsf.add_argument(
        "--model-file-dir",
        default=None,
        help="optional audio-separator --model_file_dir (model ckpt cache dir)",
    )
    dsf.add_argument("--cookies", type=Path, default=None)
    dsf.add_argument(
        "--dry-run",
        action="store_true",
        help="compute everything possible but write nothing",
    )

    diag = sub.add_parser(
        "diagnose-tracker",
        help=(
            "read-only: compare numpy-ACF vs Praat on audio already on "
            "disk (raw90/raw90b/stem90); never touches --measurements"
        ),
    )
    diag.add_argument(
        "--measurements",
        type=Path,
        default=Path("data/measurements/luna_monthly.json"),
    )
    diag.add_argument("--data-dir", type=Path, default=Path("data"))
    diag.add_argument("--stems-dir", type=Path, default=Path("data/stems_fast"))
    diag.add_argument("--model-filename", default=DEFAULT_MODEL_FILENAME)
    diag.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help="ids to compare (default: every record whose qc.pass is false)",
    )
    diag.add_argument(
        "--ids-file",
        type=Path,
        default=None,
        help=(
            "file with one video id per line, appended after --ids "
            "(accepts dash-leading ids like -DwvlhziHBI)"
        ),
    )
    diag.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("data/measurements/luna_tracker_diagnostic.json"),
    )

    args = parser.parse_args(_dash_ids_argv(argv, ret))
    if args.cmd == "roster":
        talents = filter_talents(
            fetch_channels(org=args.org, api_key=_holodex_key())
        )
        write_roster(talents, args.out)
        groups: dict[str, int] = {}
        for row in talents:
            groups[row["group"]] = groups.get(row["group"], 0) + 1
        print(f"{len(talents)} talents -> {args.out}")
        for group in sorted(groups):
            print(f"  {group}: {groups[group]}")
        return
    if args.cmd == "plan":
        payload = run_plan(args.roster, args.cache_dir, args.out, api_key=_holodex_key())
        print(
            f"{payload['talents_planned']}/{payload['roster_count']} talents, "
            f"{payload['total_picks']} picks, ~{payload['est_disk_gb']} GB -> {args.out}"
        )
        for record in payload["talents"]:
            if record["pick_count"] == 0:
                print(f"  zero eligible picks: {record['name']} ({record['group']})")
        return
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
        run_dir = new_run_dir(args.out_dir, args.label)
        write_plots(entries, run_dir, talent=args.talent)
        write_quarterly_plots(entries, run_dir, talent=args.talent)
        write_yearly_plot(entries, run_dir, talent=args.talent)
        present_keys = {
            key
            for entry in entries
            for key in (entry.get("features") or {})
        }
        for feature_key, filename, subject, subtitle, unit_label, caveat in _EXTRA_FEATURE_PLOTS:
            if feature_key in present_keys:
                write_feature_yearly_plot(
                    entries, run_dir,
                    feature_key=feature_key, filename=filename,
                    subject=subject, subtitle=subtitle, unit_label=unit_label,
                    talent=args.talent, caveat=caveat,
                )
        print(f"plots -> {run_dir}")
        return
    if args.cmd == "plot-compare":
        talents = {
            name: json.loads(Path(path).read_text(encoding="utf-8"))
            for name, path in args.talents
        }
        run_dir = new_run_dir(args.out_dir, args.label)
        write_multi_talent_plot(talents, run_dir)
        print(f"plots -> {run_dir}")
        return
    if args.cmd == "retry":
        ids = list(args.ids) if args.ids else []
        if args.ids_file is not None:
            ids.extend(
                line.strip()
                for line in args.ids_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        if args.ids_file is not None and not ids:
            parser.error("give at least one video id via --ids or --ids-file")
        summary = run_retry(
            ids or None,
            args.data_dir,
            measurements_path=args.measurements,
            windows_path=args.windows,
            stems_dir=args.stems_dir,
            model_filename=args.model_filename,
            model_file_dir=args.model_file_dir,
            dry_run=args.dry_run,
        )
        print(json.dumps(summary, indent=2))
        return
    if args.cmd == "rescue":
        ids = list(args.ids) if args.ids else []
        if args.ids_file is not None:
            ids.extend(
                line.strip()
                for line in args.ids_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        if args.ids_file is not None and not ids:
            parser.error("give at least one video id via --ids or --ids-file")
        summary = run_rescue(
            ids or None,
            args.data_dir,
            measurements_path=args.measurements,
            windows_path=args.windows,
            stems_dir=args.stems_dir,
            model_filename=args.model_filename,
            model_file_dir=args.model_file_dir,
            dry_run=args.dry_run,
        )
        print(json.dumps(summary, indent=2))
        return
    if args.cmd == "densify":
        cookies = args.cookies
        if cookies is None:
            fallback = args.data_dir / "youtube.cookies.txt"
            if fallback.is_file():
                cookies = fallback
        summary = run_densify(
            args.data_dir,
            measurements_path=args.measurements,
            video_cache_path=args.video_cache,
            windows_path=args.windows,
            stems_dir=args.stems_dir,
            target_n=args.target_n,
            model_filename=args.model_filename,
            model_file_dir=args.model_file_dir,
            cookies=cookies,
            dry_run=args.dry_run,
            log=print,
        )
        print(json.dumps(summary, indent=2))
        return
    if args.cmd == "remeasure-praat":
        summary = run_remeasure(
            args.data_dir,
            measurements_path=args.measurements,
            windows_path=args.windows,
            stems_dir=args.stems_dir,
            model_filename=args.model_filename,
            out_path=args.out,
        )
        print(json.dumps(summary, indent=2))
        return
    if args.cmd == "diagnose-tracker":
        ids = list(args.ids) if args.ids else []
        if args.ids_file is not None:
            ids.extend(
                line.strip()
                for line in args.ids_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        if args.ids_file is not None and not ids:
            parser.error("give at least one video id via --ids or --ids-file")
        results = run_diagnose(
            ids or None,
            args.data_dir,
            measurements_path=args.measurements,
            stems_dir=args.stems_dir,
            model_filename=args.model_filename,
            out_path=args.out,
        )
        print(f"{len(results)} comparison records -> {args.out}")
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
    try:
        for video_id, dest in fetch_audio_many(
            todo, args.data_dir, cookies=cookies
        ).items():
            if dest is not None:
                print(dest)
    except BotCheckDetected as exc:
        print(
            f"stopped: YouTube bot-check triggered on {exc.video_id} — "
            "this is a session/IP-level signal, not a per-video problem; "
            "further fetches in this batch were not attempted. Refresh "
            "cookies (see CLAUDE.md) before retrying.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
