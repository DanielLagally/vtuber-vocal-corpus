from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vvc.catalog import filter_videos, pick_monthly
from vvc.densify import run_densify
from vvc.diagnose import run_diagnose
from vvc.remeasure import run_remeasure
from vvc.expand import fetch_channel_videos, run_plan
from vvc.fetch import BotCheckDetected, audio_path, fetch_audio_many
from vvc.holodex import holodex_key as _holodex_key
from vvc.holodex import load_dotenv as _load_dotenv  # noqa: F401 (compat alias)
from vvc.isolate import DEFAULT_MODEL_FILENAME, isolate_vocals, vocals_path
from vvc.measure import run_monthly
from vvc.rescue import run_rescue
from vvc.retry import run_retry
from vvc.roster import fetch_channels, filter_talents, write_roster
from vvc.site_data import write_site_data
from vvc.series import (
    new_run_dir,
    write_feature_multi_talent_yearly_plot,
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
from vvc.windows import best_speech_window, raw90_path, slice_wav

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
            headers={"X-APIKEY": api_key, "User-Agent": "vvc/0.1"},
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


def _load_registry(path: Path) -> dict[str, str]:
    """``{measurements_path_str: talent_display_name}``. Missing file ->
    empty registry (the very first talent ever plotted starts one)."""
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_registry(path: Path, registry: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_registered_talents(registry: dict[str, str]) -> dict[str, list[dict]]:
    """Loads every registered talent's measurements file. A registered
    path that no longer exists (moved/renamed) is skipped with a
    warning, not a crash — the comparison still runs on whoever's left."""
    talents: dict[str, list[dict]] = {}
    for path_str, name in registry.items():
        path = Path(path_str)
        if not path.is_file():
            print(f"warning: registered talent {name!r} missing {path}, skipping", file=sys.stderr)
            continue
        talents[name] = json.loads(path.read_text(encoding="utf-8"))
    return talents


def _write_all_comparison_plots(talents: dict[str, list[dict]], run_dir: Path) -> None:
    write_multi_talent_plot(talents, run_dir)
    present_keys = {
        key
        for entries in talents.values()
        for entry in entries
        for key in (entry.get("features") or {})
    }
    for feature_key, filename, subject, subtitle, unit_label, caveat in _EXTRA_FEATURE_PLOTS:
        if feature_key in present_keys:
            write_feature_multi_talent_yearly_plot(
                talents, run_dir,
                feature_key=feature_key,
                filename=filename.replace(".png", "_multi.png"),
                subject=f"{subject} — Talent Comparison",
                subtitle=subtitle, unit_label=unit_label, caveat=caveat,
            )


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
    parser = argparse.ArgumentParser(prog="vvc")
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

    newt = sub.add_parser(
        "new-talent",
        help=(
            "bootstrap a fresh talent: cache the raw Holodex listing for "
            "channel_id and seed an empty measurements file, so densify "
            "can run immediately after"
        ),
    )
    newt.add_argument("name", help='file-path slug, e.g. "chihaya"')
    newt.add_argument("channel_id")
    newt.add_argument(
        "--cache-dir", type=Path, default=Path("data/catalog/video_cache")
    )
    newt.add_argument(
        "--measurements-dir", type=Path, default=Path("data/measurements")
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
    plo.add_argument(
        "--registry",
        type=Path,
        default=Path("data/measurements/talents.json"),
        help=(
            "measurements-path -> display-name registry; --talent auto-"
            "registers this run into it, then the cross-talent comparison "
            "plots are regenerated from every registered talent"
        ),
    )
    plo.add_argument(
        "--no-compare",
        action="store_true",
        help="skip auto-updating the registry and regenerating the "
        "cross-talent comparison plots",
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

    sit = sub.add_parser(
        "site-data",
        help=(
            "write the interactive site's data.js (docs/, served by GitHub "
            "Pages) from the "
            "talent registry — per-talent series plus the cute/mature "
            "percentile scatter, see vvc.site_data"
        ),
    )
    sit.add_argument(
        "--registry",
        type=Path,
        default=Path("data/measurements/talents.json"),
    )
    sit.add_argument(
        "--roster",
        type=Path,
        default=Path("data/catalog/roster.json"),
        help=(
            "roster.json (talents list with english_name/group) used to "
            "attach generation/branch metadata per talent; pass a "
            "nonexistent path or empty file to fall back to "
            "group=branch='Unknown'"
        ),
    )
    sit.add_argument("-o", "--out", type=Path, default=Path("docs/data.js"))

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
    dsf.add_argument(
        "--cpu-workers",
        type=int,
        default=1,
        help=(
            "clips whose window-hunt/isolate/measure stages may run "
            "concurrently (fetch always stays sequential); default 1 "
            "reproduces today's fully sequential behavior"
        ),
    )
    dsf.add_argument("--model-filename", default=DEFAULT_MODEL_FILENAME)
    dsf.add_argument(
        "--model-file-dir",
        default=None,
        help="optional audio-separator --model_file_dir (model ckpt cache dir)",
    )
    dsf.add_argument(
        "--offload-remote",
        default=None,
        help=(
            "rclone remote (e.g. 'Google Drive:vanalysis-raw-audio') to "
            "upload a QC-pass id's raw wav to, deleting it locally once "
            "confirmed uploaded; default None disables offload entirely "
            "(a QC-fail id's raw wav is never offloaded, regardless)"
        ),
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
    if args.cmd == "new-talent":
        videos = fetch_channel_videos(
            args.channel_id, api_key=_holodex_key(), cache_dir=args.cache_dir
        )
        cache_path = Path(args.cache_dir) / f"{args.channel_id}.json"
        measurements_path = Path(args.measurements_dir) / f"{args.name}_monthly.json"
        if measurements_path.is_file():
            print(f"measurements already exist, left untouched -> {measurements_path}")
        else:
            measurements_path.parent.mkdir(parents=True, exist_ok=True)
            measurements_path.write_text("[]\n", encoding="utf-8")
            print(f"measurements seeded -> {measurements_path}")
        print(f"{len(videos)} videos cached -> {cache_path}")
        print(
            "next: vvc densify --measurements "
            f"{measurements_path} --video-cache {cache_path} "
            "--target-n 3 --cpu-workers 4"
        )
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
        if not args.no_compare:
            registry = _load_registry(args.registry)
            if args.talent:
                registry[str(args.measurements)] = args.talent
                _save_registry(args.registry, registry)
            talents = _load_registered_talents(registry)
            if len(talents) >= 2:
                _write_all_comparison_plots(talents, run_dir)
                print(f"comparison plots ({', '.join(sorted(talents))}) -> {run_dir}")
        print(f"plots -> {run_dir}")
        return
    if args.cmd == "plot-compare":
        talents = {
            name: json.loads(Path(path).read_text(encoding="utf-8"))
            for name, path in args.talents
        }
        run_dir = new_run_dir(args.out_dir, args.label)
        _write_all_comparison_plots(talents, run_dir)
        print(f"plots -> {run_dir}")
        return
    if args.cmd == "site-data":
        registry = _load_registry(args.registry)
        roster = None
        if args.roster.is_file():
            roster = json.loads(args.roster.read_text(encoding="utf-8")).get("talents")
        payload = write_site_data(registry, args.out, roster=roster)
        print(f"site data ({', '.join(sorted(payload['talents']))}) -> {args.out}")
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
            cpu_workers=args.cpu_workers,
            offload_remote=args.offload_remote,
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
            dest = raw90_path(video_id, args.data_dir)
            if dest.is_file():
                print(f"skip {dest} (already windowed)")
                continue
            src = audio_path(video_id, args.data_dir)
            if not src.is_file():
                print(f"missing {src}", file=sys.stderr)
                continue
            start_s, end_s = best_speech_window(
                src, window_s=args.window_s, hop_s=args.hop_s
            )
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
        if raw90_path(video_id, args.data_dir).is_file():
            print(f"skip {dest} (already windowed, raw wav not needed)")
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
