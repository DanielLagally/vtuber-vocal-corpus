#!/usr/bin/env bash
# Multi-talent orchestrator for the 2026-09 veteran-JP batch (Roboco, Coco,
# Fubuki, Subaru, Korone, Okayu, Watame, Polka — Sora is handled separately,
# see PLAN.md's 2026-09-03 section). Each talent's stages run as separate
# subprocesses (new-talent, densify, plot) so a setup error in one talent
# can't take down the ones queued after it — run_densify has no top-level
# exception guard. Sequential across talents (fetch inside densify must
# stay sequential anyway; running two talents' densify at once would
# double up on the yt-dlp/bot-check risk).
#
# Disk policy: soft threshold 40G free -> stop starting new talents, let
# the current one finish. Hard/emergency floor 15G -> kill the in-flight
# densify and stop immediately. Bot-check (densify's "stopped_early" is
# non-null in its JSON summary) -> stop the WHOLE batch, not just this
# talent, since stale cookies fail identically for everyone queued after.
#
# Portable across machines/OSes (this repo runs on both a Linux desktop
# and an aarch64-darwin laptop, see flake.nix): resolves the repo root
# from the script's own location rather than a hardcoded path, and uses
# `df -k` + awk instead of GNU-only `df --output=avail` (BSD/macOS df
# doesn't support --output). `jq` is a flake devShell package
# (flake.nix), invoked via `direnv exec .` rather than assumed to be on
# the global PATH.
#
# Usage: run_veteran_batch.sh <name1>:<channel_id1> <name2>:<channel_id2> ...
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TARGET_N=2
CPU_WORKERS=4
OFFLOAD_REMOTE="Google Drive:vanalysis-raw-audio"
SOFT_FREE_G=40
HARD_FREE_G=15

free_gb() {
  df -k . | awk 'NR==2 {print int($4/1024/1024)}'
}

run() {
  echo "=== $* ==="
  "$@"
  local ec=$?
  echo "--- exit $ec ---"
  return $ec
}

declare -A DISPLAY_NAMES=(
  [roboco]="Robocosan"
  [coco]="Kiryu Coco"
  [fubuki]="Shirakami Fubuki"
  [subaru]="Oozora Subaru"
  [korone]="Inugami Korone"
  [okayu]="Nekomata Okayu"
  [watame]="Tsunomaki Watame"
  [polka]="Omaru Polka"
  [azki]="AZKi"
  [miko]="Sakura Miko"
  [suisei]="Hoshimachi Suisei"
  [haato]="Akai Haato"
  [matsuri]="Natsuiro Matsuri"
  [aki]="Aki Rosenthal"
  [choco]="Yuzuki Choco"
  [ayame]="Nakiri Ayame"
  [flare]="Shiranui Flare"
  [marine]="Houshou Marine"
  [noel]="Shirogane Noel"
  [pekora]="Usada Pekora"
  [towa]="Tokoyami Towa"
  [botan]="Shishiro Botan"
  [nene]="Momosuzu Nene"
  [iroha]="Kazama Iroha"
  [koyori]="Hakui Koyori"
  [laplus]="La+ Darknesss"
  [mio]="Ookami Mio"
  [achrora]="ACHRORA"
  [ao]="Hiodoshi Ao"
  [niko]="Koganei Niko"
  [su]="Mizumiya Su"
  [luna]="Himemori Luna"
  [sora]="Tokino Sora"
  [raora]="Raora Panthera"
  [kanade]="Otonose Kanade"
  [lamy]="Yukihana Lamy"
  [mori]="Mori Calliope"
  [kiara]="Takanashi Kiara"
  [irys]="IRyS"
  [ririka]="Ichijou Ririka"
  [elizabeth]="Elizabeth Rose Bloodflame"
  [gigi]="Gigi Murin"
  [lui]="Takane Lui"
)

for pair in "$@"; do
  name="${pair%%:*}"
  channel_id="${pair##*:}"
  display="${DISPLAY_NAMES[$name]:-$name}"

  fg=$(free_gb)
  echo "### [$name] pre-flight: ${fg}G free"
  if [ "$fg" -lt "$SOFT_FREE_G" ]; then
    echo "### PAUSED before $name: only ${fg}G free (< ${SOFT_FREE_G}G soft threshold). Stopping batch, no data touched."
    exit 2
  fi

  echo "##### TALENT: $name ($display, $channel_id) #####"

  run direnv exec . python -m vvc new-talent "$name" "$channel_id"

  densify_log="data/logs/${name}_densify_$(date +%Y%m%d_%H%M%S).json"
  echo "=== densify $name -> $densify_log ==="
  direnv exec . python -m vvc densify \
    --measurements "data/measurements/${name}_monthly.json" \
    --video-cache "data/catalog/video_cache/${channel_id}.json" \
    --target-n "$TARGET_N" --cpu-workers "$CPU_WORKERS" \
    --offload-remote "$OFFLOAD_REMOTE" \
    > "$densify_log" 2>&1
  densify_ec=$?
  echo "--- densify exit $densify_ec ---"
  tail -n 20 "$densify_log"

  # Bot-check: densify's JSON summary is a pretty-printed json.dumps(...)
  # appended at the end of $densify_log, NOT the whole file — everything
  # before it is yt-dlp/ffmpeg/audio-separator stdout noise. Find the
  # LAST line that is exactly "{" (json.dumps's opening brace on its own
  # line) and parse from there with jq, don't hand-scrape with grep/sed —
  # that mis-parsed a `null` value plus trailing fields as "non-null"
  # once already (2026-09-03 false positive on roboco: grep -A2 + tr -d
  # '\n' glued neighboring JSON fields onto the captured value with
  # nothing to delimit them, killing the batch after a fully successful
  # run).
  json_start=$(grep -n '^{$' "$densify_log" | tail -1 | cut -d: -f1)
  if [ -z "$json_start" ]; then
    echo "### WARNING: no JSON summary found in $densify_log (densify may have crashed before writing one) — stopping so this doesn't silently masquerade as success."
    exit 5
  fi
  stopped_early=$(tail -n "+$json_start" "$densify_log" | direnv exec . jq -r '.stopped_early // "null"' 2>/dev/null || echo "PARSE_ERROR")
  if [ "$stopped_early" = "PARSE_ERROR" ]; then
    echo "### WARNING: found a JSON summary in $densify_log but jq couldn't parse it — stopping rather than guessing."
    exit 5
  fi
  if [ "$stopped_early" != "null" ]; then
    echo "### BOT-CHECK DETECTED during $name ($stopped_early). Stopping the WHOLE batch — stale cookies would fail identically for every remaining talent. Re-export cookies before resuming (see CLAUDE.md gotcha)."
    exit 3
  fi

  run direnv exec . python -m vvc plot \
    --talent "$display" \
    --measurements "data/measurements/${name}_monthly.json" \
    --label "${name}-monthly"

  run direnv exec . python -m vvc site-data

  fg_after=$(free_gb)
  echo "### [$name] done. ${fg_after}G free."
  if [ "$fg_after" -lt "$HARD_FREE_G" ]; then
    echo "### CRITICAL: ${fg_after}G free (< ${HARD_FREE_G}G hard floor) after $name. Stopping batch immediately."
    exit 4
  fi
done

echo "=== VETERAN BATCH DONE (all requested talents attempted) ==="
