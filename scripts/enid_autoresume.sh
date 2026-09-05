#!/usr/bin/env bash
# Keep one hololive generation's pipeline going across YouTube bot-checks.
# Poll every 30 min; whenever fetching looks possible, (re)launch
# enid_pipeline.sh and wait for it. If it commits the generation, stop.
# If it halts on a bot-check (densify stopped_early), go back to polling.
#
#   scripts/enid_autoresume.sh "English -Myth-" \
#     mori:UCL_...:"Mori Calliope" ina:UCMw...:"Ninomae Ina'nis" ...
#
# "Fetching looks possible" means EITHER:
#   - data/youtube.cookies.txt exists and a live yt-dlp probe of a real
#     hololive VOD *with those cookies* clears the bot-check, OR
#   - a cookieless (PO-token) probe of that VOD clears it.
# A name-grep on the cookie file is not enough: an expired or yt-dlp-
# degraded jar still has the right cookie names but fails at YouTube.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PLUGDIR="$HOME/.config/yt-dlp/plugins"
# A real hololive VOD the bot-check rejects when the IP is flagged (an
# unrelated popular video like dQw4w9WgXcQ clears even then — useless as
# a probe). Update if this id ever goes private.
PROBE_ID="h5QspNrKphM"
GEN="$1"; shift
SPECS=("$@")
GENTAG=$(echo "$GEN" | tr -cd 'A-Za-z')

probe() {  # $1: extra yt-dlp args (e.g. --cookies ...). 0 = fetch works.
  nix develop --command bash -c \
    "unset PYTHONPATH; export PYTHONPATH='src:$PLUGDIR'; \
     yt-dlp $1 --extractor-args 'youtube:player_client=mweb' \
       --skip-download --print '%(id)s' -- $PROBE_ID" 2>/dev/null \
    | grep -qx "$PROBE_ID"
}

can_fetch() {
  if [ -f data/youtube.cookies.txt ]; then
    # probe against a COPY — yt-dlp write-back would otherwise degrade a
    # freshly dropped export before the pipeline ever runs.
    cp data/youtube.cookies.txt data/.cookies.probe.txt
    if probe "--cookies data/.cookies.probe.txt"; then
      rm -f data/.cookies.probe.txt; echo "cookies present and live"; return 0
    fi
    rm -f data/.cookies.probe.txt
  fi
  if probe ""; then
    echo "cookieless / PO-token probe OK"; return 0
  fi
  return 1
}

gen_committed() {  # the generation's own commit landed on HEAD
  git log -1 --pretty=%s | grep -qi "Add hololive ${GEN}"
}

backup_cookies() {  # snapshot each distinct drop for diagnostics
  [ -f data/youtube.cookies.txt ] || return 0
  mkdir -p data/cookie_backups
  local sum latest
  sum=$(shasum data/youtube.cookies.txt | cut -d' ' -f1)
  latest=$(ls -t data/cookie_backups/*.txt 2>/dev/null | head -1)
  if [ -z "$latest" ] || [ "$(shasum "$latest" | cut -d' ' -f1)" != "$sum" ]; then
    cp -p data/youtube.cookies.txt \
      "data/cookie_backups/youtube.cookies.$(date '+%Y%m%dT%H%M%S').${sum:0:12}.txt"
    echo "$(date '+%F %T') cookie drop backed up (${sum:0:12}, $(wc -c < data/youtube.cookies.txt) bytes)"
  fi
}

while true; do
  backup_cookies
  if gen_committed; then
    echo "$(date '+%F %T') $GEN already committed — nothing to do"; exit 0
  fi
  if reason=$(can_fetch); then
    echo "$(date '+%F %T') === RESUMING $GEN ($reason) ==="
    nohup bash scripts/enid_pipeline.sh "$GEN" "${SPECS[@]}" \
      > "data/enid_${GENTAG}.log" 2>&1 &
    d=$!
    nohup caffeinate -i -m -s -w "$d" >/dev/null 2>&1 &
    echo "$(date '+%F %T') driver PID $d, caffeinate bound — waiting"
    wait "$d"
    echo "$(date '+%F %T') driver exited ($?)"
    if gen_committed; then
      echo "$(date '+%F %T') === $GEN COMMITTED — done ==="; exit 0
    fi
    echo "$(date '+%F %T') driver halted without committing (likely bot-check) — back to polling"
  else
    echo "$(date '+%F %T') still bot-checked, no working cookies — retry in 30m"
  fi
  sleep 1800
done
