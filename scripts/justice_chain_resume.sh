#!/usr/bin/env bash
# Resume the Justice densify chain after the bot-check halt: elizabeth
# (append-only, picks up from the 8 records already on disk) then gigi.
# Same target-n 2, same sequential / halt-on-stopped_early behaviour as
# scripts/justice_chain.sh. Assumes a FRESH data/youtube.cookies.txt.
set -u
cd /Users/daniellagally/Dev/vtuber-vocal-corpus

MEMBERS=(
  "elizabeth UCW5uhrG1eCBYditmhL0Ykjw"
  "gigi      UCDHABijvPBnJm7F-KlNME3w"
)

summarize() {
  python3 - "$1" <<'PY'
import sys, json
slug = sys.argv[1]
lines = open(f"data/{slug}_densify.log").read().splitlines()
starts = [i for i, l in enumerate(lines) if l == "{"]
obj = json.loads("\n".join(lines[starts[-1]:])) if starts else {}
c = obj.get("counts", {})
se = obj.get("stopped_early")
recs = json.load(open(f"data/measurements/{slug}_monthly.json"))
mp = sum(1 for r in recs if r.get("qc", {}).get("pass"))
months = len({r["month"] for r in recs})
print(f"SUMMARY {slug}: added={c.get('added')} skipped={c.get('skipped')} "
      f"error={c.get('error')} stopped_early={se} | "
      f"corpus now {len(recs)} records / {mp} QC-pass / {months} months")
open(f"data/{slug}.stopped_early", "w").write(str(se))
PY
}

# refuse to run on a stale/partial cookie file
if ! grep -qE '\bLOGIN_INFO\b' data/youtube.cookies.txt || ! grep -qE '\bSAPISID\b' data/youtube.cookies.txt; then
  echo "!!! data/youtube.cookies.txt is missing first-party auth cookies (LOGIN_INFO / SAPISID) — aborting"
  exit 2
fi

for entry in "${MEMBERS[@]}"; do
  set -- $entry
  slug=$1 cid=$2
  echo "=== $(date '+%F %T') START densify $slug ==="
  nix develop --command bash -c \
    "export PYTHONPATH=src; python -m vvc densify \
       --measurements data/measurements/${slug}_monthly.json \
       --video-cache data/catalog/video_cache/${cid}.json \
       --target-n 2 --cpu-workers 4" \
    > "data/${slug}_densify.log" 2>&1
  echo "=== $(date '+%F %T') END densify $slug (exit $?) ==="
  summarize "$slug"
  se=$(cat "data/${slug}.stopped_early" 2>/dev/null || echo ERROR)
  if [ "$se" != "None" ]; then
    echo "!!! $(date '+%F %T') $slug stopped_early=$se — HALTING (bot-check / error)"
    exit 3
  fi
done

echo "=== $(date '+%F %T') ELIZABETH + GIGI COMPLETE — Justice gen done ==="
