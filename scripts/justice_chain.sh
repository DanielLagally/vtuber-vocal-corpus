#!/usr/bin/env bash
# One-off driver: run `vvc densify` for every hololive EN -Justice- member
# sequentially (target-n 2). Sequential, not parallel: the GPU isolate pool
# is single-worker anyway, and back-to-back YouTube fetches are what trips
# the bot-check — one talent at a time keeps the request rate sane.
# Halts the whole chain if densify reports stopped_early (bot-check).
set -u
cd /Users/daniellagally/Dev/vtuber-vocal-corpus

MEMBERS=(
  "raora     UCl69AEx4MdqMZH7Jtsm7Tig"
  "cecilia   UCvN5h1ShZtc7nly3pezRayg"
  "elizabeth UCW5uhrG1eCBYditmhL0Ykjw"
  "gigi      UCDHABijvPBnJm7F-KlNME3w"
)

summarize() {  # $1 = slug ; reads data/<slug>_densify.log, writes data/<slug>.stopped_early
  python3 - "$1" <<'PY'
import sys, json
slug = sys.argv[1]
try:
    lines = open(f"data/{slug}_densify.log").read().splitlines()
except FileNotFoundError:
    print(f"SUMMARY {slug}: (no log)"); open(f"data/{slug}.stopped_early","w").write("ERROR"); raise SystemExit
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

# Wait out the raora densify that is already running in the background.
echo "=== $(date '+%F %T') waiting for in-flight raora densify ==="
while pgrep -f "vvc densify --measurements data/measurements/raora_monthly" >/dev/null; do
  sleep 30
done
echo "=== $(date '+%F %T') in-flight raora densify ended ==="

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
    echo "!!! $(date '+%F %T') $slug stopped_early=$se — HALTING CHAIN (bot-check / error)"
    exit 3
  fi
done

echo "=== $(date '+%F %T') ALL JUSTICE DENSIFY RUNS COMPLETE ==="
