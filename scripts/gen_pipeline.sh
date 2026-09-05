#!/usr/bin/env bash
# Full corpus pipeline for one hololive generation, cookieless (PO-token
# provider must be up: scripts/potoken_server.sh).
#
#   scripts/gen_pipeline.sh "English -Myth-" mori:UCL_q... ina:UCMw... kiara:UCHs...
#
# Per member: densify (target-n 2) -> retry -> rescue, each append-only /
# QC-gated. Sequential (GPU isolate pool is single-worker; cookieless
# fetch still benefits from not hammering in parallel). Writes a
# "GEN COMPLETE" marker; halts the whole run if densify reports
# stopped_early (should not happen with PO tokens, but a real block still
# stops rather than churns).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PLUGDIR="$HOME/.config/yt-dlp/plugins"
export PYTHONPATH="src:$PLUGDIR"

GEN="$1"; shift
MEMBERS=("$@")

VV() { nix develop --command bash -c "PYTHONPATH='src:$PLUGDIR' python -m vvc $*"; }

status() {
  python3 - "$@" <<'PY'
import json, sys
for slug in sys.argv[1:]:
    try:
        d = json.load(open(f"data/measurements/{slug}_monthly.json"))
    except FileNotFoundError:
        print(f"  {slug}: (no file)"); continue
    bym = {}
    for r in d: bym.setdefault(r["month"], []).append(r)
    gaps = sorted(m for m, rs in bym.items() if not any(r["qc"]["pass"] for r in rs))
    nfail = sum(1 for r in d if not r["qc"]["pass"])
    mp = len(bym) - len(gaps)
    print(f"  {slug:12s} {len(d):3d} rec | {mp}/{len(bym)} months w/ QC-pass | {nfail} failing clips"
          + (f" | GAPS: {gaps}" if gaps else ""))
PY
}

last_json() {  # extract trailing pretty-printed JSON object from a log
  python3 - "$1" <<'PY'
import json, sys
lines = open(sys.argv[1]).read().splitlines()
st = [i for i, l in enumerate(lines) if l == "{"]
print(json.dumps(json.loads("\n".join(lines[st[-1]:])) if st else {}))
PY
}

echo "############ $(date '+%F %T')  GEN START: $GEN  (${#MEMBERS[@]} members) ############"

for spec in "${MEMBERS[@]}"; do
  slug="${spec%%:*}"; cid="${spec##*:}"
  cache="data/catalog/video_cache/${cid}.json"

  echo "==== $(date '+%F %T')  $slug  densify ===="
  VV densify --measurements "data/measurements/${slug}_monthly.json" \
             --video-cache "$cache" --target-n 2 --cpu-workers 4 \
             > "data/${slug}_densify.log" 2>&1
  se=$(last_json "data/${slug}_densify.log" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("stopped_early"))')
  echo "     densify: $(last_json "data/${slug}_densify.log" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d.get("counts"))')"
  if [ "$se" != "None" ] && [ -n "$se" ]; then
    echo "!!!! $(date '+%F %T')  $slug densify stopped_early=$se — HALTING GEN $GEN"
    exit 3
  fi

  echo "==== $(date '+%F %T')  $slug  retry ===="
  VV retry  --measurements "data/measurements/${slug}_monthly.json" > "data/${slug}_retry.log"  2>&1
  echo "     retry: $(last_json "data/${slug}_retry.log" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("counts"))')"

  echo "==== $(date '+%F %T')  $slug  rescue ===="
  VV rescue --measurements "data/measurements/${slug}_monthly.json" > "data/${slug}_rescue.log" 2>&1
  echo "     rescue: $(last_json "data/${slug}_rescue.log" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("counts"))')"

  echo "---- $slug done ----"
  status "$slug"
done

echo "############ $(date '+%F %T')  GEN COMPLETE: $GEN ############"
status "${MEMBERS[@]%%:*}"
