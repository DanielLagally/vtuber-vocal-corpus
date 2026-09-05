#!/usr/bin/env bash
# One hololive generation, end to end, disk-safe. Cookieless — the bgutil
# PO-token server (scripts/potoken_server.sh) must be up on :4416.
#
#   scripts/enid_pipeline.sh "English -Myth-" \
#     mori:UCL_qhgtOy0dy1Agp8vkySQg:"Mori Calliope" \
#     ina:UCMwGHR0BTZuLsmjY_NT5Pwg:"Ninomae Ina'nis" \
#     kiara:UCHsx4Hqa-1ORjQTh9TYDhww:"Takanashi Kiara"
#
# Per member: new-talent -> densify(target-n 2) -> retry/rescue only if
# gaps -> remeasure-praat (normalise to Praat + F1-F4 formants) -> verify.
# After all members: plot each, regenerate docs/data.js, git add + commit
# (NO push — the caller pushes). Raw audio/stems are KEPT. Halts on:
# densify stopped_early, or free disk below MIN_FREE_GB (checked before
# each member and after each densify) — a clean stop, not a crash.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PLUGDIR="$HOME/.config/yt-dlp/plugins"
MIN_FREE_GB=50   # a single veteran talent (~130 clips) can add ~30G raw+stems

GEN="$1"; shift
SPECS=("$@")

# env(1) overrides the python3.14 PYTHONPATH that yt-dlp leaks; running
# the command directly (no `bash -c "$*"`) keeps quoted args like
# --talent "Mori Calliope" intact — the old $* form split them and
# silently failed the plot/registry step under `|| true`.
VV() { nix develop --command env PYTHONPATH="src:$PLUGDIR" python -m vvc "$@"; }

free_gb() { df -g . | awk 'NR==2{print $4}'; }

check_disk() {
  local f; f=$(free_gb)
  echo "     [disk] ${f}G free"
  if [ "$f" -lt "$MIN_FREE_GB" ]; then
    echo "!!!! $(date '+%F %T') free disk ${f}G < ${MIN_FREE_GB}G — HALTING GEN $GEN"
    exit 4
  fi
}

last_json() { python3 -c "import json,sys;L=open(sys.argv[1]).read().splitlines();s=[i for i,l in enumerate(L) if l=='{'];print(json.dumps(json.loads(chr(10).join(L[s[-1]:])) if s else {}))" "$1"; }

gaps_for() {  # prints number of gap months
  python3 - "$1" <<'PY'
import json, sys
d = json.load(open(f"data/measurements/{sys.argv[1]}_monthly.json"))
bym = {}
for r in d: bym.setdefault(r["month"], []).append(r)
print(sum(1 for rs in bym.values() if not any(r["qc"]["pass"] for r in rs)))
PY
}

echo "############ $(date '+%F %T')  GEN START: $GEN  (${#SPECS[@]} members) ############"
check_disk

NAMES=()
for spec in "${SPECS[@]}"; do
  slug="${spec%%:*}"; rest="${spec#*:}"; cid="${rest%%:*}"; disp="${rest#*:}"
  NAMES+=("$disp")
  echo "======== $(date '+%F %T')  $slug ($disp)  ========"

  check_disk   # don't start a member we may not be able to finish

  VV new-talent "$slug" "$cid" 2>&1 | grep -E "seeded|cached|already" || true

  echo "---- densify ----"
  VV densify --measurements "data/measurements/${slug}_monthly.json" \
             --video-cache "data/catalog/video_cache/${cid}.json" \
             --target-n 2 --cpu-workers 2 > "data/${slug}_densify.log" 2>&1
  echo "     $(last_json "data/${slug}_densify.log" | python3 -c 'import json,sys;d=json.load(sys.stdin);print("counts",d.get("counts"),"stopped_early",d.get("stopped_early"))')"
  se=$(last_json "data/${slug}_densify.log" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("stopped_early"))')
  if [ "$se" != "None" ] && [ -n "$se" ]; then
    echo "!!!! $slug densify stopped_early=$se — HALTING GEN $GEN"; exit 3
  fi

  if [ "$(gaps_for "$slug")" -gt 0 ]; then
    echo "---- retry (gaps: $(gaps_for "$slug")) ----"
    VV retry --measurements "data/measurements/${slug}_monthly.json" > "data/${slug}_retry.log" 2>&1
    echo "     $(last_json "data/${slug}_retry.log" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("counts"))')"
  fi
  if [ "$(gaps_for "$slug")" -gt 0 ]; then
    echo "---- rescue (gaps: $(gaps_for "$slug")) ----"
    VV rescue --measurements "data/measurements/${slug}_monthly.json" > "data/${slug}_rescue.log" 2>&1
    echo "     $(last_json "data/${slug}_rescue.log" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("counts"))')"
  fi

  echo "---- remeasure-praat (formant normalise) ----"
  VV remeasure-praat --measurements "data/measurements/${slug}_monthly.json" > "data/${slug}_remeasure.log" 2>&1
  echo "     $(last_json "data/${slug}_remeasure.log" | python3 -c 'import json,sys;d=json.load(sys.stdin);print("pass",d.get("pass"),"fail",d.get("fail"))')"

  python3 - "$slug" <<'PY'
import json, sys
d = json.load(open(f"data/measurements/{sys.argv[1]}_monthly.json"))
nofmt = sum(1 for r in d if "f1_hz" not in r.get("features", {}))
bym = {}
for r in d: bym.setdefault(r["month"], []).append(r)
gaps = sorted(m for m, rs in bym.items() if not any(r["qc"]["pass"] for r in rs))
p = sum(1 for r in d if r["qc"]["pass"])
print(f"     RESULT {sys.argv[1]}: {len(d)} rec / {p} QC-pass / {len(bym)} months"
      f" | gaps: {gaps or 'none'} | missing-formants: {nofmt}")
PY

  check_disk
done

echo "======== $(date '+%F %T')  plots + site-data ========"
for spec in "${SPECS[@]}"; do
  slug="${spec%%:*}"; rest="${spec#*:}"; disp="${rest#*:}"
  VV plot --measurements "data/measurements/${slug}_monthly.json" --talent "$disp" 2>&1 | grep -E "plots ->" || true
done
VV site-data 2>&1 | grep -oE "site data \([^)]*\)" | head -c 120; echo

echo "======== $(date '+%F %T')  commit ========"
# Only this generation's members — a blind *_monthly.json glob also
# stages the empty [] placeholders that new-talent / bootstrap seed for
# talents not processed yet.
for spec in "${SPECS[@]}"; do
  git add "data/measurements/${spec%%:*}_monthly.json"
done
git add data/measurements/talents.json docs/data.js
git commit -m "Add hololive ${GEN} ($(IFS=', '; echo "${NAMES[*]}"))

densify target-n 2, Praat tracker with F1-F4 formants; retry/rescue for
QC gaps; remeasure-praat normalise. Registry + docs/data.js regenerated.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>" 2>&1 | tail -2

echo "############ $(date '+%F %T')  GEN COMMITTED: $GEN  $(git rev-parse --short HEAD) ############"
