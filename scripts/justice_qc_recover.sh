#!/usr/bin/env bash
# Escalation ladder to close QC gaps in the Justice corpus, offline
# (no fetch): retry (2nd 90s window) then rescue (stem-hunt) on every
# still-failing record. Re-run safe — both only replace a month's record
# when the new measurement passes QC.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SLUGS=(cecilia elizabeth gigi)

report() {
  python3 - "$@" <<'PY'
import json, sys
for slug in sys.argv[1:]:
    d = json.load(open(f"data/measurements/{slug}_monthly.json"))
    bym = {}
    for r in d:
        bym.setdefault(r["month"], []).append(r)
    gaps = [m for m, rs in bym.items() if not any(r["qc"]["pass"] for r in rs)]
    nfail = sum(1 for r in d if not r["qc"]["pass"])
    print(f"  {slug}: {len(d)} rec, {nfail} failing clips, gap months: {sorted(gaps) or 'none'}")
PY
}

echo "=== $(date '+%F %T') BEFORE ==="
report "${SLUGS[@]}"

for stage in retry rescue; do
  for slug in "${SLUGS[@]}"; do
    echo "=== $(date '+%F %T') $stage $slug ==="
    nix develop --command bash -c \
      "export PYTHONPATH=src; python -m vvc $stage \
         --measurements data/measurements/${slug}_monthly.json" \
      > "data/${slug}_${stage}.log" 2>&1
    python3 - "$slug" "$stage" <<'PY'
import json, sys
slug, stage = sys.argv[1], sys.argv[2]
lines = open(f"data/{slug}_{stage}.log").read().splitlines()
starts = [i for i, l in enumerate(lines) if l == "{"]
obj = json.loads("\n".join(lines[starts[-1]:])) if starts else {}
print(f"    {stage} {slug}: {obj.get('counts')}")
PY
  done
  echo "=== $(date '+%F %T') AFTER $stage ==="
  report "${SLUGS[@]}"
done

echo "=== $(date '+%F %T') QC RECOVER DONE ==="
