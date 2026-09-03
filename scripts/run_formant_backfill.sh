#!/usr/bin/env bash
# Backfill f1_hz-f4_hz (added to praat_features.stem_features 2026-09-03,
# see PLAN.md) into already-measured talents via the EXISTING
# remeasure-praat command — it already calls stem_features() on each
# record's already-resolved LOCAL audio (data/windows, data/stems_fast —
# deliberately not offloaded, see CLAUDE.md), so no new CLI and no
# re-fetch/re-isolate is needed. Sequential (remeasure.py's own
# snapshot-then-overwrite pattern isn't safe to run twice on the SAME
# file concurrently, though separate talents' files have no shared
# state — sequential here is simplicity, not a hard requirement).
#
# Multi-machine note: if running this on more than one machine at once
# (e.g. desktop + laptop) against the same repo, split the talent list
# between them — never pass the same talent to two machines running
# concurrently, and pull/push between runs so neither machine's commit
# clobbers the other's. To see which talents still need backfilling:
#   for f in data/measurements/*_monthly.json; do
#     direnv exec . python -c "
#   import json,sys
#   r=json.load(open('$f'))
#   sys.exit(0 if (not r or 'f1_hz' in r[0]['features']) else 1)" \
#       || echo "$f"
#   done
#
# Portable across machines/OSes (Linux desktop + aarch64-darwin laptop,
# see flake.nix) — resolves the repo root from the script's own
# location rather than a hardcoded path.
#
# Usage: run_formant_backfill.sh <name1> <name2> ...
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
for name in "$@"; do
  echo "##### REMEASURE: $name #####"
  direnv exec . python -m vvc remeasure-praat --measurements "data/measurements/${name}_monthly.json"
  echo "--- $name exit $? ---"
done
echo "=== FORMANT BACKFILL DONE ==="
