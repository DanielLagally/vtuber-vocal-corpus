#!/usr/bin/env bash
# Keep the bgutil PO-token HTTP provider alive on :4416. yt-dlp (with the
# bgutil plugin on PYTHONPATH) calls it to mint GVS PO tokens, which lets
# fetch run WITHOUT YouTube cookies — no more cookie-rotation bot-checks
# on long batch runs. See CLAUDE.md "PO-token provider".
#
# One-time setup (already done on this machine):
#   git clone -b 1.3.2 https://github.com/Brainicism/bgutil-ytdlp-pot-provider ~/bgutil-ytdlp-pot-provider
#   (cd ~/bgutil-ytdlp-pot-provider/server && deno install --entrypoint src/main.ts --allow-scripts)
#   curl -sL <release>/bgutil-ytdlp-pot-provider.zip | (cd ~/.config/yt-dlp/plugins && unzip -)
#
# Run:  nohup bash scripts/potoken_server.sh > ~/bgutil-server.log 2>&1 &
set -u
export DENO_DIR="${DENO_DIR:-$HOME/.cache/deno}"
FLAKE="${FLAKE:-$HOME/Dev/vtuber-vocal-corpus}"   # for `nix develop` to find deno
SERVER_DIR="$HOME/bgutil-ytdlp-pot-provider/server"
PORT=4416

while true; do
  echo "=== $(date '+%F %T') starting POT server on :$PORT ==="
  nix develop "$FLAKE" --command bash -c \
    "cd '$SERVER_DIR' && exec deno run \
       --allow-env --allow-net --allow-read --allow-write --allow-sys --allow-run --allow-ffi \
       src/main.ts --port $PORT"
  echo "=== $(date '+%F %T') POT server exited ($?); restarting in 5s ==="
  sleep 5
done
