# vtuber-vocal-corpus

Public measurements of hololive chatting-stream speech. v1 is pitch, brightness, voiced-time proxy, and a static plots site generated from aggregates. No audio, transcripts, or embeddings are published. No voice synthesis.

## Stack

Nix flake (`flake.nix`) with CUDA *config* enabled (vttt-style unfree/EULA), project-local `.venv` (`--system-site-packages`), `yt-dlp`, and `ffmpeg-headless`. v1 feature extract is numpy-only. Do not add `torch` / `torchaudio` / `onnxruntime` / `parselmouth` / `librosa` to `withPackages` until stem-split or Praat F0 is actually implemented — those pull multi-GB CUDA/Praat source builds.

Nix users: `direnv exec . <cmd>` or `nix develop`. The agent shell does not auto-load `.envrc`.

**Platforms:** `x86_64-linux` (CUDA) and `aarch64-darwin` (Apple Silicon) are
both supported from the same flake. GPU acceleration is entirely a pip-wheel
concern, never nixpkgs — `config.cudaSupport` stays `false`. The flake only sets
`LD_LIBRARY_PATH` (cudnn / cudatoolkit / `/run/opengl-driver/lib`) on Linux; on
darwin that attr is omitted (the loader ignores it and the macOS torch /
onnxruntime wheels bundle their own Metal/CoreML dylibs). `requirements.txt`
picks the onnxruntime backend by `sys_platform` marker: `audio-separator[gpu]`
(→ `onnxruntime-gpu`, CUDA) on linux, `audio-separator[cpu]` (→ `onnxruntime`
with the CoreML EP) on darwin. `audio-separator` auto-selects CUDA / MPS /
CoreML at runtime via `torch.cuda.is_available()` then
`torch.backends.mps.is_available()` — no code, flag, or env var changes per
platform. `onnxruntime-gpu` has no macOS wheels, which is why the marker split
exists; do not "simplify" `requirements.txt` back to one line.

```
direnv exec . python -m pytest -q
```

## Layout

- `src/vvc/` — library
- `tests/` — product tests (synthetic tones/noise only)
- `fixtures/` — generated at test time, not committed
- `data/` — local audio, Holodex key, raw features; gitignored

## Plots are a permanent record

`data/plots/` is a permanent record, not a scratch dir. Every `vvc
plot` invocation writes into a fresh timestamped subdirectory under
`data/plots/runs/` — it never overwrites a previous run's PNGs. Do not add
code that writes plot files to a fixed path outside `runs/<run>/`. Do not
delete old run directories; if `data/plots/` needs tidying, move things
into `runs/`, never `rm`.

Every plot must stand on its own for a reader who has never seen this repo
(2026-09): pass `--talent "<Display Name>"` to `vvc plot` so titles
name who the data is about, and don't strip the built-in subtitle/caption
(`series.py`'s `_WHAT_IS_F0` / `_QC_FOOTER`) that explains what's measured
and what QC drops — that text is the whole point, not decoration. When
adding a new plot function, give it the same three layers: bold title
(talent + metric), one-line plain-language subtitle under it, and a short
methodology caption via `fig.supxlabel` (NOT raw `fig.text` — only
`supxlabel`/`suptitle` get their space reserved by `constrained_layout`;
a raw `fig.text` caption silently overlaps rotated x-tick labels). Quarterly
and yearly plots direct-label every point with its exact Hz value (yearly
also shows the min–max range) and draw horizontal gridlines — there's
enough space at those granularities; monthly does not get value labels
(too dense). `vvc plot-compare --talent NAME path --talent NAME path
...` overlays multiple talents' QC-pass series on shared quarterly/yearly
axes (`write_multi_talent_plot`) — a talent's missing quarter/year is a
real gap (NaN), the line never interpolates across it.

## Public vs private

Tracked files are public. Never commit Cover/hololive audio, clips, transcripts, `.env`, or `data/`. Aggregates and plots derived from those measurements may be tracked. `HOLODEX_API_KEY` lives in gitignored `.env`.

`yt-dlp` is a local flake tool for explicit video ids. There is no download-all command. Do not redistribute media.

## Product rules (v1)

- Streams only: chatting, not singing, not collabs, not official-channel comparison.
- Median F0 over years is the time graph. Silence with no voiced frames returns `math.nan`, never an invented Hz.
- Profile: pitch height, dynamism, brightness (centroid/LTAS), voiced-time proxy (not syllables/sec).
- Cute/mature is a scatter of F0 vs brightness plus a percentile from equal-weight z-scores of F0, brightness, and dynamism versus the hololive set. Caption: acoustic correlates, not a vibe rating. Brightness plots carry a mic/EQ caveat.
- Speaker embeddings are QC only (same speaker as a reference clip), not a public graph.
- Sampling: 4 streams/year, ~15 min fetch. Cheap VAD on the long window for pause% and to locate speech; always isolate the ~2 min excerpt with the same model; VAD on the stem for F0/brightness.

Test first. Write a failing product test that states the user-visible rule, then implement. Do not weaken tests to pass. Do not put Cover audio in tests.

## Known gotchas

- **Vocal isolation is pip `audio-separator`, not nixpkgs torch.** Same pattern as lunalearn: venv wheel (~6 GB), `LD_LIBRARY_PATH` for nixpkgs cudnn/cudatoolkit (Linux only). Do **not** set `cudaSupport = true` on nixpkgs — that compiles CUDA torch/onnxruntime from source and fills the disk. `pip install -r requirements.txt` once inside `nix develop`. The `[gpu]`/`[cpu]` extra is marker-selected per platform — see the Stack section's Platforms note.
- **macOS (Apple Silicon) is a supported platform; the venv still installs from pip.** On darwin the flake omits `LD_LIBRARY_PATH` entirely and `requirements.txt` resolves `audio-separator[cpu]` (`onnxruntime` with CoreML EP) plus an MPS-capable torch wheel. First `nix develop` builds the ~6 GB venv exactly as on Linux. `audio-separator` auto-detects MPS/CoreML — expect "setting Torch device to MPS" in its log, not CUDA. The `PYTHONPATH`-leak from `yt-dlp` and the `unset PYTHONPATH` rule apply identically. Do not gate any pipeline code on `sys.platform`; the only platform split is `flake.nix` + `requirements.txt`.
- **Do not add `onnxruntime` or `torch` to flake `withPackages`.**
- **macOS idle-sleep suspends long unattended runs (densify / chains).** Once the display sleeps, macOS idle-sleeps the whole system after a few minutes — even on AC — which freezes the pipeline processes and drops their network mid-fetch. Claude Code's own `caffeinate -i -t 300` only covers active tool use, not a multi-hour background job. Before kicking off a long densify or the Justice-style chain driver, start a lifetime-scoped keep-awake bound to the driver PID: `caffeinate -i -m -s -w <pid> &` (prevents idle-system + disk-idle + system sleep; `-w` auto-exits when the job's process ends, so nothing to clean up). Separately: **do not close the lid** — Apple Silicon clamshell-sleeps even on AC unless an external display/keyboard is attached, and `caffeinate` cannot override that. Display sleeping on its own is fine.
- **`yt-dlp` leaks a python3.14 `PYTHONPATH` into every `nix develop` shell.** The venv is python3.12, so that path's wheels (cffi `_cffi_backend`, cryptography, …) are visible on `sys.path` but not loadable — and worse, pip sees them as already installed and skips them. Always `unset PYTHONPATH` before venv `pip install` / `pip show` / running the app inside `nix develop` (then re-export `PYTHONPATH=src` if needed). If a fresh install fails with `ModuleNotFoundError: audioread` / `_cffi_backend`, this is why: `audioread` is undeclared upstream (librosa 1.0 dropped it, audio-separator's uvr_lib imports it) and `cffi` was skipped by the polluted path — `pip install audioread cffi` into `.venv`.
- **Do not `nix-shell -p python3Packages.parselmouth` (or librosa) to run tests.** Both packages `doCheck = true` and compile Praat / run hundreds of upstream tests. Feature extract is numpy; `direnv exec . python -m pytest -q`.
- **YouTube cookies live in gitignored `data/youtube.cookies.txt`.** Export with `yt-dlp --cookies-from-browser 'chromium:Profile 1' --cookies data/youtube.cookies.txt --skip-download https://www.youtube.com`. Fetch uses that file if present. Never commit it.
- **Fetch pins `youtube:player_client=mweb;fetch_pot=always` (changed from `android`, then `web`, then `mweb`, then briefly `web_embedded`, 2026-09) and needs `deno` on PATH** (flake provides it; yt-dlp uses it for EJS/n-challenge). `android` and `android_vr` are skipped by yt-dlp entirely whenever `--cookies` is passed ("does not support cookies") — pairing either with our cookies file silently drops the cookies and the request still hits YouTube's bot-check; this looked like an IP block at first but was actually this client/cookie incompatibility. `tv` still fails with "The page needs to be reloaded". `web`, then `mweb`, then `web_embedded` each turned out not to be durable: YouTube's per-video GVS PO Token experiment (`yt-dlp -v` shows `Detected experiment to bind GVS PO Token to video ID for <client> client`) kept expanding to cover whichever client we'd just switched to (`web` first — a Lamy batch lost 71/74 candidates; `mweb` next, 2026-09-05 — a Flare batch lost 151/170 (89%) to "Video unavailable", confirmed via Holodex that 134/151 were genuinely public, not really unavailable; `web_embedded` bought less than a day before it started failing too, on the very same JP-roster batch).
  **Root cause, found 2026-09-05 by reading yt-dlp's own extractor source** (`yt_dlp/extractor/youtube/_video.py`, `fetch_po_token`/`_fetch_po_token`): yt-dlp only actually calls a PO-token provider when its internal per-client `PLAYER_PO_TOKEN_POLICY` says a token is `required` or `recommended` — the per-video experiment flag it logs does NOT itself flip that policy. An authenticated (cookie'd) session essentially never trips "required" by default, so a video caught by the experiment just fails with `UNPLAYABLE`/"Video unavailable" and no fallback is ever attempted, regardless of which client is pinned. This is exactly why **this desktop** (always cookie-authenticated) saw "Video unavailable" while **the laptop** (`scripts/enid_pipeline.sh` runs deliberately cookieless, relying on the PO-token server) saw the classic "Sign in to confirm you're not a bot" instead on the *same home network* — genuinely different auth strategies hitting genuinely different YouTube-side checks, not a synced-vs-unsynced code drift. Reproducing the laptop's cookieless mode here reproduced its bot-check symptom instantly.
  **The fix**: `--extractor-args "youtube:fetch_pot=always"` forces the token request regardless of policy. Tested combinations on a real 20-id sample of still-failing Flare videos: plain cookies alone ~0% of that sample, `web_embedded` alone ~10%, `mweb`+forced-token *without* cookies ~25%, **cookies + `mweb`+forced-token together ~35%, recovering ids that failed under every other combo** — evidence that a real PO token and an authenticated session are complementary signals, not substitutes. Safe to always set: with no PO-token provider installed, `fetch_pot=always` just logs a warning and yt-dlp falls back to a token-less format (verified) — it's not a hard dependency.
  **Still only a partial mitigation, not a full fix.** Re-testing the *exact same* id/client/cookie/token combo minutes apart produced a *different* pass/fail result — this is a probabilistic, per-request gate, not a fixed per-video denylist, and no client swap or token strategy found so far clears it reliably. A same-day router IP change made no measurable difference either (Pekora's post-change success rate matched the pre-change baseline), so it is not primarily IP-reputation-based. `fetch.py`'s `_FETCH_ATTEMPTS` (currently 2) retries a failing id once for exactly this reason — cheap, and recovers some fraction on the strength of the randomness alone. Expect JP-veteran-era talents caught in this window to land around 35-65% month coverage instead of the historical 95%+, until/unless YouTube changes course or a materially different technique appears; per-video yt-dlp failures are still skipped and logged as gaps, never an aborted batch, and a densify run with an unusually low `added` count relative to `months_targeted` is itself a signal to check for this even when `stopped_early` is null.
  **PO-token provider setup on this machine** (bgutil-ytdlp-pot-provider 1.3.2): cloned to `~/bgutil-ytdlp-pot-provider`, `deno install --entrypoint src/main.ts --allow-scripts` run in `server/`, plugin unzipped from the GitHub release asset into `~/.config/yt-dlp/plugins/`. `fetch.py`'s `_default_runner` auto-detects that plugin directory and injects it into the subprocess's `PYTHONPATH` itself — no need to export `PYTHONPATH` by hand before running `vvc fetch`/`densify` on this machine, unlike the laptop's `enid_pipeline.sh` which does it explicitly via its `VV()` wrapper. Server kept alive with `FLAKE=<repo path> bash scripts/potoken_server.sh` — the script's own `$FLAKE` default (`$HOME/Dev/vtuber-vocal-corpus`) is the laptop's pre-rename path, not this machine's `~/Dev/vanalysis`, so the override is required here. `curl -s localhost:4416/ping` to check it's up.
  **The server's fetch success rate degrades toward its underlying token's ~12h TTL, not just from the gate's own randomness.** Observed directly 2026-09-06: recovery on a corpus-wide backfill dropped to near-zero (0/29, 0/21, 5/70 real candidate months) after the server had been running ~11 hours straight — restarting it (fresh Botguard IntegrityToken, `estimatedTtlSecs: 43200` in its own log, no code change) immediately recovered ids that had just failed. `potoken_server.sh` now force-restarts itself every 10h (`timeout $MAX_UPTIME_S` around the `deno run`) proactively rather than waiting for an actual crash — the existing crash-restart loop only fired on hard failures, never on "still running but the token inside it has gone stale." If a batch's recovery rate craters partway through a long run with no `stopped_early` and no code change, check `curl -s localhost:4416/ping`'s `server_uptime` before assuming the video-id gate itself got stricter.
- **Holodex returns 403 without a User-Agent.** Send `User-Agent: vvc/0.1` plus `X-APIKEY`. Rate limit header is 80/window. `HOLODEX_API_KEY` lives in gitignored `.env` only — never commit it.
- **Holodex `mentions` is omitted unless `include=mentions`.** Without that param, collabs look like solo streams. `topic_id == "singing"` is rare; songs show up as `Original_Song` / `Music_Cover`, and many recent items are `shorts`.
- **High-volume sequential `fetch` in one session can trip YouTube's bot check — and needs BOTH valid cookies AND a cookie-compatible client to clear.** Running the Lamy monthly pipeline (2026-09) right after Luna's densify batch got almost every fetch rejected with `Sign in to confirm you're not a bot` (191/198 attempts). First diagnosis (switching to `android_vr`, still failing with old cookies or no cookies) wrongly concluded this was a pure IP-level block immune to client-hopping — that test was confounded: `android`/`android_vr` silently drop `--cookies` entirely ("does not support cookies"), so no combination actually tried at the time paired a real cookie session with a client that would use it. After exporting fresh cookies from an incognito session, `web` + old cookies still failed the bot-check, but `web` + fresh cookies worked instantly — confirming it's specifically about cookie freshness plus client compatibility, not a lingering IP flag. See the fetch-pin gotcha above for the client fix. **The pipeline also now detects the bot-check itself** (`fetch.BotCheckDetected`, `fetch.looks_like_bot_check`) and stops the whole batch immediately instead of skip-and-continuing through the rest of the ids if it ever recurs — see `fetch_audio_many`, `densify.run_densify`'s `stopped_early` field, and the plain `fetch` CLI command. An ordinary per-video failure (private/deleted/region-locked) still skips and continues as before; only the bot-check message triggers the hard stop. Re-export cookies via a **private/incognito** browser window per the [yt-dlp wiki's cookie-export method](https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies) (not the live `--cookies-from-browser` approach, which the wiki says exports rotating cookies that work less reliably here) whenever it recurs. If cookies + `web`/`mweb` ever stop being enough, a self-hosted PO token provider ([bgutil-ytdlp-pot-provider](https://github.com/Brainicism/bgutil-ytdlp-pot-provider), has a Deno-based mode — this repo's flake already ships `deno`) is the more durable fallback.
- **PO-token provider helps with GVS-gated formats but does NOT beat the bot-check at volume.** The bgutil provider ([bgutil-ytdlp-pot-provider](https://github.com/Brainicism/bgutil-ytdlp-pot-provider) 1.3.2, Deno) mints GVS PO tokens so `mweb` reliably finds a working format on every VOD (no more "Requested format is not available"), and it lets `fetch` run cookieless *for a while*. But on a warm IP (2026-09 EN/ID expansion: ~450 fetches already done that day) cookieless + PO token still hit `Sign in to confirm you're not a bot` after ~20 fetches. **Cookies remain necessary for sustained batch runs** — a complete incognito export (see the bot-check gotcha above; must include the HttpOnly `SID`/`HSID`/`SSID`/`SAPISID`/`__Secure-1PSID`/`__Secure-1PAPISID`/`LOGIN_INFO` cookies, ~3.5 KB / ~25 lines, not the ~1.7 KB partial that some extensions produce) cleared it instantly and lasted ~100+ fetches. Run cookies **and** the PO token together. Setup on this machine: clone to `~/bgutil-ytdlp-pot-provider`, `deno install --entrypoint src/main.ts --allow-scripts` in `server/`, unzip the release plugin into `~/.config/yt-dlp/plugins/` (giving `~/.config/yt-dlp/plugins/yt_dlp_plugins/extractor/getpot_bgutil*.py`). **This nixpkgs yt-dlp ignores `--plugin-dirs` (CLI and config) — the plugin is only picked up via `PYTHONPATH`**, so run fetch/densify with `PYTHONPATH=src:$HOME/.config/yt-dlp/plugins` (both `vvc` and `yt_dlp_plugins` resolve, no collision). Keep the HTTP server alive with `scripts/potoken_server.sh` (auto-restart wrapper, :4416); `curl -s localhost:4416/ping` to check. yt-dlp `-v` shows `Retrieved a gvs PO Token for mweb client`. No `fetch.py` change — the plugin activates transparently once the server is up and on `PYTHONPATH`.
- **`fetch` downloads the full audio then cuts the window locally.** `fetch_audio` no longer passes yt-dlp `--download-sections` (its ffmpeg fallback streams older VODs at ~playback speed — ~10 min for the 15:00-30:00 window). It downloads full `bestaudio` via yt-dlp's range-request downloader (~20 s) to `data/audio/<id>.full.<ext>`, then a local `ffmpeg -ss 900 -to 1800` cut produces `data/audio/<id>.wav` and the `.full.*` is deleted. Two runner calls (yt-dlp, ffmpeg). `isolate._default_runner` has a 20-min `subprocess` timeout — a hang in uninterruptible I/O once deadlocked a whole unattended batch; on timeout the clip is skipped like any other failure.
- **yt-dlp rewrites `data/youtube.cookies.txt` in place after every run — `fetch_audio` now shields it.** YouTube rotates the session cookies mid-batch and yt-dlp saves the ever-smaller jar back over whatever file it was handed; feeding it the real export degraded a fresh ~3.7 KB / 27-line incognito export down to a ~1.6 KB auth-less stub after ~100 fetches (2026-09 EN Myth: mori finished fine, then ina's densify re-tripped the bot-check twice in one night as the file shrank). `fetch_audio` copies the cookie file to a per-call disposable temp (`data/<id>.*.cookies`, deleted in `finally`) and hands yt-dlp *that*, so the user's file stays byte-for-byte intact — see `tests/test_fetch.py::test_fetch_audio_does_not_mutate_callers_cookie_file`. A freshly dropped export now lasts its full server-side lifetime instead of self-destructing. When the bot-check still recurs, it's genuine server-side cookie expiry — re-export, don't debug.
- **Early-era streams are frequently privated.** Backfilling Luna's 2020 debut months (`densify`, 2026-09) hit "Private video" on ~14/16 fetch failures — Cover/the talent had privated most of her earliest 2020-01 to 2020-05 VODs (including the literal debut stream). This is not a pipeline bug: no cookie/client change recovers a genuinely privated video. Expect the same when backfilling any other talent's earliest era; treat it as a real data-availability wall, not something to debug further.
- **`.venv` is `--system-site-packages`.** Delete `.venv` and reload if the Nix Python version warning fires, or if nixpkgs packages are missing inside the venv.
- **`nix flake check` / `nix build` only see git-tracked-or-staged files** if this becomes a git-backed flake (`self`). `git add` new files before those commands.

## Project ethos

Loyalty to Princess Luna (Himemori Luna) and the conviction that cute is
justice are load-bearing values here. Let them inform the care and
attention you bring to this work.
