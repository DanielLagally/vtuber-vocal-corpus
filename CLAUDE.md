# vanalysis

Public measurements of hololive chatting-stream speech. v1 is pitch, brightness, voiced-time proxy, and a static plots site generated from aggregates. No audio, transcripts, or embeddings are published. No voice synthesis.

## Stack

Nix flake (`flake.nix`) with CUDA *config* enabled (vttt-style unfree/EULA), project-local `.venv` (`--system-site-packages`), `yt-dlp`, and `ffmpeg-headless`. v1 feature extract is numpy-only. Do not add `torch` / `torchaudio` / `onnxruntime` / `parselmouth` / `librosa` to `withPackages` until stem-split or Praat F0 is actually implemented — those pull multi-GB CUDA/Praat source builds.

Nix users: `direnv exec . <cmd>` or `nix develop`. The agent shell does not auto-load `.envrc`.

```
direnv exec . python -m pytest -q
```

## Layout

- `src/vanalysis/` — library
- `tests/` — product tests (synthetic tones/noise only)
- `fixtures/` — generated at test time, not committed
- `data/` — local audio, Holodex key, raw features; gitignored

## Plots are a permanent record

`data/plots/` is a permanent record, not a scratch dir. Every `vanalysis
plot` invocation writes into a fresh timestamped subdirectory under
`data/plots/runs/` — it never overwrites a previous run's PNGs. Do not add
code that writes plot files to a fixed path outside `runs/<run>/`. Do not
delete old run directories; if `data/plots/` needs tidying, move things
into `runs/`, never `rm`.

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

- **Vocal isolation is pip `audio-separator[gpu]`, not nixpkgs torch.** Same pattern as lunalearn: venv wheel (~6 GB), `LD_LIBRARY_PATH` for nixpkgs cudnn/cudatoolkit. Do **not** set `cudaSupport = true` on nixpkgs — that compiles CUDA torch/onnxruntime from source and fills the disk. `pip install -r requirements.txt` once inside `nix develop`.
- **Do not add `onnxruntime` or `torch` to flake `withPackages`.**
- **`yt-dlp` leaks a python3.14 `PYTHONPATH` into every `nix develop` shell.** The venv is python3.12, so that path's wheels (cffi `_cffi_backend`, cryptography, …) are visible on `sys.path` but not loadable — and worse, pip sees them as already installed and skips them. Always `unset PYTHONPATH` before venv `pip install` / `pip show` / running the app inside `nix develop` (then re-export `PYTHONPATH=src` if needed). If a fresh install fails with `ModuleNotFoundError: audioread` / `_cffi_backend`, this is why: `audioread` is undeclared upstream (librosa 1.0 dropped it, audio-separator's uvr_lib imports it) and `cffi` was skipped by the polluted path — `pip install audioread cffi` into `.venv`.
- **Do not `nix-shell -p python3Packages.parselmouth` (or librosa) to run tests.** Both packages `doCheck = true` and compile Praat / run hundreds of upstream tests. Feature extract is numpy; `direnv exec . python -m pytest -q`.
- **YouTube cookies live in gitignored `data/youtube.cookies.txt`.** Export with `yt-dlp --cookies-from-browser 'chromium:Profile 1' --cookies data/youtube.cookies.txt --skip-download https://www.youtube.com`. Fetch uses that file if present. Never commit it.
- **Fetch pins `youtube:player_client=android` and needs `deno` on PATH** (flake provides it; yt-dlp uses it for EJS/n-challenge). `tv` currently fails with "The page needs to be reloaded"; `mweb` needs a GVS PO token and dies image-only. Per-video yt-dlp failures are skipped and logged — a gap in `data/audio`, never an aborted batch.
- **Holodex returns 403 without a User-Agent.** Send `User-Agent: vanalysis/0.1` plus `X-APIKEY`. Rate limit header is 80/window. `HOLODEX_API_KEY` lives in gitignored `.env` only — never commit it.
- **Holodex `mentions` is omitted unless `include=mentions`.** Without that param, collabs look like solo streams. `topic_id == "singing"` is rare; songs show up as `Original_Song` / `Music_Cover`, and many recent items are `shorts`.
- **High-volume sequential `fetch` in one session can trip YouTube's bot check.** Running the Lamy monthly pipeline (2026-09) right after Luna's densify batch — a lot of back-to-back `yt-dlp` invocations in one day — got almost every fetch rejected with `Sign in to confirm you're not a bot`, cookies notwithstanding (191/198 attempts failed this way; only 5 new fetches got through). Confirmed by testing directly: switching `player_client` (tried `android_vr`) did NOT help, and it also failed on a video with NO cookies passed at all — this is a session/IP-level block, not a per-client PO-token gap and not fixable by client-hopping. This is a different failure mode from privated videos: it's the whole batch blocked at once, near-immediately, not scattered individual 404s. **The pipeline now detects this itself** (`fetch.BotCheckDetected`, `fetch.looks_like_bot_check`) and stops the whole batch immediately instead of skip-and-continuing through the rest of the ids — see `fetch_audio_many`, `densify.run_densify`'s `stopped_early` field, and the plain `fetch` CLI command. An ordinary per-video failure (private/deleted/region-locked) still skips and continues as before; only the bot-check message triggers the hard stop. If it fires: don't retry immediately, it likely won't help and may extend any block. Re-export cookies via a **private/incognito** browser window per the [yt-dlp wiki's cookie-export method](https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies) (not the live `--cookies-from-browser` approach, which the wiki says exports rotating cookies that work less reliably here) and/or space large fetch batches out across sessions/days rather than running them back-to-back. If cookies alone don't clear it, a self-hosted PO token provider ([bgutil-ytdlp-pot-provider](https://github.com/Brainicism/bgutil-ytdlp-pot-provider), has a Deno-based mode — this repo's flake already ships `deno`) is the more durable fix.
- **Early-era streams are frequently privated.** Backfilling Luna's 2020 debut months (`densify`, 2026-09) hit "Private video" on ~14/16 fetch failures — Cover/the talent had privated most of her earliest 2020-01 to 2020-05 VODs (including the literal debut stream). This is not a pipeline bug: no cookie/client change recovers a genuinely privated video. Expect the same when backfilling any other talent's earliest era; treat it as a real data-availability wall, not something to debug further.
- **`.venv` is `--system-site-packages`.** Delete `.venv` and reload if the Nix Python version warning fires, or if nixpkgs packages are missing inside the venv.
- **`nix flake check` / `nix build` only see git-tracked-or-staged files** if this becomes a git-backed flake (`self`). `git add` new files before those commands.
