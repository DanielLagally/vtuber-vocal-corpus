# Record format

Measurements are stored as one JSON file per talent, with one record per clip.

## Current records

`data/measurements/v2/<talent>.json`:

| Field | Meaning |
|---|---|
| `id` | YouTube video id; `<id>@<seconds>` for a window from a one-off stretch |
| `month` | Month the stream aired (`YYYY-MM`) |
| `window` | Start and end of the clip, in seconds into the fetched audio |
| `title`, `topic_id`, `duration_s`, `available_at` | Stream metadata from Holodex |
| `reliability_band` | `high`, `mid`, or `low`, from topic and length |
| `features` | Measured values (see [`measurement.md`](measurement.md)), plus the tracker and analysis settings used |
| `qc` | Stored copy of the quality verdict (see [`qc.md`](qc.md)) |
| `separation_applied`, `model` | Whether the voice was separated, and with which model |
| `background_ratio_db` | Background level of the source |
| `source_audio` | Local file the clip was measured from |
| `measured_at`, `corpus_release` | When and in which build the clip was measured |
| `legacy` | `true` when the audio was unavailable and the features come from the original pipeline |
| `one_off`, `section`, `language` | Present on one-off stretches: the stretch in stream time, and its language if different from the talent's usual one |

A feature that could not be measured is missing or `nan`, never a substitute
value. Monthly and yearly values are computed from these records, never stored.

## Original records

`data/measurements/<talent>_monthly.json` holds the records behind the v1 site,
with a smaller feature set. `data/measurements/talents.json` maps each file to a
talent's display name.

## Site data

`docs/data.js` is generated from the current records. Per talent it holds
typical values with confidence intervals, percentiles, noise floors, trends,
monthly, quarterly, and yearly series, and closest-voice rankings.
