"use strict";

// Caption text mirrors series.py's _WHAT_IS_F0 / _QC_FOOTER wording (no
// Python import path available from a static page, so the strings are
// copied here — keep them in sync if series.py's wording changes).
const WHAT_IS_F0 =
  "Median F0 = typical pitch of voiced speech in a ~90 s clip of a public " +
  "chatting stream. Higher = higher-pitched voice.";
const QC_FOOTER =
  "QC excludes clips with too little voice, unstable pitch (BGM/tracker " +
  "error), or an implausible reading. Gaps are missing/failed data, never 0 Hz.";
const CUTE_MATURE_CAPTION =
  "Acoustic correlates, not a vibe rating. Percentile = equal-weight average " +
  "of each talent's z-scored mean F0, brightness, and pitch dynamism vs. " +
  "every other talent shown here with QC-pass data.";

// Each talent's official/fandom-recognized image color, where confidently
// known. A talent NOT in this map falls back to Plotly's default
// categorical palette (assignTalentColor below) rather than a guessed
// hex value — colors are identity, worth getting right or not at all.
// STILL NEEDS VERIFICATION: Takane Lui, Chihaya, Riona, Vivi, Hajime,
// Kanade, Raden, Ririka (fill in real values here once confirmed).
const TALENT_COLORS = {
  "Himemori Luna": "#FF6FA5",
  "Yukihana Lamy": "#4FC3F7",
  Ao: "#3A7CA5",
  Niko: "#E8973D",
  Su: "#3AA6B9",
};

// Plotly's default categorical palette (d3.schemeCategory10-derived),
// used for any talent not in TALENT_COLORS — cycled independently of
// the hardcoded talents so no two visible traces share a color.
const FALLBACK_PALETTE = [
  "#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A",
  "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
];
const _fallbackAssigned = new Map();
function assignFallbackColors(allNames) {
  // Deterministic regardless of selection/render order: walk every
  // talent alphabetically once at load, skipping anyone with a
  // hardcoded color, so the same un-mapped talent always gets the same
  // fallback color across a session.
  let i = 0;
  for (const name of [...allNames].sort()) {
    if (TALENT_COLORS[name]) continue;
    _fallbackAssigned.set(name, FALLBACK_PALETTE[i % FALLBACK_PALETTE.length]);
    i += 1;
  }
}
function talentColor(name) {
  return TALENT_COLORS[name] || _fallbackAssigned.get(name);
}

// Yearly metrics: keyed by the data.json "yearly" feature key. label/unit
// mirror the existing per-talent PNGs (__main__.py's _EXTRA_FEATURE_PLOTS).
const YEARLY_METRICS = {
  median_f0: { label: "F0 (Pitch) — Yearly", unit: "Median F0 (Hz)", caption: WHAT_IS_F0 },
  brightness_hz: {
    label: "Brightness — Yearly",
    unit: "Brightness (Hz)",
    caption:
      "Spectral centroid - a brighter/more forward vs. darker/warmer voice. " +
      "Mic/EQ and leftover BGM affect this too; trust the relative shape.",
  },
  dynamism_semitones: {
    label: "Pitch Dynamism — Yearly",
    unit: "Dynamism (semitones)",
    caption:
      "Mean semitone change between consecutive voiced frames - how much " +
      "the pitch actually moves, not just its static spread.",
  },
  jitter_local: {
    label: "Jitter — Yearly",
    unit: "Jitter (local, fraction)",
    caption: "Cycle-to-cycle pitch-period timing irregularity.",
  },
  shimmer_local: {
    label: "Shimmer — Yearly",
    unit: "Shimmer (local, fraction)",
    caption: "Cycle-to-cycle amplitude irregularity.",
  },
  hnr_db: {
    label: "Harmonics-to-Noise Ratio — Yearly",
    unit: "HNR (dB)",
    caption: "Higher = clearer/more tonal voice; lower = breathier/noisier.",
  },
  loudness_dynamics_db: {
    label: "Loudness Dynamics — Yearly",
    unit: "Loudness spread (dB)",
    caption:
      "Spread of frame loudness (RMS in dB) within a clip - animated volume " +
      "swings vs. a flat, even delivery.",
  },
};

let DATA = null;
let selectedTalents = new Set();

function main() {
  // window.SITE_DATA comes from data.js (a <script>, not a fetch() —
  // fetch("data.json") is blocked by CORS when this page is opened via
  // file://, which is the whole point of a static, no-server site).
  DATA = window.SITE_DATA;
  selectedTalents = new Set(Object.keys(DATA.talents));
  assignFallbackColors(Object.keys(DATA.talents));

  buildMetricPicker();
  buildTalentList();

  document.getElementById("select-all").addEventListener("click", () => {
    selectedTalents = new Set(Object.keys(DATA.talents));
    syncCheckboxes();
    render();
  });
  document.getElementById("select-none").addEventListener("click", () => {
    selectedTalents = new Set();
    syncCheckboxes();
    render();
  });
  const picker = document.getElementById("metric-picker");
  picker.addEventListener("change", () => {
    location.hash = picker.value;
    render();
  });

  // Deep-linkable views: #cute_mature, #yearly:brightness_hz, etc.
  const initial = location.hash.slice(1);
  if (initial && [...picker.options].some((o) => o.value === initial)) {
    picker.value = initial;
  }

  render();
}

function presentYearlyKeys() {
  const present = new Set();
  for (const talent of Object.values(DATA.talents)) {
    for (const key of Object.keys(talent.yearly || {})) present.add(key);
  }
  return Object.keys(YEARLY_METRICS).filter((key) => present.has(key));
}

function buildMetricPicker() {
  const picker = document.getElementById("metric-picker");
  const options = [
    { value: "f0_monthly", label: "F0 (Pitch) — Monthly" },
    { value: "f0_quarterly", label: "F0 (Pitch) — Quarterly" },
  ];
  for (const key of presentYearlyKeys()) {
    options.push({ value: `yearly:${key}`, label: YEARLY_METRICS[key].label });
  }
  if (Object.keys(DATA.cute_mature).length > 0) {
    options.push({ value: "cute_mature", label: "Cute × Mature (F0 vs Brightness)" });
  }
  picker.innerHTML = options
    .map((o) => `<option value="${o.value}">${o.label}</option>`)
    .join("");
}

function buildTalentList() {
  const list = document.getElementById("talent-list");
  const names = Object.keys(DATA.talents).sort();
  list.innerHTML = names
    .map(
      (name) => `
      <label>
        <input type="checkbox" value="${name}" checked />
        <span>${name}</span>
      </label>`
    )
    .join("");
  list.addEventListener("change", (event) => {
    const name = event.target.value;
    if (event.target.checked) selectedTalents.add(name);
    else selectedTalents.delete(name);
    render();
  });
}

function syncCheckboxes() {
  for (const input of document.querySelectorAll("#talent-list input")) {
    input.checked = selectedTalents.has(input.value);
  }
}

function selectedNames() {
  return Object.keys(DATA.talents)
    .filter((name) => selectedTalents.has(name))
    .sort();
}

function render() {
  const metric = document.getElementById("metric-picker").value;
  const chart = document.getElementById("chart");
  const caption = document.getElementById("caption");
  const names = selectedNames();

  if (names.length === 0) {
    Plotly.purge(chart);
    caption.textContent = "Select at least one talent.";
    return;
  }

  if (metric === "f0_monthly") {
    renderLineSeries(names, (t) => DATA.talents[t].monthly_f0_qc, "Median F0 (Hz)");
    caption.textContent = `${WHAT_IS_F0}\n\n${QC_FOOTER}`;
  } else if (metric === "f0_quarterly") {
    renderQuarterly(names);
    caption.textContent = `${WHAT_IS_F0}\n\n${QC_FOOTER}`;
  } else if (metric.startsWith("yearly:")) {
    const key = metric.slice("yearly:".length);
    renderYearly(names, key);
    caption.textContent = `${YEARLY_METRICS[key].caption}\n\n${QC_FOOTER}`;
  } else if (metric === "cute_mature") {
    renderCuteMature(names);
    caption.textContent = CUTE_MATURE_CAPTION;
  }
}

function renderLineSeries(names, seriesFn, yLabel) {
  const traces = names.map((name) => {
    const points = seriesFn(name) || [];
    return {
      type: "scatter",
      mode: "lines+markers",
      name,
      x: points.map((p) => p[0]),
      y: points.map((p) => p[1]),
      connectgaps: false,
      marker: { color: talentColor(name) },
      line: { color: talentColor(name) },
    };
  });
  Plotly.react("chart", traces, layout({ yLabel, xType: "category" }));
}

function renderQuarterly(names) {
  const traces = names.map((name) => {
    const points = DATA.talents[name].quarterly_f0 || [];
    return {
      type: "scatter",
      mode: "lines+markers",
      name,
      x: points.map((p) => p.quarter),
      y: points.map((p) => p.mean),
      connectgaps: false,
      marker: { color: talentColor(name) },
      line: { color: talentColor(name) },
    };
  });
  Plotly.react("chart", traces, layout({ yLabel: "Median F0 (Hz)", xType: "category" }));
}

function renderYearly(names, key) {
  const traces = names.map((name) => {
    const points = (DATA.talents[name].yearly && DATA.talents[name].yearly[key]) || [];
    return {
      type: "scatter",
      mode: "lines+markers",
      name,
      x: points.map((p) => p.year),
      y: points.map((p) => p.median),
      connectgaps: false,
      marker: { color: talentColor(name) },
      line: { color: talentColor(name) },
    };
  });
  Plotly.react(
    "chart",
    traces,
    layout({ yLabel: YEARLY_METRICS[key].unit, xType: "category" })
  );
}

function renderCuteMature(names) {
  const rows = names
    .map((name) => ({ name, point: DATA.cute_mature[name] }))
    .filter((row) => row.point);
  const trace = {
    type: "scatter",
    mode: "markers+text",
    text: rows.map((r) => r.name),
    textposition: "top center",
    x: rows.map((r) => r.point.f0_mean),
    y: rows.map((r) => r.point.brightness_mean),
    marker: { size: 12, color: rows.map((r) => talentColor(r.name)) },
    hovertemplate:
      "%{text}<br>F0: %{x:.0f} Hz<br>Brightness: %{y:.0f} Hz<br>Percentile: %{customdata:.0f}<extra></extra>",
    customdata: rows.map((r) => r.point.percentile),
  };
  Plotly.react(
    "chart",
    [trace],
    layout({ yLabel: "Brightness (Hz)", xLabel: "Median F0 (Hz)", xType: "linear" })
  );
}

function layout({ yLabel, xLabel, xType }) {
  const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const fg = dark ? "#e6e8ec" : "#1a1d23";
  const grid = dark ? "#2a2e38" : "#dde1e7";
  return {
    autosize: true,
    height: 480,
    margin: { t: 20, r: 20, b: 60, l: 60 },
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { color: fg },
    xaxis: {
      type: xType,
      title: xLabel || "",
      gridcolor: grid,
      tickangle: -45,
      // Plotly's default category order is "first seen across traces",
      // which interleaves each talent's own months/quarters/years in
      // whatever order their trace was added — not chronological. Every
      // category here is a sortable "YYYY-..." string, so ascending
      // string order IS chronological order.
      categoryorder: xType === "category" ? "category ascending" : undefined,
    },
    yaxis: { title: yLabel || "", gridcolor: grid },
    legend: { orientation: "h", y: -0.25 },
  };
}

window.addEventListener("resize", () => {
  const chart = document.getElementById("chart");
  if (chart && chart.data) Plotly.Plots.resize(chart);
});

main();
