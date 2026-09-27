"use strict";

// window.SITE_DATA_V2 comes from data.js (a <script>, not a fetch() —
// fetch("data.json") is blocked by CORS when this page is opened via
// file://, which is the whole point of a static, no-server site).
let DATA;

let selectedCompare = new Set();
let currentView = "home";
let compareFamily = "pitch_level";
let compareMetric = "median_f0";
let compareMode = "series";
let compareGranularity = "quarterly";
let scatterX = "median_f0";
let scatterY = "brightness_hz";
let groupBy = "member";
let tableSort = { key: "typical", dir: -1 };

// ---------------------------------------------------------------------
// Colors — ported from v1's app.js unchanged (same curated brand-color
// map, same deterministic alphabetical fallback). Colors are identity:
// keeping one shared table here rather than a second hand-maintained
// copy is what keeps a talent's color consistent between v1 and v2.
// ---------------------------------------------------------------------

const TALENT_COLORS = {
  "Aki Rosenthal": "#4E7FFC",
  "AZKi": "#FC3488",
  "Hakos Baelz": "#D72517",
  "Koseki Bijou": "#6E5BF4",
  "Yuzuki Choco": "#FE739E",
  "Elizabeth Rose Bloodflame": "#C7383B",
  "Gigi Murin": "#FDB440",
  "Himemori Luna": "#F7ABD5",
  "Hakui Koyori": "#F09CC0",
  "Inugami Korone": "#FEE039",
  "Ninomae Ina'nis": "#62567E",
  "IRyS": "#8C1236",
  "Kikirara Vivi": "#FF90CC",
  "Natsuiro Matsuri": "#FDAB45",
  "Nerissa Ravencroft": "#2233FB",
  "Ookami Mio": "#C71E3E",
  "Ouro Kronii": "#20318B",
  "Shirogane Noel": "#ACBDC5",
  "Usada Pekora": "#7EC2FE",
  "Juufuutei Raden": "#3C7C71",
  "Isaki Riona": "#FE3480",
  "Takanashi Kiara": "#FF511C",
  "Takane Lui": "#B84A67",
  "Todoroki Hajime": "#B6B9FF",
  "Tokoyami Towa": "#BA92CA",
  "Yukihana Lamy": "#6ABADF",
  "Tokino Sora": "#266AFF",
  "Robocosan": "#D192FE",
  "Shirakami Fubuki": "#43BFEF",
  "Oozora Subaru": "#E5FB67",
  "Nekomata Okayu": "#B190FC",
  "Kiryu Coco": "#F38514",
  "Tsunomaki Watame": "#F9AFB2",
  "Omaru Polka": "#B92731",
  "Mizumiya Su": "#71E5FF",
};

const FALLBACK_PALETTE = [
  "#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A",
  "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
];
const _fallbackAssigned = new Map();
function assignFallbackColors(allNames) {
  let i = 0;
  for (const name of [...allNames].sort()) {
    if (TALENT_COLORS[name]) continue;
    _fallbackAssigned.set(name, FALLBACK_PALETTE[i % FALLBACK_PALETTE.length]);
    i += 1;
  }
}
function talentColor(name) {
  return TALENT_COLORS[name] || _fallbackAssigned.get(name) || "#888888";
}

// ---------------------------------------------------------------------
// Metric metadata — display label and unit only. Which family a metric
// belongs to, and whether it's robust/experimental, comes from
// DATA.families (reference/statistics.md and reference/measurement.md),
// not duplicated here.
// ---------------------------------------------------------------------

const METRIC_META = {
  median_f0: { label: "Median F0 (pitch)", unit: "Hz", short: "pitch" },
  f0_iqr_semitones: { label: "F0 spread (IQR)", unit: "semitones", short: "pitch range" },
  dynamism_semitones: { label: "Pitch dynamism", unit: "semitones", short: "pitch movement" },
  jitter_local: { label: "Jitter", unit: "fraction", short: "jitter" },
  shimmer_local: { label: "Shimmer", unit: "fraction", short: "shimmer" },
  hnr_db: { label: "Harmonics-to-noise ratio", unit: "dB", short: "harmonicity" },
  h1h2_db: { label: "H1*–H2* (airiness)", unit: "dB", short: "airiness" },
  cpp_db: { label: "Cepstral peak prominence (clarity)", unit: "dB", short: "clarity" },
  harmonic_tilt_db_per_octave: { label: "Harmonic tilt", unit: "dB/oct", short: "tilt" },
  alpha_ratio_db: { label: "Alpha ratio (upper-band energy)", unit: "dB", short: "upper-band energy" },
  hammarberg_db: { label: "Hammarberg index", unit: "dB", short: "Hammarberg" },
  speaking_rate_syl_per_s: { label: "Speaking rate", unit: "syl/s", short: "tempo" },
  formant_dispersion_hz: { label: "Formant dispersion", unit: "Hz", short: "formant spacing" },
  brightness_hz: { label: "Brightness", unit: "Hz", short: "brightness" },
  f1_hz: { label: "Formant F1", unit: "Hz" },
  f2_hz: { label: "Formant F2", unit: "Hz" },
  f3_hz: { label: "Formant F3", unit: "Hz" },
  f4_hz: { label: "Formant F4", unit: "Hz" },
  voiced_fraction: { label: "Voiced fraction", unit: "" },
  loudness_dynamics_db: { label: "Loudness dynamics", unit: "dB" },
  background_ratio_db: { label: "Background ratio", unit: "dB" },
};

function allMetrics() {
  return DATA.families.flatMap((f) => f.metrics);
}

function familyOf(metric) {
  return DATA.families.find((f) => f.metrics.includes(metric));
}

// ---------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------

function fmt(x) {
  if (!Number.isFinite(x)) return "—";
  return Math.abs(x) < 10 ? x.toFixed(2) : x.toFixed(0);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

function fillSelect(id, values) {
  document.getElementById(id).innerHTML = values
    .map((v) => `<option value="${escapeHtml(v)}">${escapeHtml(v)}</option>`)
    .join("");
}

// Talent name with the characters a fuzzy match hit (fuzzy.js) marked.
function fuzzyLabelHtml(text, positions) {
  const hit = new Set(positions);
  return Array.from(text)
    .map((c, i) => (hit.has(i) ? `<mark>${escapeHtml(c)}</mark>` : escapeHtml(c)))
    .join("");
}

// A search box with a dropdown of fuzzy matches. Enter takes the highlighted
// match: the first one, unless the arrow keys moved it.
function attachFuzzyPicker(input, { items, onPick }) {
  const wrap = document.createElement("span");
  wrap.className = "fuzzy-wrap";
  input.parentNode.insertBefore(wrap, input);
  wrap.appendChild(input);
  const list = document.createElement("ul");
  list.className = "fuzzy-list";
  list.id = `${input.id}-options`;
  list.setAttribute("role", "listbox");
  list.hidden = true;
  wrap.appendChild(list);
  input.setAttribute("role", "combobox");
  input.setAttribute("aria-autocomplete", "list");
  input.setAttribute("aria-controls", list.id);
  input.setAttribute("aria-expanded", "false");
  input.autocomplete = "off";

  let results = [];
  let active = 0;

  function close() {
    list.hidden = true;
    input.setAttribute("aria-expanded", "false");
    input.removeAttribute("aria-activedescendant");
  }

  function render() {
    results = fuzzyFilter(input.value.trim(), items(), (n) => n);
    active = Math.min(active, Math.max(results.length - 1, 0));
    if (!results.length || document.activeElement !== input) {
      close();
      return;
    }
    list.innerHTML = results
      .map(
        (r, i) =>
          `<li id="${list.id}-${i}" role="option" class="fuzzy-option${i === active ? " active" : ""}" ` +
          `aria-selected="${i === active}" data-index="${i}">` +
          `<span class="closest-swatch" style="background:${talentColor(r.item)}"></span>` +
          `<span>${fuzzyLabelHtml(r.item, r.positions)}</span></li>`
      )
      .join("");
    list.hidden = false;
    input.setAttribute("aria-expanded", "true");
    input.setAttribute("aria-activedescendant", `${list.id}-${active}`);
    list.children[active]?.scrollIntoView({ block: "nearest" });
  }

  function pick(i) {
    const name = results[i]?.item;
    if (!name) return;
    input.value = "";
    close();
    onPick(name);
  }

  input.addEventListener("input", () => {
    active = 0;
    render();
  });
  input.addEventListener("focus", () => {
    input.select();
    render();
  });
  input.addEventListener("blur", close);
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      if (list.hidden) return render();
      const step = e.key === "ArrowDown" ? 1 : -1;
      active = (active + step + results.length) % results.length;
      render();
    } else if (e.key === "Enter") {
      e.preventDefault();
      pick(active);
    } else if (e.key === "Escape") {
      if (!list.hidden) close();
      else input.value = "";
    }
  });
  // mousedown, not click: picking must happen before the input blurs.
  list.addEventListener("mousedown", (e) => {
    const option = e.target.closest(".fuzzy-option");
    if (!option) return;
    e.preventDefault();
    pick(Number(option.dataset.index));
  });
}

function chartColors() {
  const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  return dark
    ? { fg: "#e6e8ec", grid: "#2a2e38" }
    : { fg: "#1a1d23", grid: "#dde1e7" };
}

// ---------------------------------------------------------------------
// Init and routing
// ---------------------------------------------------------------------

function main() {
  DATA = window.SITE_DATA_V2;
  assignFallbackColors(Object.keys(DATA.talents));
  document.getElementById("generated-at").textContent = DATA.generated_at;
  selectedCompare = new Set(Object.keys(DATA.talents));

  buildHomeFilters();
  buildFamilyPicker();
  buildScatterAxisPickers();

  wireNav();
  wireHomeControls();
  wireHighlightsControls();
  wireCompareControls();

  route();
  window.addEventListener("hashchange", route);
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (currentView === "compare") renderCompare();
      if (currentView === "profile" && document.getElementById("metric-strips")) renderStrips();
    }, 120);
  });
}

function route() {
  const hash = location.hash.slice(1);
  if (hash.startsWith("talent/")) {
    const { name, others } = parseTalentHash(hash.slice("talent/".length));
    showProfile(name, others);
  } else if (hash === "highlights" || hash.startsWith("highlights/")) {
    highlightsFocus = decodeURIComponent(hash.slice("highlights/".length));
    if (!DATA.highlights.signatures[highlightsFocus]) highlightsFocus = "";
    showView("highlights");
  } else if (hash === "compare") {
    showView("compare");
  } else {
    showView("home");
  }
}

function showView(view) {
  currentView = view;
  for (const el of document.querySelectorAll(".view")) el.hidden = true;
  document.getElementById(view + "-view").hidden = false;
  for (const btn of document.querySelectorAll(".nav-btn")) {
    btn.classList.toggle("active", btn.dataset.view === view);
  }
  if (view === "home") buildTalentGrid();
  if (view === "highlights") renderHighlights();
  if (view === "compare") {
    renderChipPicker();
    renderCompare();
  }
}

function showProfile(name, others = []) {
  if (!DATA.talents[name]) {
    showView("home");
    return;
  }
  currentView = "profile";
  for (const el of document.querySelectorAll(".view")) el.hidden = true;
  document.getElementById("profile-view").hidden = false;
  for (const btn of document.querySelectorAll(".nav-btn")) btn.classList.remove("active");
  renderProfile(name, others);
}

function wireNav() {
  for (const btn of document.querySelectorAll(".nav-btn")) {
    btn.addEventListener("click", () => {
      location.hash = btn.dataset.view;
    });
  }
}

// ---------------------------------------------------------------------
// Home view
// ---------------------------------------------------------------------

function buildHomeFilters() {
  const branches = new Set();
  const gens = new Set();
  for (const t of Object.values(DATA.talents)) {
    branches.add(t.branch);
    for (const g of t.group) gens.add(g);
  }
  fillSelect("home-branch-filter", ["All branches", ...[...branches].sort()]);
  fillSelect(
    "home-generation-filter",
    ["All generations", ...DATA.generation_order.filter((g) => gens.has(g))]
  );
}

function wireHomeControls() {
  const search = document.getElementById("talent-search");
  search.addEventListener("input", buildTalentGrid);
  search.addEventListener("keydown", (e) => {
    if (e.key !== "Enter") return;
    const first = homeGridNames()[0];
    if (first) location.hash = "talent/" + encodeURIComponent(first);
  });
  document.getElementById("home-branch-filter").addEventListener("change", buildTalentGrid);
  document.getElementById("home-generation-filter").addEventListener("change", buildTalentGrid);
}

function sparklineSvg(points, color) {
  if (!points || points.length < 2) return "";
  const ys = points.map((p) => p.median);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const w = 100;
  const h = 22;
  const scaleX = (i) => (points.length === 1 ? w / 2 : (i / (points.length - 1)) * w);
  const scaleY = (y) => (maxY === minY ? h / 2 : h - ((y - minY) / (maxY - minY)) * h);
  const pts = ys.map((y, i) => `${scaleX(i).toFixed(1)},${scaleY(y).toFixed(1)}`).join(" ");
  return (
    `<svg class="talent-card-spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">` +
    `<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="1.6"/></svg>`
  );
}

function talentCardHtml(name) {
  const t = DATA.talents[name];
  const legacy = t.legacy_fallback ? `<span class="legacy-tag">partial data</span>` : "";
  const spark = sparklineSvg(t.metrics.median_f0.yearly, talentColor(name));
  return (
    `<button class="talent-card" data-name="${escapeHtml(name)}">` +
    `<span class="talent-swatch" style="background:${talentColor(name)}"></span>` +
    `<span class="talent-card-body">` +
    `<span class="talent-card-name">${escapeHtml(name)}${legacy}</span>` +
    `<span class="talent-card-meta">${escapeHtml(t.group[0] || "Unknown")} · ${t.n_pass}/${t.n_clips} clips</span>` +
    spark +
    `</span></button>`
  );
}

// The home grid's talents: filtered by branch and generation, then by the
// search box, best match first.
function homeGridNames() {
  const query = document.getElementById("talent-search").value.trim();
  const branch = document.getElementById("home-branch-filter").value;
  const gen = document.getElementById("home-generation-filter").value;
  const names = Object.keys(DATA.talents)
    .sort()
    .filter((name) => branch === "All branches" || DATA.talents[name].branch === branch)
    .filter((name) => gen === "All generations" || DATA.talents[name].group.includes(gen));
  return fuzzyFilter(query, names, (n) => n).map((r) => r.item);
}

function buildTalentGrid() {
  const names = homeGridNames();
  const grid = document.getElementById("talent-grid");
  grid.innerHTML = names.map(talentCardHtml).join("");
  for (const card of grid.querySelectorAll(".talent-card")) {
    card.addEventListener("click", () => {
      location.hash = "talent/" + encodeURIComponent(card.dataset.name);
    });
  }
}

// ---------------------------------------------------------------------
// Highlights view
//
// Awards, rankings and signatures all come from DATA.highlights
// (src/vvc/highlights.py). This view only filters and formats them.
// ---------------------------------------------------------------------

let highlightsBranch = "All branches";
let highlightsFocus = "";
const highlightsExpanded = new Set();

function wireHighlightsControls() {
  const branches = new Set(
    Object.keys(DATA.highlights.signatures).map((n) => DATA.talents[n].branch)
  );
  fillSelect("highlights-branch-filter", ["All branches", ...[...branches].sort()]);
  document.getElementById("highlights-branch-filter").addEventListener("change", (e) => {
    highlightsBranch = e.target.value;
    renderHighlights();
  });
  attachFuzzyPicker(document.getElementById("highlights-focus"), {
    items: () => Object.keys(DATA.highlights.signatures).sort(),
    onPick: (name) => {
      location.hash = "highlights/" + encodeURIComponent(name);
    },
  });
  document.getElementById("highlights-focus-clear").addEventListener("click", () => {
    location.hash = "highlights";
  });
  document.getElementById("highlights-content").addEventListener("click", (e) => {
    const btn = e.target.closest(".podium-toggle");
    if (!btn) return;
    const key = btn.dataset.award;
    if (highlightsExpanded.has(key)) highlightsExpanded.delete(key);
    else highlightsExpanded.add(key);
    const card = btn.closest(".award-card");
    card.outerHTML = awardCardHtml(awardByKey(key));
    revealFocus(document.querySelector(`.award-card[data-award="${key}"]`));
  });
}

// Scroll an opened ranking so the focused talent's row is in view.
function revealFocus(card) {
  const list = card && card.querySelector(".podium-more");
  const row = list && list.querySelector(".podium-row.focus");
  if (row) list.scrollTop = row.offsetTop - list.clientHeight / 2 + row.clientHeight / 2;
}

function awardByKey(key) {
  return DATA.highlights.awards.find((a) => a.key === key);
}

// A row is in the chosen branch when every talent it names is.
function rowMembers(award, row) {
  if (award.section === "groups") return row.members;
  return row.members || [row.name];
}

function rowInBranch(award, row) {
  if (highlightsBranch === "All branches") return true;
  const members = rowMembers(award, row);
  return award.section === "groups"
    ? members.some((m) => DATA.talents[m].branch === highlightsBranch)
    : members.every((m) => DATA.talents[m].branch === highlightsBranch);
}

function signed(x, digits, unit, up, down) {
  return `${Math.abs(x).toFixed(digits)}${unit} ${x >= 0 ? up : down}`;
}

function awardValueText(award, row) {
  const v = row.value;
  switch (award.key) {
    case "one_of_a_kind":
      return `closest match ${fmtPct(v)} (${escapeHtml(row.closest)})`;
    case "voice_twins":
      return `${fmtPct(v)} voice match`;
    case "most_harmonious_generation":
    case "most_varied_generation":
      return `more alike than ${fmtPct(v)} of talent pairs`;
    case "pitch_journey":
      return `${Math.round(row.from_hz)} → ${Math.round(row.to_hz)} Hz`;
    case "longest_record":
      return `${v} months since ${escapeHtml(row.since)}`;
  }
  switch (award.metric) {
    case "median_f0":
    case "brightness_hz":
      return `${Math.round(v)} Hz`;
    case "voiced_fraction":
      return `${Math.round(v * 100)}% of the time`;
    case "speaking_rate_syl_per_s":
      return `${v.toFixed(1)} syllables/s`;
    case "dynamism_semitones":
      return `${v.toFixed(2)} semitones`;
    default:
      return `${v.toFixed(1)} ${METRIC_META[award.metric]?.unit || ""}`;
  }
}

function awardDeltaText(award, row) {
  if (award.key === "pitch_journey") {
    return `${signed(row.value, 1, " semitones", "higher", "lower")} since ${escapeHtml(row.since.slice(0, 4))}`;
  }
  if (award.key === "voice_twins") return "";
  if (award.key.endsWith("_generation")) return row.members.map(escapeHtml).join(", ");
  if (!Number.isFinite(row.delta)) return "";
  if (award.delta_unit === "semitones") {
    return `${signed(row.delta, 1, " semitones", "above", "below")} the typical talent`;
  }
  if (award.delta_unit === "percent") {
    return `${signed(row.delta, 0, "%", "above", "below")} the typical talent`;
  }
  const digits = award.metric === "dynamism_semitones" ? 2 : 1;
  const unit = METRIC_META[award.metric]?.unit || "";
  return `${signed(row.delta, digits, ` ${unit}`, "above", "below")} the typical talent`;
}

function talentLink(name) {
  const duo = DATA.highlights.duos.includes(name) ? `<span class="one-off-tag">duo</span>` : "";
  return `<a href="#talent/${encodeURIComponent(name)}">${escapeHtml(name)}</a>${duo}`;
}

function podiumRowHtml(award, row, place) {
  const members = award.section === "groups" ? [] : rowMembers(award, row);
  const swatches = members
    .map((m) => `<span class="closest-swatch" style="background:${talentColor(m)}"></span>`)
    .join("");
  const name = award.section === "groups"
    ? `<span>${escapeHtml(row.name)}</span>`
    : members.map(talentLink).join(" &amp; ");
  // Overlap is judged against the neighbour in the full ranking, so it is
  // only shown when nothing has been filtered out between them.
  const close = row.too_close && highlightsBranch === "All branches"
    ? `<span class="too-close" title="Within measurement uncertainty of the place above">≈</span>`
    : "";
  const focus = highlightsFocus && members.includes(highlightsFocus) ? " focus" : "";
  const delta = awardDeltaText(award, row);
  return (
    `<li class="podium-row${focus}">` +
    `<span class="podium-place place-${Math.min(place, 4)}">${place}</span>` +
    `<span class="podium-swatches">${swatches}</span>` +
    `<span class="podium-body"><span class="podium-name">${name}${close}</span>` +
    (delta ? `<span class="podium-delta">${delta}</span>` : "") +
    `</span><span class="podium-value">${awardValueText(award, row)}</span></li>`
  );
}

function awardCardHtml(award) {
  const rows = award.ranking.filter((row) => rowInBranch(award, row));
  if (!rows.length) return "";
  const open = highlightsExpanded.has(award.key);
  const html = rows.slice(0, 3).map((row, i) => podiumRowHtml(award, row, i + 1)).join("");
  // The focused talent's place, shown only where they are in the top half:
  // each scale has an award at both ends, so this is the flattering side.
  let preview = "";
  if (!open && highlightsFocus && award.section !== "groups") {
    const at = rows.findIndex((row) => rowMembers(award, row).includes(highlightsFocus));
    if (at >= 3 && at < rows.length / 2) {
      preview = `<ol class="podium podium-preview">${podiumRowHtml(award, rows[at], at + 1)}</ol>`;
    }
  }
  // The toggle sits directly under the podium, with everything that opens or
  // closes below it, so the arrow stays put in both states.
  const rest = rows.slice(3);
  const toggle = rest.length
    ? `<button type="button" class="podium-toggle${open ? " open" : ""}" data-award="${award.key}" ` +
      `aria-expanded="${open}" aria-controls="more-${award.key}">` +
      `<span>${open ? "Hide full ranking" : `Show all ${rows.length}`}</span>` +
      `<svg class="podium-chevron" viewBox="0 0 12 12" aria-hidden="true">` +
      `<path d="M2.5 4.5 6 8l3.5-3.5" fill="none" stroke="currentColor" stroke-width="1.6" ` +
      `stroke-linecap="round" stroke-linejoin="round"/></svg></button>`
    : "";
  const more = open
    ? `<ol class="podium podium-more" id="more-${award.key}">` +
      rest.map((row, i) => podiumRowHtml(award, row, i + 4)).join("") +
      `</ol>`
    : "";
  return (
    `<article class="award-card panel" data-award="${award.key}">` +
    `<header class="award-head"><h3>${escapeHtml(award.title)}</h3></header>` +
    `<p class="award-blurb">${escapeHtml(award.blurb)}</p>` +
    `<ol class="podium">${html}</ol>${toggle}${preview}${more}</article>`
  );
}

function signatureText(name) {
  const sig = DATA.highlights.signatures[name];
  if (!sig) return "";
  return `${escapeHtml(awardByKey(sig.award).title)} · #${sig.rank} of ${sig.of}`;
}

function signaturesHtml() {
  const names = Object.keys(DATA.highlights.signatures)
    .filter((n) => highlightsBranch === "All branches" || DATA.talents[n].branch === highlightsBranch)
    .sort();
  return (
    `<section class="highlights-section"><h2>Signatures</h2>` +
    `<p class="highlights-note">Every talent's best placing across the awards above.</p>` +
    `<div class="signature-grid">` +
    names
      .map(
        (n) =>
          `<a class="signature-card${n === highlightsFocus ? " focus" : ""}" ` +
          `href="#highlights/${encodeURIComponent(n)}">` +
          `<span class="closest-swatch" style="background:${talentColor(n)}"></span>` +
          `<span class="signature-body"><span class="signature-name">${escapeHtml(n)}</span>` +
          `<span class="signature-award">${signatureText(n)}</span></span></a>`
      )
      .join("") +
    `</div></section>`
  );
}

function renderHighlights() {
  const hl = DATA.highlights;
  document.getElementById("highlights-branch-filter").value = highlightsBranch;
  document.getElementById("highlights-focus").value = highlightsFocus;
  document.getElementById("highlights-focus-clear").hidden = !highlightsFocus;
  document.getElementById("highlights-intro").innerHTML = highlightsFocus
    ? `${talentLink(highlightsFocus)}'s signature: <strong>${signatureText(highlightsFocus)}</strong>. ` +
      `Their place is shown on every card where they rank in the top half.`
    : `Fun facts from the measurements. Talents with at least ${hl.min_clips} clips over ` +
      `${hl.min_months} months take part, and everyone gets a signature: the award they place ` +
      `highest in. Tone, pace and volume are also shaped by each talent's microphone and setup.`;
  document.getElementById("highlights-content").innerHTML =
    hl.sections
      .map((section) => {
        const cards = hl.awards
          .filter((a) => a.section === section.key)
          .map(awardCardHtml)
          .join("");
        return cards
          ? `<section class="highlights-section"><h2>${escapeHtml(section.label)}</h2>` +
            `<div class="award-grid">${cards}</div></section>`
          : "";
      })
      .join("") + signaturesHtml();
  for (const card of document.querySelectorAll(".award-card")) revealFocus(card);
}

// ---------------------------------------------------------------------
// Profile view
// ---------------------------------------------------------------------

function metricRowHtml(name, metric) {
  const m = DATA.talents[name].metrics[metric];
  const meta = METRIC_META[metric];
  if (!m || !Number.isFinite(m.typical)) {
    return (
      `<div class="metric-row"><span class="metric-name">${escapeHtml(meta.label)}</span>` +
      `<span class="metric-sub">no data</span></div>`
    );
  }
  const ci = Number.isFinite(m.ci_low)
    ? ` · 95% CI ${fmt(m.ci_low)}–${fmt(m.ci_high)}`
    : "";
  const pct = m.percentile != null ? ` · ${Math.round(m.percentile)}th percentile` : "";
  const trend =
    m.trend && Number.isFinite(m.trend.slope_per_year)
      ? ` · ${m.trend.slope_per_year >= 0 ? "+" : ""}${m.trend.slope_per_year.toFixed(2)} ${meta.unit}/yr (r² ${m.trend.r2.toFixed(2)})`
      : "";
  const nf =
    m.noise_floor && m.noise_floor.median_abs_diff != null
      ? ` · noise floor ${fmt(m.noise_floor.median_abs_diff)} ${meta.unit}`
      : "";
  return (
    `<div class="metric-row"><span class="metric-name">${escapeHtml(meta.label)}</span>` +
    `<span><span class="metric-value">${fmt(m.typical)} ${escapeHtml(meta.unit)}</span>` +
    `<span class="metric-sub">${ci}${pct}${trend}${nf}</span></span></div>`
  );
}

function familyCardHtml(name, family) {
  const badge =
    family.robust === true ? "robust" : family.robust === false ? "experimental" : "context";
  const rows = family.metrics.map((metric) => metricRowHtml(name, metric)).join("");
  return (
    `<div class="family-card"><h3>${escapeHtml(family.label)} ` +
    `<span class="badge ${badge}">${badge}</span></h3>` +
    `<p class="family-note">${escapeHtml(family.note)}</p>${rows}</div>`
  );
}

// ---------------------------------------------------------------------
// Profile comparison — the talent being viewed is pinned; up to
// MAX_COMPARE others are added from "Closest voices" or by name. The
// selection lives in the URL (#talent/<name>?vs=<a>,<b>) so a comparison
// can be shared. Every number drawn here is precomputed by
// site_data_v2.py (typical, CI, noise_units, neighbours); nothing is
// re-derived in the browser.
// ---------------------------------------------------------------------

const MAX_COMPARE = 4;
// Identity is talent colour AND a per-slot marker shape, so two talents
// with near-identical brand colours still read apart (and never by colour
// alone). Plotly symbol names, with an SVG path for the same shape.
const SLOT_SYMBOLS = ["circle", "diamond", "square", "triangle-up", "cross"];
const SLOT_GLYPHS = ["●", "◆", "■", "▲", "✚"];

let compareState = { base: null, others: [], route: "voice", table: false, expanded: new Set() };

function parseTalentHash(rest) {
  const [namePart, query] = rest.split("?");
  const name = decodeURIComponent(namePart);
  const vs = new URLSearchParams(query || "").get("vs");
  const others = vs ? vs.split(",").map(decodeURIComponent) : [];
  return { name, others };
}

function writeCompareHash() {
  const base = encodeURIComponent(compareState.base);
  const vs = compareState.others.map(encodeURIComponent).join(",");
  history.replaceState(null, "", `#talent/${base}${vs ? `?vs=${vs}` : ""}`);
}

function compareSelection() {
  return [compareState.base, ...compareState.others];
}

function slotOf(name) {
  return compareSelection().indexOf(name);
}

function svgGlyph(slot, color, size = 12) {
  const h = size / 2;
  const shapes = [
    `<circle cx="${h}" cy="${h}" r="${h - 1.5}"/>`,
    `<polygon points="${h},1 ${size - 1},${h} ${h},${size - 1} 1,${h}"/>`,
    `<rect x="2" y="2" width="${size - 4}" height="${size - 4}" rx="1"/>`,
    `<polygon points="${h},1.5 ${size - 1.5},${size - 1.5} 1.5,${size - 1.5}"/>`,
    `<path d="M${h - 1.5} 1h3v${h - 3}h${h - 3}v3h-${h - 3}v${h - 3}h-3v-${h - 3}h-${h - 3}v-3h${h - 3}z"/>`,
  ];
  return (
    `<svg class="glyph" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}" ` +
    `aria-hidden="true" fill="${color}">${shapes[slot % shapes.length]}</svg>`
  );
}

function metricShort(metric) {
  const meta = METRIC_META[metric];
  return meta ? meta.short || meta.label : metric;
}

function addToCompare(name) {
  if (name === compareState.base || compareState.others.includes(name)) return;
  if (compareState.others.length >= MAX_COMPARE) return;
  compareState.others.push(name);
  writeCompareHash();
  renderComparison();
}

function removeFromCompare(name) {
  compareState.others = compareState.others.filter((n) => n !== name);
  writeCompareHash();
  renderComparison();
}

function hasVoiceRoute(name) {
  return Boolean(DATA.talents[name].neighbours && DATA.talents[name].neighbours.voice);
}

function compareSectionHtml() {
  return (
    `<section class="compare-section panel">` +
    `<div class="compare-bar"><span class="compare-bar-label">Comparing</span>` +
    `<div id="compare-chips" class="compare-chips"></div>` +
    `<input id="compare-add" type="text" placeholder="+ add a talent…" ` +
    `aria-label="Add a talent to the comparison" /></div>` +
    `<div class="compare-grid">` +
    `<div class="closest-panel"><div class="closest-head"><h3>Closest voices</h3>` +
    `<div id="route-tabs" class="mode-tabs" role="group" aria-label="Similarity route"></div></div>` +
    `<ol id="closest-list" class="closest-list"></ol>` +
    `<p id="closest-note" class="control-hint"></p></div>` +
    `<div class="radar-panel"><div id="compare-radar" class="chart"></div>` +
    `<p class="control-hint">Distance from the median talent on each metric, in units of ` +
    `how much a talent varies between their own clips.</p></div></div>` +
    `<div class="strips-head"><h3>Metric by metric</h3>` +
    `<button id="strips-table-toggle" class="mode-btn" type="button"></button></div>` +
    `<div id="metric-strips"></div>` +
    `</section>`
  );
}

function renderProfile(name, others = []) {
  const t = DATA.talents[name];
  const legacyNote = t.legacy_fallback
    ? `<p class="control-hint">Measured with the original pipeline; some metrics are ` +
      `unavailable.</p>`
    : "";
  const oneOffNote = t.one_off
    ? `<p class="control-hint">Based on a single stream. Shown for comparison; not ranked ` +
      `against other talents.</p>`
    : "";
  compareState = {
    base: name,
    others: others.filter((n) => n !== name && DATA.talents[n]).slice(0, MAX_COMPARE),
    route: hasVoiceRoute(name) ? compareState.route : "measured",
    table: compareState.table,
    expanded: new Set(),
  };
  document.getElementById("profile-content").innerHTML =
    `<div class="profile-header">` +
    `<span class="profile-swatch" style="background:${talentColor(name)}"></span>` +
    `<h2>${escapeHtml(name)}</h2></div>` +
    `<p class="profile-meta">${escapeHtml(t.group.join(", ") || "Unknown")} (${escapeHtml(t.branch)}) &middot; ` +
    `${t.n_pass}/${t.n_clips} clips passing QC &middot; ${t.months_covered} ` +
    `month${t.months_covered === 1 ? "" : "s"} ` +
    `(${escapeHtml(t.first_month || "—")} to ${escapeHtml(t.last_month || "—")})` +
    `</p>` +
    (DATA.highlights.signatures[name]
      ? `<p class="profile-signature">Signature: <a href="#highlights/${encodeURIComponent(name)}">` +
        `${signatureText(name)}</a></p>`
      : "") +
    legacyNote +
    oneOffNote +
    compareSectionHtml() +
    `<div class="profile-families">${DATA.families.map((f) => familyCardHtml(name, f)).join("")}</div>` +
    `<p class="profile-closing">Typical value, 95% confidence interval, percentile, trend, and ` +
    `noise floor for each metric. A trend smaller than the noise floor is within measurement ` +
    `error, and trends can reflect changes in recording as well as in voice.</p>`;

  attachFuzzyPicker(document.getElementById("compare-add"), {
    items: () => Object.keys(DATA.talents).sort().filter((n) => !compareSelection().includes(n)),
    onPick: addToCompare,
  });
  document.getElementById("strips-table-toggle").addEventListener("click", () => {
    compareState.table = !compareState.table;
    renderStrips();
  });
  renderComparison();
}

function renderComparison() {
  renderCompareChips();
  renderClosest();
  renderCompareRadar();
  renderStrips();
}

function renderCompareChips() {
  const full = compareState.others.length >= MAX_COMPARE;
  document.getElementById("compare-chips").innerHTML = compareSelection()
    .map((name, slot) => {
      const color = talentColor(name);
      const remove =
        slot === 0
          ? ""
          : `<button type="button" class="chip-remove" data-remove="${escapeHtml(name)}" ` +
            `aria-label="Remove ${escapeHtml(name)} from the comparison">×</button>`;
      const label =
        slot === 0
          ? `<span>${escapeHtml(name)}</span>`
          : `<a href="#talent/${encodeURIComponent(name)}">${escapeHtml(name)}</a>`;
      return `<span class="chip selected compare-chip">${svgGlyph(slot, color)}${label}${remove}</span>`;
    })
    .join("");
  for (const btn of document.querySelectorAll("#compare-chips [data-remove]")) {
    btn.addEventListener("click", () => removeFromCompare(btn.dataset.remove));
  }
  const add = document.getElementById("compare-add");
  add.disabled = full;
  add.placeholder = full ? `up to ${MAX_COMPARE + 1} at once` : "+ add a talent…";
}

function renderClosest() {
  const base = compareState.base;
  const near = DATA.talents[base].neighbours || {};
  const tabs = document.getElementById("route-tabs");
  const routes = [];
  if (near.voice) routes.push(["voice", "Sounds like"]);
  routes.push(["measured", "Measured"]);
  tabs.innerHTML = routes
    .map(
      ([key, label]) =>
        `<button type="button" class="mode-btn${compareState.route === key ? " active" : ""}" ` +
        `data-route="${key}" aria-pressed="${compareState.route === key}">${label}</button>`
    )
    .join("");
  for (const btn of tabs.querySelectorAll("button")) {
    btn.addEventListener("click", () => {
      compareState.route = btn.dataset.route;
      renderClosest();
    });
  }
  const rows = near[compareState.route] || [];
  const full = compareState.others.length >= MAX_COMPARE;
  const list = document.getElementById("closest-list");
  list.innerHTML = rows.length
    ? rows
        .map((row) => {
          const chosen = compareSelection().includes(row.name);
          const reasons = (row.closest_metrics || []).map(metricShort).join(" · ");
          const action = chosen
            ? `<button type="button" class="closest-toggle on" data-remove="${escapeHtml(row.name)}" ` +
              `aria-label="Remove ${escapeHtml(row.name)} from the comparison">✓</button>`
            : `<button type="button" class="closest-toggle" data-add="${escapeHtml(row.name)}" ` +
              `${full ? "disabled" : ""} aria-label="Compare with ${escapeHtml(row.name)}">+</button>`;
          return (
            `<li class="closest-row">` +
            `<span class="closest-swatch" style="background:${talentColor(row.name)}"></span>` +
            `<span class="closest-body"><span class="closest-name">` +
            `<a href="#talent/${encodeURIComponent(row.name)}">${escapeHtml(row.name)}</a>` +
            `${row.one_off ? `<span class="one-off-tag">one-off</span>` : ""}</span>` +
            `<span class="closest-reasons">${reasons ? `close on ${escapeHtml(reasons)}` : ""}</span></span>` +
            `<span class="closest-score" title="Voice match ${fmtPct(row.match_pct)} · closer than ` +
            `${row.closer_than_pct.toFixed(1)}% of talent pairs">${fmtPct(row.match_pct)}</span>${action}</li>`
          );
        })
        .join("")
    : `<li class="closest-empty">No comparable talents yet.</li>`;
  for (const btn of list.querySelectorAll("[data-add]")) {
    btn.addEventListener("click", () => addToCompare(btn.dataset.add));
  }
  for (const btn of list.querySelectorAll("[data-remove]")) {
    btn.addEventListener("click", () => removeFromCompare(btn.dataset.remove));
  }
  document.getElementById("closest-note").textContent =
    compareState.route === "voice"
      ? "Voice match from a speaker-recognition model, adjusted for language. 100% means as " +
        "alike as a talent is to themselves across streams; 0% is a typical unrelated pair."
      : "Voice match from the measured voice metrics. 100% means as alike as a talent is to " +
        "themselves across streams; 0% is a typical unrelated pair.";
}

function fmtPct(x) {
  return Number.isFinite(x) ? `${Math.round(x)}%` : "—";
}

function surfaceColor() {
  return getComputedStyle(document.documentElement).getPropertyValue("--panel").trim() || "#fff";
}

function renderCompareRadar() {
  const metrics = DATA.radar_metrics || [];
  const el = document.getElementById("compare-radar");
  if (!metrics.length) {
    el.innerHTML = `<p class="control-hint">No voice metrics to compare yet.</p>`;
    return;
  }
  const { fg, grid } = chartColors();
  const labels = metrics.map(metricShort);
  let extent = 3;
  const traces = compareSelection().map((name, slot) => {
    const color = talentColor(name);
    const r = metrics.map((m) => {
      const e = DATA.talents[name].metrics[m];
      const v = e && Number.isFinite(e.noise_units) ? e.noise_units : null;
      if (v !== null) extent = Math.max(extent, Math.abs(v));
      return v;
    });
    const custom = metrics.map((m) => {
      const e = DATA.talents[name].metrics[m];
      return e && Number.isFinite(e.typical) ? `${fmt(e.typical)} ${METRIC_META[m].unit}` : "no data";
    });
    return {
      type: "scatterpolar",
      name,
      r: [...r, r[0]],
      theta: [...labels, labels[0]],
      customdata: [...custom, custom[0]],
      mode: "lines+markers",
      connectgaps: false,
      fill: "toself",
      fillcolor: color + "14",
      line: { color, width: 2 },
      marker: { color, size: 8, symbol: SLOT_SYMBOLS[slot], line: { color: surfaceColor(), width: 2 } },
      hovertemplate: `<b>${escapeHtml(name)}</b><br>%{theta}: %{customdata}<br>%{r:.1f} from median<extra></extra>`,
    };
  });
  const lim = Math.min(6, Math.ceil(extent));
  const ticks = [];
  for (let v = -lim; v <= lim; v += lim > 4 ? 2 : 1) ticks.push(v);
  Plotly.react(
    el,
    traces,
    {
      polar: {
        bgcolor: "transparent",
        radialaxis: {
          range: [-lim, lim],
          tickvals: ticks,
          ticktext: ticks.map((v) => (v === 0 ? "0 median" : String(v))),
          gridcolor: grid,
          color: fg,
          tickfont: { size: 9 },
          // Run the scale between the first two axes, not along one of them.
          angle: 90 - 180 / metrics.length,
          tickangle: 90 - 180 / metrics.length,
        },
        angularaxis: { gridcolor: grid, color: fg, direction: "clockwise" },
      },
      paper_bgcolor: "transparent",
      font: { color: fg, size: 11 },
      showlegend: false,
      margin: el.clientWidth < 480 ? { t: 30, b: 30, l: 62, r: 62 } : { t: 30, b: 30, l: 50, r: 50 },
    },
    { displayModeBar: false, responsive: true }
  );
}

// Direction hints for metrics whose sign is not obvious from the name.
const STRIP_HINTS = {
  median_f0: ["lower", "higher"],
  h1h2_db: ["pressed", "airy"],
  cpp_db: ["breathy", "clear"],
  harmonic_tilt_db_per_octave: ["soft top", "full top"],
  alpha_ratio_db: ["soft", "bright"],
  hammarberg_db: ["bright", "soft"],
  speaking_rate_syl_per_s: ["slower", "faster"],
};

function stripSvg(metric, width) {
  width = Math.max(160, Math.round(width || 600));
  const selection = compareSelection();
  const lane = 12;
  const top = 8;
  const height = top + lane * selection.length + 6;
  const all = Object.entries(DATA.talents)
    .map(([n, t]) => [n, t.metrics[metric]])
    .filter(([, e]) => e && Number.isFinite(e.typical));
  if (!all.length) return `<span class="control-hint">no data</span>`;
  let lo = Math.min(...all.map(([, e]) => e.typical));
  let hi = Math.max(...all.map(([, e]) => e.typical));
  for (const name of selection) {
    const e = DATA.talents[name].metrics[metric];
    if (e && Number.isFinite(e.ci_low)) lo = Math.min(lo, e.ci_low);
    if (e && Number.isFinite(e.ci_high)) hi = Math.max(hi, e.ci_high);
  }
  const pad = (hi - lo) * 0.04 || 1;
  lo -= pad;
  hi += pad;
  const x = (v) => (8 + ((v - lo) / (hi - lo)) * (width - 16)).toFixed(1);
  const unit = METRIC_META[metric].unit;
  const ticks = all
    .filter(([n]) => !selection.includes(n))
    .map(
      ([n, e]) =>
        `<line class="strip-other" x1="${x(e.typical)}" x2="${x(e.typical)}" y1="${top - 4}" ` +
        `y2="${height - 2}"><title>${escapeHtml(n)}: ${fmt(e.typical)} ${escapeHtml(unit)}</title></line>`
    )
    .join("");
  const marks = selection
    .map((name, slot) => {
      const e = DATA.talents[name].metrics[metric];
      if (!e || !Number.isFinite(e.typical)) return "";
      const y = top + lane * slot + lane / 2;
      const color = talentColor(name);
      const ci = Number.isFinite(e.ci_low)
        ? `<line x1="${x(e.ci_low)}" x2="${x(e.ci_high)}" y1="${y}" y2="${y}" stroke="${color}" ` +
          `stroke-width="2" stroke-linecap="round"/>`
        : "";
      const cx = Number(x(e.typical));
      const ciText = Number.isFinite(e.ci_low) ? ` (95% CI ${fmt(e.ci_low)}–${fmt(e.ci_high)})` : "";
      return (
        `<g class="strip-mark">${ci}<g transform="translate(${cx - 6},${y - 6})" class="strip-glyph">` +
        `${svgGlyph(slot, color)}</g>` +
        `<rect x="${cx - 12}" y="${y - lane / 2}" width="24" height="${lane}" fill="transparent">` +
        `<title>${escapeHtml(name)}: ${fmt(e.typical)} ${escapeHtml(unit)}${ciText}</title></rect></g>`
      );
    })
    .join("");
  const hint = STRIP_HINTS[metric];
  return (
    `<svg class="strip-svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" ` +
    `role="img" aria-label="${escapeHtml(METRIC_META[metric].label)} for the compared talents">` +
    `<line class="strip-axis" x1="8" x2="${width - 8}" y1="${height - 2}" y2="${height - 2}"/>` +
    `${ticks}${marks}</svg>` +
    `<div class="strip-scale"><span>${fmt(lo + pad)}${hint ? ` · ${hint[0]}` : ""}</span>` +
    `<span>${hint ? `${hint[1]} · ` : ""}${fmt(hi - pad)} ${escapeHtml(unit)}</span></div>`
  );
}

function stripMetrics() {
  return DATA.radar_metrics && DATA.radar_metrics.length
    ? DATA.radar_metrics
    : ["median_f0", "f0_iqr_semitones", "dynamism_semitones"];
}

function renderStrips() {
  const el = document.getElementById("metric-strips");
  const toggle = document.getElementById("strips-table-toggle");
  toggle.textContent = compareState.table ? "Show strips" : "Show as table";
  toggle.classList.toggle("active", compareState.table);
  const selection = compareSelection();
  if (compareState.table) {
    el.innerHTML =
      `<div class="table-view-wrap"><table class="data-table compare-table"><thead><tr><th>Metric</th>` +
      selection.map((n, s) => `<th>${SLOT_GLYPHS[s]} ${escapeHtml(n)}</th>`).join("") +
      `</tr></thead><tbody>` +
      stripMetrics()
        .map((m) => {
          const cells = selection
            .map((n) => {
              const e = DATA.talents[n].metrics[m];
              if (!e || !Number.isFinite(e.typical)) return `<td>—</td>`;
              const ci = Number.isFinite(e.ci_low) ? ` (${fmt(e.ci_low)}–${fmt(e.ci_high)})` : "";
              return `<td>${fmt(e.typical)}${ci}</td>`;
            })
            .join("");
          return `<tr><td class="name-cell">${escapeHtml(METRIC_META[m].label)} (${escapeHtml(METRIC_META[m].unit)})</td>${cells}</tr>`;
        })
        .join("") +
      `</tbody></table></div><p class="control-hint">Typical value (95% CI).</p>`;
    return;
  }
  el.innerHTML = stripMetrics()
    .map((m) => {
      const open = compareState.expanded.has(m);
      return (
        `<div class="strip-row"><div class="strip-label">${escapeHtml(METRIC_META[m].label)}` +
        `<button type="button" class="strip-expand" data-metric="${m}" aria-expanded="${open}">` +
        `${open ? "▾" : "▸"} over time</button></div>` +
        `<div class="strip-plot" data-metric="${m}"></div>` +
        (open ? `<div class="strip-series chart" id="strip-series-${m}"></div>` : "") +
        `</div>`
      );
    })
    .join("");
  for (const btn of el.querySelectorAll(".strip-expand")) {
    btn.addEventListener("click", () => {
      const m = btn.dataset.metric;
      if (compareState.expanded.has(m)) compareState.expanded.delete(m);
      else compareState.expanded.add(m);
      renderStrips();
    });
  }
  // Drawn at the plot's real pixel width, so marker shapes keep their size
  // and proportions on a phone instead of being scaled with the viewBox.
  for (const plot of el.querySelectorAll(".strip-plot")) {
    plot.innerHTML = stripSvg(plot.dataset.metric, plot.clientWidth);
  }
  for (const m of compareState.expanded) renderStripSeries(m);
}

function renderStripSeries(metric) {
  const el = document.getElementById(`strip-series-${metric}`);
  if (!el) return;
  const { fg, grid } = chartColors();
  const unit = METRIC_META[metric].unit;
  const traces = compareSelection().map((name, slot) => {
    const series = (DATA.talents[name].metrics[metric] || {}).quarterly || [];
    const color = talentColor(name);
    return {
      type: "scatter",
      mode: "lines+markers",
      name,
      x: series.map((p) => p.period),
      y: series.map((p) => p.median),
      line: { color, width: 2 },
      marker: { color, size: 8, symbol: SLOT_SYMBOLS[slot], line: { color: surfaceColor(), width: 2 } },
      hovertemplate: `<b>${escapeHtml(name)}</b> %{x}<br>%{y:.3~g} ${escapeHtml(unit)}<extra></extra>`,
    };
  });
  Plotly.react(
    el,
    traces,
    {
      xaxis: periodXAxis(grid),
      yaxis: { gridcolor: grid, title: { text: unit, font: { size: 10 } }, zeroline: false },
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      font: { color: fg, size: 10 },
      showlegend: false,
      hovermode: "x unified",
      margin: { t: 10, b: 60, l: 50, r: 10 },
    },
    { displayModeBar: false, responsive: true }
  );
}

// ---------------------------------------------------------------------
// Compare view
// ---------------------------------------------------------------------

// A graduated talent's stored `branch` is always "Graduated" (a status, not a
// region — see site_data.py) even though their `group` still carries their
// real generation. Branch CHIPS need the region a talent actually debuted
// in, or "JP" would silently exclude every graduated JP member — so this
// re-derives it from group[0] instead of trusting the stored field. Mirrors
// site_data.py's _branch_for_group; small and pure enough that duplicating
// it here (rather than threading it through data.js) is the simpler choice.
function regionOf(name) {
  const group = DATA.talents[name].group[0];
  if (!group) return DATA.talents[name].branch;
  if (group.startsWith("English")) return "EN";
  if (group.startsWith("Indonesia")) return "ID";
  if (group.startsWith("DEV_IS")) return "DEV_IS";
  return "JP";
}

function chipTokens() {
  const regions = [...new Set(Object.keys(DATA.talents).map(regionOf))].sort();
  const hasGraduated = Object.values(DATA.talents).some((t) => t.branch === "Graduated");
  return [
    { type: "all", label: "All" },
    ...regions.map((label) => ({ type: "region", label })),
    ...(hasGraduated ? [{ type: "graduated", label: "Graduated" }] : []),
    ...DATA.generation_order.map((label) => ({ type: "group", label })),
    ...Object.keys(DATA.talents).sort().map((label) => ({ type: "talent", label })),
  ];
}

function membersOf(token) {
  if (token.type === "talent") return [token.label];
  if (token.type === "all") return Object.keys(DATA.talents);
  if (token.type === "region") {
    return Object.keys(DATA.talents).filter((n) => regionOf(n) === token.label);
  }
  if (token.type === "graduated") {
    return Object.keys(DATA.talents).filter((n) => DATA.talents[n].branch === "Graduated");
  }
  return Object.keys(DATA.talents).filter((n) => DATA.talents[n].group.includes(token.label));
}

function renderChipPicker() {
  const query = document.getElementById("compare-search").value.trim();
  const wrap = document.getElementById("chip-picker");
  const tokens = fuzzyFilter(query, chipTokens(), (tok) => tok.label).map((r) => r.item);
  wrap.innerHTML = tokens
    .map((tok) => {
      const members = membersOf(tok);
      const allSelected = members.length > 0 && members.every((m) => selectedCompare.has(m));
      const swatch =
        tok.type === "talent"
          ? `<span class="chip-swatch" style="background:${talentColor(tok.label)}"></span>`
          : "";
      const cls = ["chip", tok.type !== "talent" ? "group" : "", allSelected ? "selected" : ""]
        .filter(Boolean)
        .join(" ");
      return (
        `<button type="button" class="${cls}" data-type="${tok.type}" ` +
        `data-label="${escapeHtml(tok.label)}">${swatch}${escapeHtml(tok.label)}</button>`
      );
    })
    .join("");
  for (const chip of wrap.querySelectorAll(".chip")) {
    chip.addEventListener("click", () => {
      const token = { type: chip.dataset.type, label: chip.dataset.label };
      const members = membersOf(token);
      const allSelected = members.every((m) => selectedCompare.has(m));
      for (const m of members) {
        if (allSelected) selectedCompare.delete(m);
        else selectedCompare.add(m);
      }
      renderChipPicker();
      renderCompare();
    });
  }
}

function buildFamilyPicker() {
  const el = document.getElementById("family-picker");
  el.innerHTML = DATA.families
    .map((f) => `<option value="${f.key}">${escapeHtml(f.label)}</option>`)
    .join("");
  el.value = compareFamily;
  el.addEventListener("change", () => {
    compareFamily = el.value;
    compareMetric = DATA.families.find((f) => f.key === compareFamily).metrics[0];
    buildMetricPicker();
    renderCompare();
  });
  buildMetricPicker();
}

function buildMetricPicker() {
  const fam = DATA.families.find((f) => f.key === compareFamily);
  const el = document.getElementById("metric-picker");
  el.innerHTML = fam.metrics
    .map((m) => `<option value="${m}">${escapeHtml(METRIC_META[m].label)}</option>`)
    .join("");
  el.value = compareMetric;
  el.onchange = () => {
    compareMetric = el.value;
    renderCompare();
  };
  document.getElementById("family-note").textContent = fam.note;
}

function buildScatterAxisPickers() {
  const opts = allMetrics()
    .map((m) => `<option value="${m}">${escapeHtml(METRIC_META[m].label)}</option>`)
    .join("");
  document.getElementById("scatter-x-picker").innerHTML = opts;
  document.getElementById("scatter-y-picker").innerHTML = opts;
  document.getElementById("scatter-x-picker").value = scatterX;
  document.getElementById("scatter-y-picker").value = scatterY;
}

function wireCompareControls() {
  const search = document.getElementById("compare-search");
  search.addEventListener("input", renderChipPicker);
  // Enter toggles the best match and clears the box for the next one.
  search.addEventListener("keydown", (e) => {
    if (e.key !== "Enter") return;
    const first = document.querySelector("#chip-picker .chip");
    if (!first) return;
    search.value = "";
    first.click();
  });
  document.getElementById("compare-select-none").addEventListener("click", () => {
    selectedCompare.clear();
    renderChipPicker();
    renderCompare();
  });
  for (const btn of document.querySelectorAll(".mode-btn")) {
    btn.addEventListener("click", () => {
      compareMode = btn.dataset.mode;
      for (const b of document.querySelectorAll(".mode-btn")) b.classList.toggle("active", b === btn);
      document.getElementById("series-controls").hidden = compareMode !== "series";
      document.getElementById("scatter-controls").hidden = compareMode !== "scatter";
      renderCompare();
    });
  }
  document.getElementById("granularity-picker").addEventListener("change", (e) => {
    compareGranularity = e.target.value;
    renderCompare();
  });
  document.getElementById("group-by-picker").addEventListener("change", (e) => {
    groupBy = e.target.value;
    renderCompare();
  });
  document.getElementById("scatter-x-picker").addEventListener("change", (e) => {
    scatterX = e.target.value;
    renderCompare();
  });
  document.getElementById("scatter-y-picker").addEventListener("change", (e) => {
    scatterY = e.target.value;
    renderCompare();
  });
}

function updateValidityBanner() {
  const fam = DATA.families.find((f) => f.key === compareFamily);
  const banner = document.getElementById("validity-banner");
  if (fam.robust === false) {
    banner.className = "validity-banner experimental";
    banner.textContent = "Experimental — " + fam.note;
  } else if (fam.robust === null) {
    banner.className = "validity-banner";
    banner.textContent = fam.note;
  } else {
    banner.className = "validity-banner robust";
    banner.textContent = "";
  }
}

// Pins the chart panel's bottom edge 20px above the viewport bottom when
// the page is scrolled to the very top — sized from the panel's real
// document position so it adapts to whatever the header/controls actually
// render at, rather than a guessed static offset. Deliberately excludes the
// legend and caption below it: those are meant to start just out of view at
// scroll-top, not be folded into "the panel".
function sizeChartPanel() {
  const chart = document.getElementById("chart");
  if (chart.hidden) return;
  const documentTop = chart.getBoundingClientRect().top + window.scrollY;
  chart.style.height = `${Math.max(360, window.innerHeight - documentTop - 20)}px`;
}

function renderCompare() {
  updateValidityBanner();
  const chart = document.getElementById("chart");
  const table = document.getElementById("table-view");
  const legend = document.getElementById("series-legend");
  const caption = document.getElementById("caption");
  legend.hidden = compareMode !== "series";
  if (compareMode !== "series") legend.innerHTML = "";
  if (selectedCompare.size === 0) {
    chart.hidden = false;
    table.hidden = true;
    sizeChartPanel();
    Plotly.purge(chart);
    caption.textContent = "Select a talent or a group chip on the left to compare.";
    return;
  }
  if (compareMode === "series") {
    chart.hidden = false;
    table.hidden = true;
    renderCompareSeries();
  } else if (compareMode === "table") {
    chart.hidden = true;
    table.hidden = false;
    renderCompareTable();
  } else {
    chart.hidden = false;
    table.hidden = true;
    renderCompareScatter();
  }
}

// Plotly's default category order is "trace" — the order each x value is
// first seen across traces, not a sort. With several talents' spans
// stitched together that jumps back and forth in time. The period strings
// (YYYY-MM / YYYY-Qn / YYYY) are fixed-width and zero-padded, so a plain
// ascending string sort is chronological.
function periodXAxis(grid) {
  return { gridcolor: grid, tickangle: -45, type: "category", categoryorder: "category ascending" };
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

// ---------------------------------------------------------------------
// "Group by" — member / generation / branch — applies across all three
// views below. Member mode is one entry per selected talent, using that
// talent's own precomputed, server-side stats (real bootstrap CI, career
// trend). Generation/branch mode is one entry per group PRESENT AMONG THE
// CURRENT SELECTION, aggregated client-side from its members' series —
// there is no server-side "group" record, so its uncertainty is a plain
// min-max range rather than a bootstrap CI, and its trend is fit over
// calendar time (member-level trends use each talent's own career time,
// which isn't comparable across members once aggregated).
// ---------------------------------------------------------------------

// A talent can hold more than one generation (e.g. Shirakami Fubuki: 1st
// Generation AND GAMERS) — grouping by generation counts her toward both,
// matching how the chip picker already treats multi-group membership.
// Grouping by branch uses regionOf (real region, graduated-inclusive) for
// the same reason region chips do: "Graduated" is a status, not a region.
function groupKeysFor(name) {
  if (groupBy === "branch") return [regionOf(name)];
  const groups = DATA.talents[name].group;
  return groups && groups.length ? groups : ["Unknown"];
}

function selectionEntries() {
  if (groupBy === "member") {
    return [...selectedCompare].sort().map((name) => ({ label: name, members: [name], isGroup: false }));
  }
  const namesByGroup = new Map();
  for (const name of selectedCompare) {
    for (const key of groupKeysFor(name)) {
      if (!namesByGroup.has(key)) namesByGroup.set(key, []);
      namesByGroup.get(key).push(name);
    }
  }
  return [...namesByGroup.keys()]
    .sort()
    .map((label) => ({ label, members: namesByGroup.get(label), isGroup: true }));
}

function entryColor(entry) {
  return entry.isGroup ? paletteColorFor(entry.label) : talentColor(entry.label);
}

// Deterministic per-label (not per-position) color, so a group's color
// stays the same across renders regardless of what else is selected.
function paletteColorFor(label) {
  let hash = 0;
  for (let i = 0; i < label.length; i++) hash = (hash * 31 + label.charCodeAt(i)) >>> 0;
  return FALLBACK_PALETTE[hash % FALLBACK_PALETTE.length];
}

// One talent (isGroup=false): pass its own series straight through, with
// its real per-point CI. Several talents (isGroup=true): aggregate across
// members at each period present in the CURRENT granularity, taking the
// median of whichever members have a value there — a period no member
// covers is a gap, never interpolated.
function aggregatedSeries(members, metric, granularity) {
  if (members.length === 1) {
    return { series: DATA.talents[members[0]].metrics[metric][granularity] || [], hasRealCi: true };
  }
  const byPeriod = new Map();
  for (const name of members) {
    for (const p of DATA.talents[name].metrics[metric][granularity] || []) {
      if (!Number.isFinite(p.median)) continue;
      if (!byPeriod.has(p.period)) byPeriod.set(p.period, []);
      byPeriod.get(p.period).push(p.median);
    }
  }
  const periods = [...byPeriod.keys()].sort();
  const series = periods.map((period) => {
    const values = byPeriod.get(period);
    return { period, median: median(values), min: Math.min(...values), max: Math.max(...values), n: values.length };
  });
  return { series, hasRealCi: false };
}

function periodToMonthIndex(period, granularity) {
  if (granularity === "yearly") return parseInt(period, 10) * 12;
  if (granularity === "monthly") {
    const [y, m] = period.split("-").map(Number);
    return y * 12 + (m - 1);
  }
  const [y, q] = period.split("-Q").map(Number);
  return y * 12 + (q - 1) * 3;
}

// Same estimator as analysis.py's theil_sen/career_trend: median of
// pairwise slopes, median intercept, ordinary r². Ported to JS because a
// group's trend can only be computed client-side — there is no server-side
// record for an ad-hoc selection of members.
function theilSenTrend(xs, ys) {
  const pairs = xs.map((x, i) => [x, ys[i]]).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
  if (pairs.length < 2) return null;
  const slopes = [];
  for (let i = 0; i < pairs.length; i++) {
    for (let j = i + 1; j < pairs.length; j++) {
      if (pairs[j][0] !== pairs[i][0]) slopes.push((pairs[j][1] - pairs[i][1]) / (pairs[j][0] - pairs[i][0]));
    }
  }
  if (!slopes.length) return null;
  const slope = median(slopes);
  const intercept = median(pairs.map(([x, y]) => y - slope * x));
  const meanY = pairs.reduce((a, [, y]) => a + y, 0) / pairs.length;
  const ssTot = pairs.reduce((a, [, y]) => a + (y - meanY) ** 2, 0);
  const ssRes = pairs.reduce((a, [x, y]) => a + (y - (intercept + slope * x)) ** 2, 0);
  return { slope_per_year: slope * 12, r2: ssTot > 0 ? 1 - ssRes / ssTot : null };
}

// One summary per entry, for the Table and Scatter views. Member entries
// use the precomputed, server-side per-talent stats directly (so numbers
// here always agree with the Profile page); group entries are aggregated
// from the currently selected granularity.
function summaryFor(entry, metric, granularity) {
  if (!entry.isGroup) {
    const name = entry.members[0];
    const m = DATA.talents[name].metrics[metric];
    const t = DATA.talents[name];
    return {
      typical: m.typical,
      low: m.ci_low,
      high: m.ci_high,
      isRange: false,
      trend: m.trend && Number.isFinite(m.trend.slope_per_year) ? m.trend : null,
      nPass: t.n_pass,
      nClips: t.n_clips,
      noiseFloor: m.noise_floor && m.noise_floor.median_abs_diff != null ? m.noise_floor.median_abs_diff : null,
    };
  }
  const { series } = aggregatedSeries(entry.members, metric, granularity);
  if (series.length === 0) {
    return { typical: NaN, low: NaN, high: NaN, isRange: true, trend: null, nPass: 0, nClips: 0, noiseFloor: null };
  }
  const values = series.map((p) => p.median);
  const xs = series.map((p) => periodToMonthIndex(p.period, granularity));
  let nPass = 0;
  let nClips = 0;
  const floors = [];
  for (const name of entry.members) {
    const t = DATA.talents[name];
    nPass += t.n_pass;
    nClips += t.n_clips;
    const nf = t.metrics[metric].noise_floor;
    if (nf && nf.median_abs_diff != null) floors.push(nf.median_abs_diff);
  }
  return {
    typical: median(values),
    low: Math.min(...values),
    high: Math.max(...values),
    isRange: true,
    trend: theilSenTrend(xs, values),
    nPass,
    nClips,
    noiseFloor: floors.length ? median(floors) : null,
  };
}

// Base band opacity shrinks as more entries overlap — several 95% CI bands
// stacked on each other turn opaque and unreadable well before six lines,
// which is what was reported. Inverse-sqrt keeps a visible band at n=1
// while backing off quickly over the first few additions and flattening out
// rather than vanishing. Hover restores one band to full visibility (see
// attachSeriesHover below) so any single member is still checkable.
function bandAlphaHex(n) {
  const frac = Math.max(0.05, Math.min(0.3, 0.3 / Math.sqrt(n)));
  return Math.round(frac * 255)
    .toString(16)
    .padStart(2, "0");
}
const BAND_HOVER_ALPHA_HEX = "6e"; // ~0.43, the hovered member's own band
const BAND_DIM_ALPHA_HEX = "08"; // ~0.03, every other band while one is hovered

// Highlights one entry's band (by its trace index), dims every other band,
// and restores the count-scaled default. Shared by hovering the line itself
// AND hovering its entry in the custom legend below the chart (see
// renderCompareSeries) — the plain HTML legend replaces Plotly's built-in
// one, which otherwise eats into the plot's own height as more entries wrap
// across legend rows inside the same fixed-size container.
function applyBandHighlight(chartEl, pairs, bandIdx) {
  const withBand = pairs.filter((p) => p.bandIdx != null);
  if (withBand.length === 0) return;
  const colors = withBand.map((p) => p.color + (p.bandIdx === bandIdx ? BAND_HOVER_ALPHA_HEX : BAND_DIM_ALPHA_HEX));
  Plotly.restyle(chartEl, { fillcolor: colors }, withBand.map((p) => p.bandIdx));
}

function clearBandHighlight(chartEl, pairs) {
  const withBand = pairs.filter((p) => p.bandIdx != null);
  if (withBand.length === 0) return;
  Plotly.restyle(
    chartEl,
    { fillcolor: withBand.map((p) => p.color + p.baseAlphaHex) },
    withBand.map((p) => p.bandIdx)
  );
}

// Listeners are re-attached on every render (old ones removed first) since
// Plotly.react keeps the same <div> but trace indices change between
// renders.
function attachSeriesHover(chartEl, pairs) {
  chartEl.removeAllListeners("plotly_hover");
  chartEl.removeAllListeners("plotly_unhover");
  chartEl.on("plotly_hover", (ev) => {
    const hit = pairs.find((p) => p.lineIdx === ev.points[0].curveNumber);
    if (hit) applyBandHighlight(chartEl, pairs, hit.bandIdx);
  });
  chartEl.on("plotly_unhover", () => clearBandHighlight(chartEl, pairs));
}

// A plain HTML legend below the fixed-height chart (see #chart's CSS) so it
// can wrap across as many rows as it needs and grow the page instead of
// shrinking the plot. Hovering an item drives the same band highlight as
// hovering the line itself.
function renderSeriesLegend(chartEl, pairs) {
  const legend = document.getElementById("series-legend");
  legend.innerHTML = pairs
    .map(
      (p, i) =>
        `<span class="legend-item" data-i="${i}"><span class="legend-swatch" ` +
        `style="background:${p.color}"></span>${escapeHtml(p.label)}</span>`
    )
    .join("");
  for (const item of legend.querySelectorAll(".legend-item")) {
    const pair = pairs[Number(item.dataset.i)];
    item.addEventListener("mouseenter", () => applyBandHighlight(chartEl, pairs, pair.bandIdx));
    item.addEventListener("mouseleave", () => clearBandHighlight(chartEl, pairs));
  }
}

function renderCompareSeries() {
  const meta = METRIC_META[compareMetric];
  const { fg, grid } = chartColors();
  const entries = selectionEntries();
  const withData = entries
    .map((entry) => ({ entry, ...aggregatedSeries(entry.members, compareMetric, compareGranularity) }))
    .filter((e) => e.series.length > 0);
  const baseAlphaHex = bandAlphaHex(withData.length);

  const traces = [];
  const hoverPairs = [];
  for (const { entry, series, hasRealCi } of withData) {
    const color = entryColor(entry);
    const xs = series.map((p) => p.period);
    const upper = series.map((p) => (hasRealCi ? p.ci_high : p.max));
    const lower = series.map((p) => (hasRealCi ? p.ci_low : p.min));
    const hasBand = upper.some((v, i) => Number.isFinite(v) && Number.isFinite(lower[i]));
    let bandIdx = null;
    if (hasBand) {
      const bandUpper = series.map((p, i) => (Number.isFinite(upper[i]) ? upper[i] : p.median));
      const bandLower = series.map((p, i) => (Number.isFinite(lower[i]) ? lower[i] : p.median));
      bandIdx = traces.length;
      traces.push({
        x: [...xs, ...xs.slice().reverse()],
        y: [...bandUpper, ...bandLower.slice().reverse()],
        fill: "toself",
        fillcolor: color + baseAlphaHex,
        mode: "lines",
        line: { width: 0 },
        showlegend: false,
        hoverinfo: "skip",
        type: "scatter",
      });
    }
    const label = entry.isGroup ? `${entry.label} (n=${new Set(entry.members).size})` : entry.label;
    const lineIdx = traces.length;
    traces.push({
      x: xs,
      y: series.map((p) => p.median),
      mode: "lines+markers",
      name: label,
      showlegend: false,
      line: { color },
      marker: { color, size: 5 },
      type: "scatter",
    });
    hoverPairs.push({ bandIdx, lineIdx, color, baseAlphaHex, label });
  }
  sizeChartPanel();
  Plotly.react(
    "chart",
    traces,
    {
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      font: { color: fg },
      xaxis: periodXAxis(grid),
      yaxis: { title: meta.label + (meta.unit ? ` (${meta.unit})` : ""), gridcolor: grid },
      showlegend: false,
      margin: { t: 20 },
    },
    { responsive: true, displayModeBar: false }
  );
  const chartEl = document.getElementById("chart");
  attachSeriesHover(chartEl, hoverPairs);
  renderSeriesLegend(chartEl, hoverPairs);
  const nf = DATA.corpus_noise_floor[compareMetric];
  const nfLine =
    nf && nf.median_abs_diff != null
      ? ` Corpus noise floor for this metric: ${fmt(nf.median_abs_diff)} ${meta.unit}` +
        (nf.median_abs_semitones != null ? ` (${nf.median_abs_semitones.toFixed(2)} semitones)` : "") +
        "."
      : "";
  const bandLine =
    groupBy === "member"
      ? "Shaded band: 95% confidence interval (quarterly and yearly views)."
      : "Shaded band: range across the selected members of each group.";
  const hoverHint =
    withData.length > 1 ? " Hover a line to highlight it." : "";
  document.getElementById("caption").textContent = bandLine + hoverHint + nfLine;
}

const TABLE_COLUMNS = [
  { key: "name", label: "Talent" },
  { key: "typical", label: "Typical" },
  { key: "ci", label: "95% CI", sortKey: "low" },
  { key: "trend", label: "Trend/yr", sortKey: "trend" },
  { key: "r2", label: "r²" },
  { key: "clips", label: "Clips (pass/total)", sortKey: "nPass" },
  { key: "noise_floor", label: "Noise floor" },
];

function renderCompareTable() {
  const meta = METRIC_META[compareMetric];
  const grouped = groupBy !== "member";
  const entries = selectionEntries();
  const rows = entries.map((entry) => {
    const s = summaryFor(entry, compareMetric, compareGranularity);
    return {
      name: entry.isGroup ? `${entry.label} (n=${new Set(entry.members).size})` : entry.label,
      typical: s.typical,
      low: s.low,
      ci: Number.isFinite(s.low) ? `${fmt(s.low)}–${fmt(s.high)}` : "—",
      trend: s.trend ? s.trend.slope_per_year : null,
      r2: s.trend && s.trend.r2 != null ? s.trend.r2 : null,
      nPass: s.nPass,
      clips: `${s.nPass}/${s.nClips}`,
      noise_floor: s.noiseFloor,
    };
  });

  const sortKey = tableSort.key === "name" ? "name" : TABLE_COLUMNS.find((c) => c.key === tableSort.key)?.sortKey || tableSort.key;
  rows.sort((a, b) => {
    const av = a[sortKey];
    const bv = b[sortKey];
    const aMissing = av == null || av === "";
    const bMissing = bv == null || bv === "";
    if (aMissing && bMissing) return 0;
    if (aMissing) return 1; // missing values always sort last, in either direction
    if (bMissing) return -1;
    if (typeof av === "string") return av.localeCompare(bv) * tableSort.dir;
    return (av - bv) * tableSort.dir;
  });

  const headerHtml = TABLE_COLUMNS.map((col) => {
    const label = col.key === "name" ? (grouped ? "Group" : "Talent") : col.key === "ci" ? (grouped ? "Range" : "95% CI") : col.label;
    const arrow = tableSort.key === col.key ? `<span class="sort-arrow">${tableSort.dir === 1 ? "▲" : "▼"}</span>` : "";
    return `<th data-key="${col.key}">${escapeHtml(label)}${arrow}</th>`;
  }).join("");
  const bodyHtml = rows
    .map(
      (r) =>
        `<tr><td class="name-cell">${escapeHtml(r.name)}</td><td>${fmt(r.typical)}</td>` +
        `<td>${r.ci}</td><td>${r.trend != null ? r.trend.toFixed(2) : "—"}</td>` +
        `<td>${r.r2 != null ? r.r2.toFixed(2) : "—"}</td><td>${r.clips}</td>` +
        `<td>${r.noise_floor != null ? fmt(r.noise_floor) : "—"}</td></tr>`
    )
    .join("");
  document.getElementById("table-view").innerHTML =
    `<table class="data-table"><thead><tr>${headerHtml}</tr></thead><tbody>${bodyHtml}</tbody></table>`;
  for (const th of document.querySelectorAll(".data-table th")) {
    th.addEventListener("click", () => {
      const key = th.dataset.key;
      if (tableSort.key === key) tableSort.dir *= -1;
      else tableSort = { key, dir: key === "name" ? 1 : -1 };
      renderCompareTable();
    });
  }
  document.getElementById("caption").textContent = grouped
    ? `Typical value: median across each group's selected members. Trend: over calendar ` +
      "time. Click a column to sort."
    : `Typical value: median of monthly values. Trend: over each talent's career. Click a ` +
      "column to sort.";
}

function renderCompareScatter() {
  const xMeta = METRIC_META[scatterX];
  const yMeta = METRIC_META[scatterY];
  const { fg, grid } = chartColors();
  const entries = selectionEntries();
  const points = entries
    .map((entry) => ({
      name: entry.isGroup ? `${entry.label} (n=${new Set(entry.members).size})` : entry.label,
      x: summaryFor(entry, scatterX, compareGranularity).typical,
      y: summaryFor(entry, scatterY, compareGranularity).typical,
      color: entryColor(entry),
    }))
    .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
  sizeChartPanel();
  Plotly.react(
    "chart",
    [
      {
        type: "scatter",
        mode: "markers+text",
        x: points.map((p) => p.x),
        y: points.map((p) => p.y),
        text: points.map((p) => p.name),
        textposition: "top center",
        textfont: { size: 9, color: fg },
        marker: { size: 12, color: points.map((p) => p.color) },
      },
    ],
    {
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      font: { color: fg },
      xaxis: { title: xMeta.label + (xMeta.unit ? ` (${xMeta.unit})` : ""), gridcolor: grid },
      yaxis: { title: yMeta.label + (yMeta.unit ? ` (${yMeta.unit})` : ""), gridcolor: grid },
      margin: { t: 20 },
    },
    { responsive: true, displayModeBar: false }
  );
  document.getElementById("caption").textContent =
    (groupBy === "member"
      ? "Each point is a talent's typical value."
      : "Each point is a group's typical value across its selected members.");
}

main();
