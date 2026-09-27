// Fuzzy search for the site's search boxes, in the style of fzf/fff:
// letters typed in order match across gaps ("pkr" finds Pekora), matches at
// the start of a word rank higher, query words may come in any order, and a
// single typo in a longer word is forgiven. Plain script, no dependencies:
// app.js uses it in the browser, tests/test_fuzzy_search.py under deno.

const FUZZY_MATCH = 16; // every matched character
const FUZZY_WORD_START = 10; // ...at the start of a word
const FUZZY_CONSECUTIVE = 6; // ...right after the previous match
const FUZZY_GAP_OPEN = 4; // skipping characters inside a match
const FUZZY_GAP_EXTEND = 1; // ...per extra character skipped
const FUZZY_TYPO = 1000; // a typo match ranks below every in-order match

// One character per character, so match positions line up with the
// original text: lower-cased, accents dropped.
function fuzzyChars(text) {
  return Array.from(String(text), (c) => c.normalize("NFD")[0].toLowerCase());
}

function fuzzyIsWordChar(c) {
  return /[\p{L}\p{N}]/u.test(c);
}

// Best in-order alignment of `token` inside `chars`, or null.
function fuzzySubsequence(token, chars) {
  const n = token.length;
  const m = chars.length;
  if (!n || n > m) return null;
  const start = chars.map((c, j) => j === 0 || !fuzzyIsWordChar(chars[j - 1]));
  let prev = null;
  const back = [];
  for (let i = 0; i < n; i++) {
    const row = new Array(m).fill(-Infinity);
    const from = new Array(m).fill(-1);
    for (let j = i; j < m; j++) {
      if (chars[j] !== token[i]) continue;
      const gain = FUZZY_MATCH + (start[j] ? FUZZY_WORD_START : 0);
      if (i === 0) {
        row[j] = gain;
        continue;
      }
      for (let k = i - 1; k < j; k++) {
        if (prev[k] === -Infinity) continue;
        const skipped = j - k - 1;
        const step = skipped
          ? -(FUZZY_GAP_OPEN + FUZZY_GAP_EXTEND * (skipped - 1))
          : FUZZY_CONSECUTIVE;
        if (prev[k] + step + gain > row[j]) {
          row[j] = prev[k] + step + gain;
          from[j] = k;
        }
      }
    }
    back.push(from);
    prev = row;
  }
  let best = -1;
  for (let j = 0; j < m; j++) if (prev[j] > (best < 0 ? -Infinity : prev[best])) best = j;
  if (best < 0 || prev[best] === -Infinity) return null;
  const positions = [];
  for (let i = n - 1, j = best; i >= 0; j = back[i][j], i--) positions.unshift(j);
  return { score: prev[best], positions };
}

// Edit distance counting a swap of two neighbours as one edit.
function fuzzyEditDistance(a, b) {
  const d = Array.from({ length: a.length + 1 }, (_, i) => [i]);
  for (let j = 1; j <= b.length; j++) d[0][j] = j;
  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost);
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) {
        d[i][j] = Math.min(d[i][j], d[i - 2][j - 2] + 1);
      }
    }
  }
  return d[a.length][b.length];
}

// A word (or the start of one, while still typing) within one typo of `token`.
function fuzzyTypo(token, chars) {
  if (token.length < 4) return null;
  const allowed = token.length >= 7 ? 2 : 1;
  let best = null;
  let j = 0;
  while (j < chars.length) {
    if (!fuzzyIsWordChar(chars[j])) {
      j++;
      continue;
    }
    let end = j;
    while (end < chars.length && fuzzyIsWordChar(chars[end])) end++;
    const word = chars.slice(j, end).join("");
    for (let len = token.length - 1; len <= token.length + 1; len++) {
      if (len > word.length) break;
      const dist = fuzzyEditDistance(token, word.slice(0, len));
      if (dist <= allowed && (!best || dist < best.dist)) {
        best = { dist, positions: Array.from({ length: len }, (_, k) => j + k) };
      }
    }
    j = end;
  }
  return best && { score: -FUZZY_TYPO * best.dist, positions: best.positions };
}

// Score of `query` against `text`, or null when some query word misses.
function fuzzyMatch(query, text) {
  const tokens = fuzzyChars(query).join("").split(/\s+/).filter(Boolean);
  const chars = fuzzyChars(text);
  let score = 0;
  const positions = new Set();
  for (const token of tokens) {
    const hit = fuzzySubsequence(token, chars) || fuzzyTypo(token, chars);
    if (!hit) return null;
    score += hit.score;
    for (const p of hit.positions) positions.add(p);
  }
  return { score, positions: [...positions].sort((a, b) => a - b) };
}

// `items` that match `query`, best first; ties keep their given order. An
// empty query keeps every item in order.
function fuzzyFilter(query, items, key) {
  const out = [];
  items.forEach((item, index) => {
    const hit = fuzzyMatch(query, key(item));
    if (hit) out.push({ item, index, ...hit });
  });
  return out.sort((a, b) => b.score - a.score || a.index - b.index);
}
