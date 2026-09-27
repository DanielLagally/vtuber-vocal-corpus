"""Product tests for the site's search boxes (docs/fuzzy.js).

Every search box ranks talents with the same fuzzy matcher, and pressing Enter
takes the first result, so "the first result" is the rule under test. The
matcher runs under deno (provided by the flake), exactly as the browser loads
it: a plain script evaluated at global scope.

User-visible rules:

1. Letters typed in order match even with gaps ("pkr" finds Pekora).
2. A match at the start of a word outranks one inside a word.
3. Words in the query may come in any order.
4. A single typo in a longer query still finds the talent.
5. Case, accents and punctuation in names do not get in the way.
6. A query that matches nothing returns nothing.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

FUZZY_JS = Path(__file__).resolve().parents[1] / "docs" / "fuzzy.js"

NAMES = [
    "Usada Pekora",
    "Himemori Luna",
    "Mori Calliope",
    "Gawr Gura",
    "Hoshimachi Suisei",
    "Ninomae Ina'nis",
    "La+ Darknesss",
    "Nanashi Mumei",
    "Takane Lui",
    "Shirogane Noel",
]

pytestmark = pytest.mark.skipif(shutil.which("deno") is None, reason="deno not on PATH")


def _search(*queries: str) -> dict[str, list[str]]:
    script = (
        f"(0, eval)(await Deno.readTextFile({json.dumps(str(FUZZY_JS))}));"
        f"const names = {json.dumps(NAMES)};"
        f"const out = {{}};"
        f"for (const q of {json.dumps(list(queries))}) "
        f"out[q] = fuzzyFilter(q, names, (n) => n).map((r) => r.item);"
        f"console.log(JSON.stringify(out));"
    )
    result = subprocess.run(
        ["deno", "eval", "--quiet", script], capture_output=True, text=True, check=True
    )
    return json.loads(result.stdout)


def test_letters_in_order_match_across_gaps():
    assert _search("pkr")["pkr"][0] == "Usada Pekora"


def test_a_word_start_outranks_a_match_inside_a_word():
    found = _search("mori", "gura")
    assert found["mori"][:2] == ["Mori Calliope", "Himemori Luna"]
    assert found["gura"][0] == "Gawr Gura"


def test_query_words_can_come_in_any_order():
    assert _search("luna hime")["luna hime"][0] == "Himemori Luna"


def test_one_typo_still_finds_the_talent():
    found = _search("suisie", "pekoar", "calilope")
    assert found["suisie"][0] == "Hoshimachi Suisei"
    assert found["pekoar"][0] == "Usada Pekora"
    assert found["calilope"][0] == "Mori Calliope"


def test_case_accents_and_punctuation_do_not_matter():
    found = _search("INANIS", "la+", "Nöel")
    assert found["INANIS"][0] == "Ninomae Ina'nis"
    assert found["la+"][0] == "La+ Darknesss"
    assert found["Nöel"][0] == "Shirogane Noel"


def test_nothing_matching_returns_nothing():
    assert _search("zzzq")["zzzq"] == []


def test_an_empty_query_keeps_the_given_order():
    assert _search("")[""] == NAMES
