"""The `reference/` tree documents methodology, not project state.

State rots: a corpus count, a run date, or a per-talent number written into
prose is wrong the next time the pipeline runs, and stale prose is worse than
no prose because it is believed. History belongs in git, which is built for
it; current numbers belong in `data/measurements/`, which is recomputable.

These tests make that rule enforceable instead of aspirational.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"

# A calendar date pins prose to a moment. `YYYY-MM` as a literal format
# placeholder is fine — it has no digits — so matching digits is enough.
_DATE = re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b")

# "8,859 clips", "76 talents", "75 months": corpus state, stale on the next run.
_CORPUS_COUNT = re.compile(
    r"\b\d[\d,]*\s+(?:clips?|records?|talents?|months?|streams?|videos?|files?)\b",
    re.IGNORECASE,
)

# "n=143", "n = 143": a specific sample, i.e. a result, not a method.
_SAMPLE_SIZE = re.compile(r"\bn\s*=\s*\d+")


def _reference_docs() -> list[Path]:
    return sorted(REFERENCE_DIR.glob("*.md"))


def test_reference_dir_exists_and_is_populated():
    assert _reference_docs(), f"no methodology docs found in {REFERENCE_DIR}"


@pytest.mark.parametrize("doc", _reference_docs(), ids=lambda p: p.name)
def test_reference_doc_carries_no_dates(doc: Path):
    hits = _DATE.findall(doc.read_text())
    assert not hits, (
        f"{doc.name} contains calendar date(s) {hits}. Methodology is timeless; "
        "put when-it-changed in the commit message."
    )


@pytest.mark.parametrize("doc", _reference_docs(), ids=lambda p: p.name)
def test_reference_doc_carries_no_corpus_counts(doc: Path):
    hits = _CORPUS_COUNT.findall(doc.read_text())
    assert not hits, (
        f"{doc.name} contains corpus count(s) {hits}. Counts are recomputed from "
        "data/measurements/, never written into prose."
    )


@pytest.mark.parametrize("doc", _reference_docs(), ids=lambda p: p.name)
def test_reference_doc_carries_no_sample_sizes(doc: Path):
    hits = _SAMPLE_SIZE.findall(doc.read_text())
    assert not hits, (
        f"{doc.name} contains sample size(s) {hits} — that is a result, and "
        "results live in the exports, not the methodology."
    )


@pytest.mark.parametrize("doc", _reference_docs(), ids=lambda p: p.name)
def test_reference_doc_does_not_cite_retired_plan_file(doc: Path):
    assert "PLAN.md" not in doc.read_text(), (
        f"{doc.name} references PLAN.md, which is retired. Point at the "
        "relevant reference/ file instead."
    )


def test_reference_index_links_every_doc():
    """A doc nobody links to is a doc nobody reads."""
    index = (REFERENCE_DIR / "README.md").read_text()
    missing = [
        doc.name
        for doc in _reference_docs()
        if doc.name != "README.md" and doc.name not in index
    ]
    assert not missing, f"reference/README.md does not link: {missing}"
