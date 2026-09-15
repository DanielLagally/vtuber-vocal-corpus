"""The published v1 site must not present its formants without warning.

v1's `formants_hz` averaged Praat's Burg output over every frame returning a
finite value, with no voiced-frame mask — so unvoiced consonants, breath,
silence and separator artifact are inside every published F1-F4 number. The
site's existing caption calls them "sensitive to the formant tracker's
parameters", which understates this: the issue is not sensitivity, it is that
the frames were never gated.

v1 is frozen, so the numbers stay. The warning is what stops the page
overstating them, which makes it load-bearing content rather than decoration —
hence a test, so it cannot be dropped silently in a later edit.

See docs/v1/LIMITATIONS.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

V1 = Path(__file__).resolve().parent.parent / "docs" / "v1"
FORMANT_KEYS = ("f1_hz", "f2_hz", "f3_hz", "f4_hz")


def _app_js() -> str:
    return (V1 / "app.js").read_text(encoding="utf-8")


def test_app_js_defines_a_shared_formant_warning():
    assert "FORMANT_WARNING" in _app_js()


def test_the_warning_says_the_frames_were_not_voiced_gated():
    """The specific defect, not a vague 'treat with care'."""
    source = _app_js()
    start = source.index("FORMANT_WARNING")
    warning = source[start : start + 600].lower()
    assert "voiced" in warning, "the warning must name voiced-frame gating"
    assert "unvoiced" in warning or "not gated" in warning or "never" in warning


@pytest.mark.parametrize("key", FORMANT_KEYS)
def test_every_formant_metric_carries_the_warning(key: str):
    """A reader who opens only F3 must still see it."""
    source = _app_js()
    start = source.index(f"{key}: {{")
    block = source[start : source.index("},", start)]
    assert "FORMANT_WARNING" in block, f"{key} caption omits the formant warning"


def test_the_radar_caption_warns_about_its_formant_axes():
    """The radar shows four formant axes at once and is the default profile view."""
    source = _app_js()
    start = source.index("const RADAR_CAPTION")
    caption = source[start : source.index(";", start)].lower()
    assert "formant" in caption


def test_the_page_links_to_the_limitations_document():
    assert "LIMITATIONS.md" in (V1 / "index.html").read_text(encoding="utf-8")


VOICE_QUALITY_KEYS = ("jitter_local", "shimmer_local", "hnr_db")


@pytest.mark.parametrize("key", VOICE_QUALITY_KEYS)
def test_voice_quality_metrics_carry_the_sustained_vowel_caveat(key: str):
    """The code's own docstrings always said these are calibrated for a
    sustained vowel rather than conversation. The site captions did not."""
    source = _app_js()
    start = source.index(f"{key}: {{")
    block = source[start : source.index("},", start)]
    assert "VOICE_QUALITY_WARNING" in block, f"{key} caption omits the caveat"


def test_hnr_warns_that_the_separator_inflates_it():
    """Measured on v1's own clips: isolation raises HNR by construction, so a
    trend in it is a candidate artifact before it is a finding."""
    source = _app_js()
    start = source.index("hnr_db: {")
    block = source[start : source.index("},", start)].lower()
    assert "isolation" in block or "separator" in block
    assert "artifact" in block
