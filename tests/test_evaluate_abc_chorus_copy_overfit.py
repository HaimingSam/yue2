import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
from evaluate_abc_chorus_copy_overfit import section_body


def test_section_body_selects_requested_occurrence():
    text = "% chorus\nV: Vocal\nA4|\n% verse\nV: Vocal\nB4|\n% chorus\nV: Vocal\nC4|\n"
    assert section_body(text, "chorus", 1) == ["V: Vocal", "A4|"]
    assert section_body(text, "chorus", 2) == ["V: Vocal", "C4|"]
