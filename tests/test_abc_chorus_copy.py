import sys
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
from abc_chorus_copy import CONTROL, PLACEHOLDER, restore_input_section, scaffold_target


def test_scaffold_and_restore_keeps_input_section_exactly():
    source = "X:1\nT:\n% chorus\nV: Vocal\nC4D4|\n"
    target = "X:1\nT:\n% intro\nV: Vocal\nZ|\n% chorus\nV: Vocal\nC4D4|\n% verse\nV: Vocal\nE4F4|\n"
    scaffold = scaffold_target(target, source, "chorus", 1)
    assert scaffold.startswith(CONTROL + "\n")
    assert PLACEHOLDER in scaffold
    restored, mode = restore_input_section(scaffold, source)
    assert mode == "placeholder"
    assert restored == target


def test_fallback_is_explicit_and_strict_mode_rejects_missing_placeholder():
    source = "X:1\nT:\n% chorus\nV: Vocal\nC4D4|\n"
    generated = "X:1\nT:\n% chorus\nV: Vocal\nG4A4|\n"
    with pytest.raises(ValueError):
        restore_input_section(generated, source)
    restored, mode = restore_input_section(generated, source, strict=False)
    assert mode == "section-fallback"
    assert "C4D4|" in restored
