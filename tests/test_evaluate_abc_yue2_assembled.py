import importlib.util
import sys
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("evaluate_abc_yue2_assembled",
                                              TOOLS / "evaluate_abc_yue2_assembled.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


SOURCE = ("X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=120\n"
          'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\nK:C\n'
          "% chorus\nV: Vocal\nC4D4E4F4|\n")


def test_candidate_filter_and_exact_assembly():
    validator = module.load_validator()
    native_input = module.restore_ins(SOURCE, validator)
    first = "% verse\nV: Vocal\nG2A2B2c2G2A2B2c2|G2A2B2c2G2A2B2c2|\nV: Ins\nC16|C16|\n"
    second = "% verse\nV: Vocal\nc2B2A2G2c2B2A2G2|c2B2A2G2c2B2A2G2|\nV: Ins\nZ2|\n"
    candidates = module.verse_candidates(native_input, first + second, validator)
    assert len(candidates) == 2
    native = module.assemble(SOURCE, candidates[0], candidates[1], validator)
    score = validator.parse_abc(native)
    assert len(score.voices["Vocal"].notes) == 40
    assert not score.voices["Ins"].notes
    assert [label for _, label, _, _ in module.section_ranges(native.splitlines())] == [
        "verse", "chorus", "verse", "chorus"]
    assert native.count("C4D4E4F4|") == 2


def test_reject_bad_verses_and_duplicate_assembly():
    validator = module.load_validator()
    original = module.restore_ins(SOURCE, validator)
    bad = "% verse\nV: Vocal\nG2A2B2c2G2A2B2c4|\nV: Ins\nZ|\n"
    dull = "% verse\nV: Vocal\ne2e2e2e2e2e2e2e2|e2e2e2e2e2e2e2e2|\nV: Ins\nZ2|\n"
    assert module.verse_candidates(original, bad + dull, validator) == []
    good = "% verse\nV: Vocal\nG2A2B2c2G2A2B2c2|G2A2B2c2G2A2B2c2|\nV: Ins\nZ2|\n"
    candidate = module.verse_candidates(original, good, validator)[0]
    with pytest.raises(ValueError, match="two different verse"):
        module.assemble(SOURCE, candidate, candidate, validator)
    assert module.verse_candidates(original, good, validator, truncated=True) == []


def test_assembly_rejects_chorus_that_sounds_different_after_key_change():
    validator = module.load_validator()
    original = module.restore_ins(SOURCE, validator)
    first = "% verse\nV: Vocal\nK:G\nG2A2B2c2G2A2B2c2|G2A2B2c2G2A2B2c2|\nV: Ins\nK:G\nZ2|\n"
    second = "% verse\nV: Vocal\nc2B2A2G2c2B2A2G2|c2B2A2G2c2B2A2G2|\nV: Ins\nZ2|\n"
    candidates = (module.verse_candidates(original, first, validator)
                  + module.verse_candidates(original, second, validator))
    assert len(candidates) == 2
    with pytest.raises(ValueError, match="sounding chorus"):
        module.assemble(SOURCE, candidates[0], candidates[1], validator)
