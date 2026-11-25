import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "tools/generate_abc_byt5.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("generate_abc_byt5", SCRIPT)
generation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = generation
SPEC.loader.exec_module(generation)


def test_validate_generated_score():
    source = ("X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=120\n"
              'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\n'
              'V: Ins clef=treble name="Ins Melody" snm="Inst."\nK:C\n'
              "% chorus\nV: Vocal\nC4D4E4F4|\nV: Ins\nZ|\n")
    assert generation.validate_abc(source) == {"valid": True, "vocal_notes": 4,
                                               "instrumental_notes": 0, "duration_seconds": 2.0}
    assert generation.validate_abc("not ABC")["valid"] is False
