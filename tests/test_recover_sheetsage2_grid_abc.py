import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("recover_sheetsage2_grid_abc",
                                              TOOLS / "recover_sheetsage2_grid_abc.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_removes_only_matching_unrepresentable_note():
    target = SimpleNamespace(pitch=79, start=299.940104, end=300.0)
    other = SimpleNamespace(pitch=77, start=299.1, end=299.8)
    midi = SimpleNamespace(instruments=[SimpleNamespace(name="Vocal", notes=[other, target]),
                                        SimpleNamespace(name="Ins", notes=[])])
    info = module.remove_unrepresentable_note(
        midi, "Vocal: MIDI note pitch=79 at 299.940104-300.000000 cannot be represented on the decoded subbeat grid")
    assert info["pitch"] == 79
    assert midi.instruments[0].notes == [other]
    with pytest.raises(ValueError, match="Cannot safely recover"):
        module.remove_unrepresentable_note(midi, "Wrong meter")
    with pytest.raises(ValueError, match="found 0"):
        module.remove_unrepresentable_note(
            midi, "Vocal: MIDI note pitch=79 at 299.940104-300.000000 cannot be represented on the decoded subbeat grid")
