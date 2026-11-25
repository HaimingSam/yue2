import importlib.util
import json
import sys
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("render_assembled_abc_audio",
                                              TOOLS / "render_assembled_abc_audio.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_validate_scores_requires_vocal_and_silent_ins(tmp_path):
    from abc_vocal_format import restore_ins

    source = ("X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=120\n"
              'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\nK:C\n'
              "% verse\nV: Vocal\nC4D4E4F4|\n")
    path = tmp_path / "example.abc"
    path.write_text(restore_ins(source))
    module.validate_scores([path])
    path.write_text(path.read_text().replace("V: Ins\nZ|", "V: Ins\nC16|"))
    with pytest.raises(ValueError, match="empty Ins"):
        module.validate_scores([path])


def test_mp3_command_uses_real_flac_input(tmp_path, monkeypatch):
    executed = []
    monkeypatch.setattr(module.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(module.subprocess, "run", lambda command, check: executed.append(command))
    source = tmp_path / "audio.flac"
    target = tmp_path / "audio.mp3"
    module.convert_mp3(source, target)
    assert executed[0][0] == "/usr/bin/ffmpeg"
    assert executed[0][-1] == str(target)
    assert str(source) in executed[0]
