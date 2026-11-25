import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/build_abc_section_pairs.py"
SPEC = importlib.util.spec_from_file_location("build_abc_section_pairs", SCRIPT)
pairs = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pairs
SPEC.loader.exec_module(pairs)
ABC_TOOLS = pairs.load_abc_tools()

HEADER = (
    "X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=120\n"
    'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\n'
    'V: Ins clef=treble name="Ins Melody" snm="Inst."\nK:C\n'
)


def write_score(tmp_path, body):
    path = tmp_path / "song.abc"
    path.write_text(HEADER + body, encoding="utf-8")
    return path


def test_prefers_first_chorus_and_preserves_target(tmp_path):
    path = write_score(tmp_path, "% verse\nV: Vocal\nC4D4E4F4|\nV: Ins\nZ|\n"
                       "% chorus\nV: Vocal\nG4A4B4c4|\nV: Ins\nZ|\n"
                       "% chorus\nV: Vocal\nc4B4A4G4|\nV: Ins\nZ|\n")
    original = path.read_text(encoding="utf-8")

    fragment, info = pairs.make_pair(path, ABC_TOOLS)

    assert info["section"] == "chorus"
    assert info["section_occurrence"] == 1
    assert info["source_start_seconds"] == 2
    assert info["source_end_seconds"] == 4
    assert fragment.count("% chorus") == 1
    assert "% verse" not in fragment
    assert len(ABC_TOOLS.parse_abc(fragment).voices["Vocal"].notes) == 4
    assert path.read_text(encoding="utf-8") == original


def test_falls_back_to_verse_when_no_chorus(tmp_path):
    path = write_score(tmp_path, "% intro\nV: Vocal\nZ|\nV: Ins\nZ|\n"
                       "% verse\nV: Vocal\nC4D4E4F4|\nV: Ins\nZ|\n")

    fragment, info = pairs.make_pair(path, ABC_TOOLS)

    assert info["section"] == "verse"
    assert "% intro" not in fragment


def test_carries_meter_and_clips_ties_at_both_boundaries(tmp_path):
    path = write_score(tmp_path, "% verse\nV: Vocal\nM:2/4\nC4D4-|\nV: Ins\nM:2/4\nZ|\n"
                       "% chorus\nV: Vocal\nD4E4-|\nV: Ins\nZ|\n"
                       "% outro\nV: Vocal\nE4F4|\nV: Ins\nZ|\n")

    fragment, info = pairs.make_pair(path, ABC_TOOLS)

    assert info["section"] == "chorus"
    assert "M:2/4" in fragment.splitlines()[:8]
    assert "D4E4|" in fragment
    notes = ABC_TOOLS.parse_abc(fragment).voices["Vocal"].notes
    assert len(notes) == 2


def test_skips_empty_chorus_for_next_valid_occurrence(tmp_path):
    path = write_score(tmp_path, "% chorus\nV: Vocal\nZ|\nV: Ins\nZ|\n"
                       "% chorus\nV: Vocal\nC4D4E4F4|\nV: Ins\nZ|\n")

    fragment, info = pairs.make_pair(path, ABC_TOOLS)

    assert info["section_occurrence"] == 2
    assert "% chorus" in fragment


def test_reports_song_without_chorus_or_verse(tmp_path):
    path = write_score(tmp_path, "% intro\nV: Vocal\nC4D4E4F4|\nV: Ins\nZ|\n")

    with pytest.raises(ValueError, match="No chorus or verse"):
        pairs.make_pair(path, ABC_TOOLS)
