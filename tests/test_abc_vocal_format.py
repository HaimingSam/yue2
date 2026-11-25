import importlib.util
import json
import sys
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
from abc_vocal_format import load_validator, restore_ins, to_vocal


HEADER = ("X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=120\n"
          'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\n'
          'V: Ins clef=treble name="Ins Melody" snm="Inst."\nK:C\n')
BODY = ("% intro\nV: Vocal\nZ2|\nV: Ins\nZ2|\n"
        "% chorus\nV: Vocal\nM:2/4\nK:G\nG4A4|B4c4|\nV: Ins\nM:2/4\nK:G\nZ2|\n"
        "% outro\nV: Vocal\nM:4/4\nC4D4E4F4|\nV: Ins\nM:4/4\nZ|\n")


def test_convert_preserves_vocal_and_restores_native_score():
    original = HEADER + BODY
    compact = to_vocal(original)
    restored = restore_ins(compact)
    validator = load_validator()
    expected = validator.parse_abc(original)
    actual = validator.parse_abc(restored)
    assert "V: Ins" not in compact
    assert [line for line in compact.splitlines() if line.startswith("% ")] == ["% intro", "% chorus", "% outro"]
    assert actual.voices["Vocal"].notes == expected.voices["Vocal"].notes
    assert actual.voices["Vocal"].bars == expected.voices["Vocal"].bars
    assert actual.voices["Vocal"].keys == expected.voices["Vocal"].keys
    assert actual.voices["Ins"].notes == []
    assert "V: Ins\nM:2/4\nK:G\nZ2|" in restored


def test_invalid_output_is_not_silently_repaired():
    compact = to_vocal(HEADER + BODY)
    with pytest.raises(ValueError):
        restore_ins(compact.replace("G4A4|B4c4|", "G4A4|B4c4"))
    with pytest.raises(ValueError):
        restore_ins(compact.replace("G4A4|B4c4|", "G4A4|B4c4|V: Ins"))


def test_inline_key_changes_are_synchronized_in_silent_ins():
    original = (HEADER + "% chorus\nV: Vocal\n"
                "z8[K:G]z8|\nV: Ins\nz8[K:G]z8|\n")
    compact = to_vocal(original)
    restored = restore_ins(compact)
    assert "V: Ins\nz8[K:G]z8|" in restored
    assert load_validator().parse_abc(restored).voices["Vocal"].keys == load_validator().parse_abc(restored).voices["Ins"].keys


def test_preparation_creates_new_splits_without_modifying_source(tmp_path):
    script = TOOLS / "prepare_abc_vocal_seq2seq.py"
    spec = importlib.util.spec_from_file_location("prepare_abc_vocal_seq2seq", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    source = tmp_path / "original"
    source.mkdir()
    records = []
    for song_id in range(30):
        directory = source / str(song_id)
        directory.mkdir()
        (directory / "input.abc").write_text(HEADER + BODY, encoding="utf-8")
        (directory / "target.abc").write_text(HEADER + BODY, encoding="utf-8")
        records.append({"id": str(song_id), "section": "chorus",
                        "input": f"{song_id}/input.abc", "target": f"{song_id}/target.abc"})
    (source / "pairs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    output = tmp_path / "vocal"
    result = module.build(source, output, 20261026)
    assert result["pairs"] == 30
    assert sum(result["splits"].values()) == 30
    assert (source / "0/input.abc").read_text() == HEADER + BODY
    assert "V: Ins" not in (output / "0/input.abc").read_text()
    assert "V: Ins" not in (output / "hf/train.json").read_text()
    with pytest.raises(FileExistsError):
        module.build(source, output, 20261026)
