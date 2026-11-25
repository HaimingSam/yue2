import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/batch_transcribe_urv.py"
SPEC = importlib.util.spec_from_file_location("batch_transcribe_urv", SCRIPT)
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)

HEADER = (
    "X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=120\n"
    'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\n'
    'V: Ins clef=treble name="Ins Melody" snm="Inst."\nK:C\n'
)


def test_moves_instrumental_notes_to_empty_vocal(tmp_path):
    score_path = tmp_path / "score.abc"
    score_path.write_text(
        HEADER + "% intro\nV: Vocal\nZ2|\nV: Ins\nC4D4E4F4|G4A4B4c4|\n"
        "% outro\nV: Vocal\nZ|\nV: Ins\nc8d8|\n", encoding="utf-8")
    source = batch.load_abc_tools().parse_abc(score_path.read_text(encoding="utf-8"))

    result = batch.check_and_repair_abc(score_path)

    corrected = batch.load_abc_tools().parse_abc(score_path.read_text(encoding="utf-8"))
    assert result["repair"] == "ins_to_vocal"
    assert result["vocal_notes_before"] == 0
    assert corrected.voices["Vocal"].notes == source.voices["Ins"].notes
    assert corrected.voices["Ins"].notes == []


def test_keeps_original_when_vocal_has_notes(tmp_path):
    score_path = tmp_path / "score.abc"
    original = HEADER + "% verse\nV: Vocal\nC4D4E4F4|\nV: Ins\nG4A4B4c4|\n"
    score_path.write_text(original, encoding="utf-8")

    result = batch.check_and_repair_abc(score_path)

    assert result["repair"] == "none"
    assert result["vocal_notes_after"] == 4
    assert result["ins_notes_after"] == 4
    assert score_path.read_text(encoding="utf-8") == original


def test_rejects_score_without_notes(tmp_path):
    score_path = tmp_path / "score.abc"
    score_path.write_text(HEADER + "% intro\nV: Vocal\nZ|\nV: Ins\nZ|\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no sounding notes"):
        batch.check_and_repair_abc(score_path)


def test_select_ids_preserves_input_order_and_rejects_unknown():
    inputs = [Path("/source/a/vocals_dereverb.wav"), Path("/source/b/vocals_dereverb.wav")]
    assert batch.select_ids(inputs, ["b"]) == inputs[1:]
    assert batch.select_ids(inputs, None) == inputs
    with pytest.raises(ValueError, match="Missing vocals_dereverb.wav"):
        batch.select_ids(inputs, ["unknown"])


def test_alternative_input_name_is_explicit(tmp_path):
    folder = tmp_path / "sample"
    folder.mkdir()
    (folder / "vocals.wav").touch()
    assert batch.collect_inputs(tmp_path, "vocals.wav") == [folder / "vocals.wav"]
    with pytest.raises(ValueError, match="Unsupported input name"):
        batch.collect_inputs(tmp_path, "../escape.wav")
