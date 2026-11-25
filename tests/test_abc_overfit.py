import json
import sys
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
from evaluate_abc_overfit import note_match, raw_pitch_tokens
from abc_vocal_format import load_validator
from prepare_abc_overfit import prepare


def test_subset_duplicates_training_only_for_memorization(tmp_path):
    source = tmp_path / "train.json"
    rows = [{"id": str(i), "text": f"input {i}", "summary": f"target {i}"} for i in range(25)]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "diagnostic"
    result = prepare(source, output, 20)
    assert result["count"] == 20
    assert (output / "train.json").read_bytes() == (output / "validation.json").read_bytes()
    assert len((output / "train.json").read_text().splitlines()) == 20
    assert source.read_text().count('"id"') == 25
    with pytest.raises(FileExistsError):
        prepare(source, output, 20)


def test_pitch_match_counts_contiguous_input_melody():
    assert note_match([[0, 60, 1], [1, 62, 1], [2, 64, 1]],
                      [[0, 65, 1], [1, 60, 1], [2, 62, 1], [3, 64, 1]]) == 1.0
    assert note_match([[0, 60, 1]], []) == 0.0
    assert raw_pitch_tokens("V: Vocal\nc2c2c2|\n", load_validator()) == 3
