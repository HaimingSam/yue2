import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/train_abc_byt5.py"
SPEC = importlib.util.spec_from_file_location("train_abc_byt5", SCRIPT)
training = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = training
SPEC.loader.exec_module(training)


class FakeTokenizer:
    def __call__(self, text=None, *, text_target=None, truncation=False):
        assert truncation is False
        value = text_target if text_target is not None else text
        return {"input_ids": list(value.encode("utf-8")) + [1]}


def test_rejects_truncation_and_preserves_data(tmp_path):
    rows = [{"id": "a", "text": "ABC", "summary": "a full ABC score"}]
    examples = training.encode_rows(rows, FakeTokenizer(), 4, 17)
    assert examples[0]["input_ids"] == [65, 66, 67, 1]
    assert len(examples[0]["labels"]) == 17
    assert len(training.EncodedScores(examples)) == 1
    with pytest.raises(ValueError, match="raise limits rather than truncating"):
        training.encode_rows(rows, FakeTokenizer(), 4, 16)


def test_read_rows_requires_both_scores(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"text": "abc", "summary": ""}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing or empty"):
        training.read_rows(path)
