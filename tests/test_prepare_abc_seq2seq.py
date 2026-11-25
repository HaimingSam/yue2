import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/prepare_abc_seq2seq.py"
SPEC = importlib.util.spec_from_file_location("prepare_abc_seq2seq", SCRIPT)
prepare_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare_module
SPEC.loader.exec_module(prepare_module)


def test_prepare_reads_scores_and_splits_by_song(tmp_path):
    pairs_root = tmp_path / "pairs"
    pairs_root.mkdir()
    records = []
    for number in range(100):
        song_id = str(number)
        pair_dir = pairs_root / song_id
        pair_dir.mkdir()
        (pair_dir / "input.abc").write_text(f"input {song_id}\n", encoding="utf-8")
        (pair_dir / "target.abc").write_text(f"target {song_id}\n", encoding="utf-8")
        records.append({"id": song_id, "input": f"{song_id}/input.abc",
                        "target": f"{song_id}/target.abc", "section": "chorus"})
    (pairs_root / "pairs.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in records), encoding="utf-8"
    )

    output_root = tmp_path / "hf"
    counts = prepare_module.prepare(pairs_root, output_root, 1234)

    assert sum(counts.values()) == 100
    for name, count in counts.items():
        lines = (output_root / f"{name}.json").read_text(encoding="utf-8").splitlines()
        assert len(lines) == count
        for line in lines:
            row = json.loads(line)
            assert row["text"] == f"input {row['id']}\n"
            assert row["summary"] == f"target {row['id']}\n"
            assert name == prepare_module.split_for(row["id"], 1234)
    with pytest.raises(FileExistsError):
        prepare_module.prepare(pairs_root, output_root, 1234)
