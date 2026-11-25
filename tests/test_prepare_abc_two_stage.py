import importlib.util
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("prepare_abc_two_stage", TOOLS / "prepare_abc_two_stage.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_structure_and_section_examples_are_separate(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    song = source / "song"
    song.mkdir()
    input_text = "X:1\nT:\n% chorus\nV: Vocal\nC4D4|\n"
    target_text = "X:1\nT:\n% intro\nV: Vocal\nZ|\n% chorus\nV: Vocal\nC4D4|\n% verse\nV: Vocal\nE4F4|\n"
    (song / "input.abc").write_text(input_text)
    (song / "target.abc").write_text(target_text)
    (source / "pairs.jsonl").write_text(json.dumps({"id": "song", "input": "song/input.abc",
        "target": "song/target.abc", "section": "chorus", "section_occurrence": 1}) + "\n")
    result = module.build(source, tmp_path / "out", 20261026)
    assert result["stage1_examples"] == 1
    assert result["stage2_examples"] == 3
    stage1 = json.loads((tmp_path / "out/stage1/hf/train.json").read_text())
    stage2 = [json.loads(line) for name in ("train", "validation", "test")
              for line in (tmp_path / f"out/stage2/hf/{name}.json").read_text().splitlines()]
    assert stage1["summary"] == "<STRUCTURE>\n% intro\n% chorus\n% verse\n</STRUCTURE>\n"
    assert stage2[1]["section"] == "chorus"
    assert "<CHORUS>" in stage2[1]["text"]
