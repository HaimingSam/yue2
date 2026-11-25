import json
import sys
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
from abc_chunk_groups import chunk_prompt, chunk_target, parse_chunk, validate_partial_vocal
from prepare_abc_chunk_groups import build
from prepare_abc_seq2seq import split_for


HEADER = ("X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=120\n"
          'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\nK:C\n')
INPUT = HEADER + "% chorus\nV: Vocal\nC4D4E4F4|\n"
TARGET = (HEADER + "% intro\nV: Vocal\nZ2|\n% verse\nV: Vocal\nC4D4E4F4|\n"
          "V: Vocal\nM:2/4\nC4D4|\n% chorus\nV: Vocal\nM:4/4\nC4D4E4F4|\n"
          "% verse\nV: Vocal\nC4D4E4F4|\n")


def test_one_group_exactly_and_strict_duration():
    group = ["V: Vocal", "C4D4E4F4|"]
    assert parse_chunk(chunk_target(group, True)) == (group, True)
    assert parse_chunk(chunk_target(group, False)) == (group, False)
    with pytest.raises(ValueError, match="1–4 measures"):
        parse_chunk("<MORE_GROUP>\nV: Vocal\n" + "C4D4E4F4|" * 5)
    with pytest.raises(ValueError, match="duration"):
        validate_partial_vocal(HEADER + "% verse\nV: Vocal\nC4D4E4|\n")
    assert validate_partial_vocal(HEADER + "% verse\nV: Vocal\nC4D4E4F4|\n") == 1


def test_prepare_chunks_uses_song_split_and_group_context(tmp_path):
    source = tmp_path / "pairs"
    source.mkdir()
    records = []
    for song_id in ("song1", "song2"):
        directory = source / song_id
        directory.mkdir()
        (directory / "input.abc").write_text(INPUT, encoding="utf-8")
        (directory / "target.abc").write_text(TARGET, encoding="utf-8")
        records.append({"id": song_id, "input": f"{song_id}/input.abc",
                        "target": f"{song_id}/target.abc", "section": "chorus"})
    (source / "pairs.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    output = tmp_path / "chunks"
    info = build(source, output, 20261026)
    assert info["groups"] == 8
    for song_id in ("song1", "song2"):
        split = split_for(song_id, 20261026)
        rows = [json.loads(line) for line in (output / "hf" / f"{split}.json").read_text().splitlines()]
        own = [row for row in rows if row["song_id"] == song_id]
        assert len(own) == 4
        verse_second = next(row for row in own if row["section_index"] == 2 and row["chunk_index"] == 2)
        assert "<PREVIOUS_GROUP>\nV: Vocal\nC4D4E4F4|" in verse_second["text"]
        assert verse_second["summary"].startswith("<END_SECTION>\nV: Vocal\nM:2/4\n")
        assert "<GROUP_INDEX>2</GROUP_INDEX>" in verse_second["text"]
    with pytest.raises(FileExistsError):
        build(source, output, 20261026)
