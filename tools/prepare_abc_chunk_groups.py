#!/usr/bin/env python3
"""Create a new Stage-2 dataset of one 1–4 measure Vocal group per example."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from abc_chunk_groups import chunk_prompt, chunk_target, split_groups
from abc_chorus_copy import section_ranges
from prepare_abc_two_stage import structure_target, write_split


ROOT = Path(__file__).resolve().parents[1]


def build(source: Path, output: Path, seed: int) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    records = [json.loads(line) for line in (source / "pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    if not records:
        raise ValueError("No input pairs")
    rows = []
    copied_sections = 0
    for record in records:
        source_text = (source / record["input"]).read_text(encoding="utf-8")
        target_lines = (source / record["target"]).read_text(encoding="utf-8").splitlines()
        structure = structure_target("\n".join(target_lines) + "\n")
        occurrences = Counter()
        for section_index, label, start, end in section_ranges(target_lines):
            occurrences[label] += 1
            if label == record["section"]:
                copied_sections += 1
                continue
            groups = split_groups(target_lines[start + 1:end])
            for chunk_index, group in enumerate(groups, 1):
                rows.append({"id": f'{record["id"]}::{section_index}::{chunk_index}',
                             "song_id": record["id"], "section": label,
                             "section_index": section_index, "occurrence": occurrences[label],
                             "chunk_index": chunk_index,
                             "text": chunk_prompt(source_text, structure, label, section_index,
                                                  occurrences[label], chunk_index,
                                                  groups[chunk_index - 2] if chunk_index > 1 else None),
                             "summary": chunk_target(group, chunk_index == len(groups))})
    counts = write_split(rows, output / "hf", seed, "song_id")
    info = {"songs": len(records), "groups": len(rows), "copied_sections": copied_sections,
            "splits": counts, "source": str(source),
            "note": "All song IDs stay in the existing song-level split; copied sections are never predicted"}
    (output / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "abc-chorus-pairs-vocal")
    parser.add_argument("--output", type=Path, default=ROOT / "abc-two-stage/stage2-chunks")
    parser.add_argument("--seed", type=int, default=20261026)
    args = parser.parse_args()
    if args.output.resolve() == args.source.resolve():
        parser.error("--output cannot overwrite the source dataset")
    print(json.dumps(build(args.source, args.output, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
