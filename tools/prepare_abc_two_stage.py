#!/usr/bin/env python3
"""Prepare independent structure-planning and section-generation datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abc_chorus_copy import section_ranges
from prepare_abc_seq2seq import split_for


ROOT = Path(__file__).resolve().parents[1]


def write_split(rows: list[dict], root: Path, seed: int, split_key: str = "id") -> dict[str, int]:
    splits = {name: [] for name in ("train", "validation", "test")}
    for row in rows:
        splits[split_for(row[split_key], seed)].append(row)
    root.mkdir(parents=True, exist_ok=True)
    for name, values in splits.items():
        (root / f"{name}.json").write_text(
            "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values), encoding="utf-8")
    return {name: len(values) for name, values in splits.items()}


def structure_target(text: str) -> str:
    labels = [label for _, label, _, _ in section_ranges(text.splitlines())]
    if not labels:
        raise ValueError("Target has no section markers")
    return "<STRUCTURE>\n" + "\n".join(f"% {label}" for label in labels) + "\n</STRUCTURE>\n"


def build(source: Path, output: Path, seed: int) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    records = [json.loads(line) for line in (source / "pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    stage1, stage2 = [], []
    for record in records:
        input_text = (source / record["input"]).read_text(encoding="utf-8")
        target_text = (source / record["target"]).read_text(encoding="utf-8")
        structure = structure_target(target_text)
        stage1.append({"id": record["id"], "text": input_text, "summary": structure,
                       "section": record["section"]})
        target_lines = target_text.splitlines()
        for occurrence, label, start, end in section_ranges(target_lines):
            prompt = (f"<SECTION>{label}</SECTION>\n<CHORUS>\n{input_text}</CHORUS>\n"
                      f"{structure}")
            stage2.append({"id": f'{record["id"]}::{occurrence}', "song_id": record["id"],
                           "text": prompt, "summary": "\n".join(target_lines[start:end]) + "\n",
                           "section": label, "section_occurrence": occurrence})
    output.mkdir(parents=True)
    stage1_counts = write_split(stage1, output / "stage1" / "hf", seed)
    stage2_counts = write_split(stage2, output / "stage2" / "hf", seed, "song_id")
    (output / "info.json").write_text(json.dumps({
        "songs": len(records), "stage1_examples": len(stage1), "stage2_examples": len(stage2),
        "stage1_counts": stage1_counts, "stage2_counts": stage2_counts,
        "note": "Stage 2 splits use song-derived IDs; all sections of a song remain in one split"},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"songs": len(records), "stage1_examples": len(stage1), "stage2_examples": len(stage2),
            "stage1": stage1_counts, "stage2": stage2_counts, "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "abc-chorus-pairs-vocal")
    parser.add_argument("--output", type=Path, default=ROOT / "abc-two-stage")
    parser.add_argument("--seed", type=int, default=20261026)
    args = parser.parse_args()
    if args.output.resolve() == args.source.resolve():
        parser.error("--output must differ from --source")
    print(json.dumps(build(args.source, args.output, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
