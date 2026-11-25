#!/usr/bin/env python3
"""Prepare a new Stage-1 dataset predicting structure and group counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abc_chorus_copy import section_ranges
from abc_chunk_groups import split_groups
from prepare_abc_two_stage import write_split


ROOT = Path(__file__).resolve().parents[1]


def counted_structure(target_text: str) -> str:
    lines = target_text.splitlines()
    plan = [f"% {label} groups={len(split_groups(lines[start + 1:end]))}"
            for _, label, start, end in section_ranges(lines)]
    if not plan:
        raise ValueError("No sections in target")
    return "<STRUCTURE>\n" + "\n".join(plan) + "\n</STRUCTURE>\n"


def build(source: Path, output: Path, seed: int) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    records = [json.loads(line) for line in (source / "pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = []
    for record in records:
        input_text = (source / record["input"]).read_text(encoding="utf-8")
        target = (source / record["target"]).read_text(encoding="utf-8")
        rows.append({"id": record["id"], "text": input_text,
                     "summary": counted_structure(target), "section": record["section"]})
    splits = write_split(rows, output / "hf", seed)
    info = {"songs": len(rows), "splits": splits, "source": str(source)}
    (output / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "abc-chorus-pairs-vocal")
    parser.add_argument("--output", type=Path, default=ROOT / "abc-two-stage/stage1-counted")
    parser.add_argument("--seed", type=int, default=20261026)
    args = parser.parse_args()
    if args.output.resolve() == args.source.resolve():
        parser.error("--output must not overwrite source scores")
    print(json.dumps(build(args.source, args.output, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
