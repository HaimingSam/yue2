#!/usr/bin/env python3
"""Select a deterministic subset of training scores for a memorization test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def prepare(source: Path, output: Path, count: int) -> dict:
    if count < 1:
        raise ValueError("--count must be positive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    if count > len(rows):
        raise ValueError(f"Requested {count}, but only {len(rows)} rows available")
    selected = rows[:count]
    if len({row["id"] for row in selected}) != count:
        raise ValueError("Duplicate song IDs in subset")
    output.mkdir(parents=True)
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)
    (output / "train.json").write_text(data, encoding="utf-8")
    (output / "validation.json").write_text(data, encoding="utf-8")
    (output / "selection.json").write_text(json.dumps({
        "source": str(source), "purpose": "memorization diagnostic only; validation equals training",
        "count": count, "ids": [row["id"] for row in selected]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output), "count": count,
            "note": "Validation duplicates training: never use its loss to claim generalization"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "abc-chorus-pairs-vocal/hf/train.json")
    parser.add_argument("--output", type=Path, default=ROOT / "abc-overfit-20/data")
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    if args.output.resolve() == args.source.parent.resolve():
        parser.error("--output cannot overwrite the source split directory")
    print(json.dumps(prepare(args.source, args.output, args.count), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
