#!/usr/bin/env python3
"""Create separate Vocal-only ABC pairs and train/validation/test JSONL splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abc_vocal_format import load_validator, to_vocal
from prepare_abc_seq2seq import prepare


ROOT = Path(__file__).resolve().parents[1]


def build(source: Path, output: Path, seed: int) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    records = [json.loads(line) for line in (source / "pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    if not records:
        raise ValueError("No source pairs found")
    validator = load_validator()
    converted = []
    nonempty_ins = 0
    for record in records:
        input_text = (source / record["input"]).read_text(encoding="utf-8")
        target_text = (source / record["target"]).read_text(encoding="utf-8")
        if validator.parse_abc(target_text).voices["Ins"].notes:
            nonempty_ins += 1
        try:
            converted.append((record, to_vocal(input_text, validator), to_vocal(target_text, validator)))
        except ValueError as exc:
            raise ValueError(f"{record['id']}: {exc}") from exc
    output.mkdir(parents=True, exist_ok=True)
    for record, vocal_input, vocal_target in converted:
        directory = output / record["id"]
        directory.mkdir()
        (directory / "input.abc").write_text(vocal_input, encoding="utf-8")
        (directory / "target.abc").write_text(vocal_target, encoding="utf-8")
    (output / "pairs.jsonl").write_text(
        "".join(json.dumps({**record, "source_input": record["input"],
                            "source_target": record["target"]}, ensure_ascii=False) + "\n"
                for record, _, _ in converted), encoding="utf-8")
    splits = prepare(output, output / "hf", seed)
    return {"pairs": len(converted), "original_targets_with_ins_notes": nonempty_ins,
            "note": "Training targets contain only Vocal; native output restores a silent Ins voice",
            "splits": splits, "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "abc-chorus-pairs")
    parser.add_argument("--output", type=Path, default=ROOT / "abc-chorus-pairs-vocal")
    parser.add_argument("--seed", type=int, default=20261026)
    args = parser.parse_args()
    if args.output.resolve() == args.source.resolve():
        parser.error("--output must differ from --source to protect the original data")
    print(json.dumps(build(args.source, args.output, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
