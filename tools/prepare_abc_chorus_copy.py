#!/usr/bin/env python3
"""Create a separate Vocal dataset whose target contains an input-section placeholder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abc_chorus_copy import scaffold_target
from prepare_abc_seq2seq import prepare


ROOT = Path(__file__).resolve().parents[1]


def build(source: Path, output: Path, seed: int) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    records = [json.loads(line) for line in (source / "pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    output.mkdir(parents=True, exist_ok=True)
    converted = []
    for record in records:
        input_text = (source / record["input"]).read_text(encoding="utf-8")
        target_text = (source / record["target"]).read_text(encoding="utf-8")
        scaffold = scaffold_target(target_text, input_text, record["section"], record["section_occurrence"])
        directory = output / record["id"]
        directory.mkdir()
        (directory / "input.abc").write_text(input_text, encoding="utf-8")
        (directory / "target.abc").write_text(scaffold, encoding="utf-8")
        converted.append({**record, "target": f'{record["id"]}/target.abc'})
    (output / "pairs.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in converted), encoding="utf-8")
    splits = prepare(output, output / "hf", seed)
    return {"pairs": len(converted), "splits": splits, "output": str(output),
            "target": "full Vocal score with the selected input section replaced by <INPUT_SECTION>"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "abc-chorus-pairs-vocal")
    parser.add_argument("--output", type=Path, default=ROOT / "abc-chorus-pairs-vocal-copychorus")
    parser.add_argument("--seed", type=int, default=20261026)
    args = parser.parse_args()
    if args.output.resolve() == args.source.resolve():
        parser.error("--output must differ from --source")
    print(json.dumps(build(args.source, args.output, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
