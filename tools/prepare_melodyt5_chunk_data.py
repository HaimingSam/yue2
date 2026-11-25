#!/usr/bin/env python3
"""Convert YuE chunk JSONL files to MelodyT5 score-to-score JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert(row: dict) -> dict:
    return {
        "id": row["id"],
        "song_id": row.get("song_id", row["id"].split("::", 1)[0]),
        "input": "%%input\n%%generation\n" + row["text"].strip() + "\n",
        "output": row["summary"].strip() + "\n",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-songs", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "validation", "test"):
        rows = []
        selected = set()
        for line in (args.source / f"{split}.json").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            song_id = row.get("song_id", row["id"].split("::", 1)[0])
            if args.max_songs is not None and song_id not in selected:
                if len(selected) >= args.max_songs:
                    continue
                selected.add(song_id)
            rows.append(convert(row))
        (args.output / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        counts[split] = len(rows)
    (args.output / "info.json").write_text(json.dumps({"counts": counts, "max_songs": args.max_songs}, indent=2) + "\n")
    print(json.dumps({"counts": counts, "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
