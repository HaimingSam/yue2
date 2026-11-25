#!/usr/bin/env python3
"""Select every short-group training row for the same 20 song IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def build(songs_path: Path, chunks_path: Path, output: Path, count: int) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    songs = [json.loads(line) for line in songs_path.read_text(encoding="utf-8").splitlines()]
    if count < 1 or count > len(songs):
        raise ValueError(f"Count must be 1–{len(songs)}")
    ids = [row["id"] for row in songs[:count]]
    if len(set(ids)) != count:
        raise ValueError("Duplicate song ID")
    selected = set(ids)
    chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines()
              if json.loads(line)["song_id"] in selected]
    if {row["song_id"] for row in chunks} != selected:
        raise ValueError("Not all selected songs have training chunks")
    output.mkdir(parents=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in chunks)
    (output / "train.json").write_text(text, encoding="utf-8")
    (output / "validation.json").write_text(text, encoding="utf-8")
    info = {"songs": ids, "groups": len(chunks), "source": str(chunks_path),
            "warning": "Validation duplicates training for memorization; not a generalization metric"}
    (output / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--songs", type=Path, default=ROOT / "abc-two-stage/stage1/hf/train.json")
    parser.add_argument("--chunks", type=Path, default=ROOT / "abc-two-stage/stage2-chunks/hf/train.json")
    parser.add_argument("--output", type=Path, default=ROOT / "abc-two-stage/stage2-chunks-overfit-20/data")
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    if args.output.resolve() in (args.songs.parent.resolve(), args.chunks.parent.resolve()):
        parser.error("--output must not overwrite source splits")
    info = build(args.songs, args.chunks, args.output, args.count)
    print(json.dumps({"songs": len(info["songs"]), "groups": info["groups"],
                      "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
