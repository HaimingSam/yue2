#!/usr/bin/env python3
"""Convert ABC section pairs into reproducible Hugging Face JSON datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DEFAULT_PAIRS_ROOT = Path(__file__).resolve().parents[1] / "abc-chorus-pairs"


def split_for(song_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{song_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10
    return "train" if bucket < 8 else "validation" if bucket == 8 else "test"


def prepare(pairs_root: Path, output_root: Path, seed: int) -> dict[str, int]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_root}")
    records = [json.loads(line) for line in (pairs_root / "pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    if not records:
        raise ValueError("pairs.jsonl is empty")
    splits: dict[str, list[dict]] = {name: [] for name in ("train", "validation", "test")}
    seen_ids: set[str] = set()
    for record in records:
        song_id = record["id"]
        if song_id in seen_ids:
            raise ValueError(f"Duplicate song ID: {song_id}")
        seen_ids.add(song_id)
        source = (pairs_root / record["input"]).read_text(encoding="utf-8")
        target = (pairs_root / record["target"]).read_text(encoding="utf-8")
        if not source.strip() or not target.strip():
            raise ValueError(f"Empty score for {song_id}")
        splits[split_for(song_id, seed)].append({
            "id": song_id, "text": source, "summary": target, "section": record["section"]
        })
    if any(not items for items in splits.values()):
        raise ValueError("An empty split was produced; choose another seed or add more data")
    output_root.mkdir(parents=True)
    for name, items in splits.items():
        (output_root / f"{name}.json").write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items), encoding="utf-8"
        )
    counts = {name: len(items) for name, items in splits.items()}
    (output_root / "split_info.json").write_text(
        json.dumps({"seed": seed, "counts": counts, "format": "JSON Lines (.json extension)"},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-root", type=Path, default=DEFAULT_PAIRS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_PAIRS_ROOT / "hf")
    parser.add_argument("--seed", type=int, default=20261026)
    args = parser.parse_args()
    print(json.dumps(prepare(args.pairs_root, args.output_root, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
