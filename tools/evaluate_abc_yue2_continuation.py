#!/usr/bin/env python3
"""Compare YuE2's untrained ABC prefix continuation on held-out chorus inputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

from abc_chorus_copy import first_section_body, section_ranges
from abc_vocal_format import load_validator, restore_ins


ROOT = Path(__file__).resolve().parents[1]


def open_prefix(request, tokenizer, vocal_abc: str) -> tuple[list[int], str]:
    from yue2.protocol import ABC_START, EOD

    native = restore_ins(vocal_abc)
    return [EOD] + tokenizer.encode(request.text()) + [ABC_START] + tokenizer.encode(native), native


def score_continuation(original: str, completion: str, validator) -> dict:
    full = original + completion
    report = {"valid": False, "exact_chorus_repeated": False, "song_like": False}
    try:
        score = validator.parse_abc(full)
        original_score = validator.parse_abc(original)
        label, body = first_section_body(original)
        positions = list(section_ranges(full.splitlines()))
        later = [(start, end) for _, found, start, end in positions[1:] if found == label]
        lines = full.splitlines()
        pitches = [note[1] for note in score.voices["Vocal"].notes[
            len(original_score.voices["Vocal"].notes):]]
        frequencies = Counter(pitches)
        report.update(valid=True,
                      exact_chorus_repeated=any(lines[start + 1:end] == body for start, end in later),
                      song_like=len(positions) > 1 and len(pitches) >= 8,
                      sections=[label for _, label, _, _ in positions],
                      generated_vocal_notes=len(pitches),
                      generated_unique_pitches=len(frequencies),
                      generated_top_pitch_share=round(max(frequencies.values()) / len(pitches), 3)
                      if pitches else None)
    except ValueError as exc:
        report["error"] = str(exc)
    return report


def evaluate(source: Path, inputs: Path, output: Path, limit: int, model: str, vae: str,
             device: str, max_tokens: int, seed: int, style: str, lyrics: str) -> dict:
    from yue2 import YuE2Pipeline
    from yue2.protocol import GenerationConfig, SongRequest
    from yue2.sampling import synchronize

    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    if not 1 <= limit <= len(rows) or max_tokens < 1:
        raise ValueError("Require 1 <= limit <= number of songs and positive max_tokens")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    validator = load_validator()
    results = []
    with YuE2Pipeline.from_pretrained(model, vae=vae, device=device, local_files_only=True) as pipe:
        sampling = replace(GenerationConfig().abc, max_tokens=max_tokens)
        for index, row in enumerate(rows[:limit]):
            song_id = row["id"]
            vocal = (inputs / song_id / "input.abc").read_text(encoding="utf-8")
            request = SongRequest(style=style, lyrics=lyrics, cot="full", seed=seed + index, id=song_id)
            prefix, original = open_prefix(request, pipe.tokenizer, vocal)
            generated_ids, timing, truncated = pipe._generate(prefix, sampling, request.seed, "abc")
            synchronize(pipe.device)
            completion = pipe.tokenizer.decode(generated_ids)
            (output / f"{song_id}.continuation.txt").write_text(completion, encoding="utf-8")
            (output / f"{song_id}.raw.abc").write_text(original + completion, encoding="utf-8")
            result = {"id": song_id, "truncated": truncated, "generated_tokens": len(generated_ids),
                      "seconds": timing.get("seconds"),
                      **score_continuation(original, completion, validator)}
            if result["valid"] and not truncated:
                (output / f"{song_id}.abc").write_text(original + completion, encoding="utf-8")
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    summary = {"songs": len(results), "valid": sum(item["valid"] and not item["truncated"] for item in results),
               "song_like": sum(item["song_like"] and not item["truncated"] for item in results),
               "exact_chorus_repeated": sum(item["exact_chorus_repeated"] and not item["truncated"]
                                            for item in results),
               "single_pitch_generated": sum(item.get("generated_unique_pitches") == 1
                                              and not item["truncated"] for item in results),
               "output": str(output), "model": model, "seed": seed, "max_tokens": max_tokens}
    (output / "report.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n"
                                                for item in results), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "abc-two-stage/stage1-counted/hf/test.json")
    parser.add_argument("--inputs", type=Path, default=ROOT / "abc-chorus-pairs-vocal")
    parser.add_argument("--output", type=Path, default=ROOT / "abc-two-stage/evaluation/yue2-continuation-5")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--model", default="m-a-p/YuE2-3B")
    parser.add_argument("--vae", default="m-a-p/YuE2-Vae")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--style", default="Mandarin pop, expressive female lead, melodic verse and chorus")
    parser.add_argument("--lyrics", default="[Verse]\nla la la la\n[Chorus]\nla la la la")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.source, args.inputs, args.output, args.limit, args.model, args.vae,
                              args.device, args.max_tokens, args.seed, args.style, args.lyrics),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
