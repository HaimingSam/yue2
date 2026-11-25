#!/usr/bin/env python3
"""Evaluate chunked full-song generation with gold or predicted structures."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import generate_abc_two_stage as inference
from abc_chorus_copy import section_ranges
from abc_vocal_format import load_validator, to_vocal
from prepare_abc_structure_counts import counted_structure


ROOT = Path(__file__).resolve().parents[1]


def evaluate(source: Path, inputs: Path, model: str, output: Path, limit: int,
             max_groups: int = 32, max_tokens: int = 256, device: str = "cuda",
             structure_model: str | None = None, sample_candidates: int = 1,
             temperature: float = 0.8, top_p: float = 0.9, seed: int | None = None) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    if limit < 1 or limit > len(rows):
        raise ValueError(f"--limit must be between 1 and {len(rows)}")
    output.mkdir(parents=True, exist_ok=True)
    if seed is not None:
        import torch
        torch.manual_seed(seed)
    validator = load_validator()
    original_load = inference.load_model
    inference.load_model = lru_cache(maxsize=2)(original_load)
    reports = []
    for row in rows[:limit]:
        score_path = output / f'{row["id"]}.abc'
        label = row["section"]
        result = {"id": row["id"], "valid": False, "copied_section_exact": False}
        input_path = inputs / row["id"] / "input.abc"
        try:
            gold_structure = None
            if structure_model is None:
                target_path = inputs / row["id"] / "target.abc"
                gold_structure = counted_structure(target_path.read_text(encoding="utf-8"))
            report = inference.run(input_path, score_path, structure_model, model,
                                   device, 256, max_tokens, chunked=True,
                                   max_groups_per_section=max_groups,
                                   structure_override=gold_structure,
                                   sample_candidates=sample_candidates,
                                   temperature=temperature, top_p=top_p)
            generated = validator.parse_abc(score_path.read_text(encoding="utf-8"))
            vocal = to_vocal(score_path.read_text(encoding="utf-8"), validator)
            input_text = input_path.read_text(encoding="utf-8")
            input_lines = input_text.splitlines()
            start = next(start for _, found, start, _ in section_ranges(input_lines) if found == label)
            end = next(end for _, found, _, end in section_ranges(input_lines) if found == label)
            source_body = input_lines[start + 1:end]
            target_lines = vocal.splitlines()
            copies = [(start, end) for _, found, start, end in section_ranges(target_lines) if found == label]
            exact = bool(copies) and all(target_lines[start + 1:end] == source_body for start, end in copies)
            pitch_counts = Counter()
            for _, section_label, start, end in section_ranges(target_lines):
                if section_label == label:
                    continue
                for line in target_lines[start + 1:end]:
                    if line.startswith(("V:", "M:", "K:")):
                        continue
                    for token in validator.TOKEN.finditer(line):
                        note = token.group("note")
                        if note is not None and note != "z":
                            pitch_counts[(token.group("acc") or "") + note + (token.group("oct") or "")] += 1
            result.update(valid=True, copied_section_exact=exact,
                          vocal_notes=len(generated.voices["Vocal"].notes),
                          generated_notes=sum(pitch_counts.values()),
                          generated_unique_pitches=len(pitch_counts),
                          generated_top_pitch_share=round(max(pitch_counts.values()) / sum(pitch_counts.values()), 3)
                          if pitch_counts else None,
                          generated_sections=report["generated_sections"])
        except (ValueError, OSError) as exc:
            result["error"] = str(exc)
        reports.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    summary = {"songs": len(reports), "valid": sum(item["valid"] for item in reports),
               "exact_copied_section": sum(item["copied_section_exact"] for item in reports),
               "with_vocal_notes": sum(item.get("vocal_notes", 0) > 0 for item in reports),
               "single_pitch_generated": sum(item.get("generated_unique_pitches") == 1 for item in reports),
               "sample_candidates": sample_candidates, "seed": seed,
               "structure_source": "predicted (holdout evaluation)" if structure_model else
               "gold structure and group counts (training-set diagnostic only)",
               "output": str(output)}
    (output / "report.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in reports), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "abc-two-stage/stage1/hf/train.json")
    parser.add_argument("--inputs", type=Path, default=ROOT / "abc-chorus-pairs-vocal")
    parser.add_argument("--model", default=str(ROOT / "abc-two-stage/stage2-chunks-overfit-20/checkpoints/best"))
    parser.add_argument("--structure-model", help="Predict structure with Stage 1 instead of using gold targets")
    parser.add_argument("--output", type=Path, default=ROOT / "abc-two-stage/stage2-chunks-overfit-20/evaluation")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-groups", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-candidates", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.source, args.inputs, args.model, args.output, args.limit,
                              args.max_groups, args.max_tokens, args.device, args.structure_model,
                              args.sample_candidates, args.temperature, args.top_p, args.seed),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
