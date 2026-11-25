#!/usr/bin/env python3
"""Generate memorized training songs and measure real Vocal output, not teacher-forced loss."""

from __future__ import annotations

import argparse
import json
import time
from difflib import SequenceMatcher
from pathlib import Path

from abc_vocal_format import load_validator, restore_ins


ROOT = Path(__file__).resolve().parents[1]


def note_match(source_notes: list, generated_notes: list) -> float:
    source_pitches = [note[1] for note in source_notes]
    generated_pitches = [note[1] for note in generated_notes]
    if not source_pitches:
        return 0.0
    matches = SequenceMatcher(None, source_pitches, generated_pitches, autojunk=False).get_matching_blocks()
    return round(max(block.size for block in matches) / len(source_pitches), 3)


def raw_pitch_tokens(text: str, validator) -> int:
    return sum(match.group("note") not in (None, "z")
               for line in text.splitlines() if "|" in line and not line.startswith(("% ", "V: "))
               for match in validator.TOKEN.finditer(line))


def evaluate(data: Path, output: Path, model_path: str, limit: int, max_new_tokens: int, device: str) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    rows = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines()]
    if not rows or limit < 1:
        raise ValueError("Expected nonempty training data and positive --limit")
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    validator = load_validator()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device).eval()
    output.mkdir(parents=True)
    reports = []
    for row in rows[:limit]:
        started = time.monotonic()
        inputs = tokenizer(row["text"], return_tensors="pt").to(device)
        with torch.inference_mode():
            tokens = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=1)
        vocal = tokenizer.decode(tokens[0], skip_special_tokens=True)
        (output / f'{row["id"]}.vocal.abc').write_text(vocal, encoding="utf-8")
        expected = validator.parse_abc(restore_ins(row["summary"], validator))
        source = validator.parse_abc(restore_ins(row["text"], validator))
        report = {"id": row["id"], "generated_tokens": len(tokens[0]),
                  "reached_token_limit": len(tokens[0]) >= max_new_tokens,
                  "raw_pitch_tokens": raw_pitch_tokens(vocal, validator),
                  "seconds": round(time.monotonic() - started, 2),
                  "exact_target": vocal.strip() == row["summary"].strip(),
                  "target_vocal_notes": len(expected.voices["Vocal"].notes)}
        try:
            native = restore_ins(vocal, validator)
            generated = validator.parse_abc(native)
            (output / f'{row["id"]}.abc').write_text(native, encoding="utf-8")
            notes = generated.voices["Vocal"].notes
            report.update(valid=True, generated_vocal_notes=len(notes),
                          input_pitch_match=note_match(source.voices["Vocal"].notes, notes),
                          exact_vocal_notes=notes == expected.voices["Vocal"].notes,
                          generated_sections=[line for line in native.splitlines() if line.startswith("% ")])
        except ValueError as exc:
            report.update(valid=False, generated_vocal_notes=0, error=str(exc))
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    (output / "report.jsonl").write_text("".join(json.dumps(report, ensure_ascii=False) + "\n" for report in reports), encoding="utf-8")
    summary = {"evaluated_training_songs": len(reports), "valid_native_abc": sum(row["valid"] for row in reports),
               "valid_with_vocal_notes": sum(row["valid"] and row["generated_vocal_notes"] > 0 for row in reports),
               "with_any_raw_pitch_tokens": sum(row["raw_pitch_tokens"] > 0 for row in reports),
               "exact_targets": sum(row["exact_target"] for row in reports),
               "output": str(output)}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "abc-overfit-20/data/train.json")
    parser.add_argument("--model", default=str(ROOT / "abc-overfit-20/checkpoints/best"))
    parser.add_argument("--output", type=Path, default=ROOT / "abc-overfit-20/evaluation")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=3072)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.data, args.output, args.model, args.limit,
                              args.max_new_tokens, args.device), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
