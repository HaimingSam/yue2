#!/usr/bin/env python3
"""Evaluate exact-section conditioning on the same songs used for training."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from abc_chorus_copy import PLACEHOLDER, restore_input_section
from abc_vocal_format import load_validator, restore_ins, to_vocal


ROOT = Path(__file__).resolve().parents[1]


def section_body(text: str, label: str, occurrence: int = 1) -> list[str]:
    lines = text.splitlines()
    markers = [(index, line[2:].strip().lower()) for index, line in enumerate(lines)
               if line.startswith("% ") and line[2:].strip().lower() == label.lower()]
    if len(markers) < occurrence:
        raise ValueError(f"Missing {label!r} section occurrence {occurrence}")
    start = markers[occurrence - 1][0]
    end = next((index for index in range(start + 1, len(lines)) if lines[index].startswith("% ")), len(lines))
    return lines[start + 1:end]


def evaluate(data: Path, model_path: str, output: Path, device: str, max_new_tokens: int) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    rows = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines()]
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    validator = load_validator()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device).eval()
    output.mkdir(parents=True)
    reports = []
    for row in rows:
        started = time.monotonic()
        source = row["text"]
        inputs = tokenizer(source, return_tensors="pt").to(device)
        with torch.inference_mode():
            tokens = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=1)
        raw = tokenizer.decode(tokens[0], skip_special_tokens=True)
        (output / f'{row["id"]}.raw.txt').write_text(raw, encoding="utf-8")
        report = {"id": row["id"], "tokens": len(tokens[0]),
                  "reached_token_limit": len(tokens[0]) >= max_new_tokens,
                  "seconds": round(time.monotonic() - started, 2),
                  "has_control": raw.startswith("<COPY_INPUT_SECTION>"),
                  "has_placeholder": PLACEHOLDER in raw}
        try:
            reconstructed, mode = restore_input_section(raw, source, strict=True)
            native = restore_ins(reconstructed, validator)
            score = validator.parse_abc(native)
            vocal = to_vocal(native, validator)
            occurrence = row.get("section_occurrence", 1)
            report.update(valid=True, restore_mode=mode,
                          vocal_notes=len(score.voices["Vocal"].notes),
                          ins_notes=len(score.voices["Ins"].notes),
                          input_section_exact=(section_body(vocal, row["section"], occurrence)
                                               == section_body(source, row["section"], 1)))
            (output / f'{row["id"]}.abc').write_text(native, encoding="utf-8")
        except ValueError as exc:
            report.update(valid=False, error=str(exc), vocal_notes=0, ins_notes=0,
                          input_section_exact=False)
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    (output / "report.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in reports), encoding="utf-8")
    summary = {"songs": len(reports),
               "control_ok": sum(row["has_control"] for row in reports),
               "placeholder_ok": sum(row["has_placeholder"] for row in reports),
               "valid_abc": sum(row["valid"] for row in reports),
               "with_vocal_notes": sum(row["vocal_notes"] > 0 for row in reports),
               "exact_input_section": sum(row["input_section_exact"] for row in reports),
               "output": str(output)}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "abc-chorus-copy-overfit-20/data/train.json")
    parser.add_argument("--model", default=str(ROOT / "abc-chorus-copy-overfit-20/checkpoints/best"))
    parser.add_argument("--output", type=Path, default=ROOT / "abc-chorus-copy-overfit-20/evaluation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=3072)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.data, args.model, args.output, args.device,
                              args.max_new_tokens), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
