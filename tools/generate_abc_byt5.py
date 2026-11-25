#!/usr/bin/env python3
"""Generate a full ABC score from a section with a fine-tuned ByT5 model."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

from abc_vocal_format import restore_ins, to_vocal
from abc_chorus_copy import restore_input_section


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "abc-chorus-pairs/checkpoints/byt5-small/best"
ABC_TOOLS = ROOT / "skills/yue2-music/scripts/abc_tools.py"


def validate_abc(text: str) -> dict:
    spec = importlib.util.spec_from_file_location("generation_abc_tools", ABC_TOOLS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load ABC validator: {ABC_TOOLS}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        score = module.parse_abc(text)
    except ValueError as exc:
        return {"valid": False, "error": str(exc)}
    return {"valid": True, "vocal_notes": len(score.voices["Vocal"].notes),
            "instrumental_notes": len(score.voices["Ins"].notes),
            "duration_seconds": round(float(score.voices["Vocal"].time * 60 / score.bpm), 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Section ABC file")
    parser.add_argument("--output", type=Path, required=True, help="Generated ABC file")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vocal-only", action="store_true", help="Use Vocal-only input; restore silent Ins in output")
    parser.add_argument("--copy-input-section", action="store_true",
                        help="Replace the generated section placeholder with the input section exactly")
    parser.add_argument("--allow-section-fallback", action="store_true",
                        help="If placeholder is absent, replace the first matching section instead")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if not args.input.is_file():
        parser.error(f"Input does not exist: {args.input}")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(args.device).eval()
    source = args.input.read_text(encoding="utf-8")
    if args.vocal_only and 'V: Ins clef=treble' in source:
        source = to_vocal(source)
    inputs = tokenizer(source, return_tensors="pt").to(args.device)
    started = time.monotonic()
    with torch.inference_mode():
        tokens = model.generate(**inputs, max_new_tokens=args.max_new_tokens, num_beams=1)
    generated = tokenizer.decode(tokens[0], skip_special_tokens=True)
    restoration_error = None
    section_restore_mode = None
    if args.copy_input_section:
        try:
            generated, section_restore_mode = restore_input_section(
                generated, source, strict=not args.allow_section_fallback)
        except ValueError as exc:
            restoration_error = str(exc)
    if args.vocal_only:
        if restoration_error is None:
            try:
                generated = restore_ins(generated)
            except ValueError as exc:
                restoration_error = str(exc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated, encoding="utf-8")
    result = {"input": str(args.input), "model": args.model, "output": str(args.output),
              "tokens": len(tokens[0]), "seconds": round(time.monotonic() - started, 2),
              "reached_token_limit": len(tokens[0]) >= args.max_new_tokens,
              "section_restore_mode": section_restore_mode,
              "abc_check": validate_abc(generated) if restoration_error is None
              else {"valid": False, "error": f"Could not restore Ins: {restoration_error}"}}
    args.output.with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
