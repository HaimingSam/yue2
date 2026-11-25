#!/usr/bin/env python3
"""Generate a full Vocal ABC in two stages: structure, then individual sections."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

from abc_chorus_copy import first_section_body, section_ranges
from abc_chunk_groups import chunk_prompt, parse_chunk, validate_partial_vocal
from abc_vocal_format import load_validator, restore_ins, to_vocal


ROOT = Path(__file__).resolve().parents[1]


def load_model(path: str, device: str):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(path).to(device).eval()
    return tokenizer, model


def generate(model, tokenizer, text: str, device: str, max_new_tokens: int,
             *, temperature: float | None = None, top_p: float = 0.9) -> str:
    import torch
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.inference_mode():
        settings = ({"do_sample": True, "temperature": temperature, "top_p": top_p}
                    if temperature is not None else {"num_beams": 1})
        tokens = model.generate(**inputs, max_new_tokens=max_new_tokens, **settings)
    return tokenizer.decode(tokens[0], skip_special_tokens=True)


def group_pitch_score(group: list[str], validator) -> tuple[int, float]:
    pitches = []
    for line in group:
        if line.startswith(("V:", "M:", "K:")):
            continue
        for token in validator.TOKEN.finditer(line):
            note = token.group("note")
            if note is not None and note != "z":
                pitches.append((token.group("acc") or "") + note + (token.group("oct") or ""))
    frequencies = Counter(pitches)
    return len(frequencies), -max(frequencies.values()) / len(pitches) if pitches else -1.0


def parse_structure(text: str) -> list[str]:
    return [label for label, _ in parse_counted_structure(text)]


def parse_counted_structure(text: str) -> list[tuple[str, int | None]]:
    match = re.search(r"<STRUCTURE>\s*(.*?)\s*</STRUCTURE>", text, re.DOTALL)
    body = match.group(1) if match else text
    entries = []
    for line in body.splitlines():
        if not line.startswith("% "):
            continue
        entry = re.fullmatch(r"% ([a-z][a-z-]*)(?: groups=([1-9][0-9]*))?", line.strip().lower())
        if not entry:
            raise ValueError(f"Invalid structure line: {line!r}")
        entries.append((entry.group(1), int(entry.group(2)) if entry.group(2) else None))
    if not entries:
        raise ValueError("Stage 1 did not generate section labels")
    if any(count is not None for _, count in entries) and any(count is None for _, count in entries):
        raise ValueError("Structure must specify group counts for every section or none")
    return entries


def extract_generated_section(text: str, label: str) -> list[str]:
    lines = text.splitlines()
    matches = [(start, end) for _, found, start, end in section_ranges(lines) if found == label]
    if not matches:
        raise ValueError(f"Stage 2 did not generate section {label!r}")
    start, end = matches[0]
    return lines[start + 1:end]


def run(input_path: Path, output_path: Path, structure_model_path: str, section_model_path: str,
        device: str, max_structure_tokens: int, max_section_tokens: int,
        chunked: bool = False, max_groups_per_section: int = 32,
        structure_override: str | None = None, sample_candidates: int = 1,
        temperature: float = 0.8, top_p: float = 0.9) -> dict:
    if sample_candidates < 1 or temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("sample_candidates and temperature must be positive; top_p must be in (0, 1]")
    if sample_candidates > 1 and not chunked:
        raise ValueError("Sampling candidates currently requires --chunked")
    source = input_path.read_text(encoding="utf-8")
    if 'V: Ins clef=treble' in source:
        source = to_vocal(source)
    if structure_override is None:
        structure_tokenizer, structure_model = load_model(structure_model_path, device)
    section_tokenizer, section_model = load_model(section_model_path, device)
    started = time.monotonic()
    structure_raw = (structure_override if structure_override is not None else
                     generate(structure_model, structure_tokenizer, source, device, max_structure_tokens))
    structure_plan = parse_counted_structure(structure_raw)
    labels = [label for label, _ in structure_plan]
    chorus_label, chorus_body = first_section_body(source)
    sections = []
    generated_sections = []
    occurrences = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    debug_dir = output_path.parent / f"{output_path.stem}.debug"
    debug_dir.mkdir(exist_ok=True)
    validator = load_validator() if sample_candidates > 1 else None
    (debug_dir / "structure.txt").write_text(structure_raw, encoding="utf-8")
    structure_text = "<STRUCTURE>\n" + "\n".join(f"% {label}" for label in labels) + "\n</STRUCTURE>\n"
    for position, (label, group_count) in enumerate(structure_plan, 1):
        occurrences[label] = occurrences.get(label, 0) + 1
        if label == chorus_label:
            sections.extend([f"% {label}", *chorus_body])
            if chunked:
                validate_partial_vocal("\n".join(source.splitlines()[:7] + sections) + "\n")
            generated_sections.append({"section": label, "source": "input"})
            continue
        if chunked:
            previous_group = None
            section_groups = []
            if group_count is not None and group_count > max_groups_per_section:
                raise ValueError(f"Section {position} ({label}) requests {group_count} groups; "
                                 f"maximum is {max_groups_per_section}")
            for group_index in range(1, (group_count or max_groups_per_section) + 1):
                prompt = chunk_prompt(source, structure_text, label, position,
                                      occurrences[label], group_index, previous_group)
                candidate_count = sample_candidates
                valid = []
                errors = []
                for candidate_index in range(1, candidate_count + 1):
                    raw_group = (generate(section_model, section_tokenizer, prompt, device, max_section_tokens,
                                          temperature=temperature, top_p=top_p)
                                 if sample_candidates > 1 else
                                 generate(section_model, section_tokenizer, prompt, device, max_section_tokens))
                    raw_path = debug_dir / f"{position:02d}-{label}-{group_index:02d}"
                    if sample_candidates > 1:
                        raw_path = debug_dir / f"{position:02d}-{label}-{group_index:02d}-candidate-{candidate_index:02d}"
                    raw_path = raw_path.with_suffix(".txt")
                    raw_path.write_text(raw_group, encoding="utf-8")
                    try:
                        group, is_last = parse_chunk(raw_group)
                        candidate = sections + [f"% {label}"]
                        for part in section_groups + [group]:
                            candidate.extend(part)
                        validate_partial_vocal("\n".join(source.splitlines()[:7] + candidate) + "\n",
                                               validator)
                        score = group_pitch_score(group, validator) if validator else (0, 0.0)
                        valid.append((score, group, is_last, raw_path))
                    except ValueError as exc:
                        errors.append(f"{raw_path}: {exc}")
                if not valid:
                    raise ValueError(f"Section {position} ({label}) group {group_index} failed: "
                                     + "; ".join(errors))
                _, group, is_last, selected_path = max(valid, key=lambda item: item[0])
                if sample_candidates > 1:
                    (debug_dir / f"{position:02d}-{label}-{group_index:02d}-selected.txt").write_text(
                        str(selected_path) + "\n", encoding="utf-8")
                section_groups.append(group)
                previous_group = group
                if group_count is None and is_last:
                    break
            else:
                if group_count is None:
                    raise ValueError(f"Section {position} ({label}) exceeded "
                                     f"{max_groups_per_section} groups; inspect {debug_dir}")
            sections.append(f"% {label}")
            for group in section_groups:
                sections.extend(group)
            generated_sections.append({"section": label, "source": "model",
                                       "groups": len(section_groups)})
            continue
        prompt = f"<SECTION>{label}</SECTION>\n<CHORUS>\n{source}</CHORUS>\n{structure_text}"
        raw_section = generate(section_model, section_tokenizer, prompt, device, max_section_tokens)
        raw_path = debug_dir / f"{position:02d}-{label}.txt"
        raw_path.write_text(raw_section, encoding="utf-8")
        try:
            body = extract_generated_section(raw_section, label)
        except ValueError as exc:
            raise ValueError(f"Section {position} ({label}) failed: {exc}; raw output: {raw_path}") from exc
        sections.extend([f"% {label}", *body])
        generated_sections.append({"section": label, "source": "model"})
    lines = source.splitlines()
    vocal = "\n".join(lines[:7] + sections) + "\n"
    vocal_path = debug_dir / "assembled.vocal.abc"
    vocal_path.write_text(vocal, encoding="utf-8")
    try:
        native = restore_ins(vocal)
    except ValueError as exc:
        raise ValueError(f"Generated Vocal ABC is invalid: {exc}; inspect {vocal_path} "
                         f"and generated sections in {debug_dir}") from exc
    output_path.write_text(native, encoding="utf-8")
    report = {"input": str(input_path), "output": str(output_path),
              "structure": labels, "generated_sections": generated_sections,
              "planned_groups": [count for _, count in structure_plan],
              "seconds": round(time.monotonic() - started, 2),
              "structure_source": "provided" if structure_override is not None else "model",
              "sample_candidates": sample_candidates,
              "temperature": temperature if sample_candidates > 1 else None,
              "top_p": top_p if sample_candidates > 1 else None,
              "vocal_only": vocal}
    output_path.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--structure-model", help="Stage 1 checkpoint (required unless --structure-file is set)")
    parser.add_argument("--structure-file", type=Path,
                        help="Use a known structure plan to isolate Stage 2 for diagnostics")
    parser.add_argument("--section-model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-structure-tokens", type=int, default=256)
    parser.add_argument("--max-section-tokens", type=int, default=1024)
    parser.add_argument("--chunked", action="store_true", help="Use the new 1–4 measure group model for Stage 2")
    parser.add_argument("--max-groups-per-section", type=int, default=32)
    parser.add_argument("--sample-candidates", type=int, default=1,
                        help="Draw this many Stage-2 candidates per group and keep the most pitch-diverse valid one")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, help="Set the sampling seed")
    args = parser.parse_args()
    if args.max_groups_per_section < 1 or args.max_section_tokens < 1:
        parser.error("--max-groups-per-section and --max-section-tokens must be positive")
    if args.structure_file is None and args.structure_model is None:
        parser.error("--structure-model or --structure-file is required")
    if args.seed is not None:
        import torch
        torch.manual_seed(args.seed)
    print(json.dumps(run(args.input, args.output, args.structure_model, args.section_model,
                         args.device, args.max_structure_tokens, args.max_section_tokens,
                         args.chunked, args.max_groups_per_section,
                         args.structure_file.read_text(encoding="utf-8") if args.structure_file else None,
                         args.sample_candidates, args.temperature, args.top_p),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
