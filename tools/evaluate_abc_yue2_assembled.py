#!/usr/bin/env python3
"""Generate new Vocal verses with YuE2; assemble an exact original chorus without training."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import replace
from itertools import combinations
from pathlib import Path

from abc_chorus_copy import first_section_body, section_ranges
from abc_vocal_format import load_validator, restore_ins, to_vocal
from evaluate_abc_yue2_continuation import open_prefix


ROOT = Path(__file__).resolve().parents[1]


def section_notes(score, label: str) -> list[list[tuple]]:
    lines = score.text.splitlines()
    bars = score.voices["Vocal"].bars
    group_times = {}
    bar_index = 0
    for line_index, voice in score.music_lines.items():
        if voice != "Vocal":
            continue
        bar_count = 0
        for bar in lines[line_index][:-1].split("|"):
            rest = re.fullmatch(r"Z([2-4])?", bar)
            bar_count += int(rest.group(1) or "1") if rest else 1
        first = bars[bar_index][0]
        last = bars[bar_index + bar_count - 1]
        group_times[line_index] = (first, last[0] + last[1])
        bar_index += bar_count
    results = []
    for _, section, start, end in section_ranges(lines):
        if section != label:
            continue
        intervals = [times for line_index, times in group_times.items() if start < line_index < end]
        section_start, section_end = intervals[0][0], intervals[-1][1]
        results.append([(note[0] - section_start, note[1], note[2])
                        for note in score.voices["Vocal"].notes
                        if section_start <= note[0] < section_end])
    return results


def verse_candidates(original: str, completion: str, validator, min_notes: int = 16,
                     truncated: bool = False) -> list[dict]:
    lines = completion.splitlines()
    positions = list(section_ranges(lines))
    original_notes = len(validator.parse_abc(original).voices["Vocal"].notes)
    candidates = []
    for _, label, start, end in positions:
        if label != "verse":
            continue
        if truncated and end == len(lines):
            continue
        section = "\n".join(lines[start:end]) + "\n"
        try:
            score = validator.parse_abc(original + section)
            vocal = to_vocal(original + section, validator)
        except ValueError:
            continue
        pitches = [note[1] for note in score.voices["Vocal"].notes[original_notes:]]
        counts = Counter(pitches)
        if len(pitches) < min_notes or len(counts) < 4 or max(counts.values()) / len(pitches) > 0.7:
            continue
        verse_lines = vocal.splitlines()
        verse_start = list(section_ranges(verse_lines))[-1][2]
        body = verse_lines[verse_start + 1:]
        if all(not line.startswith("V: Vocal") for line in body):
            continue
        candidates.append({"body": body, "notes": len(pitches), "pitches": len(counts),
                           "top_pitch_share": round(max(counts.values()) / len(pitches), 3)})
    return candidates


def assemble(vocal_input: str, first: dict, second: dict, validator) -> str:
    label, chorus = first_section_body(vocal_input)
    if label != "chorus" or first["body"] == second["body"]:
        raise ValueError("Expected a chorus input and two different verse candidates")
    sections = (["% verse", *first["body"], "% chorus", *chorus,
                 "% verse", *second["body"], "% chorus", *chorus])
    vocal = "\n".join(vocal_input.splitlines()[:7] + sections) + "\n"
    native = restore_ins(vocal, validator)
    score = validator.parse_abc(native)
    if score.voices["Ins"].notes:
        raise ValueError("Instrumental track must be empty")
    result = to_vocal(native, validator).splitlines()
    choruses = [result[start + 1:end] for _, section, start, end in section_ranges(result)
                if section == label]
    if len(choruses) != 2 or any(body != chorus for body in choruses):
        raise ValueError("Input chorus was not preserved exactly")
    original_score = validator.parse_abc(restore_ins(vocal_input, validator))
    expected = section_notes(original_score, label)[0]
    if any(notes != expected for notes in section_notes(score, label)):
        raise ValueError("Reordering changed the sounding chorus melody")
    return native


def evaluate(source: Path, inputs: Path, output: Path, limit: int, attempts: int, max_tokens: int,
             model: str, vae: str, device: str, seed: int, style: str, lyrics: str) -> dict:
    from yue2 import YuE2Pipeline
    from yue2.protocol import GenerationConfig, SongRequest

    rows = [row for line in source.read_text(encoding="utf-8").splitlines()
            if (row := json.loads(line))["section"] == "chorus"]
    if not 1 <= limit <= len(rows) or attempts < 1 or max_tokens < 1:
        raise ValueError("Require positive limits, attempts, and max_tokens")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    validator = load_validator()
    reports = []
    with YuE2Pipeline.from_pretrained(model, vae=vae, device=device, local_files_only=True) as pipe:
        sampling = replace(GenerationConfig().abc, max_tokens=max_tokens)
        for index, row in enumerate(rows[:limit]):
            song_id = row["id"]
            source_text = (inputs / song_id / "input.abc").read_text(encoding="utf-8")
            if first_section_body(source_text)[0] != "chorus":
                raise ValueError(f"{song_id}: expected chorus input")
            request = SongRequest(style=style, lyrics=lyrics, cot="full", seed=seed, id=song_id)
            prefix, native_input = open_prefix(request, pipe.tokenizer, source_text)
            candidates = []
            report = {"id": song_id, "success": False, "attempts": [], "model": model}
            for attempt in range(attempts):
                attempt_seed = seed + index * attempts + attempt
                tokens, timing, truncated = pipe._generate(prefix, sampling, attempt_seed, "abc")
                completion = pipe.tokenizer.decode(tokens)
                (output / f"{song_id}.attempt-{attempt + 1:02d}.txt").write_text(completion, encoding="utf-8")
                found = verse_candidates(native_input, completion, validator, truncated=truncated)
                report["attempts"].append({"seed": attempt_seed, "truncated": truncated,
                                           "generated_tokens": len(tokens), "seconds": timing.get("seconds"),
                                           "usable_verses": len(found)})
                for candidate in found:
                    if all(candidate["body"] != previous["body"] for previous in candidates):
                        candidates.append(candidate)
                for first, second in combinations(candidates, 2):
                    try:
                        native = assemble(source_text, first, second, validator)
                    except ValueError as exc:
                        report["assembly_error"] = str(exc)
                    else:
                        (output / f"{song_id}.abc").write_text(native, encoding="utf-8")
                        report.update(success=True, verse_metrics=[{key: item[key] for key in
                                    ("notes", "pitches", "top_pitch_share")} for item in (first, second)])
                        break
                if report["success"]:
                    break
            if not report["success"]:
                report["reason"] = report.get("assembly_error", "Fewer than two distinct usable verses")
            reports.append(report)
            print(json.dumps(report, ensure_ascii=False), flush=True)
    summary = {"songs": len(reports), "success": sum(item["success"] for item in reports),
               "attempts": attempts, "output": str(output), "seed": seed,
               "max_tokens": max_tokens, "model": model}
    (output / "report.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                                for row in reports), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "abc-two-stage/stage1-counted/hf/test.json")
    parser.add_argument("--inputs", type=Path, default=ROOT / "abc-chorus-pairs-vocal")
    parser.add_argument("--output", type=Path, default=ROOT / "abc-two-stage/evaluation/yue2-assembled-20")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--model", default="m-a-p/YuE2-3B")
    parser.add_argument("--vae", default="m-a-p/YuE2-Vae")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--style", default="Mandarin pop, expressive female lead, melodic verse and chorus")
    parser.add_argument("--lyrics", default="[Verse]\nla la la la\n[Chorus]\nla la la la")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.source, args.inputs, args.output, args.limit, args.attempts,
                              args.max_tokens, args.model, args.vae, args.device,
                              args.seed, args.style, args.lyrics), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
