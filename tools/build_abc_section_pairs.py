#!/usr/bin/env python3
"""Build native ABC section-to-full-song pairs from SheetSage2 transcriptions."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path


DEFAULT_INPUT = Path("/data2/cmq/abc-yue2")
DEFAULT_OUTPUT = Path("/data/cmq/YuE/abc-chorus-pairs")
ABC_TOOLS = Path(__file__).resolve().parents[1] / "skills/yue2-music/scripts/abc_tools.py"


def load_abc_tools():
    spec = importlib.util.spec_from_file_location("section_pair_abc_tools", ABC_TOOLS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load ABC validator: {ABC_TOOLS}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sections(lines: list[str]):
    markers = [(index, line[2:].strip().lower()) for index, line in enumerate(lines)
               if line.startswith("% ")]
    for position, (start, label) in enumerate(markers):
        end = markers[position + 1][0] if position + 1 < len(markers) else len(lines)
        yield start, end, label


def bar_count(line: str) -> int:
    return sum(int(match.group(1) or "1") if (match := re.fullmatch(r"Z([2-4])?", bar)) else 1
               for bar in line[:-1].split("|"))


def score_sections(lines: list[str], score):
    vocal_bar_cursor = 0
    for start, end, label in sections(lines):
        count = sum(bar_count(lines[index]) for index in range(start, end)
                    if score.music_lines.get(index) == "Vocal")
        if count:
            yield start, end, label, vocal_bar_cursor, vocal_bar_cursor + count
        vocal_bar_cursor += count
    if vocal_bar_cursor != len(score.voices["Vocal"].bars):
        raise ValueError("Section labels do not cover all measures")


def active_key(score, offset: Fraction) -> str:
    return next(value for time, value in reversed(score.voices["Vocal"].keys) if time <= offset)


def note_events(voice, start: Fraction, end: Fraction) -> list[tuple]:
    return [(max(onset, start) - start, pitch, min(onset + duration, end) - max(onset, start))
            for onset, pitch, duration in voice.notes if onset < end and onset + duration > start]


def extract_section(lines: list[str], score, entry, abc_tools):
    start_line, end_line, label, start_bar, end_bar = entry
    vocal = score.voices["Vocal"]
    start_time = vocal.bars[start_bar][0]
    end_start, end_length, _ = vocal.bars[end_bar - 1]
    end_time = end_start + end_length
    numerator, denominator = vocal.bars[start_bar][2]
    header = lines[:8].copy()
    header[2] = f"M:{numerator}/{denominator}"
    header[7] = f"K:{active_key(score, start_time)}"
    body = lines[start_line:end_line].copy()
    for name in ("Vocal", "Ins"):
        if not any(onset < end_time < onset + duration for onset, _, duration in score.voices[name].notes):
            continue
        music_indices = [index for index, line in enumerate(body)
                         if score.music_lines.get(start_line + index) == name]
        if not music_indices or not body[music_indices[-1]].endswith("-|"):
            raise ValueError(f"Cannot clip {name} tie at section end")
        last_index = music_indices[-1]
        body[last_index] = body[last_index][:-2] + "|"
    text = "\n".join(header + body) + "\n"
    fragment = abc_tools.parse_abc(text)
    if not fragment.voices["Vocal"].notes:
        raise ValueError("Selected section has no Vocal notes")
    if fragment.voices["Vocal"].time != end_time - start_time:
        raise ValueError("Section duration differs from full score")
    for name in ("Vocal", "Ins"):
        expected = note_events(score.voices[name], start_time, end_time)
        actual = [tuple(note) for note in fragment.voices[name].notes]
        if actual != expected:
            raise ValueError(f"{name} notes differ from full score at section boundary")
    return text, start_time, end_time, len(fragment.voices["Vocal"].notes)


def make_pair(score_path: Path, abc_tools) -> tuple[str, dict]:
    original = score_path.read_text(encoding="utf-8")
    score = abc_tools.parse_abc(original)
    lines = original.splitlines()
    entries = list(score_sections(lines, score))
    chosen_label = "chorus" if any(item[2] == "chorus" for item in entries) else "verse"
    candidates = [item for item in entries if item[2] == chosen_label]
    if not candidates:
        raise ValueError("No chorus or verse section")
    errors = []
    for occurrence, entry in enumerate(candidates, 1):
        try:
            snippet, start, end, note_count = extract_section(lines, score, entry, abc_tools)
        except ValueError as exc:
            errors.append(f"{chosen_label} #{occurrence}: {exc}")
            continue
        return snippet, {
            "id": score_path.stem,
            "input": f"{score_path.stem}/input.abc",
            "target": f"{score_path.stem}/target.abc",
            "source": str(score_path),
            "section": chosen_label,
            "section_occurrence": occurrence,
            "start_quarters": str(start),
            "end_quarters": str(end),
            "source_start_seconds": round(float(start * 60 / score.bpm), 3),
            "source_end_seconds": round(float(end * 60 / score.bpm), 3),
            "input_vocal_notes": note_count,
        }
    raise ValueError("No valid section: " + "; ".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, help="Process the first N scores, for testing")
    parser.add_argument("--exclude-manifest", type=Path,
                        help="JSONL manifest; exclude rows whose actual audio is instrumental.wav")
    parser.add_argument("--dry-run", action="store_true", help="Validate and count without writing")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    excluded_ids = set()
    if args.exclude_manifest:
        for line in args.exclude_manifest.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            actual_audio = row.get("actual_audio_input", "")
            if Path(actual_audio).name == "instrumental.wav":
                excluded_ids.add(Path(row["output"]).stem)
    scores = [path for path in sorted(args.input_root.glob("*.abc"))
              if path.stem not in excluded_ids]
    if args.limit is not None:
        scores = scores[:args.limit]
    if not scores:
        parser.error(f"No .abc files found in {args.input_root}")
    if not args.dry_run and args.output_root.exists() and any(args.output_root.iterdir()):
        parser.error(f"Output directory must be empty: {args.output_root}")
    abc_tools = load_abc_tools()
    records = []
    failures = []
    counts = Counter()
    for index, score_path in enumerate(scores, 1):
        try:
            snippet, record = make_pair(score_path, abc_tools)
            counts[record["section"]] += 1
            if not args.dry_run:
                pair_dir = args.output_root / score_path.stem
                pair_dir.mkdir(parents=True)
                (pair_dir / "input.abc").write_text(snippet, encoding="utf-8")
                (pair_dir / "target.abc").write_bytes(score_path.read_bytes())
            records.append(record)
        except (ValueError, OSError) as exc:
            failures.append({"source": str(score_path), "error": str(exc)})
        if index % 100 == 0:
            print(f"Processed {index}/{len(scores)}; pairs={len(records)} failures={len(failures)}", flush=True)
    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "pairs.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
        (args.output_root / "failures.jsonl").write_text(
            "".join(json.dumps(error, ensure_ascii=False) + "\n" for error in failures), encoding="utf-8")
    print(json.dumps({"scores": len(scores), "excluded": len(excluded_ids),
                      "pairs": len(records), "chorus": counts["chorus"],
                      "verse_fallback": counts["verse"], "failed": len(failures),
                      "output_root": str(args.output_root), "dry_run": args.dry_run}, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
