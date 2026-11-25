#!/usr/bin/env python3
"""Build an ABAB symbolic mashup from two SheetSage2/YuE2 native ABC scores."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
A_PATH = ROOT / "transcription_4b2f_pianai" / "score.abc"
B_PATH = ROOT / "transcription_trace" / "score.abc"
FULL_OUT_DIR = ROOT / "mashup_abab"
SHORT_OUT_DIR = ROOT / "mashup_abab_short"
TARGET_BPM = 76
TARGET_DENOMINATOR = 32

NOTE = re.compile(
    r"(?P<acc>\^\^|__|\^|_|=)?(?P<note>[A-Ga-gz])"
    r"(?P<oct>[,']*)(?P<duration>[0-9]*)(?P<tie>-?)"
)


@dataclass
class Section:
    label: str
    occurrence: int
    lines: list[str]


@dataclass
class SourceScore:
    name: str
    path: Path
    denominator: int
    bpm: int
    key: str
    sections: list[Section]


def parse_header(lines: list[str]) -> tuple[int, int, str]:
    denominator = int(lines[3].split(":", 1)[1].split("/", 1)[1])
    bpm = int(lines[4].split("=", 1)[1])
    key = lines[7].split(":", 1)[1]
    return denominator, bpm, key


def split_sections(path: Path, name: str) -> SourceScore:
    lines = path.read_text(encoding="utf-8").splitlines()
    denominator, bpm, key = parse_header(lines)
    sections: list[Section] = []
    counts: dict[str, int] = {}
    label = "section"
    body: list[str] = []

    def flush() -> None:
        nonlocal body
        if not body:
            return
        occurrence = counts.get(label, 0)
        counts[label] = occurrence + 1
        sections.append(Section(label=label, occurrence=occurrence, lines=body))
        body = []

    for line in lines[8:]:
        if line.startswith("% "):
            flush()
            label = line[2:].strip() or "section"
        else:
            body.append(line)
    flush()
    return SourceScore(name=name, path=path, denominator=denominator, bpm=bpm, key=key, sections=sections)


def scale_music_line(line: str, factor: int) -> str:
    if factor == 1:
        return line
    output: list[str] = []
    index = 0
    while index < len(line):
        char = line[index]
        if char == '"':
            end = line.find('"', index + 1)
            if end == -1:
                raise ValueError(f"Unclosed chord/string in {line!r}")
            output.append(line[index : end + 1])
            index = end + 1
            continue
        if line.startswith("[K:", index):
            end = line.find("]", index + 1)
            if end == -1:
                raise ValueError(f"Unclosed key token in {line!r}")
            output.append(line[index : end + 1])
            index = end + 1
            continue
        match = NOTE.match(line, index)
        if match:
            duration = int(match.group("duration") or "1") * factor
            output.append(
                f"{match.group('acc') or ''}{match.group('note')}{match.group('oct')}"
                f"{'' if duration == 1 else duration}{match.group('tie')}"
            )
            index = match.end()
            continue
        output.append(char)
        index += 1
    return "".join(output)


def normalize_lines(section: Section, source: SourceScore) -> list[str]:
    if TARGET_DENOMINATOR % source.denominator:
        raise ValueError(f"Cannot integer-scale L:1/{source.denominator} to L:1/{TARGET_DENOMINATOR}")
    factor = TARGET_DENOMINATOR // source.denominator
    result: list[str] = []
    for line in section.lines:
        if line.endswith("|") and not line.startswith(("V:", "M:", "K:")):
            result.append(scale_music_line(line, factor))
        else:
            result.append(line)
    return result


def find_section(source: SourceScore, label: str, occurrence: int) -> Section:
    for section in source.sections:
        if section.label == label and section.occurrence == occurrence:
            return section
    available = ", ".join(f"{s.label}#{s.occurrence + 1}" for s in source.sections)
    raise KeyError(f"{source.name} has no {label} occurrence {occurrence + 1}; available: {available}")


def write_score(out_dir: Path, sources: dict[str, SourceScore], recipe: list[tuple[str, str, int]],
                recipe_description: str) -> dict:
    a = sources["A"]
    lines = [
        "X:1",
        "T:",
        "M:4/4",
        f"L:1/{TARGET_DENOMINATOR}",
        f"Q:1/4={TARGET_BPM}",
        'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"',
        'V: Ins clef=treble name="Ins Melody" snm="Inst."',
        f"K:{a.key}",
    ]
    manifest_sections = []
    for source_key, label, occurrence in recipe:
        source = sources[source_key]
        section = find_section(source, label, occurrence)
        lines.append(f"% {source_key} {label} {occurrence + 1}")
        lines.extend(normalize_lines(section, source))
        manifest_sections.append(
            {
                "source": source.name,
                "source_path": str(source.path),
                "label": label,
                "occurrence": occurrence + 1,
                "source_bpm": source.bpm,
                "source_unit": f"1/{source.denominator}",
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    score_path = out_dir / "score.abc"
    score_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "score": str(score_path),
        "target_bpm": TARGET_BPM,
        "target_unit": f"1/{TARGET_DENOMINATOR}",
        "target_key": a.key,
        "recipe": recipe_description,
        "sections": manifest_sections,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def build() -> dict:
    a = split_sections(A_PATH, "A_pianai")
    b = split_sections(B_PATH, "B_daoxiang")
    if a.key != b.key:
        raise ValueError(f"Key mismatch: {a.key} vs {b.key}")

    sources = {"A": a, "B": b}
    full_recipe = [
        ("A", "intro", 0),
        ("A", "verse", 0),
        ("A", "pre-chorus", 0),
        ("A", "chorus", 0),
        ("B", "verse", 0),
        ("B", "chorus", 0),
        ("A", "verse", 1),
        ("A", "pre-chorus", 1),
        ("A", "chorus", 1),
        ("B", "verse", 1),
        ("B", "chorus", 1),
        ("B", "outro", 0),
    ]
    short_recipe = [
        ("A", "intro", 0),
        ("A", "verse", 0),
        ("A", "pre-chorus", 0),
        ("B", "chorus", 0),
        ("A", "chorus", 0),
        ("B", "verse", 0),
        ("B", "chorus", 0),
        ("B", "outro", 0),
    ]
    return {
        "full": write_score(
            FULL_OUT_DIR,
            sources,
            full_recipe,
            "A intro/verse/pre/chorus, B verse/chorus, A verse/pre/chorus, B verse/chorus/outro",
        ),
        "short": write_score(
            SHORT_OUT_DIR,
            sources,
            short_recipe,
            "A intro/verse/pre, B chorus, A chorus, B verse/chorus/outro",
        ),
    }


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
