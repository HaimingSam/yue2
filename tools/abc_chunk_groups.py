"""Shared training and inference format for short Vocal music groups."""

from __future__ import annotations

import re

from abc_vocal_format import load_validator


MORE_GROUP = "<MORE_GROUP>"
END_SECTION = "<END_SECTION>"


def group_bars(music: str) -> int:
    if not music.endswith("|") or not music[:-1]:
        raise ValueError("Music group must end with a barline")
    count = 0
    for bar in music[:-1].split("|"):
        if not bar:
            raise ValueError("Empty measure")
        rest = re.fullmatch(r"Z([2-4])?", bar)
        count += int(rest.group(1) or "1") if rest else 1
    if not 1 <= count <= 4:
        raise ValueError(f"Music group must contain 1–4 measures; got {count}")
    return count


def split_groups(body: list[str]) -> list[list[str]]:
    groups = []
    cursor = 0
    while cursor < len(body):
        if body[cursor] != "V: Vocal":
            raise ValueError(f"Expected V: Vocal, got {body[cursor]!r}")
        start = cursor
        cursor += 1
        while cursor < len(body) and body[cursor].startswith(("M:", "K:")):
            cursor += 1
        if cursor >= len(body):
            raise ValueError("Vocal music group is incomplete")
        group_bars(body[cursor])
        cursor += 1
        groups.append(body[start:cursor])
    if not groups:
        raise ValueError("Section contains no music group")
    return groups


def chunk_prompt(source: str, structure: str, label: str, section_index: int,
                 occurrence: int, chunk_index: int, previous_group: list[str] | None) -> str:
    previous = "\n".join(previous_group) + "\n" if previous_group else "<NONE>\n"
    return (f"<SECTION>{label}</SECTION>\n<SECTION_INDEX>{section_index}</SECTION_INDEX>\n"
            f"<OCCURRENCE>{occurrence}</OCCURRENCE>\n<GROUP_INDEX>{chunk_index}</GROUP_INDEX>\n"
            f"<PREVIOUS_GROUP>\n{previous}</PREVIOUS_GROUP>\n"
            f"<CHORUS>\n{source}</CHORUS>\n{structure}")


def chunk_target(group: list[str], is_last: bool) -> str:
    return f"{END_SECTION if is_last else MORE_GROUP}\n" + "\n".join(group) + "\n"


def parse_chunk(text: str) -> tuple[list[str], bool]:
    lines = text.strip().splitlines()
    if len(lines) < 3 or lines[0] not in (MORE_GROUP, END_SECTION):
        raise ValueError(f"Expected {MORE_GROUP} or {END_SECTION} followed by one Vocal group")
    groups = split_groups(lines[1:])
    if len(groups) != 1:
        raise ValueError(f"Expected one Vocal group, got {len(groups)}")
    return groups[0], lines[0] == END_SECTION


def validate_partial_vocal(text: str, validator=None) -> int:
    """Check every completed group, including durations, without requiring a closing tie."""
    validator = validator or load_validator()
    lines = text.splitlines()
    if len(lines) < 10 or not lines[2].startswith("M:") or not lines[3].startswith("L:1/") or not lines[6].startswith("K:"):
        raise ValueError("Incomplete Vocal header")
    meter = validator.meter_value(lines[2][2:])
    denominator = int(lines[3].split("/", 1)[1])
    if denominator > 1024 or denominator & (denominator - 1):
        raise ValueError("Unsupported Vocal unit length")
    from fractions import Fraction
    key = lines[6][2:]
    validator.key_accidentals(key)
    voice = validator.Voice(meter, key, keys=[(Fraction(0), key)])
    music_groups = 0
    cursor = 7
    while cursor < len(lines):
        if lines[cursor].startswith("% "):
            cursor += 1
            continue
        if lines[cursor] != "V: Vocal":
            raise ValueError(f"Expected V: Vocal at line {cursor + 1}")
        cursor += 1
        while cursor < len(lines) and lines[cursor].startswith(("M:", "K:")):
            field, value = lines[cursor].split(":", 1)
            if field == "M":
                voice.meter = validator.meter_value(value)
            else:
                validator.key_accidentals(value)
                voice.key = value
                voice.keys.append((voice.time, value))
            cursor += 1
        if cursor >= len(lines):
            raise ValueError("Missing Vocal music line")
        music = lines[cursor]
        group_bars(music)
        for bar in music[:-1].split("|"):
            rest = re.fullmatch(r"Z([2-4])?", bar)
            for expanded in (["Z"] * int(rest.group(1) or "1")) if rest else [bar]:
                validator.parse_bar(expanded, voice, Fraction(1, denominator),
                                    f"Vocal group {music_groups + 1}, bar {len(voice.bars) + 1}")
        music_groups += 1
        cursor += 1
    return music_groups
