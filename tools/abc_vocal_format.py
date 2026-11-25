"""Convert between native two-voice ABC and a compact Vocal-only training format."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path


ABC_TOOLS = Path(__file__).resolve().parents[1] / "skills/yue2-music/scripts/abc_tools.py"
VOCAL_DEFINITION = 'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"'
INS_DEFINITION = 'V: Ins clef=treble name="Ins Melody" snm="Inst."'


def load_validator():
    spec = importlib.util.spec_from_file_location("vocal_format_abc_tools", ABC_TOOLS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load ABC validator: {ABC_TOOLS}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def to_vocal(text: str, validator=None) -> str:
    validator = validator or load_validator()
    score = validator.parse_abc(text)
    lines = text.splitlines()
    result = lines[:6] + lines[7:8]
    cursor = 8
    while cursor < len(lines):
        while cursor < len(lines) and lines[cursor].startswith("% "):
            result.append(lines[cursor])
            cursor += 1
        if cursor == len(lines):
            break
        if lines[cursor] != "V: Vocal":
            raise ValueError(f"Expected Vocal group at line {cursor + 1}")
        result.append(lines[cursor])
        cursor += 1
        while cursor < len(lines) and lines[cursor].startswith(("M:", "K:")):
            result.append(lines[cursor])
            cursor += 1
        result.append(lines[cursor])
        cursor += 1
        if lines[cursor] != "V: Ins":
            raise ValueError(f"Expected Ins group at line {cursor + 1}")
        cursor += 1
        while cursor < len(lines) and lines[cursor].startswith(("M:", "K:")):
            cursor += 1
        cursor += 1
    vocal = "\n".join(result) + "\n"
    restored = validator.parse_abc(restore_ins(vocal, validator))
    if (restored.voices["Vocal"].notes != score.voices["Vocal"].notes
            or restored.voices["Vocal"].bars != score.voices["Vocal"].bars
            or restored.voices["Vocal"].keys != score.voices["Vocal"].keys
            or restored.voices["Vocal"].chords != score.voices["Vocal"].chords):
        raise ValueError("Vocal score changed during conversion")
    return vocal


def silent_bar_with_keys(body: str, validator) -> str:
    events = []
    cursor = 0
    offset = 0
    while cursor < len(body):
        if body[cursor].isspace():
            cursor += 1
            continue
        match = validator.TOKEN.match(body, cursor)
        if match is None:
            raise ValueError(f"Unsupported Vocal token at {body[cursor:cursor + 24]!r}")
        cursor = match.end()
        if match.group("key") is not None:
            events.append((offset, match.group(0)))
        elif match.group("note") is not None:
            offset += int(match.group("duration") or "1")
    if not events:
        return "Z"
    result = []
    position = 0
    for event_offset, key in events + [(offset, "")]:
        remaining = event_offset - position
        while remaining:
            duration = max(value for value in validator.DURATIONS if value <= remaining)
            result.append(f"z{duration if duration > 1 else ''}")
            remaining -= duration
        result.append(key)
        position = event_offset
    return "".join(result)


def restore_ins(text: str, validator=None) -> str:
    validator = validator or load_validator()
    lines = text.splitlines()
    if len(lines) < 10 or lines[0:2] != ["X:1", "T:"] or lines[5] != VOCAL_DEFINITION or not lines[6].startswith("K:"):
        raise ValueError("Expected Vocal-only ABC with native header")
    result = lines[:6] + [INS_DEFINITION, lines[6]]
    cursor = 7
    groups = 0
    while cursor < len(lines):
        while cursor < len(lines) and lines[cursor].startswith("% "):
            result.append(lines[cursor])
            cursor += 1
        if cursor == len(lines):
            raise ValueError("Section comment without music")
        if lines[cursor] != "V: Vocal":
            raise ValueError(f"Expected V: Vocal at line {cursor + 1}")
        result.append(lines[cursor])
        cursor += 1
        fields = []
        while cursor < len(lines) and lines[cursor].startswith(("M:", "K:")):
            fields.append(lines[cursor])
            result.append(lines[cursor])
            cursor += 1
        if cursor == len(lines):
            raise ValueError("Missing Vocal music line")
        music = lines[cursor]
        if not music.endswith("|"):
            raise ValueError("Vocal music line must end with a barline")
        result.append(music)
        count = 0
        silent_bars = []
        for bar in music[:-1].split("|"):
            match = re.fullmatch(r"Z([2-4])?", bar)
            if match:
                expanded = int(match.group(1) or "1")
                silent_bars.extend(["Z"] * expanded)
                count += expanded
            else:
                silent_bars.append(silent_bar_with_keys(bar, validator))
                count += 1
        if not 1 <= count <= 4:
            raise ValueError(f"Expected 1–4 measures, got {count}")
        silent_music = f"Z{count if count > 1 else ''}|" if all(bar == "Z" for bar in silent_bars) else "|".join(silent_bars) + "|"
        result.extend(["V: Ins", *fields, silent_music])
        groups += 1
        cursor += 1
    if not groups:
        raise ValueError("No Vocal music groups")
    restored = "\n".join(result) + "\n"
    validator.parse_abc(restored)
    return restored
