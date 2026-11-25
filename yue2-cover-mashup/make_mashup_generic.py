#!/usr/bin/env python3
"""Rearrange two native YuE2 ABC scores and map timestamped lyrics to the result."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


ABC_TOOLS = Path(__file__).resolve().parents[1] / "skills/yue2-music/scripts/abc_tools.py"
NOTE = re.compile(r"(?P<acc>\^\^|__|\^|_|=)?(?P<note>[A-Ga-gz])(?P<oct>[,']*)(?P<duration>[0-9]*)(?P<tie>-?)")


def load_abc_tools():
    spec = importlib.util.spec_from_file_location("mashup_abc_tools", ABC_TOOLS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load ABC validator: {ABC_TOOLS}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class Section:
    label: str
    occurrence: int
    lines: list[str]
    start_quarters: Fraction
    end_quarters: Fraction
    meter: tuple[int, int]
    bars: list[tuple[Fraction, Fraction, tuple[int, int]]]
    events: dict[str, list[tuple[Fraction, int, Fraction]]]


@dataclass
class Source:
    key: str
    score_path: Path
    lyrics_path: Path
    header: list[str]
    bpm: int
    denominator: int
    tonic_key: str
    vocal_pitches: list[int]
    instrument_pitches: list[int]
    sections: dict[tuple[str, int], Section]
    lyrics: list[dict]
    audio_offset_seconds: float


@dataclass
class TransformState:
    source_accidentals: dict[str, int]
    target_accidentals: dict[str, int]
    pending_tie: tuple[int, int] | None = None


DEFAULT_STYLE = (
    "Mandarin pop mashup duet, alternate singer A and singer B by section, "
    "follow the supplied combined melody score as the main melodic guide, "
    "use natural phrasing and smooth transitions."
)


def resolve_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def safe_id(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-")
    return value[:160] or "mashup"


def key_pitch(key: str) -> int:
    match = re.fullmatch(r"([A-Ga-g])([#b]?)(m)?", key)
    if not match:
        raise ValueError(f"Unsupported key {key!r}")
    return {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[match.group(1).upper()] + {
        "#": 1, "b": -1, "": 0}[match.group(2)]


def key_shift(source_key: str, target_key: str) -> int:
    return (key_pitch(target_key) - key_pitch(source_key) + 6) % 12 - 6


def best_target_key(sources: dict[str, Source], vocal_range: tuple[int, int]) -> str:
    candidates = [source.tonic_key for source in sources.values()]
    scored = []
    for candidate in candidates:
        shifts = []
        for source in sources.values():
            delta = key_shift(source.tonic_key, candidate)
            if any(not vocal_range[0] <= pitch + delta <= vocal_range[1] for pitch in source.vocal_pitches):
                break
            if any(not 0 <= pitch + delta <= 127 for pitch in source.instrument_pitches):
                break
            shifts.append(abs(delta))
        if len(shifts) == len(sources):
            scored.append((sum(shifts), max(shifts), candidates.index(candidate), candidate))
    if not scored:
        raise ValueError(f"Neither source key fits vocal_pitch_range {list(vocal_range)} after transposition; "
                         "adjust the range or provide a suitable target_key")
    return min(scored)[3]


def section_alias(label: str) -> str:
    chinese = {
        "前奏": "intro", "前引": "intro", "主歌": "verse", "主歌一": "verse", "主歌二": "verse",
        "副歌前": "pre-chorus", "导歌": "pre-chorus", "副歌": "chorus", "高潮": "chorus",
        "桥段": "bridge", "桥": "bridge", "结尾": "outro", "尾奏": "outro", "间奏": "interlude",
        "静音": "silence", "空白": "silence",
    }
    if label.strip() in chinese:
        return chinese[label.strip()]
    value = re.sub(r"[^a-z0-9]", "", label.lower())
    if value in {"intro", "introduction", "opening", "前奏", "前引"}:
        return "intro"
    if value in {"verse", "a", "主歌", "主歌一", "主歌二"}:
        return "verse"
    if value in {"prechorus", "pre", "副歌前", "导歌"}:
        return "pre-chorus"
    if value in {"chorus", "hook", "refrain", "副歌", "高潮"}:
        return "chorus"
    if value in {"bridge", "桥段", "桥"}:
        return "bridge"
    if value in {"outro", "ending", "结尾", "尾奏"}:
        return "outro"
    if value in {"interlude", "间奏", "间奏段"}:
        return "interlude"
    if value in {"silence", "静音", "空白"}:
        return "silence"
    return value


def auto_recipe(sources: dict[str, Source]) -> list[dict]:
    """Build a conservative ABAB recipe from conventional section labels."""
    available = {key: list(source.sections) for key, source in sources.items()}

    def pick(source_key: str, aliases: tuple[str, ...], occurrence: int = 1):
        for label, number in available[source_key]:
            if number == occurrence and section_alias(label) in aliases:
                return {"source": source_key, "section": label, "occurrence": number}
        return None

    def add(source_key: str, aliases: tuple[str, ...], occurrence: int = 1, instrumental: bool = False):
        item = pick(source_key, aliases, occurrence)
        if item is not None:
            if instrumental:
                item["instrumental"] = True
            result.append(item)

    result: list[dict] = []
    add("A", ("intro", "interlude"), instrumental=True)
    add("A", ("verse", "chorus", "bridge"))
    add("A", ("pre-chorus",))
    add("A", ("chorus", "verse"))
    add("B", ("verse", "chorus", "bridge"))
    add("B", ("chorus", "verse"))
    add("A", ("verse", "chorus"), occurrence=2)
    add("A", ("pre-chorus",), occurrence=2)
    add("A", ("chorus", "verse"), occurrence=2)
    add("B", ("verse", "chorus"), occurrence=2)
    add("B", ("chorus", "verse"), occurrence=2)
    add("B", ("outro", "interlude"), instrumental=True)
    if not result:
        raise ValueError("Cannot infer sections; provide recipe explicitly or add % intro/% verse/% chorus labels")
    return result


def compressed_recipe(sources: dict[str, Source]) -> list[dict]:
    """Build the short song-like recipe used by the existing condensed mashup."""
    available = {key: list(source.sections) for key, source in sources.items()}

    def pick(source_key: str, aliases: tuple[str, ...], occurrence: int = 1):
        for alias in aliases:
            for label, number in available[source_key]:
                if number == occurrence and section_alias(label) == alias:
                    return {"source": source_key, "section": label, "occurrence": number}
        return None

    def add(source_key: str, aliases: tuple[str, ...], occurrence: int = 1,
            instrumental: bool = False) -> None:
        item = pick(source_key, aliases, occurrence)
        if item is not None:
            if instrumental:
                item["instrumental"] = True
            result.append(item)

    result: list[dict] = []
    add("A", ("intro", "interlude"), instrumental=True)
    add("A", ("verse", "pre-chorus", "chorus", "bridge"))
    add("A", ("pre-chorus", "chorus", "verse"))
    add("B", ("chorus", "verse", "bridge"))
    add("A", ("chorus", "verse", "bridge"))
    add("B", ("verse", "chorus", "bridge"))
    add("B", ("chorus", "verse", "bridge"))
    add("B", ("outro", "interlude"), instrumental=True)
    if not result:
        raise ValueError("Cannot infer compressed sections; provide recipe explicitly or add section labels")
    return result


def load_lyrics(path: Path, unit: str) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else data.get("data") or data.get("lyrics")
    if not isinstance(items, list):
        raise ValueError(f"{path}: expected a list or a 'data'/'lyrics' list")
    multiplier = {"seconds": 1, "milliseconds": 0.001}[unit]
    result = []
    for index, item in enumerate(items, 1):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: lyric item {index} must be an object")
        text = " ".join(str(item.get("text", "")).split())
        if not text:
            continue
        words = [word for word in item.get("words", []) if word.get("word") and word.get("start") is not None and word.get("end") is not None]
        try:
            start = float(min(word["start"] for word in words) if words else item["start"]) * multiplier
            end = float(max(word["end"] for word in words) if words else item["end"]) * multiplier
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}: invalid timestamps on lyric item {index}") from exc
        if not (0 <= start < end < float("inf")):
            raise ValueError(f"{path}: invalid time interval on lyric item {index}")
        result.append({"start": start, "end": end, "text": text,
                       "words": [{"word": str(word["word"]), "start": float(word["start"]) * multiplier,
                                  "end": float(word["end"]) * multiplier} for word in words]})
    return sorted(result, key=lambda item: item["start"])


def syllable_tokens(text: str) -> list[str]:
    return re.findall(r"[\u3400-\u9fff]|[A-Za-z]+(?:'[A-Za-z]+)?", text)


def align_lyric_line(aligned: dict, lyric: dict, section: Section, section_start: Fraction,
                     bpm: int, source: Source, semitones: int, max_error_ms: float) -> dict:
    tokens = syllable_tokens(aligned["text"])
    report = {"text": aligned["text"], "source": source.key, "section": aligned["section"],
              "occurrence": aligned["occurrence"], "target_seconds": [aligned["target_start"], aligned["target_end"]],
              "status": "review", "syllable_count": len(tokens), "candidate_note_count": 0,
              "syllables": [], "rests_seconds": [], "max_onset_error_ms": None,
              "suggestion": None}
    if not tokens:
        report["suggestion"] = "未识别可对齐音节，请人工检查歌词文本"
        return report
    try:
        from pypinyin import Style, lazy_pinyin
        chinese = lazy_pinyin("".join(token for token in tokens if len(token) == 1 and "\u3400" <= token <= "\u9fff"),
                              style=Style.TONE3)
    except ImportError:
        chinese = []
    pronunciations = iter(chinese)
    word_tokens = [token for word in lyric["words"] for token in syllable_tokens(word["word"])]
    precise_words = bool(lyric["words"]) and word_tokens == tokens
    expected = []
    if precise_words:
        for word in lyric["words"]:
            parts = syllable_tokens(word["word"])
            if not parts:
                continue
            start = max(word["start"], source.audio_offset_seconds + float(section.start_quarters * 60 / source.bpm),
                        aligned["source_start"])
            end = min(word["end"], source.audio_offset_seconds + float(section.end_quarters * 60 / source.bpm),
                      aligned["source_end"])
            if not start < end:
                precise_words = False
                break
            for part in range(len(parts)):
                expected.append((float(section_start * 60 / bpm) +
                                 (start + (end - start) * part / len(parts)
                                  - float(section.start_quarters * 60 / source.bpm)
                                  - source.audio_offset_seconds) * source.bpm / bpm,
                                 float(section_start * 60 / bpm) +
                                 (start + (end - start) * (part + 1) / len(parts)
                                  - float(section.start_quarters * 60 / source.bpm)
                                  - source.audio_offset_seconds) * source.bpm / bpm))
    if not precise_words:
        duration = aligned["target_end"] - aligned["target_start"]
        expected = [(aligned["target_start"] + duration * index / len(tokens),
                     aligned["target_start"] + duration * (index + 1) / len(tokens))
                    for index in range(len(tokens))]
    notes = []
    for index, (onset, pitch, duration) in enumerate(section.events["Vocal"]):
        start = float((section_start + onset) * 60 / bpm)
        end = float((section_start + onset + duration) * 60 / bpm)
        if aligned["target_start"] <= (start + end) / 2 < aligned["target_end"]:
            notes.append({"index": index, "onset_quarters": str(section_start + onset),
                          "duration_quarters": str(duration), "midi_pitch": pitch + semitones,
                          "start_seconds": round(start, 3), "end_seconds": round(end, 3)})
    report["candidate_note_count"] = len(notes)
    cursor = aligned["target_start"]
    for note in notes:
        if note["start_seconds"] > cursor + 0.03:
            report["rests_seconds"].append([round(cursor, 3), note["start_seconds"]])
        cursor = max(cursor, note["end_seconds"])
    if cursor + 0.03 < aligned["target_end"]:
        report["rests_seconds"].append([round(cursor, 3), aligned["target_end"]])
    report["time_basis"] = "word_timestamps" if precise_words else "uniform_sentence_estimate"
    if len(notes) < len(tokens):
        report["suggestion"] = (f"{len(tokens)} 个音节只有 {len(notes)} 个候选 Vocal 音符；"
                                "请调整歌词、乐句边界或人工配音")
        return report

    paths: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {(0, 0): (0.0, ())}

    def update(position: tuple[int, int], candidate: tuple[float, tuple[int, ...]]) -> None:
        if position not in paths or candidate[0] < paths[position][0]:
            paths[position] = candidate

    scale = max(0.25, aligned["target_end"] - aligned["target_start"])
    for syllable_index in range(len(tokens) + 1):
        for note_index in range(len(notes)):
            current = paths.get((syllable_index, note_index))
            if current is None:
                continue
            cost, assigned = current
            update((syllable_index, note_index + 1), (cost + 0.35, assigned + (-1,)))
            if syllable_index < len(tokens):
                error = abs(notes[note_index]["start_seconds"] - expected[syllable_index][0])
                update((syllable_index + 1, note_index + 1),
                       (cost + error / scale, assigned + (syllable_index,)))
            if syllable_index > 0:
                previous = next((offset for offset in range(note_index - 1, -1, -1)
                                 if assigned[offset] >= 0), None)
                gap = (max(0, notes[note_index]["start_seconds"] - notes[previous]["end_seconds"])
                       if previous is not None else 0)
                update((syllable_index, note_index + 1),
                       (cost + 0.12 + gap / scale, assigned + (syllable_index - 1,)))
    solution = paths.get((len(tokens), len(notes)))
    if solution is None:
        report["suggestion"] = "未找到满足顺序的音节—音符映射，请人工配音"
        return report
    assignments = solution[1]
    errors = []
    unresolved = False
    for index, token in enumerate(tokens):
        matched = [note for note, assigned in zip(notes, assignments) if assigned == index]
        is_chinese = len(token) == 1 and "\u3400" <= token <= "\u9fff"
        pronunciation = next(pronunciations, None) if is_chinese else None
        if pronunciation is None:
            unresolved = True
        error = (matched[0]["start_seconds"] - expected[index][0]) * 1000
        errors.append(abs(error))
        report["syllables"].append({"text": token, "pronunciation_estimate": pronunciation,
                                    "expected_seconds": [round(value, 3) for value in expected[index]],
                                    "notes": matched, "onset_error_ms": round(error)})
    report["max_onset_error_ms"] = round(max(errors))
    report["mean_abs_onset_error_ms"] = round(sum(errors) / len(errors))
    report["pronunciation_source"] = "pypinyin_estimate" if chinese else "unavailable"
    if unresolved or not precise_words or max(errors) > max_error_ms:
        report["suggestion"] = "读音、逐字时间或起音误差需要复核；可修改歌词、选段或人工配音"
    else:
        report["status"] = "aligned_estimate"
    return report


def load_source(key: str, config: dict, base: Path, abc_tools) -> Source:
    score_path = resolve_path(base, config["score"])
    lyrics_path = resolve_path(base, config["lyrics"])
    lines = score_path.read_text(encoding="utf-8").splitlines()
    score = abc_tools.parse_abc("\n".join(lines) + "\n")
    if score.voices["Vocal"].chords:
        raise ValueError(f"{score_path}: cot=melody requires chord-free input; strip chord symbols first")
    if any(line.startswith("K:") for line in lines[8:]):
        raise ValueError(f"{score_path}: inline key changes are not supported; normalize the key first")
    sections: dict[tuple[str, int], Section] = {}
    counts: dict[str, int] = {}
    label = None
    body: list[str] = []
    bar_cursor = 0
    vocal_bars = 0

    def flush() -> None:
        nonlocal body, bar_cursor, vocal_bars
        if label is None or not body:
            return
        if vocal_bars == 0:
            raise ValueError(f"{score_path}: section {label!r} contains no Vocal bars")
        occurrence = counts.get(label, 0) + 1
        counts[label] = occurrence
        bars = score.voices["Vocal"].bars
        if bar_cursor + vocal_bars > len(bars):
            raise ValueError(f"{score_path}: section bars exceed source score")
        start = bars[bar_cursor][0]
        last_start, last_length, _ = bars[bar_cursor + vocal_bars - 1]
        end = last_start + last_length
        for voice in score.voices.values():
            if any(onset < boundary < onset + duration for boundary in (start, end)
                   for onset, _, duration in voice.notes):
                raise ValueError(f"{score_path}: section {label!r} cuts a tied note at a boundary")
        section_bars = [(onset - start, length, meter) for onset, length, meter
                        in bars[bar_cursor:bar_cursor + vocal_bars]]
        events = {name: [(onset - start, pitch, duration) for onset, pitch, duration in voice.notes
                         if start <= onset < end] for name, voice in score.voices.items()}
        sections[(label, occurrence)] = Section(label, occurrence, body, start, end,
                                                bars[bar_cursor][2], section_bars, events)
        bar_cursor += vocal_bars
        vocal_bars = 0
        body = []

    for index in range(8, len(lines)):
        line = lines[index]
        if line.startswith("% "):
            flush()
            label = line[2:].strip()
            if not label:
                raise ValueError(f"{score_path}: empty section name at line {index + 1}")
        else:
            if label is None:
                raise ValueError(f"{score_path}: music before the first % section label")
            body.append(line)
            if score.music_lines.get(index) == "Vocal":
                bars = line[:-1].split("|")
                vocal_bars += sum(int(bar[1:] or "1") if re.fullmatch(r"Z[2-4]?", bar) else 1 for bar in bars)
    flush()
    if bar_cursor != len(score.voices["Vocal"].bars):
        raise ValueError(f"{score_path}: section boundaries do not cover the score")
    return Source(key, score_path, lyrics_path, lines[:8], score.bpm, int(lines[3].split("/")[1]), lines[7][2:],
                  [pitch for _, pitch, _ in score.voices["Vocal"].notes],
                  [pitch for _, pitch, _ in score.voices["Ins"].notes], sections,
                  load_lyrics(lyrics_path, config.get("timestamp_unit", "seconds")),
                  float(config.get("audio_offset_seconds", 0)))


def pitch_value(note: str, octave: str, accidental: str | None,
                key_accidentals: dict[str, int], local: dict[str, int]) -> int:
    letter = note.upper()
    value = 60 + {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[letter]
    value += 12 * (octave.count("'") - octave.count(","))
    if note.islower():
        value += 12
    if accidental:
        value += {"=": 0, "_": -1, "__": -2, "^": 1, "^^": 2}[accidental]
        local[letter] = {"=": 0, "_": -1, "__": -2, "^": 1, "^^": 2}[accidental]
    else:
        value += local.get(letter, key_accidentals[letter])
    return value


def spell_pitch(value: int, target_key_accidentals: dict[str, int]) -> tuple[str, str, str]:
    naturals = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    candidates = []
    for letter, natural in naturals.items():
        for register in range(-5, 6):
            base = 60 + natural + 12 * register
            delta = value - base
            if -2 <= delta <= 2:
                note = letter.lower() if register >= 1 else letter
                marks = "'" * max(0, register - 1) + "," * max(0, -register)
                if delta == 0:
                    accidental = "=" if target_key_accidentals[letter] else ""
                else:
                    accidental = {1: "^", 2: "^^", -1: "_", -2: "__"}.get(delta)
                if accidental is not None:
                    candidates.append((abs(delta), len(marks), letter, note, marks, accidental))
    if not candidates:
        raise ValueError(f"Cannot spell MIDI pitch {value} in target key")
    _, _, _, note, marks, accidental = min(candidates)
    return accidental, note, marks


def transform_line(line: str, factor: int, source_key: str, target_key: str,
                   semitones: int = 0, state: TransformState | None = None) -> str:
    if not line.endswith("|") or line.startswith(("V:", "M:", "K:")):
        return line
    abc_tools = load_abc_tools()
    source_accidentals = abc_tools.key_accidentals(source_key)
    target_accidentals = abc_tools.key_accidentals(target_key)
    if state is None:
        state = TransformState({}, {})
    output = []
    index = 0
    while index < len(line):
        if line[index] == '"' or line.startswith("[K:", index):
            closer = '"' if line[index] == '"' else "]"
            end = line.find(closer, index + 1)
            if end == -1:
                raise ValueError(f"Unclosed token in {line!r}")
            output.append(line[index:end + 1])
            index = end + 1
            continue
        match = NOTE.match(line, index)
        if match:
            duration = int(match.group("duration") or "1") * factor
            if match.group("note") == "z":
                output.append(f"z{duration}{match.group('tie')}")
            else:
                letter = match.group("note").upper()
                written = 60 + {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[letter]
                written += 12 * (match.group("note").islower() + match.group("oct").count("'") - match.group("oct").count(","))
                pitch = pitch_value(match.group("note"), match.group("oct"), match.group("acc"),
                                    source_accidentals, state.source_accidentals)
                if state.pending_tie is not None:
                    previous_pitch, previous_written = state.pending_tie
                    if not match.group("acc") and written == previous_written:
                        pitch = previous_pitch
                    elif pitch != previous_pitch:
                        raise ValueError(f"Tie changes pitch in {line!r}")
                state.pending_tie = (pitch, written) if match.group("tie") else None
                accidental, note, marks = spell_pitch(pitch + semitones, target_accidentals)
                target_letter = note.upper()
                if not accidental and state.target_accidentals.get(target_letter, target_accidentals[target_letter]):
                    accidental = "="
                if accidental:
                    state.target_accidentals[target_letter] = {"=": 0, "_": -1, "__": -2,
                                                               "^": 1, "^^": 2}[accidental]
                output.append(f"{accidental}{note}{marks}{duration}{match.group('tie')}")
            index = match.end()
        else:
            output.append(line[index])
            index += 1
        if index > 0 and line[index - 1] == "|":
            state.source_accidentals.clear()
            state.target_accidentals.clear()
    return "".join(output)


def normalize_section(lines: list[str], section: Section, factor: int,
                      source: Source, target_key: str, semitones: int) -> list[str]:
    result = []
    meter = f"M:{section.meter[0]}/{section.meter[1]}"
    states = {"Vocal": TransformState({}, {}), "Ins": TransformState({}, {})}
    voice = None
    initialized_voices = set()
    for index, line in enumerate(lines):
        if line in ("V: Vocal", "V: Ins"):
            voice = line[3:]
        result.append(transform_line(line, factor, source.tonic_key, target_key,
                                     semitones, states[voice] if voice is not None else None))
        if line in ("V: Vocal", "V: Ins") and voice not in initialized_voices:
            initialized_voices.add(voice)
            if index + 1 == len(lines) or not lines[index + 1].startswith("M:"):
                result.append(meter)
    return result


def verify_sections(score, emissions: list[tuple[Section, Fraction, int]]) -> None:
    total_quarters = (emissions[-1][1] + emissions[-1][0].end_quarters
                      - emissions[-1][0].start_quarters) if emissions else Fraction(0)
    for name, voice in score.voices.items():
        if voice.time != total_quarters:
            raise ValueError(f"{name}: rebuilt score duration differs from selected sections")
    for index, (section, target_start, semitones) in enumerate(emissions, 1):
        target_end = target_start + section.end_quarters - section.start_quarters
        for name, voice in score.voices.items():
            actual_bars = [(onset - target_start, length, meter)
                           for onset, length, meter in voice.bars if target_start <= onset < target_end]
            if actual_bars != section.bars:
                raise ValueError(f"Section #{index} {name}: measure positions, durations or meters changed")
            actual_events = [(onset - target_start, pitch, duration)
                             for onset, pitch, duration in voice.notes if target_start <= onset < target_end]
            expected_events = [(onset, pitch + semitones, duration)
                               for onset, pitch, duration in section.events[name]]
            if actual_events != expected_events:
                raise ValueError(f"Section #{index} {name}: note onsets, pitches or durations changed")


def build(config_path: Path, output: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    abc_tools = load_abc_tools()
    if not isinstance(config.get("sources"), dict) or not all(key in config["sources"] for key in ("A", "B")):
        raise ValueError("sources.A and sources.B are required")
    sources = {key: load_source(key, config["sources"][key], config_path.parent, abc_tools) for key in ("A", "B")}
    first, second = sources["A"], sources["B"]
    vocal_range = config.get("vocal_pitch_range", [48, 88])
    if (not isinstance(vocal_range, (list, tuple)) or len(vocal_range) != 2
            or any(type(bound) is not int for bound in vocal_range)
            or not 0 <= vocal_range[0] <= vocal_range[1] <= 127):
        raise ValueError("vocal_pitch_range must be two MIDI pitches between 0 and 127")
    target_key = config.get("target_key") or best_target_key(sources, tuple(vocal_range))
    try:
        abc_tools.key_accidentals(target_key)
    except Exception as exc:
        raise ValueError(f"Unsupported target_key {target_key!r}") from exc
    target_meter = config.get("target_meter") or first.header[2][2:]
    try:
        meter_numerator, meter_denominator = (int(value) for value in target_meter.split("/", 1))
        if meter_numerator <= 0 or meter_denominator <= 0:
            raise ValueError
    except (AttributeError, ValueError):
        raise ValueError(f"Invalid target_meter {target_meter!r}; use for example '4/4'")
    target_bpm = int(config.get("target_bpm") or round((first.bpm + second.bpm) / 2))
    target_denominator = int(config.get("target_denominator", max(first.denominator, second.denominator)))
    if target_bpm <= 0 or target_denominator <= 0:
        raise ValueError("Target BPM and denominator must be positive")
    max_tempo_ratio = config.get("max_tempo_ratio", 1.5)
    if (type(max_tempo_ratio) not in (int, float) or not math.isfinite(max_tempo_ratio)
            or max_tempo_ratio < 1):
        raise ValueError("max_tempo_ratio must be a finite number at least 1")
    allow_meter_changes = config.get("allow_meter_changes", True)
    if type(allow_meter_changes) is not bool:
        raise ValueError("allow_meter_changes must be a boolean")
    alignment_mode = config.get("syllable_alignment", "report")
    if alignment_mode not in ("report", "strict", "off"):
        raise ValueError("syllable_alignment must be report, strict or off")
    max_alignment_error_ms = config.get("max_alignment_error_ms", 700)
    if (type(max_alignment_error_ms) not in (int, float) or not math.isfinite(max_alignment_error_ms)
            or max_alignment_error_ms < 0):
        raise ValueError("max_alignment_error_ms must be a nonnegative finite number")
    compression = str(config.get("compression", config.get("mode", "full"))).strip().lower()
    compression_aliases = {"full": "full", "normal": "full", "default": "full",
                           "compressed": "compressed", "compact": "compressed", "short": "compressed"}
    if compression not in compression_aliases:
        raise ValueError("compression must be one of: full, compressed")
    compression = compression_aliases[compression]
    recipe = config.get("recipe")
    if recipe is None:
        recipe = compressed_recipe(sources) if compression == "compressed" else auto_recipe(sources)
    lines = first.header.copy()
    lines[2] = f"M:{meter_numerator}/{meter_denominator}"
    lines[3] = f"L:1/{target_denominator}"
    lines[4] = f"Q:1/4={target_bpm}"
    lines[7] = f"K:{target_key}"
    manifest = []
    timed_lyrics = []
    syllable_reports = []
    request_blocks = []
    emissions: list[tuple[Section, Fraction, int]] = []
    elapsed_quarters = Fraction(0)
    for index, entry in enumerate(recipe, 1):
        source = sources[entry["source"]]
        semitones = key_shift(source.tonic_key, target_key)
        label = entry["section"]
        occurrence = int(entry.get("occurrence", 1))
        if occurrence < 1 or (label, occurrence) not in source.sections:
            raise ValueError(f"Recipe #{index}: {source.key} has no {label} #{occurrence}; available: {sorted(source.sections)}")
        if target_denominator % source.denominator:
            raise ValueError(f"Cannot represent {source.key} L:1/{source.denominator} in L:1/{target_denominator}")
        section = source.sections[(label, occurrence)]
        tempo_ratio = max(source.bpm / target_bpm, target_bpm / source.bpm)
        if tempo_ratio > max_tempo_ratio:
            raise ValueError(f"Recipe #{index}: {source.key} tempo ratio {tempo_ratio:.3f} "
                             f"exceeds max_tempo_ratio {max_tempo_ratio}; choose another target_bpm or limit")
        if not allow_meter_changes and any(bar_meter != (meter_numerator, meter_denominator)
                                           for _, _, bar_meter in section.bars):
            raise ValueError(f"Recipe #{index}: {source.key} contains a meter change; "
                             "allow_meter_changes is false")
        if any(not vocal_range[0] <= pitch + semitones <= vocal_range[1]
               for pitch in source.vocal_pitches):
            raise ValueError(f"{source.key}: transposed Vocal exceeds vocal_pitch_range {vocal_range}")
        if any(not 0 <= pitch + semitones <= 127 for pitch in source.instrument_pitches):
            raise ValueError(f"{source.key}: transposed Ins exceeds MIDI range")
        lines.append(f"% {source.key} {label} {occurrence}")
        lines.extend(normalize_section(section.lines, section, target_denominator // source.denominator,
                                       source, target_key, semitones))
        emissions.append((section, elapsed_quarters, semitones))
        window_start = float(section.start_quarters * 60 / source.bpm) + source.audio_offset_seconds
        window_end = float(section.end_quarters * 60 / source.bpm) + source.audio_offset_seconds
        target_start = float(elapsed_quarters * 60 / target_bpm)
        entry_lyrics = []
        entry_alignment = []
        if not entry.get("instrumental", False):
            for lyric in source.lyrics:
                if window_start <= (lyric["start"] + lyric["end"]) / 2 < window_end:
                    clipped_start = max(window_start, lyric["start"])
                    clipped_end = min(window_end, lyric["end"])
                    if clipped_end <= clipped_start:
                        continue
                    aligned = {"text": lyric["text"], "source_start": lyric["start"],
                               "source_end": lyric["end"],
                               "target_start": round(target_start + (clipped_start - window_start) * source.bpm / target_bpm, 3),
                               "target_end": round(target_start + (clipped_end - window_start) * source.bpm / target_bpm, 3),
                               "source": source.key, "section": label, "occurrence": occurrence}
                    entry_lyrics.append(aligned)
                    timed_lyrics.append(aligned)
                    if alignment_mode != "off":
                        alignment = align_lyric_line(aligned, lyric, section, elapsed_quarters,
                                                     target_bpm, source, semitones, max_alignment_error_ms)
                        if alignment_mode == "strict" and alignment["status"] != "aligned_estimate":
                            raise ValueError(f"Recipe #{index} {source.key} {label}: lyric {lyric['text']!r} "
                                             f"needs manual alignment: {alignment['suggestion']}")
                        entry_alignment.append(alignment)
                        syllable_reports.append(alignment)
        request_blocks.append(f"[{source.key} {label} {occurrence}]" +
                              ("\n" + "\n".join(item["text"] for item in entry_lyrics) if entry_lyrics else ""))
        section_quarters = section.end_quarters - section.start_quarters
        manifest.append({"source": source.key, "section": label, "occurrence": occurrence,
                         "transpose_semitones": semitones,
                         "bar_count": len(section.bars),
                         "meter_sequence": [f"{bar_meter[0]}/{bar_meter[1]}" for _, _, bar_meter in section.bars],
                         "start_quarters": str(elapsed_quarters),
                         "duration_quarters": str(section_quarters),
                         "source_bpm": source.bpm, "tempo_ratio": round(tempo_ratio, 4),
                         "source_seconds": [round(window_start, 3), round(window_end, 3)],
                         "target_seconds": [round(target_start, 3), round(target_start + float(section_quarters * 60 / target_bpm), 3)],
                         "lyric_count": len(entry_lyrics),
                         "alignment_review_count": sum(item["status"] == "review" for item in entry_alignment)})
        elapsed_quarters += section_quarters
    score_text = "\n".join(lines) + "\n"
    verify_sections(abc_tools.parse_abc(score_text), emissions)
    default_id = f"{first.score_path.parent.name}_{second.score_path.parent.name}_mashup"
    request = {"id": safe_id(config.get("id", default_id)),
               "style": config.get("style", DEFAULT_STYLE), "lyrics": "\n\n".join(request_blocks),
               "cot": "melody", "seed": config.get("seed", 831001)}
    alignment_data = {"mode": alignment_mode, "max_alignment_error_ms": max_alignment_error_ms,
                      "line_count": len(syllable_reports),
                      "mapped_line_count": sum(bool(item["syllables"]) for item in syllable_reports),
                      "review_count": sum(item["status"] == "review" for item in syllable_reports),
                      "lines": syllable_reports}
    output.mkdir(parents=True, exist_ok=True)
    for filename, value in (("score.abc", score_text), ("request.json", json.dumps(request, ensure_ascii=False, indent=2) + "\n"),
                            ("aligned_lyrics.json", json.dumps(timed_lyrics, ensure_ascii=False, indent=2) + "\n"),
                            ("syllable_alignment.json", json.dumps(alignment_data, ensure_ascii=False, indent=2) + "\n"),
                            ("manifest.json", json.dumps({"sources": {key: {"score": str(source.score_path),
                            "lyrics": str(source.lyrics_path), "bpm": source.bpm} for key, source in sources.items()},
                            "target_bpm": target_bpm, "target_denominator": target_denominator,
                            "max_tempo_ratio": max_tempo_ratio, "allow_meter_changes": allow_meter_changes,
                            "target_key": target_key, "target_meter": f"{meter_numerator}/{meter_denominator}",
                            "vocal_pitch_range": vocal_range,
                            "transposition_semitones": {key: key_shift(source.tonic_key, target_key)
                                                       for key, source in sources.items()},
                            "syllable_alignment": alignment_mode,
                            "alignment_review_count": alignment_data["review_count"],
                            "compression": compression,
                            "sections": manifest}, ensure_ascii=False, indent=2) + "\n")):
        (output / filename).write_text(value, encoding="utf-8")
    return {"output": str(output), "sections": len(manifest), "lyric_lines": len(timed_lyrics),
            "duration_seconds": round(float(elapsed_quarters * 60 / target_bpm), 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compression", choices=("full", "compressed"),
                        help="Override config compression mode; explicit recipe still wins")
    args = parser.parse_args()
    if args.compression:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        config["compression"] = args.compression
        temporary_config = args.config.with_name(f".{args.config.name}.compression.json")
        temporary_config.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        try:
            summary = build(temporary_config, args.output.resolve())
        finally:
            temporary_config.unlink(missing_ok=True)
    else:
        summary = build(args.config.resolve(), args.output.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
