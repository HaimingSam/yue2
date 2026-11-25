#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent

A_SCORE = ROOT / "transcription_4b2f_pianai" / "score.abc"
B_SCORE = ROOT / "transcription_trace" / "score.abc"
A_LYRICS = Path("/data/cmq/datas/audio/pianai.json")
B_LYRICS = Path("/data/cmq/datas/audio/daoxiang.json")
MASHUP_SCORE = ROOT / "mashup_abab" / "score.abc"
MASHUP_MANIFEST = ROOT / "mashup_abab" / "manifest.json"

CORRECTIONS = {
    "追你偏爱": "对你偏爱",
    "讲不清偏爱": "讲不听偏爱",
    "没有别套路能走": "没有别条路能走",
    "跌倒了就不该继续往前走": "跌倒了就不敢继续往前走",
    "为什么人要这么的脆弱堕落": "为什么人要这么的脆弱 堕落",
    "还记得你说这是唯一的城堡 谁知道像狗又继续奔跑": "还记得你说家是唯一的城堡 随着稻香河流继续奔跑",
    "微微笑 小时候的梦我知道 小时候的梦我知道": "微微笑 小时候的梦我知道",
    "躲在水果堆里 蜜蜂给叮到怕了": "偷摘水果被蜜蜂给叮到怕了",
    "不过吉他太聪明总跟紧追": "午后吉他在虫鸣中更清脆",
    "想见的更要永远的依靠": "乡间的歌谣永远的依靠",
    "谁知道像河流继续奔跑": "随着稻香河流继续奔跑",
    "乡间大哥要永远的依靠": "乡间的歌谣永远的依靠",
}


@dataclass
class SectionWindow:
    label: str
    occurrence: int
    start: float
    end: float


def expanded_bar_count(line: str) -> int:
    count = 0
    for bar in line[:-1].split("|"):
        bar = bar.strip()
        if not bar:
            continue
        match = re.fullmatch(r"Z([2-4])?", bar)
        count += int(match.group(1) or "1") if match else 1
    return count


def source_windows(score_path: Path) -> list[SectionWindow]:
    lines = score_path.read_text(encoding="utf-8").splitlines()
    bpm = int(lines[4].split("=", 1)[1])
    seconds_per_measure = 4 * 60 / bpm
    label = None
    current_bars = 0
    current_start = 0
    sections: list[tuple[str, int, int]] = []
    cursor = 8
    while cursor < len(lines):
        if lines[cursor].startswith("% "):
            if label is not None:
                sections.append((label, current_start, current_bars))
                current_start = current_bars
            label = lines[cursor][2:].strip()
            cursor += 1
            continue
        if lines[cursor] == "V: Vocal":
            cursor += 1
            while cursor < len(lines) and lines[cursor].startswith(("M:", "K:")):
                cursor += 1
            current_bars += expanded_bar_count(lines[cursor])
        cursor += 1
    if label is not None:
        sections.append((label, current_start, current_bars))

    counts: dict[str, int] = {}
    windows: list[SectionWindow] = []
    for label, start_bar, end_bar in sections:
        occurrence = counts.get(label, 0) + 1
        counts[label] = occurrence
        windows.append(
            SectionWindow(
                label=label,
                occurrence=occurrence,
                start=start_bar * seconds_per_measure,
                end=end_bar * seconds_per_measure,
            )
        )
    return windows


def load_lyric_lines(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    lines = []
    source_items = data.get("data") or data["lyrics"]
    for item in source_items:
        text = " ".join(str(item.get("text", "")).split())
        if text:
            words = [w for w in item.get("words", []) if w.get("word") and w.get("start") is not None and w.get("end") is not None]
            if words:
                start = min(float(w["start"]) for w in words)
                end = max(float(w["end"]) for w in words)
            else:
                start = float(item["start"])
                end = float(item["end"])
            for before, after in CORRECTIONS.items():
                text = text.replace(before, after)
            lines.append({"start": start, "end": end, "text": text})
    return lines


def find_window(windows: list[SectionWindow], label: str, occurrence: int) -> SectionWindow:
    for window in windows:
        if window.label == label and window.occurrence == occurrence:
            return window
    raise KeyError(f"Missing window {label} #{occurrence}")


def select_lines(lines: list[dict], window: SectionWindow) -> list[dict]:
    selected = []
    for line in lines:
        midpoint = (line["start"] + line["end"]) / 2
        if window.start <= midpoint < window.end:
            selected.append(line)
    return selected


def build() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    a_windows = source_windows(A_SCORE)
    b_windows = source_windows(B_SCORE)
    lyrics = {"A": load_lyric_lines(A_LYRICS), "B": load_lyric_lines(B_LYRICS)}
    windows = {"A": a_windows, "B": b_windows}
    source_names = {"A": "singer A", "B": "singer B"}
    source_labels = {"A": "pianai", "B": "daoxiang"}

    manifest = json.loads(MASHUP_MANIFEST.read_text(encoding="utf-8"))
    blocks: list[str] = []
    extraction = []
    for index, section in enumerate(manifest["sections"], 1):
        source_key = "A" if section["source"].startswith("A_") else "B"
        label = section["label"]
        occurrence = int(section["occurrence"])
        if label in {"intro", "outro"}:
            blocks.append(f"[melodic {label} {source_key}{occurrence}]")
            extraction.append({"index": index, "source": source_key, "label": label, "occurrence": occurrence,
                               "lines": [], "note": "instrumental section"})
            continue
        window = find_window(windows[source_key], label, occurrence)
        selected = select_lines(lyrics[source_key], window)
        if not selected and occurrence > 1:
            fallback = find_window(windows[source_key], label, 1)
            selected = select_lines(lyrics[source_key], fallback)
            window = fallback
        blocks.append(f"[{source_names[source_key]} {source_labels[source_key]} {label} {occurrence}]")
        for line in selected:
            blocks.append(line["text"])
        extraction.append(
            {
                "index": index,
                "source": source_key,
                "label": label,
                "occurrence": occurrence,
                "source_time": [round(window.start, 3), round(window.end, 3)],
                "lines": selected,
            }
        )

    lyrics_text = "\n".join(blocks).strip() + "\n"
    style = (
        "Mandarin pop mashup duet, alternating singer A and singer B. Singer A is emotional rock-pop "
        "with clear powerful phrasing; singer B is warm nostalgic folk-rap pop. Acoustic guitar, piano, "
        "rounded bass, light hip-hop drums, subtle strings and airy pads. Preserve the supplied timestamp-aligned "
        "ABAB score structure and use smooth transitions between song sections."
    )
    request = {
        "id": "mashup_abab_timestamped_pianai_daoxiang",
        "style": style,
        "lyrics": lyrics_text,
        "cot": "melody",
        "seed": 831001,
    }
    (OUT / "aligned_lyrics.txt").write_text(lyrics_text, encoding="utf-8")
    (OUT / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "extraction_manifest.json").write_text(
        json.dumps({"score": str(MASHUP_SCORE), "source_windows": {
            "A": [w.__dict__ for w in a_windows],
            "B": [w.__dict__ for w in b_windows],
        }, "sections": extraction}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"request": str(OUT / "request.json"), "lyrics": str(OUT / "aligned_lyrics.txt"),
            "manifest": str(OUT / "extraction_manifest.json")}


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
