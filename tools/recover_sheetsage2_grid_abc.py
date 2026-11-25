#!/usr/bin/env python3
"""Rebuild a failed SheetSage2 ABC after dropping only unrepresentable MIDI notes."""

from __future__ import annotations

import argparse
import importlib
import json
import re
from pathlib import Path

from batch_transcribe_urv import check_and_repair_abc


NOTE_ERROR = re.compile(
    r"(?P<voice>Vocal|Ins): MIDI note pitch=(?P<pitch>\d+) at "
    r"(?P<start>[\d.]+)-(?P<end>[\d.]+) cannot be represented on the decoded subbeat grid"
)


def remove_unrepresentable_note(midi, error: str) -> dict:
    match = NOTE_ERROR.search(error)
    if match is None:
        raise ValueError(f"Cannot safely recover unrelated notation error: {error}")
    name = match.group("voice")
    pitch = int(match.group("pitch"))
    start = float(match.group("start"))
    end = float(match.group("end"))
    matches = [(instrument, note) for instrument in midi.instruments
               if name.lower() in instrument.name.lower()
               for note in instrument.notes
               if note.pitch == pitch and abs(note.start - start) < 0.001
               and abs(note.end - end) < 0.001]
    if len(matches) != 1:
        raise ValueError(f"Expected one MIDI note for grid error, found {len(matches)}: {error}")
    instrument, note = matches[0]
    instrument.notes.remove(note)
    return {"voice": name, "pitch": pitch, "start": start, "end": end}


def rebuild(artifact_dir: Path, notation, output: Path, max_drops: int = 8) -> dict:
    import pretty_midi

    directory = artifact_dir / "notation"
    midi = pretty_midi.PrettyMIDI(str(directory / "song_melody.mid"))
    beats = notation.read_beats(directory / "song_beats.txt")
    keys = notation.read_keys(directory / "song_keys.txt")
    structures = notation.read_structures(directory / "song_structures.txt")
    removed = []
    for _ in range(max_drops + 1):
        try:
            abc, _ = notation.generate_abc_from_data(midi, beats, [], keys, structures,
                                                      melody_only=True)
            if output.exists():
                raise FileExistsError(f"Refusing to overwrite existing score: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(abc, encoding="utf-8")
            try:
                check = check_and_repair_abc(output)
            except ValueError:
                output.unlink()
                raise
            report = {"output": str(output), "source_artifacts": str(artifact_dir),
                      "removed_unrepresentable_notes": removed, "abc_check": check}
            output.with_suffix(".recovery.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return report
        except ValueError as exc:
            if len(removed) >= max_drops:
                raise ValueError(f"Exceeded --max-drops={max_drops}: {exc}") from exc
            removed.append(remove_unrepresentable_note(midi, str(exc)))
    raise AssertionError("Unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--max-drops", type=int, default=8)
    args = parser.parse_args()
    if args.max_drops < 1 or args.max_seconds <= 0:
        parser.error("--max-drops and --max-seconds must be positive")
    if args.output.exists() or args.artifacts.exists() and any(args.artifacts.iterdir()):
        parser.error("Choose a new score path and an empty artifacts directory")
    from transformers import AutoModel
    import torch

    torch.set_num_threads(4)
    model = AutoModel.from_pretrained(
        "/data/cmq/YuE/models/SheetSage2",
        base_model_path=("/data/cmq/hf-cache/hub/models--m-a-p--MERT-v2-FullSong"
                         "/snapshots/d8ba1c745e733b3908ce6ad16ebeb17ac7600a42"),
        local_files_only=True, trust_remote_code=True).eval().to("cuda:0")
    try:
        model.transcribe(str(args.audio), output_dir=str(args.artifacts),
                         prompts=["timestamp", "downbeat_meter", "structure", "key", "melody_vocal"],
                         melody_only=True, max_seconds=args.max_seconds, dtype="bf16", preset="default")
    except RuntimeError as exc:
        if NOTE_ERROR.search(str(exc)) is None:
            raise
    else:
        raise RuntimeError("Transcription succeeded; use the standard batch script instead")
    namespace = model.__class__.__module__.rsplit(".", 1)[0]
    notation = importlib.import_module(namespace + ".notation_sheetsage2")
    print(json.dumps(rebuild(args.artifacts, notation, args.output, args.max_drops),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
