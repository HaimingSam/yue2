#!/usr/bin/env python3
"""Audit all input IDs and copy explicitly documented fallback ABC scores."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from batch_transcribe_urv import check_and_repair_abc, collect_inputs


DEFAULT_INPUT = Path("/data2/cmq/datas/003-muraka_urv_separated")
DEFAULT_OUTPUT = Path("/data2/cmq/abc-yue2-003")
DEFAULT_FALLBACKS = {
    "3780908535368734195": (Path("/data2/cmq/abc-yue2-003-retry-vocals"), "vocals.wav"),
    "3780908535370668050": (Path("/data2/cmq/abc-yue2-003-retry-grid"),
                            "vocals_dereverb.wav (one unrepresentable note removed)"),
    "3780908535373745344": (Path("/data2/cmq/abc-yue2-003-retry-instrumental"), "instrumental.wav"),
    "3780908535378527761": (Path("/data2/cmq/abc-yue2-003-retry-instrumental"), "instrumental.wav"),
}


def finalize(inputs: Path, output: Path, fallbacks: dict = DEFAULT_FALLBACKS,
             copy_fallbacks: bool = False) -> dict:
    source_paths = collect_inputs(inputs)
    ids = {path.parent.name for path in source_paths}
    manifest = output / "batch_manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    latest = {}
    for row in rows:
        latest[Path(row["input"]).parent.name] = row
    missing = sorted(ids - {path.stem for path in output.glob("*.abc")})
    recovered = []
    if copy_fallbacks:
        for song_id in missing:
            if song_id not in fallbacks:
                continue
            source_root, stem = fallbacks[song_id]
            source = source_root / f"{song_id}.abc"
            if not source.is_file():
                continue
            import tempfile

            with tempfile.TemporaryDirectory(dir=str(output)) as temp:
                staged = Path(temp) / f"{song_id}.abc"
                shutil.copyfile(source, staged)
                check = check_and_repair_abc(staged)
                destination = output / f"{song_id}.abc"
                if destination.exists():
                    continue
                staged.replace(destination)
            entry = {"status": "recovered", "input": str(inputs / song_id / "vocals_dereverb.wav"),
                     "output": str(destination), "actual_audio_input": str(inputs / song_id / stem.split(" (")[0]),
                     "source_description": stem, "source_abc": str(source), "abc_check": check}
            with manifest.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
                stream.flush()
            recovered.append(entry)
        missing = sorted(ids - {path.stem for path in output.glob("*.abc")})
    unexpected = sorted({path.stem for path in output.glob("*.abc")} - ids)
    summary = {"inputs": len(ids), "abc_files": len(ids) - len(missing),
               "missing": missing, "unexpected_files": unexpected,
               "recovered_this_run": recovered,
               "recovery_records": sum(row["status"] == "recovered" for row in rows) + len(recovered),
               "last_known_failures": [{"id": song_id, "error": row.get("error")}
                                       for song_id, row in sorted(latest.items()) if row["status"] == "failed"
                                       and song_id in missing],
               "source": str(inputs), "output": str(output)}
    (output / "completion_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--copy-fallbacks", action="store_true")
    args = parser.parse_args()
    summary = finalize(args.input_root, args.output_root, copy_fallbacks=args.copy_fallbacks)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["missing"] or summary["unexpected_files"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
