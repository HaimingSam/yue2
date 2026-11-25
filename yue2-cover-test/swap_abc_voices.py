#!/usr/bin/env python3
"""Swap the native ABC Vocal and Ins voice bodies while preserving the grid."""

from __future__ import annotations

import argparse
from pathlib import Path


VOCAL = 'V: Vocal'
INS = 'V: Ins'


def swap(text: str) -> str:
    lines = text.splitlines()
    if len(lines) < 9:
        raise ValueError("ABC score is too short")
    output = lines[:8]
    cursor = 8
    while cursor < len(lines):
        if lines[cursor].startswith("% "):
            output.append(lines[cursor])
            cursor += 1
            continue
        if lines[cursor] != VOCAL:
            raise ValueError(f"Expected {VOCAL!r} at line {cursor + 1}")
        cursor += 1
        vocal_body = []
        while cursor < len(lines) and lines[cursor] != INS:
            if lines[cursor].startswith("% "):
                raise ValueError("Section comment appeared inside a voice group")
            vocal_body.append(lines[cursor])
            cursor += 1
        if cursor >= len(lines) or lines[cursor] != INS:
            raise ValueError("Missing Ins voice after Vocal voice")
        cursor += 1
        ins_body = []
        while cursor < len(lines) and lines[cursor] not in (VOCAL, INS) and not lines[cursor].startswith("% "):
            ins_body.append(lines[cursor])
            cursor += 1
        output.extend((VOCAL, *ins_body, INS, *vocal_body))
    return "\n".join(output) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(swap(args.source.read_text(encoding="utf-8")), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
