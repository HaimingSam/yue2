"""Build targets that use an exact input-section placeholder."""

from __future__ import annotations

from pathlib import Path


PLACEHOLDER = "<INPUT_SECTION>"
CONTROL = "<COPY_INPUT_SECTION>"


def section_ranges(lines: list[str]):
    markers = [(index, line[2:].strip().lower()) for index, line in enumerate(lines)
               if line.startswith("% ")]
    for occurrence, (start, label) in enumerate(markers, 1):
        end = markers[occurrence][0] if occurrence < len(markers) else len(lines)
        yield occurrence, label, start, end


def first_section_body(text: str) -> tuple[str, list[str]]:
    lines = text.splitlines()
    for _, label, start, end in section_ranges(lines):
        return label, lines[start + 1:end]
    raise ValueError("Input has no section marker")


def scaffold_target(target: str, input_text: str, label: str, occurrence: int) -> str:
    target_lines = target.splitlines()
    input_label, input_body = first_section_body(input_text)
    if input_label != label.lower():
        raise ValueError(f"Input section {input_label!r} differs from expected {label!r}")
    matches = [(start, end) for item, found, start, end in section_ranges(target_lines)
               if found == label.lower() for _ in [item]]
    if len(matches) < occurrence:
        raise ValueError(f"Target has fewer than {occurrence} {label!r} sections")
    start, end = matches[occurrence - 1]
    result = [CONTROL] + target_lines[:start] + [target_lines[start], PLACEHOLDER] + target_lines[end:]
    if PLACEHOLDER not in result:
        raise ValueError("Failed to insert section placeholder")
    return "\n".join(result) + "\n"


def restore_input_section(generated: str, input_text: str, strict: bool = True) -> tuple[str, str]:
    label, input_body = first_section_body(input_text)
    lines = generated.splitlines()
    if lines and lines[0] == CONTROL:
        lines = lines[1:]
    if PLACEHOLDER in lines:
        index = lines.index(PLACEHOLDER)
        result = lines[:index] + input_body + lines[index + 1:]
        return "\n".join(result) + "\n", "placeholder"
    if strict:
        raise ValueError(f"Generated output does not contain {PLACEHOLDER}")
    for _, found, start, end in section_ranges(lines):
        if found == label:
            result = lines[:start] + input_body + lines[end:]
            return "\n".join(result) + "\n", "section-fallback"
    result = lines + input_body
    return "\n".join(result) + "\n", "append-fallback"
