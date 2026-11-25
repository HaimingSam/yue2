#!/usr/bin/env python3
"""Batch YuE2 cover generation from yue2-test.xlsx.

This runner intentionally lives under the output directory. It orchestrates the
public helper scripts without changing YuE2 core code.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
import zipfile
import xml.etree.ElementTree as ET


ROOT = Path("/data/cmq/YuE")
EXCEL = ROOT / "yue2-test.xlsx"
OUTPUT_ROOT = ROOT / "yue2-cover-results"
DOWNLOADS = OUTPUT_ROOT / "downloads"
GENERATED = OUTPUT_ROOT / "generated"
WORK = OUTPUT_ROOT / "work"
REQUESTS = WORK / "requests"
TRANSCRIPTIONS = WORK / "transcriptions"
YUE2_RUNS = WORK / "yue2_runs"
REPORT = OUTPUT_ROOT / "report.json"

YUE2_PYTHON = ROOT / ".venv/bin/python"
SHEETSAGE_PYTHON = ROOT / ".venv-sheetsage2/bin/python"
TRANSCRIBE_SCRIPT = ROOT / "skills/yue2-music/scripts/transcribe.py"
RUN_YUE2_SCRIPT = ROOT / "skills/yue2-music/scripts/run_yue2.py"
SHEETSAGE_MODEL = ROOT / "models/SheetSage2"
SHEETSAGE_BASE_MODEL = ROOT / "models/MERT-v2-FullSong"
YUE2_MODEL = ROOT / "models/YuE2-3B"
YUE2_VAE = ROOT / "models/YuE2-Vae"

NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_report(cases: list[dict]) -> None:
    summary = {
        "total_rows": len(cases),
        "processed": sum(1 for case in cases if case["status"] != "skipped"),
        "succeeded": sum(1 for case in cases if case["status"] == "success"),
        "failed": sum(1 for case in cases if case["status"] == "failed"),
        "skipped": sum(1 for case in cases if case["status"] == "skipped"),
    }
    REPORT.write_text(
        json.dumps({"summary": summary, "cases": cases}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def cell_column(ref: str) -> int:
    match = re.match(r"([A-Z]+)", ref)
    if not match:
        raise ValueError(f"Invalid cell reference: {ref}")
    value = 0
    for char in match.group(1):
        value = value * 26 + ord(char) - 64
    return value


def parse_xlsx(path: Path) -> list[tuple[int, dict[int, str]]]:
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", NS):
                shared.append("".join(text.text or "" for text in item.findall(".//a:t", NS)))

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        first_sheet = workbook.find("a:sheets/a:sheet", NS)
        if first_sheet is None:
            raise ValueError("Workbook has no sheets")
        rel_id = first_sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = None
        for rel in rels.findall("rel:Relationship", NS):
            if rel.attrib.get("Id") == rel_id:
                target = rel.attrib["Target"]
                break
        if not target:
            raise ValueError("Cannot resolve first worksheet")
        sheet_path = "xl/" + target.lstrip("/") if not target.startswith("xl/") else target
        sheet = ET.fromstring(archive.read(sheet_path))

        def value(cell: ET.Element) -> str:
            cell_type = cell.attrib.get("t")
            if cell_type == "inlineStr":
                return "".join(text.text or "" for text in cell.findall(".//a:t", NS)).strip()
            node = cell.find("a:v", NS)
            if node is None:
                return ""
            text = node.text or ""
            if cell_type == "s":
                return shared[int(text)].strip() if text else ""
            return text.strip()

        rows: list[tuple[int, dict[int, str]]] = []
        for row in sheet.findall("a:sheetData/a:row", NS):
            values: dict[int, str] = {}
            for cell in row.findall("a:c", NS):
                values[cell_column(cell.attrib["r"])] = value(cell)
            rows.append((int(row.attrib.get("r", "0")), values))
        return rows


def slug_from_parts(row_number: int, label: str, url: str) -> str:
    candidates = [label]
    parsed = urllib.parse.urlparse(url)
    if parsed.path:
        candidates.append(urllib.parse.unquote(Path(parsed.path).stem))
    for candidate in candidates:
        text = Path(candidate.strip()).stem if candidate else ""
        text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
        if text:
            return f"row_{row_number}_{text[:80]}"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10] if url else ""
    return f"row_{row_number}_{digest}" if digest else f"row_{row_number}"


def extension_from_url(url: str) -> str:
    suffix = Path(urllib.parse.urlparse(url).path).suffix.lower()
    return suffix if suffix in {".mp3", ".m4a", ".wav", ".flac", ".aac", ".ogg"} else ".audio"


def download(url: str, destination: Path, timeout: int) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return
    tmp = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "yue2-cover-batch/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    tmp.replace(destination)


def run_command(command: list[str], timeout: int | None, env: dict[str, str]) -> tuple[int, str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )
    return completed.returncode, completed.stdout[-8000:]


def cuda_available() -> bool:
    command = [str(YUE2_PYTHON), "-c", "import torch; print(int(torch.cuda.is_available()))"]
    completed = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False)
    return completed.stdout.strip().endswith("1")


def convert_audio(source: Path, destination: Path) -> None:
    import soundfile as sf

    audio, sample_rate = sf.read(source, always_2d=True)
    sf.write(destination, audio, sample_rate, subtype="FLOAT")


def available_work_dir(base: Path) -> Path:
    if not base.exists() or not any(base.iterdir()):
        return base
    for index in range(2, 1000):
        candidate = base.with_name(f"{base.name}_attempt_{index}")
        if not candidate.exists() or not any(candidate.iterdir()):
            return candidate
    raise RuntimeError(f"Cannot find a fresh work directory for {base}")


def native_review_reason(yue2_dir: Path) -> str:
    result_path = yue2_dir / "result.json"
    if not result_path.is_file():
        return ""
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    truncated = result.get("truncated")
    if isinstance(truncated, dict) and any(truncated.values()):
        return f"Native YuE2 result has review flags: truncated={truncated}"
    return ""


def prepare_case(row_number: int, row: dict[int, str]) -> dict:
    url = row.get(2, "").strip()
    prompt = row.get(3, "").strip()
    lyrics = row.get(4, "").strip()
    return {
        "row": row_number,
        "source_url": url,
        "prompt": prompt,
        "lyrics_preview": lyrics[:200],
        "downloaded_audio_path": None,
        "generated_output_path": None,
        "status": "skipped",
        "reason": "",
        "command_used": [],
        "started_at": now(),
        "finished_at": None,
        "_lyrics": lyrics,
        "_label": row.get(1, "").strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Process only the first N data rows")
    parser.add_argument("--start-row", type=int, default=2)
    parser.add_argument("--download-timeout", type=int, default=300)
    parser.add_argument("--transcribe-timeout", type=int, default=None)
    parser.add_argument("--generate-timeout", type=int, default=None)
    parser.add_argument("--require-cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--online-models", action="store_true", help="Allow HF model downloads during helper calls")
    parser.add_argument("--sheetsage-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--yue2-device", default="cuda")
    args = parser.parse_args()

    for directory in (DOWNLOADS, GENERATED, WORK, REQUESTS, TRANSCRIPTIONS, YUE2_RUNS):
        directory.mkdir(parents=True, exist_ok=True)

    rows = parse_xlsx(EXCEL)
    if not rows:
        raise ValueError("No rows in workbook")
    headers = rows[0][1]
    expected = {2: "歌曲链接", 3: "prompt", 4: "歌词"}
    mismatches = {column: headers.get(column, "") for column, name in expected.items() if name.lower() not in headers.get(column, "").lower()}
    if mismatches:
        raise ValueError(f"Unexpected headers in B/C/D: {mismatches}; all headers: {headers}")

    data_rows = [(row_number, row) for row_number, row in rows if row_number >= args.start_row]
    if args.limit is not None:
        data_rows = data_rows[: args.limit]

    env = os.environ.copy()
    if "HF_HOME" not in env:
        shared_cache = Path("/data/cmq/hf-cache")
        env["HF_HOME"] = str(shared_cache if shared_cache.exists() else OUTPUT_ROOT / "hf-cache")
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    if args.online_models:
        model_flag: list[str] = []
    else:
        model_flag = ["--offline"]

    has_cuda = cuda_available()
    cases: list[dict] = []
    for row_number, row in data_rows:
        case = prepare_case(row_number, row)
        try:
            missing = []
            if not case["source_url"]:
                missing.append("source_url")
            if not case["prompt"]:
                missing.append("prompt")
            if not case["_lyrics"]:
                missing.append("lyrics")
            if missing:
                case["status"] = "skipped"
                case["reason"] = "Missing required fields: " + ", ".join(missing)
                continue

            slug = slug_from_parts(row_number, case["_label"], case["source_url"])
            audio_path = DOWNLOADS / f"{slug}{extension_from_url(case['source_url'])}"
            wav_path = GENERATED / f"{slug}.wav"
            request_path = REQUESTS / f"{slug}.json"
            transcription_dir = TRANSCRIPTIONS / slug
            yue2_dir = YUE2_RUNS / slug
            generated_flac = yue2_dir / "audio.flac"

            download_command = f"download {case['source_url']} -> {audio_path}"
            case["command_used"].append(download_command)
            download(case["source_url"], audio_path, args.download_timeout)
            case["downloaded_audio_path"] = str(audio_path)

            if wav_path.is_file() and wav_path.stat().st_size > 0:
                case["command_used"].append(f"reuse existing {wav_path}")
                case["generated_output_path"] = str(wav_path)
                case["status"] = "success"
                review = native_review_reason(yue2_dir)
                case["reason"] = "Reused existing generated wav" + (f"; {review}" if review else "")
                continue

            if args.require_cuda and not has_cuda:
                case["status"] = "failed"
                case["reason"] = "CUDA unavailable in current environment; YuE2 cover generation requires a BF16 NVIDIA GPU baseline"
                continue

            request = {
                "id": slug,
                "style": case["prompt"],
                "lyrics": case["_lyrics"],
                "cot": "melody",
                "seed": 831001 + row_number,
            }
            request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            score = transcription_dir / "score.abc"
            if score.is_file():
                case["command_used"].append(f"reuse existing {score}")
            else:
                transcription_dir = available_work_dir(transcription_dir)
                score = transcription_dir / "score.abc"
                transcribe_command = [
                    str(SHEETSAGE_PYTHON),
                    str(TRANSCRIBE_SCRIPT),
                    str(audio_path),
                    "--output",
                    str(transcription_dir),
                    "--task",
                    "melody-full",
                    "--model",
                    str(SHEETSAGE_MODEL),
                    "--device",
                    args.sheetsage_device,
                    "--dtype",
                    "bf16" if args.sheetsage_device == "cuda" else "fp32",
                ] + model_flag
                if SHEETSAGE_BASE_MODEL.exists():
                    transcribe_command.extend(["--base-model", str(SHEETSAGE_BASE_MODEL)])
                case["command_used"].append(" ".join(transcribe_command))
                code, output = run_command(transcribe_command, args.transcribe_timeout, env)
                if code != 0:
                    case["status"] = "failed"
                    case["reason"] = f"SheetSage2 transcription failed with exit {code}: {output.strip()}"
                    continue

            if not score.is_file():
                case["status"] = "failed"
                case["reason"] = f"SheetSage2 did not create {score}"
                continue

            if generated_flac.is_file():
                case["command_used"].append(f"reuse existing {generated_flac}")
                generation_warning = ""
            else:
                yue2_dir = available_work_dir(yue2_dir)
                generated_flac = yue2_dir / "audio.flac"
                generation_warning = ""
                generate_command = [
                    str(YUE2_PYTHON),
                    str(RUN_YUE2_SCRIPT),
                    "generate",
                    "--request",
                    str(request_path),
                    "--cot",
                    "melody",
                    "--abc-file",
                    str(score),
                    "--output",
                    str(yue2_dir),
                    "--device",
                    args.yue2_device,
                ]
                if YUE2_MODEL.exists():
                    generate_command.extend(["--model", str(YUE2_MODEL)])
                if YUE2_VAE.exists():
                    generate_command.extend(["--vae", str(YUE2_VAE)])
                generate_command += model_flag
                case["command_used"].append(" ".join(generate_command))
                code, output = run_command(generate_command, args.generate_timeout, env)
                if code != 0:
                    if generated_flac.is_file():
                        generation_warning = (
                            f"YuE2 helper returned exit {code}, but audio.flac was produced; "
                            f"check native run artifacts for review flags. Tail: {output.strip()}"
                        )
                    else:
                        case["status"] = "failed"
                        case["reason"] = f"YuE2 cover generation failed with exit {code}: {output.strip()}"
                        continue

            if not generated_flac.is_file():
                case["status"] = "failed"
                case["reason"] = f"YuE2 did not create {generated_flac}"
                continue

            convert_command = f"soundfile convert {generated_flac} -> {wav_path}"
            case["command_used"].append(convert_command)
            convert_audio(generated_flac, wav_path)
            case["generated_output_path"] = str(wav_path)
            case["status"] = "success"
            review = native_review_reason(yue2_dir)
            case["reason"] = "; ".join(part for part in (generation_warning, review) if part)
        except subprocess.TimeoutExpired as exc:
            case["status"] = "failed"
            case["reason"] = f"Command timed out after {exc.timeout} seconds"
        except Exception as exc:
            case["status"] = "failed"
            case["reason"] = f"{type(exc).__name__}: {exc}"
        finally:
            case["finished_at"] = now()
            case.pop("_lyrics", None)
            case.pop("_label", None)
            cases.append(case)
            write_report(cases)

    write_report(cases)
    summary = json.loads(REPORT.read_text(encoding="utf-8"))["summary"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
