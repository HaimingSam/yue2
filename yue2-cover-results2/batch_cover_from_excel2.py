#!/usr/bin/env python3
"""Batch YuE2 cover generation from yue2-test2.xlsx.

Column mapping for this workbook:
  A: source song URL
  B: lyrics
  C: prompt/style
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess


ROOT = Path("/data/cmq/YuE")
BASE_SCRIPT = ROOT / "yue2-cover-results" / "batch_cover_from_excel.py"
EXCEL = ROOT / "yue2-test2.xlsx"
OUTPUT_ROOT = ROOT / "yue2-cover-results2"
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


def load_base_module():
    spec = importlib.util.spec_from_file_location("yue2_cover_batch_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helper script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()


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


def header_matches(headers: dict[int, str]) -> bool:
    column_a = headers.get(1, "").strip().lower()
    column_b = headers.get(2, "").strip().lower()
    column_c = headers.get(3, "").strip().lower()
    link_ok = any(token in column_a for token in ("链接", "url", "song", "audio", "线上数据", "数据"))
    lyrics_ok = any(token in column_b for token in ("歌词", "lyrics", "lyric", "asr"))
    prompt_ok = any(token in column_c for token in ("prompt", "style", "风格"))
    return link_ok and lyrics_ok and prompt_ok


def prepare_case(row_number: int, row: dict[int, str]) -> dict:
    url = row.get(1, "").strip()
    lyrics = row.get(2, "").strip()
    prompt = row.get(3, "").strip()
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
        "started_at": base.now(),
        "finished_at": None,
        "_lyrics": lyrics,
        "_label": "",
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

    rows = base.parse_xlsx(EXCEL)
    if not rows:
        raise ValueError("No rows in workbook")
    headers = rows[0][1]
    if not header_matches(headers):
        raise ValueError(f"Unexpected headers in A/B/C: {headers}")

    data_rows = [(row_number, row) for row_number, row in rows if row_number >= args.start_row]
    if args.limit is not None:
        data_rows = data_rows[: args.limit]

    env = os.environ.copy()
    if "HF_HOME" not in env:
        shared_cache = Path("/data/cmq/hf-cache")
        env["HF_HOME"] = str(shared_cache if shared_cache.exists() else OUTPUT_ROOT / "hf-cache")
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    model_flag: list[str] = [] if args.online_models else ["--offline"]

    has_cuda = base.cuda_available()
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

            slug = base.slug_from_parts(row_number, case["_label"], case["source_url"])
            audio_path = DOWNLOADS / f"{slug}{base.extension_from_url(case['source_url'])}"
            wav_path = GENERATED / f"{slug}.wav"
            request_path = REQUESTS / f"{slug}.json"
            transcription_dir = TRANSCRIPTIONS / slug
            yue2_dir = YUE2_RUNS / slug
            generated_flac = yue2_dir / "audio.flac"

            case["command_used"].append(f"download {case['source_url']} -> {audio_path}")
            base.download(case["source_url"], audio_path, args.download_timeout)
            case["downloaded_audio_path"] = str(audio_path)

            if wav_path.is_file() and wav_path.stat().st_size > 0:
                case["command_used"].append(f"reuse existing {wav_path}")
                case["generated_output_path"] = str(wav_path)
                case["status"] = "success"
                review = base.native_review_reason(yue2_dir)
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
                "seed": 832001 + row_number,
            }
            request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            score = transcription_dir / "score.abc"
            if score.is_file():
                case["command_used"].append(f"reuse existing {score}")
            else:
                transcription_dir = base.available_work_dir(transcription_dir)
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
                code, output = base.run_command(transcribe_command, args.transcribe_timeout, env)
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
                yue2_dir = base.available_work_dir(yue2_dir)
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
                code, output = base.run_command(generate_command, args.generate_timeout, env)
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

            case["command_used"].append(f"soundfile convert {generated_flac} -> {wav_path}")
            base.convert_audio(generated_flac, wav_path)
            case["generated_output_path"] = str(wav_path)
            case["status"] = "success"
            review = base.native_review_reason(yue2_dir)
            case["reason"] = "; ".join(part for part in (generation_warning, review) if part)
        except subprocess.TimeoutExpired as exc:
            case["status"] = "failed"
            case["reason"] = f"Command timed out after {exc.timeout} seconds"
        except Exception as exc:
            case["status"] = "failed"
            case["reason"] = f"{type(exc).__name__}: {exc}"
        finally:
            case["finished_at"] = base.now()
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
