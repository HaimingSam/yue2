#!/usr/bin/env python3
"""Batch YuE2 cover generation from /data/cmq/datas/audio/fl/* inputs."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess


ROOT = Path("/data/cmq/YuE")
INPUT_ROOT = Path("/data/cmq/datas/audio/fl")
OUTPUT_ROOT = ROOT / "yue2-cover-results3"
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


def safe_slug(index: int, source_dir: Path) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", source_dir.name).strip("._-")
    if not text:
        text = hashlib.sha1(str(source_dir).encode("utf-8")).hexdigest()[:10]
    return f"row_{index}_{text[:80]}"


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


def available_work_dir(base: Path) -> Path:
    if not base.exists():
        return base
    for index in range(2, 1000):
        candidate = base.with_name(f"{base.name}_attempt_{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Cannot find a fresh work directory for {base}")


def convert_audio(source: Path, destination: Path) -> None:
    import soundfile as sf

    audio, sample_rate = sf.read(source, always_2d=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, audio, sample_rate, subtype="FLOAT")


def detect_language(lyrics: str) -> str:
    cjk = sum(1 for char in lyrics if "\u4e00" <= char <= "\u9fff")
    latin = sum(1 for char in lyrics if char.isascii() and char.isalpha())
    lowered = lyrics.lower()
    spanish_hits = sum(
        token in lowered
        for token in (" que ", " si ", " corazón", " vida", " amor", " dolor", " no ", " la ", " el ", " los ")
    )
    if cjk > max(20, latin // 3):
        return "Chinese"
    if spanish_hits >= 3 or any(char in lowered for char in "áéíóúñ¿¡"):
        return "Spanish"
    if latin:
        return "English"
    return "Unknown"


PROMPTS = {
    "Chinese": [
        "Mandarin pop ballad, emotional lead vocal, warm piano, soft drums, lush strings, polished modern C-pop production, heartfelt chorus",
        "Chinese pop rock ballad, expressive vocal delivery, clean guitars, steady drums, warm bass, dramatic chorus, radio-ready arrangement",
        "Mandarin acoustic pop, intimate vocal, gentle guitar, piano pads, restrained percussion, bittersweet emotional atmosphere",
        "Chinese R&B pop ballad, smooth vocal phrasing, warm keys, soft groove, layered harmonies, clean contemporary mix",
        "Mandarin cinematic pop, tender verse, soaring chorus, piano and strings, emotional late-night atmosphere",
    ],
    "Spanish": [
        "Spanish Latin pop ballad, emotional male vocal, warm acoustic guitar, soft percussion, dramatic chorus, polished romantic production",
        "Latin pop rock, Spanish vocal, driving drums, melodic guitars, heartfelt chorus, clean modern radio mix",
        "Spanish urban ballad, expressive vocal delivery, atmospheric keys, subtle reggaeton pulse, emotional hook, polished mix",
        "Spanish acoustic pop, intimate vocal, nylon guitar, warm bass, gentle percussion, melancholic storytelling mood",
    ],
    "English": [
        "English pop ballad, expressive lead vocal, warm piano, soft drums, subtle pads, emotional chorus, clean modern production",
        "English acoustic pop rock, intimate vocal, guitar-driven arrangement, steady drums, heartfelt chorus, polished mix",
        "English R&B pop, smooth vocal, warm keys, soft groove, layered harmonies, contemporary clean mix",
    ],
    "Unknown": [
        "emotional pop ballad, expressive lead vocal, warm piano, soft drums, melodic bass, polished modern arrangement",
        "acoustic pop, intimate vocal, gentle guitar, subtle percussion, heartfelt chorus, clean mix",
    ],
}


def random_prompt(language: str, rng: random.Random) -> str:
    return rng.choice(PROMPTS.get(language, PROMPTS["Unknown"]))


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


def collect_inputs() -> list[tuple[Path, Path]]:
    cases: list[tuple[Path, Path]] = []
    for audio_path in sorted(INPUT_ROOT.glob("*/vocals_dereverb.wav")):
        lyrics_path = audio_path.parent / "lyrics.txt"
        cases.append((audio_path, lyrics_path))
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prompt-seed", type=int, default=None)
    parser.add_argument("--base-yue-seed", type=int, default=933000)
    parser.add_argument("--transcribe-timeout", type=int, default=None)
    parser.add_argument("--generate-timeout", type=int, default=None)
    parser.add_argument("--online-models", action="store_true")
    parser.add_argument("--sheetsage-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--yue2-device", default="cuda")
    args = parser.parse_args()

    for directory in (GENERATED, WORK, REQUESTS, TRANSCRIPTIONS, YUE2_RUNS):
        directory.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("HF_HOME", "/data/cmq/hf-cache")
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    model_flag: list[str] = [] if args.online_models else ["--offline"]
    rng = random.Random(args.prompt_seed if args.prompt_seed is not None else int(dt.datetime.now().timestamp() * 1000))

    inputs = collect_inputs()
    if args.limit is not None:
        inputs = inputs[: args.limit]

    cases: list[dict] = []
    for index, (audio_path, lyrics_path) in enumerate(inputs, start=1):
        slug = safe_slug(index, audio_path.parent)
        lyrics = lyrics_path.read_text(encoding="utf-8", errors="replace").strip() if lyrics_path.is_file() else ""
        language = detect_language(lyrics)
        prompt = random_prompt(language, rng)
        request_path = REQUESTS / f"{slug}.json"
        score_dir = TRANSCRIPTIONS / slug
        score = score_dir / "score.abc"
        yue2_dir = YUE2_RUNS / slug
        generated_flac = yue2_dir / "audio.flac"
        wav_path = GENERATED / f"{slug}.wav"
        seed = args.base_yue_seed + index

        case = {
            "row": index,
            "id": slug,
            "source_audio_path": str(audio_path),
            "lyrics_path": str(lyrics_path),
            "detected_language": language,
            "prompt": prompt,
            "lyrics_preview": lyrics[:200],
            "downloaded_audio_path": str(audio_path),
            "generated_output_path": str(wav_path),
            "status": "pending",
            "reason": "",
            "command_used": [],
            "started_at": now(),
            "finished_at": None,
        }
        try:
            missing = []
            if not audio_path.is_file():
                missing.append("source_audio")
            if not lyrics:
                missing.append("lyrics")
            if missing:
                case["status"] = "skipped"
                case["reason"] = "Missing required fields: " + ", ".join(missing)
                continue

            request = {
                "id": slug,
                "style": prompt,
                "lyrics": lyrics,
                "cot": "melody",
                "seed": seed,
            }
            request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            if wav_path.is_file() and wav_path.stat().st_size > 0 and not args.overwrite:
                case["command_used"].append(f"reuse existing {wav_path}")
                case["status"] = "success"
                case["reason"] = "Reused existing generated wav"
                continue

            if score.is_file():
                case["command_used"].append(f"reuse existing {score}")
            else:
                score_dir = available_work_dir(score_dir)
                score = score_dir / "score.abc"
                transcribe_command = [
                    str(SHEETSAGE_PYTHON),
                    str(TRANSCRIBE_SCRIPT),
                    str(audio_path),
                    "--output",
                    str(score_dir),
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

            if generated_flac.is_file() and not args.overwrite:
                case["command_used"].append(f"reuse existing {generated_flac}")
            else:
                yue2_dir = available_work_dir(yue2_dir) if args.overwrite or yue2_dir.exists() else yue2_dir
                generated_flac = yue2_dir / "audio.flac"
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
                if code != 0 and not generated_flac.is_file():
                    case["status"] = "failed"
                    case["reason"] = f"YuE2 cover generation failed with exit {code}: {output.strip()}"
                    continue

            if not generated_flac.is_file():
                case["status"] = "failed"
                case["reason"] = f"YuE2 did not create {generated_flac}"
                continue

            case["command_used"].append(f"soundfile convert {generated_flac} -> {wav_path}")
            convert_audio(generated_flac, wav_path)
            case["generated_output_path"] = str(wav_path)
            case["status"] = "success"
            case["reason"] = native_review_reason(yue2_dir)
        except subprocess.TimeoutExpired as exc:
            case["status"] = "failed"
            case["reason"] = f"Command timed out after {exc.timeout} seconds"
        except Exception as exc:
            case["status"] = "failed"
            case["reason"] = f"{type(exc).__name__}: {exc}"
        finally:
            case["finished_at"] = now()
            cases.append(case)
            write_report(cases)

    write_report(cases)
    summary = json.loads(REPORT.read_text(encoding="utf-8"))["summary"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
