#!/usr/bin/env python3
"""Resample the existing yue2-test2 cover inputs into a new output directory.

The original downloaded audio, lyrics, prompt, and SheetSage2 score are reused.
The YuE2 seed and prompt can be overridden independently.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess


ROOT = Path("/data/cmq/YuE")
RESULTS_ROOT = ROOT / "yue2-cover-results2"
OLD_REPORT = RESULTS_ROOT / "report.json"
DOWNLOADS = RESULTS_ROOT / "downloads"
OLD_REQUESTS = RESULTS_ROOT / "work" / "requests"
OLD_TRANSCRIPTIONS = RESULTS_ROOT / "work" / "transcriptions"
GENERATED = RESULTS_ROOT / "generated2"
WORK = RESULTS_ROOT / "work" / "generated2"
REQUESTS = WORK / "requests"
YUE2_RUNS = WORK / "yue2_runs"
REPORT = GENERATED / "report.json"

YUE2_PYTHON = ROOT / ".venv/bin/python"
RUN_YUE2_SCRIPT = ROOT / "skills/yue2-music/scripts/run_yue2.py"
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
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps({"summary": summary, "cases": cases}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


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


def available_work_dir(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.name}_attempt_{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Cannot find a fresh work directory for {path}")


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


def make_case(source_case: dict, request: dict, source_audio: Path, output_wav: Path, seed: int) -> dict:
    return {
        "row": source_case.get("row"),
        "source_url": source_case.get("source_url", ""),
        "prompt": request.get("style", ""),
        "lyrics_preview": str(request.get("lyrics", ""))[:200],
        "downloaded_audio_path": str(source_audio),
        "generated_output_path": str(output_wav),
        "resample_seed": seed,
        "status": "pending",
        "reason": "",
        "command_used": [],
        "started_at": None,
        "finished_at": None,
    }


def main() -> int:
    global GENERATED, WORK, REQUESTS, YUE2_RUNS, REPORT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed-offset", type=int, default=100000)
    parser.add_argument("--prompt", default=None, help="Override style/prompt for every request")
    parser.add_argument("--output-dir", type=Path, default=GENERATED)
    parser.add_argument("--work-dir", type=Path, default=WORK)
    parser.add_argument("--generate-timeout", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    GENERATED = args.output_dir
    WORK = args.work_dir
    REQUESTS = WORK / "requests"
    YUE2_RUNS = WORK / "yue2_runs"
    REPORT = GENERATED / "report.json"

    if not OLD_REPORT.is_file():
        raise FileNotFoundError(OLD_REPORT)
    old_report = json.loads(OLD_REPORT.read_text(encoding="utf-8"))
    source_cases = old_report.get("cases", [])
    if args.limit is not None:
        source_cases = source_cases[: args.limit]

    for directory in (GENERATED, WORK, REQUESTS, YUE2_RUNS):
        directory.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("HF_HOME", "/data/cmq/hf-cache")
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    model_flag = ["--offline"]

    cases: list[dict] = []
    for source_case in source_cases:
        source_audio = Path(source_case.get("downloaded_audio_path", ""))
        if not source_audio.is_file():
            stem = ""
            source_audio = DOWNLOADS / stem
        else:
            stem = source_audio.stem
        source_request = OLD_REQUESTS / f"{stem}.json"
        score = OLD_TRANSCRIPTIONS / stem / "score.abc"
        output_wav = GENERATED / f"{stem}.wav"

        try:
            request = json.loads(source_request.read_text(encoding="utf-8"))
        except Exception:
            request = {}
        original_seed = request.get("seed")
        if isinstance(original_seed, (int, float)) and not isinstance(original_seed, bool):
            seed = int(original_seed) + args.seed_offset
        else:
            seed = 932000 + int(source_case.get("row") or len(cases) + 2)
        request["seed"] = seed
        if args.prompt is not None:
            request["style"] = args.prompt
        request["id"] = request.get("id") or stem
        request_path = REQUESTS / f"{stem}.json"
        request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        case = make_case(source_case, request, source_audio, output_wav, seed)
        cases.append(case)

    write_report(cases)

    for case in cases:
        case["started_at"] = now()
        try:
            source_audio = Path(case["downloaded_audio_path"])
            stem = source_audio.stem
            request_path = REQUESTS / f"{stem}.json"
            score = OLD_TRANSCRIPTIONS / stem / "score.abc"
            output_wav = GENERATED / f"{stem}.wav"
            request = json.loads(request_path.read_text(encoding="utf-8"))

            missing = []
            if not source_audio.is_file():
                missing.append(f"audio={source_audio}")
            if not score.is_file():
                missing.append(f"score={score}")
            if not request.get("style"):
                missing.append("prompt/style")
            if not request.get("lyrics"):
                missing.append("lyrics")
            if missing:
                case["status"] = "skipped"
                case["reason"] = "Missing required resample input: " + ", ".join(missing)
                continue

            if output_wav.is_file() and output_wav.stat().st_size > 0 and not args.overwrite:
                case["status"] = "success"
                case["reason"] = "Reused existing generated2 wav"
                case["command_used"].append(f"reuse existing {output_wav}")
                continue

            yue2_dir = available_work_dir(YUE2_RUNS / stem)
            generated_flac = yue2_dir / "audio.flac"
            command = [
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
                "cuda",
            ]
            if YUE2_MODEL.exists():
                command.extend(["--model", str(YUE2_MODEL)])
            if YUE2_VAE.exists():
                command.extend(["--vae", str(YUE2_VAE)])
            command += model_flag
            case["command_used"].extend(
                [
                    f"reuse existing {score}",
                    " ".join(command),
                ]
            )
            code, output = run_command(command, args.generate_timeout, env)
            if code != 0 and not generated_flac.is_file():
                case["status"] = "failed"
                case["reason"] = f"YuE2 generation failed with exit {code}: {output.strip()}"
                continue
            if not generated_flac.is_file():
                case["status"] = "failed"
                case["reason"] = f"YuE2 did not create {generated_flac}"
                continue

            import soundfile as sf

            audio, sample_rate = sf.read(generated_flac, always_2d=True)
            output_wav.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_wav, audio, sample_rate, subtype="FLOAT")
            case["command_used"].append(f"soundfile convert {generated_flac} -> {output_wav}")
            case["status"] = "success"
            review = native_review_reason(yue2_dir)
            case["reason"] = review
        except subprocess.TimeoutExpired as exc:
            case["status"] = "failed"
            case["reason"] = f"Command timed out after {exc.timeout} seconds"
        except Exception as exc:
            case["status"] = "failed"
            case["reason"] = f"{type(exc).__name__}: {exc}"
        finally:
            case["finished_at"] = now()
            write_report(cases)

    summary = json.loads(REPORT.read_text(encoding="utf-8"))["summary"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
