#!/usr/bin/env python3
"""Batch YuE2 cover generation from downloaded audio and request JSON files."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess


ROOT = Path("/data/cmq/YuE")
OUTPUT_ROOT = ROOT / "yue2-cover-results"
DEFAULT_AUDIO_DIR = OUTPUT_ROOT / "downloads"
DEFAULT_REQUEST_DIR = OUTPUT_ROOT / "work" / "requests"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "generated2"
DEFAULT_WORK_DIR = OUTPUT_ROOT / "work" / "generated2"

YUE2_PYTHON = ROOT / ".venv/bin/python"
SHEETSAGE_PYTHON = ROOT / ".venv-sheetsage2/bin/python"
TRANSCRIBE_SCRIPT = ROOT / "skills/yue2-music/scripts/transcribe.py"
RUN_YUE2_SCRIPT = ROOT / "skills/yue2-music/scripts/run_yue2.py"
SHEETSAGE_MODEL = ROOT / "models/SheetSage2"
SHEETSAGE_BASE_MODEL = ROOT / "models/MERT-v2-FullSong"
YUE2_MODEL = ROOT / "models/YuE2-3B"
YUE2_VAE = ROOT / "models/YuE2-Vae"

AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".flac", ".aac", ".ogg"}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def row_number(name: str) -> int:
    match = re.match(r"row_(\d+)(?:_|$)", name)
    return int(match.group(1)) if match else 10**9


def case_sort_key(name: str) -> tuple[int, str]:
    return (row_number(name), name)


def write_report(report_path: Path, cases: list[dict]) -> None:
    summary = {
        "total_rows": len(cases),
        "processed": sum(1 for case in cases if case["status"] != "skipped"),
        "succeeded": sum(1 for case in cases if case["status"] == "success"),
        "failed": sum(1 for case in cases if case["status"] == "failed"),
        "skipped": sum(1 for case in cases if case["status"] == "skipped"),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
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


def cuda_available() -> bool:
    command = [str(YUE2_PYTHON), "-c", "import torch; print(int(torch.cuda.is_available()))"]
    completed = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False)
    return completed.stdout.strip().endswith("1")


def convert_audio(source: Path, destination: Path) -> None:
    import soundfile as sf

    audio, sample_rate = sf.read(source, always_2d=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, audio, sample_rate, subtype="FLOAT")


def available_work_dir(base: Path) -> Path:
    if not base.exists():
        return base
    for index in range(2, 1000):
        candidate = base.with_name(f"{base.name}_attempt_{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Cannot find a fresh work directory for {base}")


def load_requests(request_dir: Path) -> dict[str, Path]:
    requests: dict[str, Path] = {}
    for path in sorted(request_dir.glob("*.json")):
        if path.stem in requests:
            raise ValueError(f"Duplicate request stem: {path.stem}")
        requests[path.stem] = path
    return requests


def load_audio(audio_dir: Path) -> dict[str, Path]:
    audio: dict[str, Path] = {}
    for path in sorted(audio_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTS:
            continue
        if path.stem in audio:
            raise ValueError(f"Duplicate audio stem: {path.stem}")
        audio[path.stem] = path
    return audio


def read_request(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    prompt = (data.get("style") or data.get("prompt") or data.get("tags") or "").strip()
    lyrics = (data.get("lyrics") or data.get("lyric") or "").strip()
    request = {
        "id": (data.get("id") or path.stem).strip(),
        "style": prompt,
        "lyrics": lyrics,
        "cot": (data.get("cot") or "melody").strip(),
    }
    if data.get("seed") is not None:
        request["seed"] = data["seed"]
    if data.get("cfg_scale") is not None:
        request["cfg_scale"] = data["cfg_scale"]
    return request


def find_existing_score(stem: str, generated2_work: Path, legacy_work: Path) -> Path | None:
    search_roots = [
        generated2_work / "transcriptions",
        legacy_work / "transcriptions",
        legacy_work / "yue2_runs",
    ]
    for root in search_roots:
        exact = root / stem / "score.abc"
        if exact.is_file():
            return exact
    candidates: list[Path] = []
    for root in search_roots:
        if root.is_dir():
            candidates.extend(path / "score.abc" for path in root.glob(f"{stem}*") if (path / "score.abc").is_file())
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--request-dir", type=Path, default=DEFAULT_REQUEST_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--online-models", action="store_true", help="Allow model downloads during helper calls")
    parser.add_argument("--sheetsage-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--yue2-device", default="cuda")
    parser.add_argument("--transcribe-timeout", type=int, default=None)
    parser.add_argument("--generate-timeout", type=int, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report or args.output_dir / "report.json"

    audio_by_stem = load_audio(args.audio_dir)
    requests_by_stem = load_requests(args.request_dir)
    stems = sorted(set(audio_by_stem) | set(requests_by_stem), key=case_sort_key)
    if args.limit is not None:
        stems = stems[: args.limit]

    env = os.environ.copy()
    if "HF_HOME" not in env:
        shared_cache = Path("/data/cmq/hf-cache")
        env["HF_HOME"] = str(shared_cache if shared_cache.exists() else OUTPUT_ROOT / "hf-cache")
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    model_flag: list[str] = [] if args.online_models else ["--offline"]
    has_cuda = cuda_available()

    generated2_requests = args.work_dir / "requests"
    generated2_transcriptions = args.work_dir / "transcriptions"
    generated2_yue2_runs = args.work_dir / "yue2_runs"
    for directory in (generated2_requests, generated2_transcriptions, generated2_yue2_runs):
        directory.mkdir(parents=True, exist_ok=True)

    cases: list[dict] = []
    for stem in stems:
        audio_path = audio_by_stem.get(stem)
        source_request_path = requests_by_stem.get(stem)
        case = {
            "row": row_number(stem) if row_number(stem) != 10**9 else None,
            "id": stem,
            "source_audio_path": str(audio_path) if audio_path else None,
            "request_path": str(source_request_path) if source_request_path else None,
            "prompt": "",
            "lyrics_preview": "",
            "generated_output_path": None,
            "status": "skipped",
            "reason": "",
            "command_used": [],
            "started_at": now(),
            "finished_at": None,
        }
        try:
            if audio_path is None:
                case["reason"] = f"No matching audio file in {args.audio_dir}"
                continue
            if source_request_path is None:
                case["reason"] = f"No matching request JSON in {args.request_dir}"
                continue

            request = read_request(source_request_path)
            case["prompt"] = request["style"]
            case["lyrics_preview"] = request["lyrics"][:200]
            missing = []
            if not request["style"]:
                missing.append("prompt/style")
            if not request["lyrics"]:
                missing.append("lyrics")
            if request["cot"] != "melody":
                missing.append("cot=melody")
            if missing:
                case["reason"] = "Missing or unsupported required fields: " + ", ".join(missing)
                continue

            output_wav = args.output_dir / f"{stem}.wav"
            case["generated_output_path"] = str(output_wav)
            if output_wav.is_file() and output_wav.stat().st_size > 0 and not args.overwrite:
                case["status"] = "success"
                case["reason"] = "Reused existing generated wav"
                case["command_used"].append(f"reuse existing {output_wav}")
                continue

            if args.require_cuda and not has_cuda:
                case["status"] = "failed"
                case["reason"] = "CUDA unavailable in current environment"
                continue

            request_path = generated2_requests / f"{stem}.json"
            request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            score = find_existing_score(stem, args.work_dir, OUTPUT_ROOT / "work")
            if score is not None:
                case["command_used"].append(f"reuse existing {score}")
            else:
                transcription_dir = available_work_dir(generated2_transcriptions / stem)
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

            yue2_dir = available_work_dir(generated2_yue2_runs / stem)
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
            generation_warning = ""
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

            case["command_used"].append(f"soundfile convert {generated_flac} -> {output_wav}")
            convert_audio(generated_flac, output_wav)
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
            cases.append(case)
            write_report(report_path, cases)

    write_report(report_path, cases)
    summary = json.loads(report_path.read_text(encoding="utf-8"))["summary"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
