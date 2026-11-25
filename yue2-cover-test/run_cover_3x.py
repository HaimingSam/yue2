#!/usr/bin/env python3
"""Run three YuE2 cover generations for each yue2-cover-test request."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import soundfile as sf


ROOT = Path("/data/cmq/YuE")
TEST_ROOT = ROOT / "yue2-cover-test"
REQUEST_DIR = TEST_ROOT / "work" / "requests"
OUTPUT_DIR = TEST_ROOT / "generated"
WORK_DIR = TEST_ROOT / "work" / "generated"
REPORT_PATH = OUTPUT_DIR / "report.json"
SCRIPT_DIR = ROOT / "skills" / "yue2-music" / "scripts"
YUE2_PYTHON = ROOT / ".venv" / "bin" / "python"
RUN_YUE2_SCRIPT = SCRIPT_DIR / "run_yue2.py"
RUNNER_CODE = """
import os
import runpy
import sys

os.environ["HF_HOME"] = "/data/cmq/hf-cache"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
sys.path.insert(0, "/data/cmq/YuE/skills/yue2-music/scripts")
sys.argv = [
    "run_yue2.py",
    "generate",
    "--request", sys.argv[1],
    "--cot", "melody",
    "--abc-file", sys.argv[2],
    "--output", sys.argv[3],
    "--device", "cuda",
    "--offline",
]
runpy.run_path("/data/cmq/YuE/skills/yue2-music/scripts/run_yue2.py", run_name="__main__")
""".strip()

SCORE_BY_STEM = {
    "88ef_lyx1": TEST_ROOT / "work" / "transcriptions" / "88ef_lyx1_attempt_2" / "score.abc",
    "a424_hbc1": TEST_ROOT / "work" / "transcriptions" / "a424_hbc1" / "score.abc",
    "bfed_sly2": TEST_ROOT / "work" / "transcriptions" / "bfed_sly2" / "score.abc",
    "f8e2_hjf3": TEST_ROOT / "work" / "transcriptions" / "f8e2_hjf3" / "score.abc",
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def write_report(cases: list[dict]) -> None:
    summary = {
        "total": len(cases),
        "complete": sum(1 for case in cases if case.get("status") == "complete"),
        "failed": sum(1 for case in cases if case.get("status") == "failed"),
        "needs_review": sum(1 for case in cases if case.get("status") == "needs_review"),
    }
    write_json(REPORT_PATH, {"summary": summary, "cases": cases})


def convert_audio(source: Path, destination: Path) -> None:
    audio, sample_rate = sf.read(source, always_2d=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, audio, sample_rate, subtype="FLOAT")


def candidate_run_dirs(run_root: Path, variant_id: str) -> list[Path]:
    candidates = [run_root / variant_id]
    candidates.extend(sorted(run_root.glob(f"{variant_id}_attempt_*")))
    return candidates


def completed_run_dir(run_root: Path, variant_id: str) -> Path | None:
    for candidate in candidate_run_dirs(run_root, variant_id):
        if (candidate / "audio.flac").is_file():
            return candidate
    return None


def available_run_dir(run_root: Path, variant_id: str) -> Path:
    base = run_root / variant_id
    if not base.exists():
        return base
    for index in range(2, 1000):
        candidate = run_root / f"{variant_id}_attempt_{index}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Cannot find a fresh run directory for {variant_id}")


def run_generation(request_path: Path, score_path: Path, run_dir: Path) -> int:
    env = os.environ.copy()
    env.pop("HF_HOME", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    command = [
        str(YUE2_PYTHON),
        "-c",
        RUNNER_CODE,
        str(request_path),
        str(score_path),
        str(run_dir),
    ]
    completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
    return completed.returncode


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    request_out = WORK_DIR / "requests"
    run_root = WORK_DIR / "yue2_runs"
    request_out.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    cases: list[dict] = []
    for source_request in sorted(REQUEST_DIR.glob("*.json")):
        stem = source_request.stem
        score_path = SCORE_BY_STEM.get(stem)
        if score_path is None or not score_path.is_file():
            cases.append({"id": stem, "status": "failed", "reason": f"Missing score.abc for {stem}"})
            write_report(cases)
            continue

        base_request = read_json(source_request)
        base_seed = int(base_request.get("seed", 100000))
        for variant in range(1, 4):
            variant_id = f"{stem}_v{variant}"
            request = dict(base_request)
            request["id"] = variant_id
            request["seed"] = base_seed + variant - 1
            request_path = request_out / f"{variant_id}.json"
            wav_path = OUTPUT_DIR / f"{variant_id}.wav"
            existing_run_dir = completed_run_dir(run_root, variant_id)
            run_dir = existing_run_dir or available_run_dir(run_root, variant_id)
            case = {
                "id": variant_id,
                "source_request": str(source_request),
                "request_path": str(request_path),
                "score_path": str(score_path),
                "run_dir": str(run_dir),
                "output_wav": str(wav_path),
                "seed": request["seed"],
                "status": "running",
            }
            cases.append(case)
            write_report(cases)

            if wav_path.is_file() and wav_path.stat().st_size > 0:
                case["status"] = "complete"
                case["reason"] = "Reused existing wav"
                write_report(cases)
                continue

            write_json(request_path, request)
            try:
                code = 0 if existing_run_dir else run_generation(request_path, score_path, run_dir)
                flac_path = run_dir / "audio.flac"
                run_json_path = run_dir / "run.json"
                result = {}
                if run_json_path.is_file():
                    run_json = read_json(run_json_path)
                    results = run_json.get("results") or []
                    if results:
                        result = results[0]
                if flac_path.is_file():
                    convert_audio(flac_path, wav_path)
                    case["audio_seconds"] = result.get("audio_seconds")
                    case["truncated"] = result.get("truncated")
                    case["identity"] = result.get("identity")
                    case["status"] = result.get("status") or ("complete" if code == 0 else "needs_review")
                    if code != 0 and case["status"] == "complete":
                        case["status"] = "needs_review"
                else:
                    case["status"] = "failed"
                    case["reason"] = f"YuE2 did not create {flac_path}"
            except Exception as exc:
                case["status"] = "failed"
                case["reason"] = f"{type(exc).__name__}: {exc}"
                case["traceback"] = traceback.format_exc()[-4000:]
            finally:
                write_report(cases)
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                gc.collect()

    write_report(cases)
    return 1 if any(case.get("status") == "failed" for case in cases) else 0


if __name__ == "__main__":
    raise SystemExit(main())
