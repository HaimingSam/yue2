#!/usr/bin/env python3
"""Batch-transcribe URV vocal stems to ABC with one persistent worker per GPU."""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path


DEFAULT_INPUT_ROOT = Path("/data/cmq/datas/1000-1-urv_separated")
DEFAULT_OUTPUT_ROOT = Path("/data2/cmq/abc-yue2")
DEFAULT_MODEL = Path("/data/cmq/YuE/models/SheetSage2")
DEFAULT_MERT = Path(
    "/data/cmq/hf-cache/hub/models--m-a-p--MERT-v2-FullSong"
    "/snapshots/d8ba1c745e733b3908ce6ad16ebeb17ac7600a42"
)

_MODEL = None
_ARGS = None
_ABC_TOOLS = None


def parse_gpu_ids(value: str | None) -> list[int]:
    if value:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
        if not result:
            raise ValueError("--gpus must contain at least one GPU index")
        if any(item < 0 for item in result):
            raise ValueError("GPU indices must be non-negative")
        return result
    import torch

    count = torch.cuda.device_count()
    if count == 0:
        raise RuntimeError("No CUDA GPUs detected; pass --gpus only when CUDA is available")
    return list(range(count))


def collect_inputs(root: Path, input_name: str = "vocals_dereverb.wav") -> list[Path]:
    if input_name not in {"vocals_dereverb.wav", "vocals.wav", "instrumental.wav"}:
        raise ValueError(f"Unsupported input name: {input_name}")
    inputs = sorted(root.glob(f"*/{input_name}"))
    if not inputs:
        raise FileNotFoundError(f"No */{input_name} files found under {root}")
    names = [path.parent.name for path in inputs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Output filename collision for directory names: {duplicates[:5]}")
    return inputs


def output_path(input_path: Path, output_root: Path) -> Path:
    return output_root / f"{input_path.parent.name}.abc"


def select_ids(inputs: list[Path], selected: list[str] | None) -> list[Path]:
    if selected is None:
        return inputs
    requested = set(selected)
    matches = [path for path in inputs if path.parent.name in requested]
    missing = requested - {path.parent.name for path in matches}
    if missing:
        raise ValueError(f"Missing vocals_dereverb.wav for IDs: {sorted(missing)}")
    return matches


def load_abc_tools():
    global _ABC_TOOLS
    if _ABC_TOOLS is None:
        script = Path(__file__).resolve().parents[1] / "skills/yue2-music/scripts/abc_tools.py"
        spec = importlib.util.spec_from_file_location("batch_abc_tools", script)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load ABC validator: {script}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _ABC_TOOLS = module
    return _ABC_TOOLS


def block_music_indices(lines: list[str], start: int, end: int) -> list[int]:
    indices = []
    for index in range(start + 1, end):
        line = lines[index]
        if not line or line.startswith(("M:", "K:", "%", "V:")):
            continue
        indices.append(index)
    return indices


def voice_blocks(lines: list[str], name: str) -> list[tuple[int, int]]:
    starts = [index for index, line in enumerate(lines) if line == f"V: {name}"]
    all_voice_starts = [index for index, line in enumerate(lines) if line.startswith("V: ")]
    blocks = []
    for start in starts:
        end = next((candidate for candidate in all_voice_starts if candidate > start), len(lines))
        blocks.append((start, end))
    return blocks


def silence_music_line(line: str, abc_tools) -> str:
    def replace(match):
        if match.group("note") is None:
            return match.group(0)
        return "z" + (match.group("duration") or "")

    return abc_tools.TOKEN.sub(replace, line)


def check_and_repair_abc(score_path: Path) -> dict:
    """Validate ABC and move Ins melody to Vocal when Vocal is empty."""
    abc_tools = load_abc_tools()
    original = score_path.read_text(encoding="utf-8")
    score = abc_tools.parse_abc(original)
    vocal_notes = len(score.voices["Vocal"].notes)
    ins_notes = len(score.voices["Ins"].notes)
    if vocal_notes == 0 and ins_notes == 0:
        raise ValueError("ABC has no sounding notes in either Vocal or Ins")
    check = {"vocal_notes_before": vocal_notes, "ins_notes_before": ins_notes,
             "repair": "none"}
    if vocal_notes == 0 and ins_notes > 0:
        lines = original.splitlines()
        vocal_blocks = voice_blocks(lines, "Vocal")
        ins_blocks = voice_blocks(lines, "Ins")
        if len(vocal_blocks) != len(ins_blocks):
            raise ValueError("Cannot repair ABC: Vocal and Ins voice block counts differ")
        replacements: dict[int, str] = {}
        for vocal_block, ins_block in zip(vocal_blocks, ins_blocks):
            vocal_indices = block_music_indices(lines, *vocal_block)
            ins_indices = block_music_indices(lines, *ins_block)
            if len(vocal_indices) != len(ins_indices):
                raise ValueError("Cannot repair ABC: Vocal and Ins music-line counts differ")
            for vocal_index, ins_index in zip(vocal_indices, ins_indices):
                replacements[vocal_index] = lines[ins_index]
                replacements[ins_index] = silence_music_line(lines[ins_index], abc_tools)
        repaired = "\n".join(replacements.get(index, line) for index, line in enumerate(lines)) + "\n"
        repaired_score = abc_tools.parse_abc(repaired)
        repaired_vocal = len(repaired_score.voices["Vocal"].notes)
        repaired_ins = len(repaired_score.voices["Ins"].notes)
        if repaired_vocal == 0 or repaired_ins != 0:
            raise ValueError(f"ABC repair failed: Vocal={repaired_vocal}, Ins={repaired_ins}")
        score_path.write_text(repaired, encoding="utf-8")
        check.update({"repair": "ins_to_vocal", "vocal_notes_after": repaired_vocal,
                      "ins_notes_after": repaired_ins})
    else:
        check.update({"vocal_notes_after": vocal_notes, "ins_notes_after": ins_notes})
    return check


def initialize_worker(gpu_id: int, options: dict) -> None:
    global _MODEL, _ARGS
    os.environ.setdefault("HF_HOME", options["hf_home"])
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import torch
    from transformers import AutoModel

    torch.set_num_threads(options["threads"])
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] loading SheetSage2", flush=True)
    _MODEL = AutoModel.from_pretrained(
        options["model"],
        base_model_path=options["mert"],
        local_files_only=True,
        trust_remote_code=True,
    ).eval().to(device)
    _ARGS = {**options, "device": device, "gpu_id": gpu_id}
    print(f"[GPU {gpu_id}] model ready", flush=True)


def transcribe_one(input_path_str: str) -> dict:
    input_path = Path(input_path_str)
    output_root = Path(_ARGS["output_root"])
    final_path = output_path(input_path, output_root)
    if final_path.is_file() and final_path.stat().st_size > 0 and not _ARGS["force"]:
        return {"status": "skipped", "input": str(input_path), "output": str(final_path),
                "gpu": _ARGS["gpu_id"]}

    work_parent = Path(_ARGS["work_root"])
    work_parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f"{input_path.parent.name}-", dir=work_parent))
    started = time.monotonic()
    try:
        prompts = ["timestamp", "downbeat_meter", "structure", "key"]
        prompts.append("melody_vocal" if _ARGS["task"] == "melody-vocal" else "melody_full")
        options = {
            "output_dir": str(work_dir),
            "prompts": prompts,
            "dtype": _ARGS["dtype"],
            "preset": _ARGS["preset"],
            "max_seconds": _ARGS["max_seconds"],
        }
        if _ARGS["task"] != "full":
            options["melody_only"] = True
        result = _MODEL.transcribe(str(input_path), **options)
        score = result.get("abc")
        score_path = work_dir / "score.abc"
        if not score or not score_path.is_file() or score_path.stat().st_size == 0:
            raise RuntimeError(f"No usable ABC produced: {result.get('abc_error')}")
        abc_check = check_and_repair_abc(score_path)

        output_root.mkdir(parents=True, exist_ok=True)
        staged = final_path.with_suffix(".abc.part")
        shutil.copyfile(score_path, staged)
        os.replace(staged, final_path)
        return {"status": "completed", "input": str(input_path), "output": str(final_path),
                "gpu": _ARGS["gpu_id"], "seconds": round(time.monotonic() - started, 3),
                "warnings": result.get("warnings", []), "abc_check": abc_check}
    except Exception as exc:
        return {"status": "failed", "input": str(input_path), "output": str(final_path),
                "gpu": _ARGS["gpu_id"], "seconds": round(time.monotonic() - started, 3),
                "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--input-name", choices=("vocals_dereverb.wav", "vocals.wav", "instrumental.wav"),
                        default="vocals_dereverb.wav", help="Stem to transcribe; use alternatives only for audited retries")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--mert", type=Path, default=DEFAULT_MERT)
    parser.add_argument("--hf-home", default="/data/cmq/hf-cache")
    parser.add_argument("--gpus", help="Comma-separated physical GPU IDs; default: all visible GPUs")
    parser.add_argument("--workers-per-gpu", type=int, default=1,
                        help="Model processes per GPU; keep at 1 unless VRAM allows duplication")
    parser.add_argument("--task", choices=("full", "melody-full", "melody-vocal"), default="melody-vocal")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--preset", choices=("default", "paper"), default="default")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only-ids", nargs="+", help="Retry only these input directory IDs")
    parser.add_argument("--force", action="store_true", help="Recompute existing ABC files")
    parser.add_argument("--dry-run", action="store_true", help="List work without loading models")
    parser.add_argument("--work-root", type=Path, default=Path("/tmp/yue2-urv-transcribe"))
    args = parser.parse_args()
    if args.workers_per_gpu < 1 or args.threads < 1:
        parser.error("--workers-per-gpu and --threads must be positive")
    inputs = select_ids(collect_inputs(args.input_root.resolve(), args.input_name), args.only_ids)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        inputs = inputs[:args.limit]
    gpus = parse_gpu_ids(args.gpus)
    jobs = [path for path in inputs if args.force or not output_path(path, args.output_root).is_file()]
    skipped = len(inputs) - len(jobs)
    print(json.dumps({"inputs": len(inputs), "pending": len(jobs), "existing_skipped": skipped,
                      "gpus": gpus, "workers": len(gpus) * args.workers_per_gpu,
                      "output_root": str(args.output_root.resolve())}, ensure_ascii=False), flush=True)
    if args.dry_run or not jobs:
        for path in inputs:
            print(f"{path}\t{output_path(path, args.output_root)}")
        return 0

    options = {"model": str(args.model.resolve()), "mert": str(args.mert.resolve()),
               "hf_home": args.hf_home, "task": args.task, "dtype": args.dtype,
               "preset": args.preset, "max_seconds": args.max_seconds, "threads": args.threads,
               "output_root": str(args.output_root.resolve()), "work_root": str(args.work_root.resolve()),
               "force": args.force}
    worker_gpus = [gpu for gpu in gpus for _ in range(args.workers_per_gpu)]
    context = mp.get_context("spawn")
    completed = failed = 0
    manifest_path = args.output_root / "batch_manifest.jsonl"
    args.output_root.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as manifest:
        processes = []
        queues = []
        result_queue = context.Queue()
        for gpu_id in worker_gpus:
            queue = context.Queue()
            process = context.Process(target=worker_loop, args=(gpu_id, options, queue, result_queue))
            process.start()
            processes.append(process)
            queues.append(queue)
        for index, path in enumerate(jobs):
            queues[index % len(queues)].put(str(path))
        for queue in queues:
            queue.put(None)
        remaining = len(jobs)
        while remaining:
            result = result_queue.get()
            manifest.write(json.dumps(result, ensure_ascii=False) + "\n")
            manifest.flush()
            remaining -= 1
            if result["status"] == "failed":
                failed += 1
                print(f"[FAIL] {result['input']}: {result['error']}", flush=True)
            else:
                completed += 1
                print(f"[{result['status']}] {result['output']} GPU={result['gpu']}", flush=True)
        for process in processes:
            process.join()
    print(json.dumps({"completed": completed, "failed": failed, "skipped": skipped,
                      "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))
    return 1 if failed else 0


def worker_loop(gpu_id: int, options: dict, job_queue, result_queue) -> None:
    try:
        initialize_worker(gpu_id, options)
    except Exception as exc:
        error = {"status": "failed", "gpu": gpu_id,
                 "error": f"Worker initialization failed: {type(exc).__name__}: {exc}",
                 "traceback": traceback.format_exc()}
        while True:
            job = job_queue.get()
            if job is None:
                return
            input_path = Path(job)
            result_queue.put({**error, "input": str(input_path),
                              "output": str(output_path(input_path, Path(options["output_root"])))})
    while True:
        job = job_queue.get()
        if job is None:
            return
        result_queue.put(transcribe_one(job))


if __name__ == "__main__":
    raise SystemExit(main())
