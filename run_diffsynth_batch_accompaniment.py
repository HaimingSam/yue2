import argparse
import json
import os
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchaudio


DIFFSYNTH_ROOT = Path("/data/cmq/DiffSynth-Studio")
DEFAULT_INPUT_DIR = Path("/data/cmq/YuE/yue2-cover-results/generated")
DEFAULT_OUTPUT_DIR = Path("/data/cmq/YuE/diffsynth-music-results/generated_batch_generic_prompt")
DEFAULT_PROMPT = "instrumental backing track, full arrangement, clean mix, no vocals, no singing, no humming"
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def safe_stem(name):
    stem = Path(name).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem or "audio"


def save_wav(path, audio, sample_rate=48000):
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), audio.detach().cpu(), sample_rate)


def read_report(path):
    if not path.exists():
        return {
            "summary": {
                "total_rows": 0,
                "processed": 0,
                "succeeded": 0,
                "failed": 0,
                "skipped": 0,
            },
            "cases": [],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def collect_inputs(input_dir):
    files = [
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.name)


def probe_audio_info(audio_path):
    audio, sample_rate = torchaudio.load(str(audio_path))
    return {
        "sample_rate": sample_rate,
        "num_frames": audio.shape[-1],
        "num_channels": audio.shape[-2],
        "duration_sec": audio.shape[-1] / sample_rate,
    }


def build_pipeline():
    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "True")
    os.chdir(DIFFSYNTH_ROOT)
    sys.path.insert(0, str(DIFFSYNTH_ROOT))

    from diffsynth.diffusion.template import TemplatePipeline
    from diffsynth.pipelines.diffsynth_music import DiffSynthMusicPipeline, ModelConfig

    vram_config = {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": "disk",
        "onload_device": "disk",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }
    vram_config_cpu = {
        "offload_dtype": torch.bfloat16,
        "offload_device": "cpu",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }
    vram_config_fp32 = {
        "offload_dtype": torch.float32,
        "offload_device": "cpu",
        "onload_dtype": torch.float32,
        "onload_device": "cpu",
        "preparing_dtype": torch.float32,
        "preparing_device": "cuda",
        "computation_dtype": torch.float32,
        "computation_device": "cuda",
    }

    pipe = DiffSynthMusicPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="DiffSynth-Studio/DiffSynth-Music", origin_file_pattern="transformer/model.safetensors", **vram_config),
            ModelConfig(model_id="DiffSynth-Studio/DiffSynth-Music", origin_file_pattern="conditioner/model.safetensors", **vram_config),
            ModelConfig(model_id="DiffSynth-Studio/DiffSynth-Music", origin_file_pattern="text_encoder/model.safetensors", **vram_config),
            ModelConfig(model_id="DiffSynth-Studio/DiffSynth-Music", origin_file_pattern="vae/model.safetensors", **vram_config_cpu),
            ModelConfig(model_id="DiffSynth-Studio/DiffSynth-Music", origin_file_pattern="track_separator/model.safetensors", **vram_config_fp32),
        ],
        tokenizer_config=ModelConfig(model_id="DiffSynth-Studio/DiffSynth-Music", origin_file_pattern="text_encoder/"),
        vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
    )

    template = TemplatePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="DiffSynth-Studio/DiffSynth-Music", origin_file_pattern="template_control/"),
            ModelConfig(model_id="DiffSynth-Studio/DiffSynth-Music", origin_file_pattern="template_prosody/"),
            ModelConfig(model_id="DiffSynth-Studio/DiffSynth-Music", origin_file_pattern="template_reference/"),
        ],
        lazy_loading=True,
    )
    return pipe, template


def case_outputs(output_dir, stem):
    return {
        "source_48k": output_dir / "source_48k" / f"{stem}_source_48k.wav",
        "extracted_vocals": output_dir / "vocals" / f"{stem}_extracted_vocals.wav",
        "vocal_with_generated_accompaniment": output_dir / "mixed" / f"{stem}_vocal_with_generated_accompaniment.wav",
        "generated_accompaniment_only": output_dir / "accompaniment" / f"{stem}_generated_accompaniment_only.wav",
    }


def load_or_extract_vocals(pipe, audio_path, outputs):
    from diffsynth.core.data.operators import LoadMultiTrackAudio

    if outputs["source_48k"].exists():
        source_audio, _ = torchaudio.load(str(outputs["source_48k"]))
    else:
        source_audio = LoadMultiTrackAudio(division_factor=3840)(str(audio_path))
        save_wav(outputs["source_48k"], source_audio, 48000)

    if outputs["extracted_vocals"].exists():
        vocals, _ = torchaudio.load(str(outputs["extracted_vocals"]))
    else:
        vocals = pipe.extract_track(source_audio, track="vocals")
        save_wav(outputs["extracted_vocals"], vocals, 48000)
    return source_audio, vocals


def generate_one_segment(pipe, template, vocals, prompt, seed, num_inference_steps, cfg_scale):
    return template(
        pipe,
        prompt=prompt,
        negative_prompt=pipe.default_negative_prompt,
        lyrics="",
        duration=vocals.shape[1] / 48000,
        seed=seed,
        tiled=True,
        cfg_scale=cfg_scale,
        num_inference_steps=num_inference_steps,
        template_inputs=[{"model_id": 0, "audio": vocals}],
        negative_template_inputs=[{"model_id": 0, "audio": vocals}],
        target_audio=vocals,
        target_track="vocals",
    )


def process_one(pipe, template, audio_path, output_dir, prompt, seed, num_inference_steps, cfg_scale, chunk_duration_sec=None, output_name=None):
    stem = safe_stem(output_name or audio_path.name)
    outputs = case_outputs(output_dir, stem)
    input_info = probe_audio_info(audio_path)
    _, vocals = load_or_extract_vocals(pipe, audio_path, outputs)

    duration = vocals.shape[1] / 48000
    chunks = []
    chunk_paths = []
    if chunk_duration_sec and duration > chunk_duration_sec:
        chunk_frames = int(chunk_duration_sec * 48000)
        chunk_frames = max(3840, chunk_frames // 3840 * 3840)
        chunk_dir = output_dir / "chunks" / stem
        chunk_dir.mkdir(parents=True, exist_ok=True)
        for chunk_index, start in enumerate(range(0, vocals.shape[1], chunk_frames), start=1):
            chunk = vocals[:, start:start + chunk_frames]
            if chunk.shape[1] < 3840:
                continue
            print(
                f"  segment {chunk_index}: {start / 48000:.2f}s-{(start + chunk.shape[1]) / 48000:.2f}s",
                flush=True,
            )
            mixed_chunk = generate_one_segment(
                pipe,
                template,
                chunk,
                prompt,
                seed + chunk_index - 1,
                num_inference_steps,
                cfg_scale,
            )
            chunk_path = chunk_dir / f"{stem}_mixed_chunk_{chunk_index:03d}.wav"
            save_wav(chunk_path, mixed_chunk, 48000)
            chunks.append(mixed_chunk.detach().cpu())
            chunk_paths.append(str(chunk_path))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if not chunks:
            raise RuntimeError("segmented generation produced no chunks")
        mixed = torch.cat(chunks, dim=-1)
        generation_mode = f"segmented_{chunk_duration_sec:g}s"
    else:
        mixed = generate_one_segment(
            pipe,
            template,
            vocals,
            prompt,
            seed,
            num_inference_steps,
            cfg_scale,
        )
        generation_mode = "full"

    save_wav(outputs["vocal_with_generated_accompaniment"], mixed, 48000)

    accompaniment = pipe.extract_track(mixed, track=["drums", "bass", "other"])
    save_wav(outputs["generated_accompaniment_only"], accompaniment, 48000)

    return {
        "source_audio_path": str(audio_path),
        "input_info": input_info,
        "duration_sec": duration,
        "generation_mode": generation_mode,
        "chunk_paths": chunk_paths,
        "outputs": {key: str(path) for key, path in outputs.items()},
    }


def update_summary(report):
    cases = report["cases"]
    report["summary"] = {
        "total_rows": len(cases),
        "processed": sum(case["status"] in {"success", "failed"} for case in cases),
        "succeeded": sum(case["status"] == "success" for case in cases),
        "failed": sum(case["status"] == "failed" for case in cases),
        "skipped": sum(case["status"] == "skipped" for case in cases),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--input-file")
    parser.add_argument("--input-name")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--chunk-duration-sec", type=float)
    parser.add_argument("--min-duration-sec", type=float, default=1.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only-failed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    report_path = output_dir / "report.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_file:
        audio_files = [Path(args.input_file)]
    else:
        audio_files = collect_inputs(input_dir)
    if args.limit is not None:
        audio_files = audio_files[:args.limit]

    command_used = " ".join(sys.argv)
    report = read_report(report_path)
    existing_by_name = {case.get("input_name"): case for case in report.get("cases", [])}
    cases = list(report.get("cases", [])) if args.input_file else []
    for index, audio_path in enumerate(audio_files, start=1):
        input_name = args.input_name or audio_path.name
        previous = existing_by_name.get(input_name)
        if previous is not None:
            previous["source_audio_path"] = str(audio_path)
            previous["prompt"] = args.prompt
            previous["command_used"] = command_used
            if not args.input_file:
                cases.append(previous)
        else:
            case = {
                "index": len(cases) + 1 if args.input_file else index,
                "input_name": input_name,
                "source_audio_path": str(audio_path),
                "prompt": args.prompt,
                "status": "pending",
                "reason": "",
                "outputs": {},
                "command_used": command_used,
                "started_at": None,
                "finished_at": None,
            }
            cases.append(case)
    report = {
        "summary": report.get("summary", {}),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "prompt": args.prompt,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "cfg_scale": args.cfg_scale,
        "started_at": report.get("started_at") or utc_now(),
        "cases": cases,
    }
    update_summary(report)
    write_report(report_path, report)

    pipe, template = build_pipeline()

    target_input_name = args.input_name or (Path(args.input_file).name if args.input_file else None)
    for case in report["cases"]:
        if args.input_file and case["input_name"] != target_input_name:
            continue
        if args.only_failed and case["status"] != "failed":
            continue
        if case["status"] == "success" and not args.overwrite:
            continue

        audio_path = Path(case["source_audio_path"])
        input_info = probe_audio_info(audio_path)
        input_duration = input_info["duration_sec"]
        if input_duration < args.min_duration_sec:
            case["status"] = "skipped"
            case["reason"] = f"input duration {input_duration:.3f}s is shorter than min duration {args.min_duration_sec:.3f}s"
            case["input_info"] = input_info
            case["finished_at"] = utc_now()
            update_summary(report)
            write_report(report_path, report)
            continue

        stem = safe_stem(audio_path.name)
        if args.input_name:
            stem = safe_stem(args.input_name)
        outputs = case_outputs(output_dir, stem)
        if (
            not args.overwrite
            and outputs["vocal_with_generated_accompaniment"].exists()
            and outputs["generated_accompaniment_only"].exists()
        ):
            case["status"] = "skipped"
            case["reason"] = "outputs already exist"
            case["outputs"] = {key: str(path) for key, path in outputs.items()}
            case["finished_at"] = utc_now()
            update_summary(report)
            write_report(report_path, report)
            continue

        case["status"] = "running"
        case["reason"] = ""
        case["prompt"] = args.prompt
        case["started_at"] = utc_now()
        case["finished_at"] = None
        update_summary(report)
        write_report(report_path, report)

        print(f"[{case['index']}/{len(report['cases'])}] processing {audio_path.name}", flush=True)
        try:
            result = process_one(
                pipe,
                template,
                audio_path,
                output_dir,
                args.prompt,
                args.seed,
                args.num_inference_steps,
                args.cfg_scale,
                args.chunk_duration_sec,
                case["input_name"],
            )
            case.update(result)
            case["status"] = "success"
            case["reason"] = ""
        except Exception as exc:
            case["status"] = "failed"
            case["reason"] = repr(exc)
            case["traceback"] = traceback.format_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            case["finished_at"] = utc_now()
            update_summary(report)
            write_report(report_path, report)
            print(f"[{case['index']}/{len(report['cases'])}] {case['status']} {audio_path.name}: {case['reason']}", flush=True)

    report["finished_at"] = utc_now()
    update_summary(report)
    write_report(report_path, report)


if __name__ == "__main__":
    main()
