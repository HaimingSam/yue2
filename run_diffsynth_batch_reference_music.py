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
DEFAULT_AUDIO_DIR = Path("/data/cmq/YuE/yue2-cover-results/downloads")
DEFAULT_REQUEST_DIR = Path("/data/cmq/YuE/yue2-cover-results/work/requests")
DEFAULT_OUTPUT_DIR = Path("/data/cmq/YuE/diffsynth-music-results/reference_music_batch")
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
REFERENCE_MIN_FRAMES = 1920 * 256


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def safe_stem(name):
    stem = Path(name).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem or "audio"


def save_wav(path, audio, sample_rate=48000):
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), audio.detach().cpu(), sample_rate)


def pad_reference_audio(audio):
    if audio.shape[-1] >= REFERENCE_MIN_FRAMES:
        return audio
    pad_frames = REFERENCE_MIN_FRAMES - audio.shape[-1]
    return torch.nn.functional.pad(audio, (0, pad_frames))


def load_audio_48k(path, division_factor=3840):
    waveform, sample_rate = torchaudio.load(str(path))
    if waveform.dim() == 2 and waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    if sample_rate != 48000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 48000)
        sample_rate = 48000
    if division_factor:
        waveform = waveform[:, : waveform.shape[1] // division_factor * division_factor]
    return waveform, sample_rate


def probe_audio_info(path):
    waveform, sample_rate = torchaudio.load(str(path))
    return {
        "sample_rate": sample_rate,
        "num_frames": waveform.shape[-1],
        "num_channels": waveform.shape[-2],
        "duration_sec": waveform.shape[-1] / sample_rate,
    }


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def collect_inputs(audio_dir):
    return sorted(
        [
            path
            for path in audio_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ],
        key=lambda p: p.name,
    )


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


def split_lyrics(lyrics, chunk_count):
    if chunk_count <= 1 or not lyrics.strip():
        return [lyrics]
    blocks = re.split(r"\n\s*\n", lyrics.strip())
    blocks = [block.strip() for block in blocks if block.strip()]
    if not blocks:
        lines = [line for line in lyrics.splitlines() if line.strip()]
        blocks = lines or [lyrics]
    buckets = [[] for _ in range(chunk_count)]
    bucket_lengths = [0 for _ in range(chunk_count)]
    target = max(1, sum(len(block) for block in blocks) / chunk_count)
    bucket_id = 0
    for block in blocks:
        if bucket_id < chunk_count - 1 and bucket_lengths[bucket_id] >= target:
            bucket_id += 1
        buckets[bucket_id].append(block)
        bucket_lengths[bucket_id] += len(block)
    return ["\n\n".join(bucket).strip() for bucket in buckets]


def output_paths(output_dir, stem):
    return {
        "source_48k": output_dir / "source_48k" / f"{stem}_source_48k.wav",
        "generated_song": output_dir / "generated" / f"{stem}_reference_song.wav",
    }


def generate_one(pipe, template, reference_audio, prompt, lyrics, duration, seed, steps, cfg_scale):
    reference_audio = pad_reference_audio(reference_audio)
    return template(
        pipe,
        prompt=prompt,
        negative_prompt="",
        lyrics=lyrics,
        duration=duration,
        seed=seed,
        tiled=True,
        cfg_scale=cfg_scale,
        num_inference_steps=steps,
        template_inputs=[{"model_id": 2, "audio": reference_audio}],
    )


def process_case(pipe, template, audio_path, request_path, output_dir, args):
    stem = safe_stem(audio_path.name)
    paths = output_paths(output_dir, stem)
    request = read_json(request_path)
    prompt = request.get("prompt") or request.get("style") or ""
    lyrics = request.get("lyrics") or ""
    seed = int(request.get("seed") or args.seed)

    input_info = probe_audio_info(audio_path)
    if input_info["duration_sec"] < args.min_duration_sec:
        return {
            "status": "skipped",
            "reason": f"input duration {input_info['duration_sec']:.3f}s is shorter than min duration {args.min_duration_sec:.3f}s",
            "input_info": input_info,
            "prompt": prompt,
            "lyrics_preview": lyrics[:200],
            "outputs": {},
        }

    reference_audio, _ = load_audio_48k(audio_path)
    save_wav(paths["source_48k"], reference_audio, 48000)
    duration = reference_audio.shape[-1] / 48000
    if args.max_duration_sec:
        duration = min(duration, args.max_duration_sec)
        reference_audio = reference_audio[:, : int(duration * 48000)]

    chunk_paths = []
    if args.chunk_duration_sec and duration > args.chunk_duration_sec:
        chunk_frames = int(args.chunk_duration_sec * 48000)
        chunk_frames = max(3840, chunk_frames // 3840 * 3840)
        chunk_count = (reference_audio.shape[-1] + chunk_frames - 1) // chunk_frames
        lyric_chunks = split_lyrics(lyrics, chunk_count)
        chunks = []
        chunk_dir = output_dir / "chunks" / stem
        chunk_dir.mkdir(parents=True, exist_ok=True)
        for chunk_index, start in enumerate(range(0, reference_audio.shape[-1], chunk_frames), start=1):
            ref_chunk = reference_audio[:, start:start + chunk_frames]
            if ref_chunk.shape[-1] < 3840:
                continue
            chunk_duration = ref_chunk.shape[-1] / 48000
            chunk_lyrics = lyric_chunks[chunk_index - 1] if chunk_index - 1 < len(lyric_chunks) else ""
            print(
                f"  segment {chunk_index}/{chunk_count}: {start / 48000:.2f}s-{(start + ref_chunk.shape[-1]) / 48000:.2f}s",
                flush=True,
            )
            generated_chunk = generate_one(
                pipe,
                template,
                ref_chunk,
                prompt,
                chunk_lyrics,
                chunk_duration,
                seed + chunk_index - 1,
                args.num_inference_steps,
                args.cfg_scale,
            )
            chunk_path = chunk_dir / f"{stem}_reference_chunk_{chunk_index:03d}.wav"
            save_wav(chunk_path, generated_chunk, 48000)
            chunks.append(generated_chunk.detach().cpu())
            chunk_paths.append(str(chunk_path))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if not chunks:
            raise RuntimeError("segmented generation produced no chunks")
        generated = torch.cat(chunks, dim=-1)
        generation_mode = f"segmented_{args.chunk_duration_sec:g}s"
    else:
        generated = generate_one(
            pipe,
            template,
            reference_audio,
            prompt,
            lyrics,
            duration,
            seed,
            args.num_inference_steps,
            args.cfg_scale,
        )
        generation_mode = "full"

    save_wav(paths["generated_song"], generated, 48000)
    return {
        "status": "success",
        "reason": "",
        "input_info": input_info,
        "prompt": prompt,
        "lyrics_preview": lyrics[:200],
        "duration_sec": generated.shape[-1] / 48000,
        "seed": seed,
        "generation_mode": generation_mode,
        "chunk_paths": chunk_paths,
        "outputs": {key: str(path) for key, path in paths.items()},
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
    parser.add_argument("--audio-dir", default=str(DEFAULT_AUDIO_DIR))
    parser.add_argument("--request-dir", default=str(DEFAULT_REQUEST_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--min-duration-sec", type=float, default=1.0)
    parser.add_argument("--max-duration-sec", type=float)
    parser.add_argument("--chunk-duration-sec", type=float)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    request_dir = Path(args.request_dir)
    output_dir = Path(args.output_dir)
    report_path = output_dir / "report.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = collect_inputs(audio_dir)
    if args.limit:
        audio_files = audio_files[: args.limit]

    cases = []
    for index, audio_path in enumerate(audio_files, start=1):
        stem = safe_stem(audio_path.name)
        request_path = request_dir / f"{stem}.json"
        cases.append(
            {
                "index": index,
                "input_name": audio_path.name,
                "source_audio_path": str(audio_path),
                "request_path": str(request_path),
                "status": "pending",
                "reason": "",
                "outputs": {},
                "started_at": None,
                "finished_at": None,
                "command_used": " ".join(sys.argv),
            }
        )

    report = {
        "summary": {},
        "audio_dir": str(audio_dir),
        "request_dir": str(request_dir),
        "output_dir": str(output_dir),
        "started_at": utc_now(),
        "num_inference_steps": args.num_inference_steps,
        "cfg_scale": args.cfg_scale,
        "chunk_duration_sec": args.chunk_duration_sec,
        "cases": cases,
    }
    update_summary(report)
    write_json(report_path, report)

    pipe, template = build_pipeline()

    for case in report["cases"]:
        audio_path = Path(case["source_audio_path"])
        request_path = Path(case["request_path"])
        generated_path = output_paths(output_dir, safe_stem(audio_path.name))["generated_song"]

        if generated_path.exists() and not args.overwrite:
            case["status"] = "skipped"
            case["reason"] = "output already exists"
            case["outputs"] = {"generated_song": str(generated_path)}
            case["finished_at"] = utc_now()
            update_summary(report)
            write_json(report_path, report)
            continue
        if not request_path.exists():
            case["status"] = "skipped"
            case["reason"] = "matching request json not found"
            case["finished_at"] = utc_now()
            update_summary(report)
            write_json(report_path, report)
            continue

        case["status"] = "running"
        case["started_at"] = utc_now()
        update_summary(report)
        write_json(report_path, report)
        print(f"[{case['index']}/{len(report['cases'])}] processing {audio_path.name}", flush=True)

        try:
            result = process_case(pipe, template, audio_path, request_path, output_dir, args)
            case.update(result)
        except Exception as exc:
            case["status"] = "failed"
            case["reason"] = repr(exc)
            case["traceback"] = traceback.format_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            case["finished_at"] = utc_now()
            update_summary(report)
            write_json(report_path, report)
            print(f"[{case['index']}/{len(report['cases'])}] {case['status']} {audio_path.name}: {case['reason']}", flush=True)

    report["finished_at"] = utc_now()
    update_summary(report)
    write_json(report_path, report)


if __name__ == "__main__":
    main()
