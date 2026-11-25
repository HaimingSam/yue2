import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchaudio


DIFFSYNTH_ROOT = Path("/data/cmq/DiffSynth-Studio")
INPUT_AUDIO = Path("/data/cmq/YuE/yue2-cover-results/downloads/row_2_song_001_eed08e24.mp3")
OUTPUT_DIR = Path("/data/cmq/YuE/diffsynth-music-results/row_2_song_001_eed08e24")
DEFAULT_PROMPT = (
    "instrumental Mandarin pop rock backing track, energetic drums, bright guitar, "
    "warm bass, clean full arrangement, no vocals, no singing, no humming"
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_wav(path, audio, sample_rate=48000):
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), audio.detach().cpu(), sample_rate)


def tagged_path(stem, tag):
    suffix = f"_{tag}" if tag else ""
    return OUTPUT_DIR / f"row_2_{stem}{suffix}.wav"


def main():
    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "True")
    os.chdir(DIFFSYNTH_ROOT)
    sys.path.insert(0, str(DIFFSYNTH_ROOT))

    from diffsynth.core.data.operators import LoadMultiTrackAudio
    from diffsynth.diffusion.template import TemplatePipeline
    from diffsynth.pipelines.diffsynth_music import DiffSynthMusicPipeline, ModelConfig

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_tag = os.environ.get("DIFFSYNTH_OUTPUT_TAG", "").strip()
    prompt = os.environ.get("DIFFSYNTH_MUSIC_PROMPT", DEFAULT_PROMPT)
    report_path = OUTPUT_DIR / (f"report_{output_tag}.json" if output_tag else "report.json")
    report = {
        "started_at": utc_now(),
        "input_audio": str(INPUT_AUDIO),
        "output_dir": str(OUTPUT_DIR),
        "output_tag": output_tag,
        "status": "running",
        "outputs": {},
    }

    try:
        input_info = torchaudio.info(str(INPUT_AUDIO))
        report["input_info"] = {
            "sample_rate": input_info.sample_rate,
            "num_frames": input_info.num_frames,
            "num_channels": input_info.num_channels,
            "duration_sec": input_info.num_frames / input_info.sample_rate,
        }

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

        source_audio = LoadMultiTrackAudio(division_factor=3840)(str(INPUT_AUDIO))
        source_path = tagged_path("source_48k", output_tag)
        save_wav(source_path, source_audio, 48000)
        report["outputs"]["source_48k"] = str(source_path)

        vocals = pipe.extract_track(source_audio, track="vocals")
        vocals_path = tagged_path("extracted_vocals", output_tag)
        save_wav(vocals_path, vocals, 48000)
        report["outputs"]["extracted_vocals"] = str(vocals_path)

        duration = vocals.shape[1] / 48000

        mixed = template(
            pipe,
            prompt=prompt,
            negative_prompt=pipe.default_negative_prompt,
            lyrics="",
            duration=duration,
            seed=42,
            tiled=True,
            cfg_scale=4,
            num_inference_steps=30,
            template_inputs=[{"model_id": 0, "audio": vocals}],
            negative_template_inputs=[{"model_id": 0, "audio": vocals}],
            target_audio=vocals,
            target_track="vocals",
        )
        mixed_path = tagged_path("vocal_with_generated_accompaniment", output_tag)
        save_wav(mixed_path, mixed, 48000)
        report["outputs"]["vocal_with_generated_accompaniment"] = str(mixed_path)

        accompaniment = pipe.extract_track(mixed, track=["drums", "bass", "other"])
        accompaniment_path = tagged_path("generated_accompaniment_only", output_tag)
        save_wav(accompaniment_path, accompaniment, 48000)
        report["outputs"]["generated_accompaniment_only"] = str(accompaniment_path)

        report["prompt"] = prompt
        report["duration_sec"] = duration
        report["status"] = "success"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        raise
    finally:
        report["finished_at"] = utc_now()
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
