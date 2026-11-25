"""Extract frozen voice/ VAE features; never invent semantic supervision."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import soundfile as sf
import torch

from yue2.storage import identity, sha256_file, write_json
from yue2.voice_training import weight_id


def read_rows(root):
    root = Path(root).resolve()
    rows = []
    singers = {}
    sample_ids = set()
    for split in ("train", "validation"):
        for line in (root / f"audio_{split}.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["split"] != split or row["sample_id"] in sample_ids:
                raise ValueError("Invalid or duplicate sample split")
            if row["singer_id"] in singers and singers[row["singer_id"]] != split:
                raise ValueError("Singer overlap across splits")
            singers[row["singer_id"]] = split
            sample_ids.add(row["sample_id"])
            rows.append(row)
    if not rows:
        raise ValueError("Empty audio manifests")
    return sorted(rows, key=lambda row: row["sample_id"])


def configure(root, name, config):
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}_config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError(f"Changed extraction configuration: {path}; use a new output directory")
    write_json(path, config)


def load_cached(path, expected):
    path = Path(path)
    metadata_path = Path(str(path) + ".json")
    if not path.exists() and not metadata_path.exists():
        return None
    if not path.exists() or not metadata_path.exists():
        raise ValueError(f"Incomplete feature cache: {path}")
    metadata = json.loads(metadata_path.read_text())
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Feature metadata mismatch: {path}")
    if sha256_file(path) != metadata["array_sha256"]:
        raise ValueError(f"Feature checksum mismatch: {path}")
    array = np.load(path, allow_pickle=False)
    if array.dtype != np.float32 or list(array.shape) != metadata["shape"] or not np.isfinite(array).all():
        raise ValueError(f"Invalid cached feature array: {path}")
    return array


def save_array(path, array, metadata):
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Non-finite extracted features")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)
    write_json(str(path) + ".json", dict(metadata, shape=list(array.shape),
                                         array_sha256=sha256_file(path)))


def audio_path(root, row, field):
    path = root / row[field]
    source_metadata = json.loads((root / row["metadata"]).read_text())
    digest = sha256_file(path)
    if digest != source_metadata["files"][path.name]["sha256"]:
        raise ValueError(f"Prepared audio changed: {path}")
    return path, digest


def resample_mono_16k(path):
    info = sf.info(path)
    if info.samplerate == 16000 and info.channels == 1:
        audio, _ = sf.read(path, dtype="float32")
    else:
        result = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-threads", "1",
                                 "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"],
                                check=True, capture_output=True, timeout=120)
        audio = np.frombuffer(result.stdout, dtype="<f4").copy()
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all() or np.max(np.abs(audio)) < 1e-6:
        raise ValueError(f"Invalid reference waveform: {path}")
    return torch.from_numpy(audio)[None]


class SpeechBrainVoiceEncoder(torch.nn.Module):
    """Local frozen ECAPA-TDNN, matching the pinned VoxCeleb recipe's encode_batch."""

    def __init__(self, directory):
        super().__init__()
        from speechbrain.lobes.features import Fbank
        from speechbrain.processing.features import InputNormalization
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
        self.features = Fbank(n_mels=80)
        self.normalization = InputNormalization(norm_type="sentence", std_norm=False)
        self.embedding = ECAPA_TDNN(input_size=80, channels=[1024, 1024, 1024, 1024, 3072],
                                   kernel_sizes=[5, 3, 3, 3, 1], dilations=[1, 2, 3, 4, 1],
                                   attention_channels=128, lin_neurons=192)
        state = torch.load(Path(directory) / "embedding_model.ckpt", map_location="cpu", weights_only=True)
        self.embedding.load_state_dict(state, strict=True)
        self.eval().requires_grad_(False)

    def forward(self, waveform):
        lengths = torch.ones(waveform.shape[0], device=waveform.device)
        features = self.normalization(self.features(waveform), lengths)
        return self.embedding(features, lengths).squeeze(1)


def speaker_spec(directory):
    return dict(repository="speechbrain/spkrec-ecapa-voxceleb",
                weights_sha256=sha256_file(Path(directory) / "embedding_model.ckpt"),
                recipe_sha256=sha256_file(Path(directory) / "hyperparams.yaml"),
                speechbrain_version=importlib.metadata.version("speechbrain"),
                torchaudio_version=importlib.metadata.version("torchaudio"),
                sample_rate=16000, preprocessing="mono_float32_sentence_mean_fbank80_v1",
                output="encode_batch_normalize_false_192d", license="Apache-2.0_model_card")


def similarity_report(references, targets):
    references = np.asarray(references, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if references.ndim != 2 or references.shape != targets.shape or len(references) < 2:
        return dict(count=len(references), note="Need at least two distinct IDs")
    references = references / np.linalg.norm(references, axis=1, keepdims=True)
    targets = targets / np.linalg.norm(targets, axis=1, keepdims=True)
    similarities = targets @ references.T
    positive = np.diag(similarities)
    negative = similarities[np.arange(len(targets)), np.roll(np.arange(len(targets)), 1)]
    return dict(count=len(targets), same_song_cosine_mean=float(np.mean(positive)),
                same_song_cosine_median=float(np.median(positive)),
                different_id_cosine_mean=float(np.mean(negative)),
                different_id_cosine_median=float(np.median(negative)),
                same_song_reference_retrieval_top1=float(np.mean(np.argmax(similarities, axis=1) == np.arange(len(targets)))),
                pairwise_auc=float(np.mean(positive[:, None] > negative[None, :]) +
                                   0.5 * np.mean(positive[:, None] == negative[None, :])),
                caveat="Same-song non-overlapping crops, not cross-song or verified singer identity")


@torch.inference_mode()
def references(args):
    root, output = Path(args.audio_root).resolve(), Path(args.output).resolve()
    all_rows = read_rows(root)
    unique = {}
    for row in all_rows:
        unique.setdefault(row["song_id"], row)
    rows = list(unique.values())
    if args.limit:
        rows = rows[:args.limit]
    spec = speaker_spec(args.encoder)
    encoder_id = identity(spec)
    configure(output, "reference", dict(audio_root=str(root), encoder_id=encoder_id, spec=spec))
    model = SpeechBrainVoiceEncoder(args.encoder).to(args.device)
    embeddings, probes, completed, errors = [], [], [], []
    started = time.monotonic()
    for index, row in enumerate(rows, 1):
        try:
            values = []
            for field, name in (("reference_audio_16k", "reference.npy"),
                                ("target_vocal_audio", "identity_probe.npy")):
                source, audio_hash = audio_path(root, row, field)
                expected = dict(encoder_id=encoder_id, audio_sha256=audio_hash,
                                source_audio=str(source), sample_rate=16000)
                path = output / "songs" / row["song_id"] / name
                value = load_cached(path, expected)
                if value is None:
                    value = model(resample_mono_16k(source).to(args.device))[0].float().cpu().numpy()
                    if value.shape != (192,) or np.linalg.norm(value) < 1e-8:
                        raise ValueError("Unexpected ECAPA embedding")
                    save_array(path, value, expected)
                values.append(value)
            embeddings.append(values[0])
            probes.append(values[1])
            completed.append(row)
        except Exception as error:
            errors.append(dict(song_id=row["song_id"], error=str(error)))
        if index % 50 == 0 or index == len(rows):
            print(f"References {index}/{len(rows)}, errors={len(errors)}, elapsed={time.monotonic()-started:.0f}s", flush=True)
    report = dict(encoder_id=encoder_id, songs=len(completed), errors=errors,
                  overall=similarity_report(embeddings, probes), elapsed_seconds=time.monotonic()-started,
                  license=spec["license"], note="Speaker model evaluated as an experimental singing identity reference")
    for split in ("train", "validation"):
        indices = [index for index, row in enumerate(completed) if row["split"] == split]
        report[split] = similarity_report([embeddings[index] for index in indices], [probes[index] for index in indices])
    write_json(output / "reference_report.json", report)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    if errors:
        raise RuntimeError("Some reference features failed; see reference_report.json")


@torch.inference_mode()
def latents(args):
    from yue2.modeling_vae import YuE2VAE
    root, output = Path(args.audio_root).resolve(), Path(args.output).resolve()
    rows = read_rows(root)
    if args.limit:
        rows = rows[:args.limit]
    vae_id = weight_id(args.vae)
    configure(output, "latent", dict(audio_root=str(root), vae_path=str(Path(args.vae).resolve()),
                                      vae_id=vae_id, sample=False, dtype="float32"))
    vae = YuE2VAE.from_pretrained(args.vae, local_files_only=True, device=args.device)
    shapes, errors = Counter(), []
    started = time.monotonic()
    for index, row in enumerate(rows, 1):
        try:
            source, audio_hash = audio_path(root, row, "target_audio")
            expected = dict(vae_id=vae_id, audio_sha256=audio_hash, source_audio=str(source), sample=False)
            path = output / "songs" / row["song_id"] / (Path(row["target_audio"]).stem + ".npy")
            value = load_cached(path, expected)
            if value is None:
                audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
                if sample_rate != 48000 or audio.shape[1] != 2:
                    raise ValueError("Expected 48 kHz stereo target")
                value = vae.encode(torch.from_numpy(audio.T.copy())[None]).squeeze(0).T.cpu().numpy()
                if value.ndim != 2 or value.shape[1] != 64 or not 0 < value.shape[0] <= 500:
                    raise ValueError("Unexpected VAE latent shape")
                save_array(path, value, dict(expected, frames=len(value)))
                if not (output / "reconstruction_check.wav").exists():
                    reconstructed = vae.decode(torch.from_numpy(value.T.copy())[None])[0].T.cpu().numpy()
                    sf.write(output / "reconstruction_check.wav", reconstructed, 48000, subtype="FLOAT")
                    write_json(output / "reconstruction_check.json", dict(sample_id=row["sample_id"],
                               source_audio=str(source), source_frames=len(audio), decoded_frames=len(reconstructed),
                               finite=bool(np.isfinite(reconstructed).all()),
                               peak=float(np.max(np.abs(reconstructed))), vae_id=vae_id,
                               note="Listen against source; finite output is not an audio-quality evaluation"))
            shapes[str(list(value.shape))] += 1
        except Exception as error:
            errors.append(dict(sample_id=row["sample_id"], error=str(error)))
        if index % 50 == 0 or index == len(rows):
            print(f"Latents {index}/{len(rows)}, errors={len(errors)}, elapsed={time.monotonic()-started:.0f}s", flush=True)
    report = dict(vae_id=vae_id, attempted=len(rows), completed=sum(shapes.values()), shapes=dict(shapes),
                  errors=errors, elapsed_seconds=time.monotonic()-started, sample=False)
    write_json(output / "latent_report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if errors:
        raise RuntimeError("Some latent features failed; see latent_report.json")


def index_features(args):
    root, output = Path(args.audio_root).resolve(), Path(args.output).resolve()
    rows = read_rows(root)
    reference_config = json.loads((output / "reference_config.json").read_text())
    latent_config = json.loads((output / "latent_config.json").read_text())
    base_id = weight_id(args.base_model)
    encoder_id, vae_id = reference_config["encoder_id"], latent_config["vae_id"]
    records = []
    for row in rows:
        directory = Path("songs") / row["song_id"]
        reference_path = directory / "reference.npy"
        target_path = directory / (Path(row["target_audio"]).stem + ".npy")
        reference = load_cached(output / reference_path, dict(encoder_id=encoder_id))
        target = load_cached(output / target_path, dict(vae_id=vae_id, sample=False))
        if reference is None or target is None:
            raise ValueError(f"Missing features for {row['sample_id']}")
        if reference.shape != (192,) or target.ndim != 2 or target.shape[1] != 64:
            raise ValueError("Feature dimensions mismatch")
        record = dict(row, reference_embedding=str(reference_path), target_latent=str(target_path),
                      encoder_id=encoder_id, vae_id=vae_id, base_id=base_id,
                      stage="features_without_semantics", alignment="missing_native_semantics",
                      training_ready=False, missing=["prefix", "aligned_semantic_tokens"],
                      target_frames=len(target), reference_dim=len(reference))
        for key in ("reference_audio", "reference_audio_16k", "target_audio", "target_vocal_audio", "metadata"):
            record[key] = os.path.relpath(root / row[key], output)
        records.append(record)
    for split in ("train", "validation"):
        path = output / f"features_{split}.jsonl"
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records if row["split"] == split))
        temporary.replace(path)
    report = dict(samples=len(records), songs=len({row["song_id"] for row in records}),
                  samples_per_split=dict(Counter(row["split"] for row in records)),
                  encoder_id=encoder_id, vae_id=vae_id, base_id=base_id,
                  reference_dim=192, target_shapes=dict(Counter(str([row["target_frames"], 64]) for row in records)),
                  arrays_hash_verified=True, training_ready=False, missing=["prefix", "aligned_semantic_tokens"])
    write_json(output / "feature_summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    reference_parser = commands.add_parser("references")
    reference_parser.add_argument("--encoder", required=True)
    reference_parser.set_defaults(action=references)
    latent_parser = commands.add_parser("latents")
    latent_parser.add_argument("--vae", required=True)
    latent_parser.set_defaults(action=latents)
    index_parser = commands.add_parser("index")
    index_parser.add_argument("--base-model", required=True)
    index_parser.set_defaults(action=index_features)
    for command in (reference_parser, latent_parser, index_parser):
        command.add_argument("--audio-root", required=True)
        command.add_argument("--output", required=True)
    for command in (reference_parser, latent_parser):
        command.add_argument("--device", default="cuda")
        command.add_argument("--limit", type=int)
        command.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args(argv)
    if hasattr(args, "cpu_threads"):
        if args.cpu_threads < 1 or (args.limit is not None and args.limit < 1):
            parser.error("cpu-threads and limit must be positive")
        torch.set_num_threads(args.cpu_threads)
    args.action(args)


if __name__ == "__main__":
    main()
