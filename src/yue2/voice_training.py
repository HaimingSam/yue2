"""Standalone experimental training utilities: python -m yue2.voice_training --help."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from .nar import song_chunks
from .protocol import CODEC_SIZE
from .storage import identity, model_identity, sha256_file, write_json
from .voice_adapter import VoiceAdapter, VoiceChunkFlow, flow_loss, synthesize_voice


def load_array(path, *, integer=False):
    array = np.load(path, allow_pickle=False)
    if integer and array.dtype.kind not in "iu":
        raise ValueError(f"Expected integer token array: {path}")
    if not integer and array.dtype.kind != "f":
        raise ValueError(f"Expected floating point array: {path}")
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite array: {path}")
    return torch.from_numpy(array.astype(np.int64 if integer else np.float32, copy=True))


def load_manifest(path, *, allow_unaligned=False, max_frames=500):
    path = Path(path).resolve()
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("song_id", "singer_id", "encoder_id", "vae_id", "base_id"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"{path}:{line_number}: missing {key}")
        alignment = row.get("alignment")
        if alignment not in {"paired", "unverified_ar"}:
            raise ValueError("alignment must be paired or unverified_ar")
        if alignment == "unverified_ar" and not allow_unaligned:
            raise ValueError("AR-generated c is not a label for original audio; use --allow-unaligned-experiment only for an explicit experiment")
        semantic_dir = path.parent / row["semantic_dir"]
        target_path = path.parent / row["target_latent"]
        reference_path = path.parent / row["reference_embedding"]
        for array_path, key in ((target_path, "vae_id"), (reference_path, "encoder_id")):
            sidecar = Path(str(array_path) + ".json")
            if sidecar.exists() and json.loads(sidecar.read_text()).get(key) != row[key]:
                raise ValueError(f"{key} does not match preprocessing metadata: {sidecar}")
        prefix = load_array(semantic_dir / "prefix.npy", integer=True)
        codec = load_array(semantic_dir / "semantic.npy", integer=True)
        target = load_array(target_path)
        reference = load_array(reference_path)
        if prefix.ndim != 1 or not len(prefix) or (prefix < 0).any():
            raise ValueError("prefix must be a nonempty nonnegative integer vector")
        if codec.ndim != 1 or not 0 < len(codec) <= max_frames:
            raise ValueError(f"Expected 1..{max_frames} semantic frames; prepare short clips, do not silently crop")
        if (codec < 0).any() or (codec >= CODEC_SIZE).any():
            raise ValueError("Semantic IDs must be unshifted, in [0,32768)")
        if target.shape != (len(codec), 64):
            raise ValueError("Target must be [semantic_frames,64]; no automatic stretching or truncation")
        if reference.ndim != 1 or not len(reference) or reference.norm() < 1e-8:
            raise ValueError("Reference must be one nonzero embedding vector")
        source_hashes = {str(source.resolve()): sha256_file(source) for source in
                         (semantic_dir / "prefix.npy", semantic_dir / "semantic.npy",
                          target_path, reference_path)}
        records.append(dict(row, input_hashes=source_hashes, prefix=prefix.tolist(), codec=codec.tolist(),
                            target=target, reference=reference))
    if not records:
        raise ValueError("Manifest is empty")
    return records


def validate_dataset(train, validation):
    if {row["song_id"] for row in train} & {row["song_id"] for row in validation}:
        raise ValueError("Training and validation must not share song_id")
    if {row["singer_id"] for row in train} & {row["singer_id"] for row in validation}:
        raise ValueError("This first experiment requires held-out validation singers")
    records = train + validation
    for key in ("encoder_id", "vae_id", "base_id"):
        if len({row[key] for row in records}) != 1:
            raise ValueError(f"All samples must use the same {key}")
    if len({len(row["reference"]) for row in records}) != 1:
        raise ValueError("Reference embedding dimensions differ")


def make_engine(model, adapter, row, checkpoint_layers=False):
    chunks = song_chunks(row["prefix"], row["codec"], seed=0)
    if len(chunks) != 1:
        raise ValueError("Training accepts one short chunk per sample")
    if max(chunks[0].ar_tokens) >= model.config.vocab_size:
        raise ValueError("Prefix contains IDs outside this model vocabulary")
    return VoiceChunkFlow(model, chunks[0], adapter, row["reference"],
                          checkpoint_layers=checkpoint_layers)


def load_base(path, device):
    from .modeling_yue2 import YuE2ForCausalLM
    if not Path(path).is_dir():
        raise ValueError("Supply an existing local YuE2 model directory; no implicit downloads")
    dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
    return YuE2ForCausalLM.from_pretrained(
        path, local_files_only=True, torch_dtype=dtype,
        low_cpu_mem_usage=True).to(device).eval().requires_grad_(False)


def weight_id(path):
    return identity(model_identity(path))


def save_adapter(path, adapter, metadata):
    from safetensors.torch import save_file
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    save_file({key: value.detach().cpu().contiguous()
               for key, value in adapter.state_dict().items()}, str(path / "adapter.safetensors"))
    write_json(path / "adapter.json", dict(metadata, version=1, spec=adapter.spec,
                                           weights_sha256=sha256_file(path / "adapter.safetensors")))


def load_adapter(path, *, base_id, encoder_id, device):
    from safetensors.torch import load_file
    path = Path(path)
    metadata = json.loads((path / "adapter.json").read_text())
    if metadata["version"] != 1:
        raise ValueError("Unsupported adapter version")
    if metadata["base_id"] != base_id or metadata["encoder_id"] != encoder_id:
        raise ValueError("Adapter base weights or reference encoder identity mismatch")
    if sha256_file(path / "adapter.safetensors") != metadata["weights_sha256"]:
        raise ValueError("Adapter weights checksum mismatch")
    adapter = VoiceAdapter(**metadata["spec"])
    adapter.load_state_dict(load_file(str(path / "adapter.safetensors")), strict=True)
    return adapter.to(device).eval(), metadata


@torch.no_grad()
def validation_loss(model, adapter, rows):
    losses = []
    adapter.eval()
    for row in rows:
        engine = make_engine(model, adapter, row)
        try:
            generator = torch.Generator(device="cpu").manual_seed(2026)
            noise = torch.randn(row["target"].shape, generator=generator)
            losses.append(flow_loss(engine, row["target"], time=0.5, noise=noise).item())
        finally:
            engine.close()
    adapter.train()
    return sum(losses) / len(losses)


def train(args):
    if args.steps < 1 or args.save_every < 1 or args.max_frames < 1:
        raise ValueError("steps, save-every and max-frames must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Learning rate must be finite and positive")
    rows = load_manifest(args.manifest, allow_unaligned=args.allow_unaligned_experiment,
                         max_frames=args.max_frames)
    validation = load_manifest(args.validation, allow_unaligned=args.allow_unaligned_experiment,
                               max_frames=args.max_frames)
    validate_dataset(rows, validation)
    base_id = weight_id(args.model)
    if rows[0]["base_id"] != base_id:
        raise ValueError("Manifest base_id differs from supplied model weights")
    if any(row["alignment"] == "unverified_ar" for row in rows + validation):
        print("WARNING: unaligned AR/original pairs; decreasing loss does NOT establish voice fidelity.", flush=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    randomizer = random.Random(args.seed)
    model = load_base(args.model, args.device)
    start = model.config.num_hidden_layers // 2 if args.start_layer is None else args.start_layer
    adapter = VoiceAdapter(model.config.hidden_size, model.config.num_hidden_layers,
                           len(rows[0]["reference"]), args.rank, start).to(args.device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.01)
    metadata = dict(base_id=base_id, encoder_id=rows[0]["encoder_id"], vae_id=rows[0]["vae_id"],
                    seed=args.seed, learning_rate=args.lr, max_frames=args.max_frames,
                    manifest_sha256=sha256_file(args.manifest),
                    validation_sha256=sha256_file(args.validation),
                    input_hashes={key: value for row in rows + validation
                                  for key, value in row["input_hashes"].items()},
                    unaligned_experiment=any(row["alignment"] == "unverified_ar" for row in rows + validation),
                    objective="linear_clean_t0_noise_t1_uniform_time", batch_size=1,
                    checkpoint_layers=args.checkpoint_layers)
    write_json(output / "run.json", metadata)
    with (output / "metrics.jsonl").open("w") as log:
        order = []
        for step in range(1, args.steps + 1):
            if not order:
                order = list(range(len(rows)))
                randomizer.shuffle(order)
            row = rows[order.pop()]
            engine = make_engine(model, adapter, row, args.checkpoint_layers)
            try:
                optimizer.zero_grad(set_to_none=True)
                loss = flow_loss(engine, row["target"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                metric = dict(step=step, loss=loss.item(), grad_norm=norm.item())
            finally:
                engine.close()
            if step % args.save_every == 0 or step == args.steps:
                metric["validation_loss"] = validation_loss(model, adapter, validation)
                save_adapter(output / f"step-{step:06d}", adapter, dict(metadata, step=step))
            log.write(json.dumps(metric, allow_nan=False) + "\n")
            log.flush()
            print(json.dumps(metric), flush=True)


def encode(args):
    import soundfile as sf
    from .modeling_vae import YuE2VAE
    output = Path(args.output)
    if output.suffix != ".npy" or output.exists():
        raise ValueError("Use a new .npy output path")
    info = sf.info(args.audio)
    if info.samplerate != 48000 or info.channels != 2:
        raise ValueError("Target must be 48 kHz stereo; explicitly resample before encoding")
    if not 0 < info.duration <= args.max_seconds:
        raise ValueError("Prepare short target clips before encoding")
    audio, _ = sf.read(args.audio, dtype="float32", always_2d=True)
    vae = YuE2VAE.from_pretrained(args.vae, local_files_only=True, device=args.device)
    latent = vae.encode(torch.from_numpy(audio.T.copy())[None]).squeeze(0).T.cpu().numpy()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, latent)
    write_json(str(output) + ".json", dict(vae_id=weight_id(args.vae),
               audio_sha256=sha256_file(args.audio), frames=len(latent), sample=False))


def embed(args):
    import soundfile as sf
    output = Path(args.output)
    if output.suffix != ".npy" or output.exists():
        raise ValueError("Use a new .npy output path")
    info = sf.info(args.audio)
    if info.samplerate != args.sample_rate or info.channels != 1:
        raise ValueError("Reference must be mono at the explicitly specified encoder sample rate")
    if not 0 < info.duration <= args.max_seconds:
        raise ValueError("Prepare a short reference crop first")
    audio, _ = sf.read(args.audio, dtype="float32")
    if not np.isfinite(audio).all() or np.max(np.abs(audio)) < 1e-6:
        raise ValueError("Reference is silent or non-finite")
    encoder = torch.jit.load(args.encoder, map_location=args.device).eval()
    with torch.no_grad():
        embedding = encoder(torch.from_numpy(audio)[None].to(args.device))
    if not isinstance(embedding, torch.Tensor) or embedding.ndim != 2 or embedding.shape[0] != 1:
        raise ValueError("Encoder wrapper must accept [1,samples] and return [1,embedding_dim]")
    embedding = embedding[0].detach().float().cpu()
    if not torch.isfinite(embedding).all() or embedding.norm() < 1e-8:
        raise ValueError("Encoder returned an invalid embedding")
    encoder_id = identity(dict(sha256=sha256_file(args.encoder), sample_rate=args.sample_rate,
                               preprocessing="mono_float32_no_normalization_v1"))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, embedding.numpy())
    write_json(str(output) + ".json", dict(encoder_id=encoder_id,
               audio_sha256=sha256_file(args.audio), dimension=len(embedding)))


def render(args):
    import soundfile as sf
    from .modeling_vae import YuE2VAE
    output = Path(args.output)
    if output.exists() or output.suffix != ".wav":
        raise ValueError("Use a new .wav output path")
    if args.steps < 1:
        raise ValueError("steps must be positive")
    adapter, metadata = load_adapter(args.adapter, base_id=weight_id(args.model),
                                      encoder_id=args.encoder_id, device=args.device)
    if weight_id(args.vae) != metadata["vae_id"]:
        raise ValueError("VAE weights differ from training targets")
    model = load_base(args.model, args.device)
    directory = Path(args.semantic_dir)
    prefix = load_array(directory / "prefix.npy", integer=True)
    codec = load_array(directory / "semantic.npy", integer=True)
    if prefix.ndim != 1 or codec.ndim != 1:
        raise ValueError("Prefix and semantic arrays must be vectors")
    reference = load_array(args.reference_embedding)
    latent = synthesize_voice(model, prefix.tolist(), codec.tolist(), args.seed,
                              adapter, reference, steps=args.steps, strength=args.strength)
    del model, adapter
    if torch.device(args.device).type == "cuda":
        torch.cuda.empty_cache()
    vae = YuE2VAE.from_pretrained(args.vae, decoder_only=True, local_files_only=True, device=args.device)
    audio = vae.decode_tiled(latent.T[None], output_device="cpu")[0].T.numpy()
    if not np.isfinite(audio).all():
        raise FloatingPointError("Non-finite decoded audio")
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, audio, vae.config.sample_rate, subtype="FLOAT")
    write_json(str(output) + ".json", dict(adapter=str(Path(args.adapter).resolve()),
               base_id=metadata["base_id"], vae_id=metadata["vae_id"], encoder_id=args.encoder_id,
               reference_sha256=sha256_file(args.reference_embedding),
               prefix_sha256=sha256_file(directory / "prefix.npy"),
               semantic_sha256=sha256_file(directory / "semantic.npy"),
               seed=args.seed, steps=args.steps, strength=args.strength))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    identify = commands.add_parser("identify", help="Print local checkpoint fingerprint for manifests")
    identify.add_argument("--model", required=True)
    identify.set_defaults(run=lambda args: print(weight_id(args.model)))
    encoding = commands.add_parser("encode", help="Encode a short 48 kHz stereo target with the frozen VAE")
    encoding.add_argument("--audio", required=True)
    encoding.add_argument("--vae", required=True)
    encoding.add_argument("--max-seconds", type=float, default=20)
    encoding.set_defaults(run=encode)
    embedding = commands.add_parser("embed", help="Use a trusted pretrained TorchScript identity-encoder wrapper")
    embedding.add_argument("--audio", required=True)
    embedding.add_argument("--encoder", required=True)
    embedding.add_argument("--sample-rate", type=int, required=True)
    embedding.add_argument("--max-seconds", type=float, default=30)
    embedding.set_defaults(run=embed)
    training = commands.add_parser("train", help="Train only the new NAR adapter, batch size 1")
    training.add_argument("--model", required=True)
    training.add_argument("--manifest", required=True)
    training.add_argument("--validation", required=True)
    training.add_argument("--steps", type=int, default=1000)
    training.add_argument("--save-every", type=int, default=100)
    training.add_argument("--lr", type=float, default=1e-4)
    training.add_argument("--rank", type=int, default=64)
    training.add_argument("--start-layer", type=int)
    training.add_argument("--max-frames", type=int, default=500)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--checkpoint-layers", action="store_true")
    training.add_argument("--allow-unaligned-experiment", action="store_true")
    training.set_defaults(run=train)
    rendering = commands.add_parser("render", help="Render saved semantic tokens with an explicitly supplied adapter")
    rendering.add_argument("--model", required=True)
    rendering.add_argument("--vae", required=True)
    rendering.add_argument("--adapter", required=True)
    rendering.add_argument("--semantic-dir", required=True)
    rendering.add_argument("--reference-embedding", required=True)
    rendering.add_argument("--encoder-id", required=True)
    rendering.add_argument("--strength", type=float, default=1.0)
    rendering.add_argument("--steps", type=int, default=32)
    rendering.add_argument("--seed", type=int, default=42)
    rendering.set_defaults(run=render)
    for command in (encoding, embedding, training, rendering):
        command.add_argument("--device", default="cuda")
        command.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
