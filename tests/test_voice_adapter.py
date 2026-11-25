"""CPU regression and training checks; no release weights or downloads."""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM
from yue2.nar import Chunk, CachedNAR
from yue2.voice_adapter import VoiceAdapter, VoiceChunkFlow, flow_loss, synthesize_voice
from yue2.voice_training import (load_adapter, load_manifest, main, save_adapter,
                                 validate_dataset, weight_id)


@pytest.fixture
def model():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(731)
        config = YuE2Config(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=4,
                            vocab_size=32, max_position_embeddings=128,
                            latent_dim=64, max_latent_frames=128)
        value = YuE2ForCausalLM(config).eval().requires_grad_(False)
    yield value
    torch.set_num_threads(previous_threads)


def adapter_for(model):
    return VoiceAdapter(model.config.hidden_size, model.config.num_hidden_layers, 8, rank=4)


def chunk():
    return Chunk([1, 2, 3], torch.randn(5, 64))


@pytest.mark.parametrize("shift", [1.0, 2.0])
def test_zero_adapter_and_disabled_adapter_preserve_base(model, shift):
    model.config.timestep_shift = shift
    item = chunk()
    base = CachedNAR(model, item)
    adapter = adapter_for(model)
    modified = VoiceChunkFlow(model, item, adapter, torch.randn(8))
    for raw_time in (-4.0, 0.0, 3.0):
        torch.testing.assert_close(modified.velocity(item.noise, raw_time),
                                   base.velocity(item.noise, raw_time), rtol=0, atol=0)
    torch.testing.assert_close(modified.solve(steps=3), base.solve(steps=3), rtol=0, atol=0)
    with torch.no_grad():
        for projection in adapter.modulations.values():
            projection.weight.normal_()
    disabled = VoiceChunkFlow(model, item, adapter, torch.randn(8), strength=0)
    torch.testing.assert_close(disabled.solve(steps=3), base.solve(steps=3), rtol=0, atol=0)


@pytest.mark.parametrize("checkpoint_layers", [False, True])
def test_training_only_updates_adapter(model, checkpoint_layers):
    before = {key: value.clone() for key, value in model.state_dict().items()}
    adapter = adapter_for(model)
    original_adapter = {key: value.clone() for key, value in adapter.state_dict().items()}
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.01)
    item = chunk()
    reference = torch.randn(8)
    for iteration in range(2):
        engine = VoiceChunkFlow(model, item, adapter, reference,
                                checkpoint_layers=checkpoint_layers)
        loss = flow_loss(engine, torch.randn(5, 64), time=0.4)
        optimizer.zero_grad()
        loss.backward()
        assert all(parameter.grad is None for parameter in model.parameters())
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                   for parameter in adapter.parameters())
        optimizer.step()
    assert any(not torch.equal(value, original_adapter[key])
               for key, value in adapter.state_dict().items())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    first = VoiceChunkFlow(model, item, adapter, reference).solve(steps=2)
    second = VoiceChunkFlow(model, item, adapter, -reference).solve(steps=2)
    assert not torch.equal(first, second)


def test_checkpointing_matches_gradients(model):
    item = chunk()
    reference, target = torch.randn(8), torch.randn(5, 64)
    adapter = adapter_for(model)
    gradients = []
    for enabled in (False, True):
        adapter.zero_grad()
        engine = VoiceChunkFlow(model, item, adapter, reference, checkpoint_layers=enabled)
        flow_loss(engine, target, noise=item.noise, time=0.3).backward()
        gradients.append([parameter.grad.clone() for parameter in adapter.parameters()])
    for actual, expected in zip(*gradients):
        torch.testing.assert_close(actual, expected)


def test_flow_direction():
    target, noise = torch.randn(4, 64), torch.randn(4, 64)

    class ExactFlow:
        device = "cpu"
        chunk = Chunk([], torch.empty(4, 64))

        def velocity(self, state, raw_t):
            torch.testing.assert_close(state, 0.75 * target + 0.25 * noise)
            assert abs(raw_t - np.log(1 / 3)) < 1e-7
            return noise - target

    assert flow_loss(ExactFlow(), target, noise=noise, time=0.25).item() == 0


def test_explicit_synthesis_with_multiple_chunks(model):
    items = [chunk(), chunk()]
    adapter = adapter_for(model)
    expected = torch.cat([CachedNAR(model, item).solve(steps=2) for item in items])
    with patch("yue2.voice_adapter.song_chunks", return_value=items):
        actual = synthesize_voice(model, [1], [2], 42, adapter, torch.randn(8), steps=2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_input_and_freeze_guards(model):
    adapter = adapter_for(model)
    with pytest.raises(ValueError, match="nonzero"):
        adapter.encode_reference(torch.zeros(8))
    with pytest.raises(ValueError, match="embedding"):
        adapter.encode_reference(torch.ones(7))
    with pytest.raises(ValueError, match="strength"):
        VoiceChunkFlow(model, chunk(), adapter, torch.ones(8), strength=float("nan"))
    model.requires_grad_(True)
    with pytest.raises(ValueError, match="Freeze"):
        VoiceChunkFlow(model, chunk(), adapter, torch.ones(8))


def test_adapter_roundtrip_and_identity_checks(model, tmp_path):
    adapter = adapter_for(model)
    directory = tmp_path / "adapter"
    save_adapter(directory, adapter, dict(base_id="base", encoder_id="encoder", vae_id="vae"))
    restored, metadata = load_adapter(directory, base_id="base", encoder_id="encoder", device="cpu")
    assert metadata["vae_id"] == "vae"
    for key, value in adapter.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="mismatch"):
        load_adapter(directory, base_id="other", encoder_id="encoder", device="cpu")
    with pytest.raises(FileExistsError):
        save_adapter(directory, adapter, {})
    with (directory / "adapter.safetensors").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_adapter(directory, base_id="base", encoder_id="encoder", device="cpu")


def write_sample(directory, name, base_id="base", alignment="paired"):
    source = directory / name
    source.mkdir()
    np.save(source / "prefix.npy", np.array([1, 2], dtype=np.int32))
    np.save(source / "semantic.npy", np.array([0, 1, 2], dtype=np.int32))
    np.save(source / "target.npy", np.zeros((3, 64), dtype=np.float32))
    np.save(source / "reference.npy", np.ones(8, dtype=np.float32))
    row = dict(song_id=name, singer_id=name, encoder_id="encoder", base_id=base_id,
               vae_id="vae", alignment=alignment, semantic_dir=name,
               target_latent=f"{name}/target.npy", reference_embedding=f"{name}/reference.npy")
    manifest = directory / f"{name}.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    return manifest


def test_manifest_rejects_unaligned_and_length_mismatch(tmp_path):
    manifest = write_sample(tmp_path, "song", alignment="unverified_ar")
    with pytest.raises(ValueError, match="not a label"):
        load_manifest(manifest)
    assert len(load_manifest(manifest, allow_unaligned=True)) == 1
    np.save(tmp_path / "song/target.npy", np.zeros((4, 64), dtype=np.float32))
    with pytest.raises(ValueError, match="no automatic"):
        load_manifest(manifest, allow_unaligned=True)


def test_dataset_rejects_leakage_and_encoder_mismatch(tmp_path):
    train = load_manifest(write_sample(tmp_path, "train"))
    validation = load_manifest(write_sample(tmp_path, "validation"))
    validate_dataset(train, validation)
    with pytest.raises(ValueError, match="song_id"):
        validate_dataset(train, train)
    validation[0]["encoder_id"] = "other"
    with pytest.raises(ValueError, match="encoder_id"):
        validate_dataset(train, validation)


def test_manifest_checks_preprocessing_metadata(tmp_path):
    manifest = write_sample(tmp_path, "song")
    (tmp_path / "song/reference.npy.json").write_text(json.dumps({"encoder_id": "wrong"}))
    with pytest.raises(ValueError, match="preprocessing metadata"):
        load_manifest(manifest)


def test_embed_command_with_torchscript_wrapper(tmp_path):
    import soundfile as sf

    class TestEncoder(torch.nn.Module):
        def forward(self, waveform):
            return torch.stack((waveform.mean(-1), waveform.square().mean(-1)), dim=-1)

    source = tmp_path / "reference.wav"
    sf.write(source, np.full(1600, 0.1, dtype=np.float32), 16000)
    encoder = tmp_path / "test-only-encoder.ts"
    torch.jit.trace(TestEncoder(), torch.ones(1, 1600)).save(str(encoder))
    output = tmp_path / "embedding.npy"
    main(["embed", "--audio", str(source), "--encoder", str(encoder),
          "--sample-rate", "16000", "--output", str(output), "--device", "cpu"])
    assert np.load(output).shape == (2,)
    assert json.loads(Path(str(output) + ".json").read_text())["encoder_id"]


def test_cli_real_training_smoke(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = YuE2Config(hidden_size=8, intermediate_size=16, num_hidden_layers=2,
                            num_attention_heads=2, num_key_value_heads=1, head_dim=4,
                            vocab_size=184704, max_position_embeddings=64, max_latent_frames=64)
        model = YuE2ForCausalLM(config)
        model_path = tmp_path / "model"
        model.save_pretrained(model_path)
        base_id = weight_id(model_path)
        train = write_sample(tmp_path, "train", base_id)
        validation = write_sample(tmp_path, "validation", base_id)
        output = tmp_path / "run"
        main(["train", "--model", str(model_path), "--manifest", str(train),
              "--validation", str(validation), "--output", str(output),
              "--device", "cpu", "--steps", "2", "--save-every", "1", "--rank", "4",
              "--checkpoint-layers"])
        assert (output / "step-000002/adapter.safetensors").exists()
        assert weight_id(model_path) == base_id
        metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        assert len(metrics) == 2 and all(np.isfinite(row["validation_loss"]) for row in metrics)
    finally:
        torch.set_num_threads(previous_threads)
