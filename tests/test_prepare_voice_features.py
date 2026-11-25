"""Feature cache and manifest tests without downloaded model dependencies."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location("prepare_voice_features", Path(__file__).parents[1] / "tools/prepare_voice_features.py")
features = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(features)


def test_feature_cache_checks_content_identity_and_shape(tmp_path):
    path = tmp_path / "reference.npy"
    expected = dict(encoder_id="encoder", audio_sha256="audio")
    assert features.load_cached(path, expected) is None
    features.save_array(path, np.ones(192), expected)
    result = features.load_cached(path, expected)
    assert result.shape == (192,) and result.dtype == np.float32
    with pytest.raises(ValueError, match="metadata mismatch"):
        features.load_cached(path, dict(encoder_id="different"))
    with path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        features.load_cached(path, expected)


def test_partial_cache_and_changed_config_are_rejected(tmp_path):
    path = tmp_path / "reference.npy"
    np.save(path, np.ones(192, dtype=np.float32))
    with pytest.raises(ValueError, match="Incomplete"):
        features.load_cached(path, {})
    features.configure(tmp_path, "reference", dict(encoder_id="first"))
    features.configure(tmp_path, "reference", dict(encoder_id="first"))
    with pytest.raises(ValueError, match="Changed"):
        features.configure(tmp_path, "reference", dict(encoder_id="second"))


def test_similarity_report_is_a_same_song_proxy():
    embeddings = np.eye(4, dtype=np.float32)
    report = features.similarity_report(embeddings, embeddings)
    assert report["same_song_reference_retrieval_top1"] == 1
    assert report["different_id_cosine_mean"] == 0
    assert report["pairwise_auc"] == 1
    assert "not cross-song" in report["caveat"]
    assert features.similarity_report([], [])["count"] == 0


def write_audio_manifests(root):
    root.mkdir()
    for split, song in (("train", "song-a"), ("validation", "song-b")):
        row = dict(sample_id=f"{song}-01", song_id=song, singer_id=song, split=split,
                   reference_audio=f"clips/{song}/reference.wav",
                   reference_audio_16k=f"clips/{song}/reference_16k.wav",
                   target_audio=f"clips/{song}/target_01.wav",
                   target_vocal_audio=f"clips/{song}/target_vocal_01.wav",
                   metadata=f"clips/{song}/metadata.json")
        (root / f"audio_{split}.jsonl").write_text(json.dumps(row) + "\n")


def test_feature_index_stays_explicitly_not_trainable(tmp_path, monkeypatch):
    root, output = tmp_path / "audio", tmp_path / "features"
    write_audio_manifests(root)
    output.mkdir()
    (output / "reference_config.json").write_text(json.dumps(dict(encoder_id="encoder")))
    (output / "latent_config.json").write_text(json.dumps(dict(vae_id="vae")))
    for song in ("song-a", "song-b"):
        directory = output / "songs" / song
        features.save_array(directory / "reference.npy", np.ones(192), dict(encoder_id="encoder"))
        features.save_array(directory / "target_01.npy", np.zeros((375, 64)), dict(vae_id="vae", sample=False))
    monkeypatch.setattr(features, "weight_id", lambda path: "base")
    features.index_features(SimpleNamespace(audio_root=root, output=output, base_model="unused"))
    row = json.loads((output / "features_train.jsonl").read_text())
    assert row["training_ready"] is False
    assert row["alignment"] == "missing_native_semantics"
    assert row["missing"] == ["prefix", "aligned_semantic_tokens"]
    assert "semantic_dir" not in row
    assert row["target_frames"] == 375
    assert (output / row["reference_embedding"]).is_file()
    assert not (output / "train.jsonl").exists()


def test_audio_manifest_split_leakage_rejected(tmp_path):
    root = tmp_path / "audio"
    write_audio_manifests(root)
    path = root / "audio_validation.jsonl"
    row = json.loads(path.read_text())
    row["singer_id"] = "song-a"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Singer overlap"):
        features.read_rows(root)
