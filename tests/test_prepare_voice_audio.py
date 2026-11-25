"""Audio preparation checks using synthetic signals, not private recordings."""
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import soundfile as sf


SPEC = importlib.util.spec_from_file_location("prepare_voice_audio", Path(__file__).parents[1] / "tools/prepare_voice_audio.py")
preparation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preparation)


def test_selection_is_active_and_nonoverlapping():
    levels = np.full(1500, -100.0)
    levels[100:500] = -20
    levels[700:1100] = -25
    selected = preparation.select_windows(levels)
    assert len(selected["targets"]) == 2
    assert selected["reference"]["active_fraction"] >= 0.7
    windows = [selected["reference"], *selected["targets"]]
    for index, first in enumerate(windows):
        for second in windows[index + 1:]:
            assert preparation.disjoint(first, second, 3)


def test_silence_and_short_audio_rejected():
    with pytest.raises(ValueError, match="active"):
        preparation.select_windows(np.full(1000, -100.0))
    with pytest.raises(ValueError):
        preparation.select_windows(np.full(150, -20.0))


def test_duplicate_sources_cannot_leak_across_splits():
    rows = [dict(song_id=str(index), source_sha256=str(index // 2)) for index in range(20)]
    first = preparation.assign_splits(rows, 42, 0.1)
    second = preparation.assign_splits(list(reversed(rows)), 42, 0.1)
    assert first == second
    assert set(first.values()) == {"train", "validation"}
    for index in range(0, 20, 2):
        assert first[str(index)] == first[str(index + 1)]


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg tools unavailable")
def test_end_to_end_audio_only_and_resume(tmp_path):
    vocals, sources = tmp_path / "vocals", tmp_path / "sources"
    rate = 8000
    time_axis = np.arange(rate * 70) / rate
    waveform = (0.15 * np.sin(2 * np.pi * 200 * time_axis)).astype(np.float32)
    for index in range(3):
        vocal_dir, source_dir = vocals / str(index), sources / str(index)
        vocal_dir.mkdir(parents=True)
        source_dir.mkdir(parents=True)
        sf.write(vocal_dir / "vocals_dereverb.wav", waveform, rate, subtype="FLOAT")
        sf.write(source_dir / "source.wav", waveform * (index + 1), rate, subtype="FLOAT")
    output = tmp_path / "prepared"
    preparation.main(["audit", "--vocals", str(vocals), "--sources", str(sources),
                      "--output", str(output), "--workers", "2"])
    args = ["prepare", "--output", str(output), "--workers", "2"]
    preparation.main(args)
    summary = json.loads((output / "prepare_summary.json").read_text())
    assert summary["prepared_songs"] == 3
    assert summary["prepared_samples"] == 6
    assert summary["training_ready"] is False
    row = json.loads((output / "audio_train.jsonl").read_text().splitlines()[0])
    assert row["training_ready"] is False
    assert "aligned_semantic_tokens" in row["missing"]
    assert not (output / "train.jsonl").exists()
    target = output / row["target_audio"]
    before = target.stat().st_mtime_ns
    info = sf.info(target)
    assert (info.frames, info.samplerate, info.channels) == (720000, 48000, 2)
    preparation.main(["verify", "--output", str(output)])
    assert json.loads((output / "verification.json").read_text())["checked_audio_files"] == 18
    preparation.main(args)
    assert target.stat().st_mtime_ns == before
    (output / row["reference_audio"]).write_bytes(b"corrupted")
    preparation.main(args)
    errors = json.loads((output / "prepare_errors.json").read_text())
    assert len(errors) == 1 and "changed" in errors[0]["error"]
    with pytest.raises(ValueError, match="configuration"):
        preparation.main(args + ["--target-seconds", "10"])
