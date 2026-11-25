"""Audit and prepare audio-only NAR reference/target material without fabricating tokens."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import time

import numpy as np
import soundfile as sf


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run(command, timeout=180):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace")[-2000:])
    return result.stdout


def stat_record(path):
    value = Path(path).stat()
    return dict(bytes=value.st_size, mtime_ns=value.st_mtime_ns)


def inspect_pair(song_id, vocal_path, source_path):
    try:
        vocal = sf.info(vocal_path)
        probe = json.loads(run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                               "-show_entries", "stream=codec_name,sample_rate,channels,duration,start_time:format=duration",
                               "-of", "json", str(source_path)]))
        stream = probe["streams"][0]
        duration = float(stream.get("duration", probe["format"]["duration"]))
        if not math.isfinite(duration) or duration <= 0 or vocal.duration <= 0:
            raise ValueError("Invalid duration")
        return dict(song_id=song_id, singer_id=song_id, singer_id_is_assumed=True,
                    vocal_path=str(vocal_path), source_path=str(source_path),
                    vocal_stat=stat_record(vocal_path), source_stat=stat_record(source_path),
                    source_sha256=digest(source_path),
                    vocal=dict(duration=vocal.duration, frames=vocal.frames,
                               sample_rate=vocal.samplerate, channels=vocal.channels, subtype=vocal.subtype),
                    source=dict(duration=duration, sample_rate=int(stream["sample_rate"]),
                                channels=int(stream["channels"]), codec=stream["codec_name"],
                                start_time=float(stream.get("start_time", 0))),
                    duration_difference=abs(vocal.duration - duration), status="ok")
    except Exception as error:
        return dict(song_id=song_id, status="error", error=str(error),
                    vocal_path=str(vocal_path), source_path=str(source_path))


def audit(args):
    vocals_root, sources_root = Path(args.vocals).resolve(), Path(args.sources).resolve()
    vocals = {path.parent.name: path for path in vocals_root.glob("*/vocals_dereverb.wav")}
    sources = {}
    ambiguous = {}
    for directory in sorted(sources_root.iterdir()):
        if directory.is_dir():
            candidates = [directory / name for name in ("source.mp3", "source.m4a", "source.wav", "source.flac")
                          if (directory / name).is_file()]
            if len(candidates) == 1:
                sources[directory.name] = candidates[0]
            elif len(candidates) > 1:
                ambiguous[directory.name] = [str(path) for path in candidates]
    paired = sorted(vocals.keys() & sources.keys())
    if not paired:
        raise ValueError("No matching source/vocal IDs")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "audit_config.json", dict(vocals_root=str(vocals_root), sources_root=str(sources_root)))
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(inspect_pair, song_id, vocals[song_id], sources[song_id]) for song_id in paired]
        for future in as_completed(jobs):
            rows.append(future.result())
            if len(rows) % 100 == 0:
                print(f"Audited {len(rows)}/{len(paired)}", flush=True)
    rows.sort(key=lambda row: row["song_id"])
    write_json(output / "inventory.json", rows)
    okay = [row for row in rows if row["status"] == "ok"]
    summary = dict(vocal_files=len(vocals), source_files=len(sources), paired=len(paired),
                   readable=len(okay), errors=len(rows) - len(okay), ambiguous=ambiguous,
                   vocal_only=sorted(vocals.keys() - sources.keys()),
                   source_only=sorted(sources.keys() - vocals.keys()),
                   source_extensions=dict(Counter(Path(row["source_path"]).suffix for row in rows)),
                   source_sample_rates=dict(Counter(str(row["source"]["sample_rate"]) for row in okay)),
                   vocal_sample_rates=dict(Counter(str(row["vocal"]["sample_rate"]) for row in okay)),
                   duration_difference_over_025=[row["song_id"] for row in okay if row["duration_difference"] > 0.25],
                   duration_difference_over_1=[row["song_id"] for row in okay if row["duration_difference"] > 1],
                   total_source_hours=sum(row["source"]["duration"] for row in okay) / 3600,
                   duration_range_seconds=[min(row["source"]["duration"] for row in okay),
                                           max(row["source"]["duration"] for row in okay)] if okay else [],
                   exact_duplicate_sources=len(okay) - len({row["source_sha256"] for row in okay}))
    write_json(output / "audit_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def assign_splits(rows, seed, validation_fraction):
    groups = sorted({row["source_sha256"] for row in rows})
    if len(groups) < 2:
        raise ValueError("Need at least two distinct source groups")
    random.Random(seed).shuffle(groups)
    validation_count = min(len(groups) - 1, max(1, math.ceil(len(groups) * validation_fraction)))
    validation = set(groups[:validation_count])
    return {row["song_id"]: "validation" if row["source_sha256"] in validation else "train"
            for row in rows}


def disjoint(first, second, gap):
    return first["end"] + gap <= second["start"] or second["end"] + gap <= first["start"]


def activity_envelope(path, duration):
    audio = np.frombuffer(run(["ffmpeg", "-nostdin", "-v", "error", "-threads", "1",
                               "-i", str(path), "-t", str(duration), "-vn", "-ac", "1",
                               "-ar", "8000", "-f", "f32le", "pipe:1"]), dtype="<f4")
    if not np.isfinite(audio).all():
        raise ValueError("Non-finite vocal waveform")
    frames = len(audio) // 800
    if frames < 1:
        raise ValueError("Empty vocal waveform")
    audio = audio[:frames * 800].reshape(frames, 800)
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2, axis=1))
    return 20 * np.log10(np.maximum(rms, 1e-10))


def window_candidates(levels, seconds, threshold, min_active, margin=2):
    count = round(seconds * 10)
    active = levels >= threshold
    candidates = []
    for start in range(round(margin * 10), len(levels) - count - round(margin * 10) + 1, 10):
        section = levels[start:start + count]
        fraction = float(np.mean(active[start:start + count]))
        if fraction >= min_active:
            candidates.append(dict(start=start / 10, end=start / 10 + seconds,
                                   duration=seconds, active_fraction=fraction,
                                   median_vocal_db=float(np.median(section)),
                                   score=fraction + float(np.clip(np.median(section), -60, 0)) / 300))
    return sorted(candidates, key=lambda item: (-item["score"], item["start"]))


def select_windows(levels, reference_seconds=12, target_seconds=15, targets=2,
                   min_active=0.7, gap=3):
    threshold = max(-50.0, float(np.percentile(levels, 90)) - 25.0)
    references = window_candidates(levels, reference_seconds, threshold, min_active)
    candidates = window_candidates(levels, target_seconds, threshold, min_active)
    if not references or not candidates:
        raise ValueError("No sufficiently active reference/target windows")
    best = None
    for reference in references[:100]:
        chosen = []
        for target in candidates:
            if disjoint(reference, target, gap) and all(disjoint(previous, target, gap) for previous in chosen):
                chosen.append(target)
                if len(chosen) == targets:
                    break
        if chosen:
            quality = (len(chosen), reference["score"] + sum(target["score"] for target in chosen))
            if best is None or quality > best[0]:
                best = (quality, reference, chosen)
    if best is None:
        raise ValueError("No non-overlapping reference/target combination")
    return dict(reference=best[1], targets=sorted(best[2], key=lambda target: target["start"]),
                activity_threshold_db=threshold, activity_method="8khz_mono_100ms_rms_energy_not_speech_vad")


def extract(source, output, window, rate, channels):
    run(["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", str(source),
         "-ss", str(window["start"]), "-t", str(window["duration"]), "-vn", "-map", "0:a:0",
         "-ac", str(channels), "-ar", str(rate), "-c:a", "pcm_f32le", "-threads", "1", str(output)])
    info = sf.info(output)
    expected_frames = round(window["duration"] * rate)
    if info.frames != expected_frames or info.samplerate != rate or info.channels != channels:
        raise ValueError(f"Incorrect clip shape for {output.name}: {info}")
    audio, _ = sf.read(output, dtype="float32", always_2d=True)
    if not np.isfinite(audio).all() or np.max(np.abs(audio)) < 1e-6:
        raise ValueError(f"Silent or non-finite clip: {output.name}")
    return dict(frames=info.frames, sample_rate=rate, channels=channels, subtype=info.subtype,
                peak=float(np.max(np.abs(audio))), rms=float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))),
                fraction_abs_ge_1=float(np.mean(np.abs(audio) >= 1)), sha256=digest(output))


def verify_saved(directory, row, config_id, split):
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata["config_id"] != config_id or metadata["song_id"] != row["song_id"] or metadata["split"] != split:
        raise ValueError("Existing output has incompatible preparation metadata")
    for name, expected in metadata["files"].items():
        if digest(directory / name) != expected["sha256"]:
            raise ValueError(f"Existing output changed: {directory / name}")
    return metadata


def prepare_song(row, split, root, config, config_id):
    song_id = row["song_id"]
    try:
        for kind in ("vocal", "source"):
            if stat_record(row[f"{kind}_path"]) != row[f"{kind}_stat"]:
                raise ValueError(f"{kind} source changed since audit; do not reuse this inventory")
        directory = root / "clips" / song_id
        if directory.exists():
            return dict(status="ok", metadata=verify_saved(directory, row, config_id, split), reused=True)
        if row["duration_difference"] > config["max_duration_difference"]:
            raise ValueError("Source/vocal duration difference exceeds permitted threshold")
        duration = min(row["source"]["duration"], row["vocal"]["duration"])
        if duration > config["max_song_seconds"]:
            raise ValueError("Song exceeds maximum analysis duration")
        levels = activity_envelope(row["vocal_path"], duration)
        selection = select_windows(levels, config["reference_seconds"], config["target_seconds"],
                                   config["targets_per_song"], config["min_active"], config["gap_seconds"])
        with tempfile.TemporaryDirectory(prefix=song_id + "-", dir=root / ".staging") as temporary:
            staging = Path(temporary)
            files = {}
            for name, rate in (("reference.wav", 48000), ("reference_16k.wav", 16000)):
                files[name] = extract(row["vocal_path"], staging / name, selection["reference"], rate, 1)
            for index, window in enumerate(selection["targets"], 1):
                name = f"target_{index:02d}.wav"
                files[name] = extract(row["source_path"], staging / name, window, 48000, 2)
                name = f"target_vocal_{index:02d}.wav"
                files[name] = extract(row["vocal_path"], staging / name, window, 48000, 1)
            warnings = ["song_id_used_as_singer_id_not_identity_verified",
                        "energy_screening_does_not_detect_harmony_or_separation_artifacts",
                        "source_vocal_time_offset_not_measured"]
            if row["source"]["channels"] == 1:
                warnings.append("mono_source_duplicated_to_stereo")
            if any(info["fraction_abs_ge_1"] > 0.001 for info in files.values()):
                warnings.append("peak_at_or_above_full_scale_manual_review")
            if row["duration_difference"] > 0.25:
                warnings.append("source_vocal_duration_difference_over_250ms")
            metadata = dict(song_id=song_id, singer_id=song_id, singer_id_is_assumed=True,
                            split=split, config_id=config_id, source=row, selection=selection,
                            files=files, warnings=warnings, stage="audio_only", training_ready=False)
            write_json(staging / "metadata.json", metadata)
            staging.rename(directory)
        return dict(status="ok", metadata=metadata, reused=False)
    except Exception as error:
        return dict(status="error", song_id=song_id, error=str(error))


def audio_records(metadata):
    song_id = metadata["song_id"]
    directory = f"clips/{song_id}"
    for index, target in enumerate(metadata["selection"]["targets"], 1):
        yield dict(sample_id=f"{song_id}-{index:02d}", song_id=song_id, singer_id=song_id,
                   singer_id_is_assumed=True, split=metadata["split"],
                   reference_audio=f"{directory}/reference.wav",
                   reference_audio_16k=f"{directory}/reference_16k.wav",
                   target_audio=f"{directory}/target_{index:02d}.wav",
                   target_vocal_audio=f"{directory}/target_vocal_{index:02d}.wav",
                   reference_start=metadata["selection"]["reference"]["start"],
                   reference_seconds=metadata["selection"]["reference"]["duration"],
                   target_start=target["start"], target_seconds=target["duration"],
                   metadata=f"{directory}/metadata.json", stage="audio_only", training_ready=False,
                   missing=["reference_embedding", "target_latent", "prefix", "aligned_semantic_tokens"],
                   source_sha256=metadata["source"]["source_sha256"], warnings=metadata["warnings"])


def prepare(args):
    root = Path(args.output).resolve()
    inventory = json.loads((root / "inventory.json").read_text())
    rows = [row for row in inventory if row["status"] == "ok"]
    config = dict(version=1, seed=args.seed, validation_fraction=args.validation_fraction,
                  reference_seconds=args.reference_seconds, target_seconds=args.target_seconds,
                  targets_per_song=args.targets_per_song, min_active=args.min_active,
                  gap_seconds=args.gap_seconds, max_duration_difference=args.max_duration_difference,
                  max_song_seconds=args.max_song_seconds, inventory_sha256=digest(root / "inventory.json"))
    config_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config_path = root / "prepare_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Preparation configuration changed; use a new output root")
    write_json(config_path, config)
    splits = assign_splits(rows, args.seed, args.validation_fraction)
    write_json(root / "split_assignments.json", splits)
    (root / "clips").mkdir(exist_ok=True)
    (root / ".staging").mkdir(exist_ok=True)
    selected = rows if args.limit is None else rows[:args.limit]
    missing_count = sum(not (root / "clips" / row["song_id"]).exists() for row in selected)
    bytes_per_song = (args.reference_seconds * (48000 + 16000) +
                      args.targets_per_song * args.target_seconds * 48000 * 3) * 4
    if shutil.disk_usage(root).free < missing_count * bytes_per_song * 1.1 + 2 * 1024 ** 3:
        raise ValueError("Insufficient free space for clips plus safety margin")
    outcomes = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(prepare_song, row, splits[row["song_id"]], root, config, config_id)
                for row in selected]
        for future in as_completed(jobs):
            outcomes.append(future.result())
            if len(outcomes) % 25 == 0 or len(outcomes) == len(selected):
                errors = sum(row["status"] == "error" for row in outcomes)
                print(f"Prepared {len(outcomes)}/{len(selected)}, errors={errors}, elapsed={time.monotonic()-started:.0f}s", flush=True)
    metadata = sorted([outcome["metadata"] for outcome in outcomes if outcome["status"] == "ok"],
                      key=lambda row: row["song_id"])
    records = [record for song in metadata for record in audio_records(song)]
    for split in ("train", "validation"):
        path = root / f"audio_{split}.jsonl"
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                                     for record in records if record["split"] == split))
        temporary.replace(path)
    errors = sorted([outcome for outcome in outcomes if outcome["status"] == "error"], key=lambda row: row["song_id"])
    write_json(root / "prepare_errors.json", errors)
    summary = dict(audited_readable=len(rows), attempted=len(selected), prepared_songs=len(metadata),
                   prepared_samples=len(records), skipped_songs=len(errors),
                   songs_per_split=dict(Counter(song["split"] for song in metadata)),
                   samples_per_split=dict(Counter(row["split"] for row in records)),
                   total_reference_hours=sum(song["selection"]["reference"]["duration"] for song in metadata) / 3600,
                   total_target_hours=sum(row["target_seconds"] for row in records) / 3600,
                   output_audio_bytes=sum((root / "clips" / song["song_id"] / name).stat().st_size
                                          for song in metadata for name in song["files"]),
                   warning_counts=dict(Counter(warning for song in metadata for warning in song["warnings"])),
                   min_reference_active_fraction=min((song["selection"]["reference"]["active_fraction"] for song in metadata), default=None),
                   min_target_active_fraction=min((target["active_fraction"] for song in metadata for target in song["selection"]["targets"]), default=None),
                   elapsed_seconds=time.monotonic() - started, training_ready=False,
                   missing=["pretrained_identity_encoder_and_embeddings", "target_vae_latents", "aligned_semantic_conditions"])
    write_json(root / "prepare_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def verify(args):
    root = Path(args.output).resolve()
    records = []
    sample_ids = set()
    group_splits = {}
    for split in ("train", "validation"):
        for line in (root / f"audio_{split}.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["sample_id"] in sample_ids or row["split"] != split or row["training_ready"] is not False:
                raise ValueError("Invalid sample identity, split or readiness flag")
            sample_ids.add(row["sample_id"])
            for key in (row["singer_id"], row["source_sha256"]):
                if key in group_splits and group_splits[key] != split:
                    raise ValueError("Singer/source leakage across splits")
                group_splits[key] = split
            records.append(row)
    if not records:
        raise ValueError("No prepared audio records")
    config = json.loads((root / "prepare_config.json").read_text())
    directories = sorted({root / Path(row["metadata"]).parent for row in records})
    checked_files = 0
    for directory in directories:
        metadata = json.loads((directory / "metadata.json").read_text())
        windows = [metadata["selection"]["reference"], *metadata["selection"]["targets"]]
        for index, first in enumerate(windows):
            for second in windows[index + 1:]:
                if not disjoint(first, second, config["gap_seconds"]):
                    raise ValueError(f"Overlapping windows: {directory}")
        for name, expected in metadata["files"].items():
            path = directory / name
            info = sf.info(path)
            if (info.frames, info.samplerate, info.channels, info.subtype) != (
                    expected["frames"], expected["sample_rate"], expected["channels"], expected["subtype"]):
                raise ValueError(f"Audio header mismatch: {path}")
            if digest(path) != expected["sha256"]:
                raise ValueError(f"Audio hash mismatch: {path}")
            checked_files += 1
    for row in records:
        for field in ("reference_audio", "reference_audio_16k", "target_audio", "target_vocal_audio"):
            if not (root / row[field]).is_file():
                raise ValueError(f"Missing manifest audio: {row[field]}")
    report = dict(status="passed", checked_songs=len(directories), checked_samples=len(records),
                  checked_audio_files=checked_files, all_audio_hashes_verified=True,
                  singer_and_exact_source_split_disjoint=True, nonoverlapping_windows=True,
                  training_ready=False, note="Audio integrity only; semantic alignment and identity not verified")
    write_json(root / "verification.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit_parser = commands.add_parser("audit")
    audit_parser.add_argument("--vocals", required=True)
    audit_parser.add_argument("--sources", required=True)
    audit_parser.set_defaults(action=audit)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--seed", type=int, default=20261009)
    prepare_parser.add_argument("--validation-fraction", type=float, default=0.1)
    prepare_parser.add_argument("--reference-seconds", type=int, default=12)
    prepare_parser.add_argument("--target-seconds", type=int, default=15)
    prepare_parser.add_argument("--targets-per-song", type=int, default=2)
    prepare_parser.add_argument("--min-active", type=float, default=0.7)
    prepare_parser.add_argument("--gap-seconds", type=int, default=3)
    prepare_parser.add_argument("--max-duration-difference", type=float, default=1.0)
    prepare_parser.add_argument("--max-song-seconds", type=float, default=1200)
    prepare_parser.add_argument("--limit", type=int)
    prepare_parser.set_defaults(action=prepare)
    verify_parser = commands.add_parser("verify")
    verify_parser.set_defaults(action=verify)
    for command in (audit_parser, prepare_parser, verify_parser):
        command.add_argument("--output", required=True)
        command.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.command == "prepare":
        if not 0 < args.validation_fraction < 1 or not 0 < args.min_active <= 1:
            parser.error("Invalid validation/active fraction")
        if min(args.reference_seconds, args.target_seconds, args.targets_per_song, args.max_song_seconds) <= 0:
            parser.error("Durations and target count must be positive")
        if args.gap_seconds < 0 or not math.isfinite(args.max_duration_difference) or args.max_duration_difference < 0:
            parser.error("Gap/duration difference must be nonnegative")
        if args.limit is not None and args.limit < 1:
            parser.error("limit must be positive")
    args.action(args)


if __name__ == "__main__":
    main()
