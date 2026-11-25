#!/usr/bin/env python3
"""Render assembled Vocal ABC scores into listenable YuE2 FLAC/MP3 files."""

from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
from pathlib import Path

from abc_vocal_format import load_validator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STYLE = "Mandarin pop, clear expressive female voice, gentle piano, bass and drums, melodic singing"
DEFAULT_LYRICS = ("[Verse]\n啦啦啦啦啦啦啦啦\n啦啦啦啦啦啦啦啦\n啦啦啦啦啦啦啦啦\n啦啦啦啦啦啦啦啦\n"
                  "[Chorus]\n啦啦啦啦啦啦啦啦\n啦啦啦啦啦啦啦啦\n啦啦啦啦啦啦啦啦\n啦啦啦啦啦啦啦啦\n"
                  "[Verse]\n啦啦啦啦啦啦啦啦\n啦啦啦啦啦啦啦啦\n啦啦啦啦啦啦啦啦\n啦啦啦啦啦啦啦啦\n"
                  "[Chorus]\n啦啦啦啦啦啦啦啦\n啦啦啦啦啦啦啦啦\n啦啦啦啦啦啦啦啦\n啦啦啦啦啦啦啦啦")


def validate_scores(paths: list[Path]) -> None:
    validator = load_validator()
    for path in paths:
        score = validator.parse_abc(path.read_text(encoding="utf-8"))
        if not score.voices["Vocal"].notes or score.voices["Ins"].notes:
            raise ValueError(f"Expected a Vocal melody and empty Ins track: {path}")


def convert_mp3(flac_path: Path, mp3_path: Path, seconds: int | None = None) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg is required for MP3 preview output")
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(flac_path)]
    if seconds is not None:
        command.extend(["-t", str(seconds)])
    subprocess.run(command + ["-codec:a", "libmp3lame", "-b:a", "192k", str(mp3_path)], check=True)


def render(scores: Path, output: Path, limit: int | None, model: str, vae: str,
           style: str, lyrics: str, seed: int, device: str) -> dict:
    from yue2 import YuE2Pipeline

    paths = sorted(scores.glob("*.abc"))
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        paths = paths[:limit]
    if not paths:
        raise ValueError(f"No ABC scores found in {scores}")
    validate_scores(paths)
    if not shutil.which("ffmpeg"):
        raise FileNotFoundError("ffmpeg is required for MP3 preview output")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    with YuE2Pipeline.from_pretrained(model, vae=vae, device=device, local_files_only=True) as pipe:
        for index, path in enumerate(paths):
            folder = output / path.stem
            folder.mkdir(exist_ok=True)
            score_text = path.read_text(encoding="utf-8")
            source_copy = folder / "score.abc"
            if source_copy.exists() and source_copy.read_text(encoding="utf-8") != score_text:
                raise ValueError(f"Existing output uses a different ABC score: {folder}")
            source_copy.write_text(score_text, encoding="utf-8")
            request = {"style": style, "lyrics": lyrics, "abc": score_text,
                       "cot": "melody", "seed": seed + index, "id": path.stem}
            request_path = folder / "request.json"
            if request_path.exists() and json.loads(request_path.read_text(encoding="utf-8")) != request:
                raise ValueError(f"Existing output uses a different request: {folder}")
            request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            flac = folder / "audio.flac"
            mp3 = folder / "audio.mp3"
            preview = folder / "preview-60s.mp3"
            report_path = folder / "result.json"
            if flac.exists() and report_path.exists():
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if not mp3.exists():
                    convert_mp3(flac, mp3)
                if not preview.exists():
                    convert_mp3(flac, preview, seconds=60)
                report["mp3"] = str(mp3)
                report["preview"] = str(preview)
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
                results.append(report)
                print(json.dumps(report, ensure_ascii=False), flush=True)
                continue
            try:
                song = pipe(**request)
                song.save(flac)
                convert_mp3(flac, mp3)
                convert_mp3(flac, preview, seconds=60)
                report = {"id": path.stem, "flac": str(flac), "mp3": str(mp3), "preview": str(preview),
                          "audio_seconds": round(len(song.audio) / song.sample_rate, 2),
                          "truncated": song.truncated, "generation_seconds": round(song.timing["e2e_seconds"], 2)}
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
            except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as exc:
                report = {"id": path.stem, "error": str(exc)}
                (folder / "error.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                                     encoding="utf-8")
            results.append(report)
            (output / "progress.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n"
                                                        for item in results), encoding="utf-8")
            print(json.dumps(report, ensure_ascii=False), flush=True)
    summary = {"requested": len(paths), "rendered": sum("flac" in result for result in results),
               "truncated": sum(any(result.get("truncated", {}).values()) for result in results),
               "output": str(output), "model": model, "vae": vae,
               "results": results}
    playlist = ["#EXTM3U"]
    previews = ["#EXTM3U"]
    for result in results:
        if "mp3" in result:
            playlist.extend([f'#EXTINF:{round(result["audio_seconds"])},{result["id"]}',
                             str(Path(result["mp3"]).relative_to(output))])
            previews.extend([f'#EXTINF:{min(round(result["audio_seconds"]), 60)},{result["id"]}',
                             str(Path(result["preview"]).relative_to(output))])
    (output / "playlist.m3u").write_text("\n".join(playlist) + "\n", encoding="utf-8")
    (output / "previews.m3u").write_text("\n".join(previews) + "\n", encoding="utf-8")
    players = []
    for result in results:
        if "mp3" not in result:
            continue
        song_id = html.escape(result["id"])
        preview = html.escape(str(Path(result["preview"]).relative_to(output)))
        mp3 = html.escape(str(Path(result["mp3"]).relative_to(output)))
        players.append(f'<section><h2>{song_id}</h2><p>{result["audio_seconds"]:.1f} 秒 · '
                       f'<a href="{mp3}" download>下载完整 MP3</a></p>'
                       f'<label>前 60 秒试听<audio controls preload="none" src="{preview}"></audio></label>'
                       f'<label>完整歌曲<audio controls preload="none" src="{mp3}"></audio></label></section>')
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>YuE2 ABC 试听</title><style>body{max-width:820px;margin:2rem auto;padding:0 1rem;'
            'font:16px/1.6 system-ui}section{border:1px solid #ddd;border-radius:12px;'
            'padding:1rem;margin:1rem 0}audio{display:block;width:100%;margin:.3rem 0 1rem}'
            'label{display:block}</style><h1>YuE2 ABC 试听</h1>'
            '<p>使用占位歌词「啦」；每首提供前 60 秒与完整版。</p>'
            + "".join(players) + '</html>')
    (output / "index.html").write_text(page, encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=ROOT / "abc-two-stage/evaluation/yue2-assembled-20")
    parser.add_argument("--output", type=Path, default=ROOT / "abc-two-stage/audio/yue2-assembled-20")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default="m-a-p/YuE2-3B")
    parser.add_argument("--vae", default="m-a-p/YuE2-Vae")
    parser.add_argument("--style", default=DEFAULT_STYLE)
    parser.add_argument("--lyrics-file", type=Path)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    lyrics = args.lyrics_file.read_text(encoding="utf-8") if args.lyrics_file else DEFAULT_LYRICS
    report = render(args.scores, args.output, args.limit, args.model, args.vae,
                    args.style, lyrics, args.seed, args.device)
    print(json.dumps({key: value for key, value in report.items() if key != "results"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
