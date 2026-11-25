import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parent

DEFAULT_AUDIO = "/data/cmq/YuE/yue2-cover-test/download/f8e2_hjf3.mp3"
DEFAULT_OUTPUT_ROOT = "/data/cmq/YuE/vocal_to_accompaniment_server_local_runs"
DEFAULT_SHEETSAGE_PYTHON = "/data/cmq/YuE/.venv-sheetsage2/bin/python"
DEFAULT_YUE_PYTHON = "/data/cmq/YuE/.venv/bin/python"
DEFAULT_SHEETSAGE_DIR = "/data/cmq/YuE/models/SheetSage2"
DEFAULT_MERT_DIR = (
    "/data/cmq/hf-cache/hub/models--m-a-p--MERT-v2-FullSong/"
    "snapshots/d8ba1c745e733b3908ce6ad16ebeb17ac7600a42"
)
DEFAULT_MODEL_DIR = (
    "/data/cmq/hf-cache/hub/models--m-a-p--YuE2-3B/"
    "snapshots/29b3558dd46954a0cd9021dc76d5c91864a0f1c7"
)
DEFAULT_VAE_DIR = (
    "/data/cmq/hf-cache/hub/models--m-a-p--YuE2-Vae/"
    "snapshots/9a94e1d0ea9f8087e98f77fa88df4a4068104d2a"
)

LYRICS = (
    "[Verse]\n"
    "听见冬天的离开，我在某年某月醒过来。\n"
    "我想，我等，我期待，未来却不能理智安排。\n"
    "阴天傍晚，车窗外，未来有一个人在等待。\n"
    "向左，向右，向前看，爱要拐几个弯才来。\n\n"
    "[Chorus]\n"
    "我遇见谁，会有怎样的对白？\n"
    "我等的人，他在多远的未来？\n"
    "我听见风来自地铁和人海，我排着队，拿着爱的号码牌。\n"
)


def existing_path(value: str, description: str) -> str:
    path = Path(value)
    if not path.exists():
        raise SystemExit(f"{description} 不存在：{path}")
    return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local vocal-to-accompaniment smoke test.")
    parser.add_argument("--audio", default=DEFAULT_AUDIO)
    parser.add_argument("--style", default="Mandarin pop, full accompaniment, drums, bass, piano and guitar")
    parser.add_argument("--key", default="Bbm")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="Print the command and check paths without running models.")
    args = parser.parse_args()

    audio = existing_path(args.audio, "audio")
    sheetsage_python = existing_path(DEFAULT_SHEETSAGE_PYTHON, "SheetSage2 Python")
    yue_python = existing_path(DEFAULT_YUE_PYTHON, "YuE2 Python")
    sheetsage_dir = existing_path(DEFAULT_SHEETSAGE_DIR, "SheetSage2")
    mert_dir = existing_path(DEFAULT_MERT_DIR, "MERT-v2-FullSong")
    model_dir = existing_path(DEFAULT_MODEL_DIR, "YuE2-3B")
    vae_dir = existing_path(DEFAULT_VAE_DIR, "YuE2-Vae")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / f"hjf3_{time.strftime('%Y%m%d-%H%M%S')}"

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        handle.write(LYRICS)
        lyrics_path = handle.name

    command = [
        yue_python,
        str(ROOT / "vocal_to_yue2.py"),
        "cover",
        "--audio",
        audio,
        "--lyrics",
        lyrics_path,
        "--style",
        args.style,
        "--correction-key",
        args.key,
        "--out",
        str(output),
        "--device",
        args.device,
        "--sheetsage-python",
        sheetsage_python,
        "--sheetsage-dir",
        sheetsage_dir,
        "--mert-dir",
        mert_dir,
        "--yue-python",
        yue_python,
        "--model",
        model_dir,
        "--vae",
        vae_dir,
    ]

    print(" ".join(subprocess.list2cmdline([item]) for item in command))
    if args.dry_run:
        print(f"dry-run ok; output would be: {output}")
        return 0

    env = os.environ.copy()
    env["HF_HOME"] = "/data/cmq/hf-cache"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    try:
        completed = subprocess.run(command, env=env, check=False)
        return completed.returncode
    finally:
        Path(lyrics_path).unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
