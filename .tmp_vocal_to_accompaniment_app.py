import os
from pathlib import Path
import random
import shutil
import tempfile
import time
import traceback
from urllib.parse import urlparse

from vocal_to_yue2 import Yue2Cover


ROOT = Path(__file__).resolve().parent
OUTPUT_FILE = Path(os.getenv("OUTPUT_FILE", str(ROOT / "result.mp3")))

# Local defaults for this machine. Environment variables still override them,
# which keeps the image usable with mounted checkpoints in other deployments.
SHEETSAGE_DIR = os.getenv("SHEETSAGE_DIR", "/data/cmq/YuE/models/SheetSage2")
MERT_DIR = os.getenv(
    "MERT_DIR",
    "/data/cmq/hf-cache/hub/models--m-a-p--MERT-v2-FullSong/snapshots/d8ba1c745e733b3908ce6ad16ebeb17ac7600a42",
)
YUE2_MODEL_DIR = os.getenv(
    "YUE2_MODEL_DIR",
    "/data/cmq/hf-cache/hub/models--m-a-p--YuE2-3B/snapshots/29b3558dd46954a0cd9021dc76d5c91864a0f1c7",
)
YUE2_VAE_DIR = os.getenv(
    "YUE2_VAE_DIR",
    "/data/cmq/hf-cache/hub/models--m-a-p--YuE2-Vae/snapshots/9a94e1d0ea9f8087e98f77fa88df4a4068104d2a",
)
DEVICE = os.getenv("DEVICE", "cuda:0")

service = Yue2Cover(
    sheetsage_dir=SHEETSAGE_DIR,
    mert_dir=MERT_DIR,
    model=YUE2_MODEL_DIR,
    vae=YUE2_VAE_DIR,
    device=DEVICE,
)
PROMPT_FILE = Path(os.getenv("PROMPT_FILE", str(ROOT / "prompts.txt")))


def _is_url(value):
    return value.startswith("http://") or value.startswith("https://")


def choose_style():
    """Pick one non-empty line from the local prompt file."""
    if not PROMPT_FILE.is_file():
        raise RuntimeError(f"Prompt 文件不存在：{PROMPT_FILE}")
    prompts = [line.strip() for line in PROMPT_FILE.read_text(encoding="utf-8").splitlines()]
    prompts = [line for line in prompts if line and not line.startswith("#")]
    if not prompts:
        raise RuntimeError(f"Prompt 文件是空的：{PROMPT_FILE}")
    return random.choice(prompts)


def callback(message):
    temp_download_dir = None
    try:
        data = message.data()
        print(f"收到消息：类型{type(data)}")
        print(f"收到消息：内容{data}")
        print(f"apollo 配置 {message.get_config()}")

        missing = [name for name in ("audio_url", "lyrics", "key") if not str(data.get(name) or "").strip()]
        if missing:
            return message.fail(data={}, code=1, msg="缺少参数：" + "、".join(missing))

        audio = str(data["audio_url"]).strip()
        lyrics = data["lyrics"] if isinstance(data["lyrics"], str) else str(data["lyrics"])
        key = str(data["key"]).strip()

        temp_download_dir = tempfile.mkdtemp(prefix="vocal-to-accompaniment-")
        download_s = time.time()
        if _is_url(audio):
            filename = os.path.basename(urlparse(audio).path) or "vocal.wav"
            audio_path = os.path.join(temp_download_dir, filename)
            message.download({audio: audio_path})
        else:
            audio_path = audio
        download_time = (time.time() - download_s) * 1000

        style = choose_style()
        infer_s = time.time()
        report = service.run(audio=audio_path, lyrics=lyrics, style=style, out=str(OUTPUT_FILE), key=key)
        infer_time = (time.time() - infer_s) * 1000

        upload_s = time.time()
        result_urls = message.upload([report["audio"]])
        upload_time = (time.time() - upload_s) * 1000
        return message.ok({
            "url": result_urls[0],
            "status": report["status"],
            "download_time": download_time,
            "infer_time": infer_time,
            "upload_time": upload_time,
        })
    except Exception as exc:
        traceback.print_stack()
        return message.fail(data={}, code=1, msg=f"{exc}")
    finally:
        if temp_download_dir and os.path.isdir(temp_download_dir):
            shutil.rmtree(temp_download_dir, ignore_errors=True)


from jimu_sdk.jimu_sdk import JimuSdks

algo_name = os.getenv("ALGO_NAME", "algo_vocal_to_accompaniment")
print(f"algo_name {algo_name}")
mq_client = JimuSdks(
    algo_name=algo_name,
    callback=callback,
)
try:
    mq_client.start_consumer()
except Exception as exc:
    traceback.print_stack()
    print(f"发生异常：{str(exc)}")
    mq_client.shutdown()
