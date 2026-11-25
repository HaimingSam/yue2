from pathlib import Path
import json

import torch
from yue2 import YuE2Pipeline
from yue2.storage import verify_result, write_json


ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
MODEL = Path("/data/cmq/hf-cache/hub/models--m-a-p--YuE2-3B/snapshots/29b3558dd46954a0cd9021dc76d5c91864a0f1c7")
VAE = Path("/data/cmq/hf-cache/hub/models--m-a-p--YuE2-Vae/snapshots/9a94e1d0ea9f8087e98f77fa88df4a4068104d2a")
REQUEST = ROOT / "request.json"
ABC = BASE / "mashup_abab" / "score.abc"
OUTPUT = ROOT / "yue2_api_run"
SEMANTIC_SAMPLING = {"max_tokens": 12000}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available before YuE2 initialization")
    torch.cuda.current_device()
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"Nonempty output directory: {OUTPUT}")
    data = json.loads(REQUEST.read_text(encoding="utf-8"))
    data["abc"] = ABC.read_text(encoding="utf-8")
    data["cot"] = "melody"
    write_json(ROOT / "effective_request.json", data)
    write_json(ROOT / "semantic_sampling.json", SEMANTIC_SAMPLING)
    with YuE2Pipeline.from_pretrained(
        MODEL,
        vae=VAE,
        local_files_only=True,
        device="cuda",
        memory_budget_gib=24,
        progress=True,
    ) as pipe:
        result = pipe(**data, semantic_sampling=SEMANTIC_SAMPLING)
        receipt = result.save_artifacts(OUTPUT)
    verify_result(OUTPUT)
    print(json.dumps({"output": str(OUTPUT), "receipt": receipt}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
