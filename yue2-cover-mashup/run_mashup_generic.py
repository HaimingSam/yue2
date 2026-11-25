#!/usr/bin/env python3
"""Render a prepared generic mashup with a local YuE2 model and VAE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Directory containing score.abc and request.json")
    parser.add_argument("--output", type=Path, required=True, help="New, empty generation directory")
    parser.add_argument("--model", type=Path, required=True, help="Local YuE2 checkpoint directory")
    parser.add_argument("--vae", type=Path, required=True, help="Local YuE2 VAE directory")
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--memory-budget-gib", type=int, default=24)
    args = parser.parse_args()
    if args.max_tokens <= 0 or args.memory_budget_gib <= 0:
        parser.error("--max-tokens and --memory-budget-gib must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {args.output}")

    import torch
    from yue2 import YuE2Pipeline
    from yue2.storage import verify_result, write_json

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this generation runner")
    data = json.loads((args.input / "request.json").read_text(encoding="utf-8"))
    data["abc"] = (args.input / "score.abc").read_text(encoding="utf-8")
    data["cot"] = "melody"
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "effective_request.json", data)
    sampling = {"max_tokens": args.max_tokens}
    with YuE2Pipeline.from_pretrained(
        args.model,
        vae=args.vae,
        local_files_only=True,
        device="cuda",
        memory_budget_gib=args.memory_budget_gib,
        progress=True,
    ) as pipeline:
        result = pipeline(**data, semantic_sampling=sampling)
        receipt = result.save_artifacts(args.output)
    verify_result(args.output)
    if result.truncated["abc"] or result.truncated["semantic"]:
        raise RuntimeError("Generation truncated; artifacts were saved, inspect the result and retry in a new directory")
    print(json.dumps({"output": str(args.output), "receipt": receipt}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
