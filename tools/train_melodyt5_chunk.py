#!/usr/bin/env python3
"""Fine-tune the official MelodyT5 checkpoint on YuE chunk pairs."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path("/data/cmq/melodyt5"))
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=20261104)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    sys.path.insert(0, str(args.repo))
    from config import CHAR_NUM_LAYERS, PATCH_NUM_LAYERS
    from utils import MelodyT5, Patchilizer
    from transformers import GPT2Config

    patch_config = GPT2Config(num_hidden_layers=PATCH_NUM_LAYERS, max_length=256,
                              max_position_embeddings=256, vocab_size=1)
    char_config = GPT2Config(num_hidden_layers=CHAR_NUM_LAYERS, max_length=64,
                             max_position_embeddings=64, vocab_size=128)
    model = MelodyT5(patch_config, char_config)
    checkpoint = torch.load(args.pretrained, map_location="cpu")
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    allowed_suffixes = (".attn.bias", ".attn.masked_bias", ".crossattention.bias", ".crossattention.masked_bias")
    unexpected = [key for key in unexpected if not key.endswith(allowed_suffixes)]
    if missing or unexpected:
        raise RuntimeError(f"Incompatible checkpoint: missing={missing}, unexpected={unexpected}")
    device = torch.device("cuda")
    model.to(device)
    patchilizer = Patchilizer()

    def read(split):
        rows = [json.loads(line) for line in (args.data / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        return [(torch.tensor(patchilizer.encode(row["input"], add_special_patches=True)),
                 torch.tensor(patchilizer.encode(row["output"], add_special_patches=True))) for row in rows]

    train, valid = read("train"), read("validation")
    train = train[:len(train) - len(train) % args.batch_size]
    valid = valid[:len(valid) - len(valid) % args.batch_size]

    def collate(batch):
        inputs, outputs = zip(*batch)
        inputs = pad_sequence(inputs, batch_first=True).to(device)
        outputs = pad_sequence(outputs, batch_first=True).to(device)
        input_masks = (inputs != 0).any(dim=2).long()
        output_masks = (outputs != 0).any(dim=2).long()
        return inputs, input_masks, outputs, output_masks

    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    args.output.mkdir(parents=True, exist_ok=True)
    step = 0
    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model(*batch)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            total += loss.item()
            if step % 10 == 0:
                print(json.dumps({"epoch": epoch + 1, "step": step, "train_loss": total / 10}), flush=True)
                total = 0.0
            if args.max_steps > 0 and step >= args.max_steps:
                break
        model.eval()
        eval_total = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                eval_total += model(*batch).item()
        eval_loss = eval_total / max(1, len(valid_loader))
        print(json.dumps({"epoch": epoch + 1, "step": step, "eval_loss": eval_loss}), flush=True)
        if eval_loss < best:
            best = eval_loss
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "epoch": epoch + 1, "step": step, "eval_loss": eval_loss}, args.output / "weights.pth")
        if args.max_steps > 0 and step >= args.max_steps:
            break
    print(json.dumps({"train": len(train), "validation": len(valid), "best_eval_loss": best,
                      "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
