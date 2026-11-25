#!/usr/bin/env python3
"""Fine-tune ByT5 to expand a short ABC section into a full song score."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


DEFAULT_DATA = Path(__file__).resolve().parents[1] / "abc-chorus-pairs/hf"


def read_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows or any(not row.get("text") or not row.get("summary") for row in rows):
        raise ValueError(f"Missing or empty text/summary field in {path}")
    return rows


def encode_rows(rows: list[dict], tokenizer, max_source: int, max_target: int) -> list[dict]:
    examples = []
    for row in rows:
        source = tokenizer(row["text"], truncation=False)["input_ids"]
        target = tokenizer(text_target=row["summary"], truncation=False)["input_ids"]
        if len(source) > max_source or len(target) > max_target:
            raise ValueError(
                f"{row.get('id', '<unknown>')}: source={len(source)}/{max_source} "
                f"target={len(target)}/{max_target}; raise limits rather than truncating ABC"
            )
        examples.append({"input_ids": source, "attention_mask": [1] * len(source), "labels": target})
    return examples


class EncodedScores:
    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        return self.examples[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="JSONL split directory; for Vocal-only training use abc-chorus-pairs-vocal/hf")
    parser.add_argument("--model", default="google/byt5-small")
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA.parent / "checkpoints/byt5-small")
    parser.add_argument("--max-source-length", type=int, default=3072)
    parser.add_argument("--max-target-length", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=-1, help="Set to e.g. 2 for a GPU smoke test")
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--eval-save-steps", type=int,
                        help="Evaluate and save every N steps rather than each epoch")
    parser.add_argument("--inspect-only", action="store_true", help="Check token lengths; do not load model weights")
    args = parser.parse_args()
    if args.eval_save_steps is not None and args.eval_save_steps < 1:
        parser.error("--eval-save-steps must be positive")

    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")

    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    training_rows = read_rows(args.data / "train.json")
    validation_rows = read_rows(args.data / "validation.json")
    training = encode_rows(training_rows, tokenizer, args.max_source_length, args.max_target_length)
    validation = encode_rows(validation_rows, tokenizer, args.max_source_length, args.max_target_length)
    print(json.dumps({"train": len(training), "validation": len(validation),
                      "max_source_tokens": max(len(row["input_ids"]) for row in training + validation),
                      "max_target_tokens": max(len(row["labels"]) for row in training + validation)},
                     ensure_ascii=False, indent=2), flush=True)
    if args.inspect_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training; run the script where GPUs are visible")

    from transformers import (AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    settings = Seq2SeqTrainingArguments(
        output_dir=str(args.output),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        eval_strategy=("steps" if args.eval_save_steps else "epoch") if args.max_steps == -1 else "no",
        save_strategy=("steps" if args.eval_save_steps else "epoch") if args.max_steps == -1 else "no",
        eval_steps=args.eval_save_steps,
        save_steps=args.eval_save_steps or 500,
        load_best_model_at_end=args.max_steps == -1,
        save_total_limit=2,
        logging_steps=10 if args.max_steps == -1 else 1,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=settings,
        train_dataset=EncodedScores(training),
        eval_dataset=EncodedScores(validation) if args.max_steps == -1 else None,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )
    trainer.train()
    if args.max_steps == -1:
        trainer.save_model(str(args.output / "best"))
        tokenizer.save_pretrained(str(args.output / "best"))


if __name__ == "__main__":
    main()
