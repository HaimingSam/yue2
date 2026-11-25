# Exact input-section conditioning

This experiment leaves `abc-chorus-pairs/` and `abc-chorus-pairs-vocal/`
unchanged. `prepare_abc_chorus_copy.py` creates
`abc-chorus-pairs-vocal-copychorus/`; each target is the complete Vocal ABC,
except that the selected chorus (or verse fallback) is replaced by the literal
line `<INPUT_SECTION>`. The model learns where the input section belongs and
how to generate the surrounding song, while `generate_abc_byt5.py` replaces
that marker with the input section byte-for-byte before restoring the silent
`Ins` track.

```sh
cd /data/cmq/YuE
.venv/bin/python -B tools/prepare_abc_chorus_copy.py
HF_HOME=/data/cmq/hf-cache CUDA_VISIBLE_DEVICES=7 \
  .venv/bin/python -B tools/train_abc_byt5.py \
  --data abc-chorus-pairs-vocal-copychorus/hf \
  --model abc-chorus-pairs-vocal/checkpoints/byt5-small/best \
  --output abc-chorus-pairs-vocal-copychorus/checkpoints/byt5-small \
  --epochs 5
```

Generate with strict placeholder restoration:

```sh
HF_HOME=/data/cmq/hf-cache CUDA_VISIBLE_DEVICES=7 \
  .venv/bin/python -B tools/generate_abc_byt5.py \
  --vocal-only --copy-input-section \
  --model abc-chorus-pairs-vocal-copychorus/checkpoints/byt5-small/best \
  --input abc-chorus-pairs-vocal-copychorus/3780908535454470894/input.abc \
  --output abc-chorus-pairs-vocal-copychorus/evaluation/3780908535454470894.generated.abc
```

Without `--allow-section-fallback`, a model output that omits
`<INPUT_SECTION>` is rejected instead of silently pretending to preserve the
chorus. The JSON report records `section_restore_mode` and ABC validation.

The current five-epoch checkpoint emits `<COPY_INPUT_SECTION>` at the start,
but did not yet emit `<INPUT_SECTION>` on the first held-out test. Strict
generation therefore rejects that result intentionally: it never claims that
the input was preserved when the model omitted the placeholder. The previous
checkpoint is retained at `abc-chorus-pairs-vocal-copychorus-v1/`.
