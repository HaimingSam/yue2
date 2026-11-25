# Vocal-only ABC training

The original `abc-chorus-pairs/` dataset stays untouched. The conversion writes
`abc-chorus-pairs-vocal/` with the same song IDs and the same train/validation/test
split. Each score keeps its native Vocal notes, meters, key changes, and section
markers, but omits the `Ins` definition and music groups. It does not alter rests
in the Vocal track. All 993 source targets have an empty instrumental track.

```sh
cd /data/cmq/YuE
.venv/bin/python -B tools/prepare_abc_vocal_seq2seq.py
HF_HOME=/data/cmq/hf-cache CUDA_VISIBLE_DEVICES=7 \
  .venv/bin/python -B tools/train_abc_byt5.py \
  --data abc-chorus-pairs-vocal/hf \
  --output abc-chorus-pairs-vocal/checkpoints/byt5-small --epochs 5
```

Generate a native two-voice ABC from either a Vocal-only `input.abc` in the
new directory or the original two-voice input. `--vocal-only` tells the model
to predict only Vocal; after decoding it recreates matching silent `Ins`
groups, including matching meter and key changes. It reports invalid or
truncated outputs rather than repairing the model's notes or structure.

```sh
HF_HOME=/data/cmq/hf-cache CUDA_VISIBLE_DEVICES=7 \
  .venv/bin/python -B tools/generate_abc_byt5.py \
  --vocal-only \
  --model abc-chorus-pairs-vocal/checkpoints/byt5-small/best \
  --input abc-chorus-pairs-vocal/3780908535454471632/input.abc \
  --output abc-chorus-pairs-vocal/evaluation/3780908535454471632.generated.abc
```

The sibling `.json` report includes `abc_check.valid`, Vocal and Ins note counts,
and whether generation reached the token limit. A valid ABC does not by itself
prove that the generated melody is musically useful; compare it against the
held-out target and other test songs before using it.

The first hold-out test (`3780908535454471632`) produces structurally valid
native ABC but **zero Vocal notes**. This workflow removes the predictable
instrumental output, but the current checkpoint has not yet learned to expand
that test chorus into a usable complete melody.
