# Twenty-song memorization diagnostic

`tools/prepare_abc_overfit.py` copies the first 20 rows of the existing
Vocal-only **training** split to a separate `abc-overfit-20/data/` directory.
The diagnostic `validation.json` deliberately duplicates `train.json`, so its
loss measures memorization, **not generalization**; the original validation
and test splits are untouched. Starting weights are the existing Vocal-only
ByT5 checkpoint, which was previously trained on these 20 songs.

```sh
cd /data/cmq/YuE
.venv/bin/python -B tools/prepare_abc_overfit.py
HF_HOME=/data/cmq/hf-cache CUDA_VISIBLE_DEVICES=7 \
  .venv/bin/python -B tools/train_abc_byt5.py \
  --data abc-overfit-20/data \
  --model abc-chorus-pairs-vocal/checkpoints/byt5-small/best \
  --output abc-overfit-20/checkpoints \
  --epochs 80 --learning-rate 2e-4 --eval-save-steps 30
HF_HOME=/data/cmq/hf-cache CUDA_VISIBLE_DEVICES=7 \
  .venv/bin/python -B tools/evaluate_abc_overfit.py \
  --data abc-overfit-20/data/train.json \
  --model abc-overfit-20/checkpoints/best \
  --output abc-overfit-20/evaluation --limit 5
```

Check `abc-overfit-20/evaluation/summary.json` and `report.jsonl` for native
ABC validity, actual Vocal note counts, exact target matches, and the longest
contiguous matching pitch sequence from the input chorus (a diagnostic proxy,
not proof the entire chorus was preserved). The `.vocal.abc` files hold raw
model output and `.abc` files, when present, hold the restored two-voice score.
If the model cannot generate useful notes for songs it just trained on, focus
on target representation, decoding, and conditioning before collecting more
training data. If it memorizes those songs but fails on untouched test songs,
more diverse data may help generalization.
