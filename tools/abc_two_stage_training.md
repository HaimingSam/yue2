# Two-stage Vocal ABC generation

The original datasets remain unchanged. Prepare a separate dataset:

```sh
cd /data/cmq/YuE
.venv/bin/python -B tools/prepare_abc_two_stage.py
```

Stage 1 maps the input chorus to a short structure plan. Stage 2 maps the
input chorus plus a requested section and the plan to one Vocal section. The
inference script inserts the input chorus directly and only asks Stage 2 to
generate other sections.

Train separate checkpoints:

```sh
HF_HOME=/data/cmq/hf-cache CUDA_VISIBLE_DEVICES=7 \
  .venv/bin/python -B tools/train_abc_byt5.py \
  --data abc-two-stage/stage1/hf \
  --output abc-two-stage/checkpoints/stage1 --epochs 10

HF_HOME=/data/cmq/hf-cache CUDA_VISIBLE_DEVICES=7 \
  .venv/bin/python -B tools/train_abc_byt5.py \
  --data abc-two-stage/stage2/hf \
  --output abc-two-stage/checkpoints/stage2 --epochs 5
```

Generate a native two-voice ABC:

```sh
HF_HOME=/data/cmq/hf-cache CUDA_VISIBLE_DEVICES=7 \
  .venv/bin/python -B tools/generate_abc_two_stage.py \
  --input abc-chorus-pairs-vocal/3780908535454470894/input.abc \
  --output abc-two-stage/evaluation/3780908535454470894.generated.abc \
  --structure-model abc-two-stage/checkpoints/stage1/best \
  --section-model abc-two-stage/checkpoints/stage2/best
```

If the model generates malformed ABC, inference deliberately fails without
writing the final `.abc`. The raw structure, generated section texts, and
assembled Vocal score remain in `<output_stem>.debug/` beside the requested
output. This is a generation-quality failure, not evidence that the empty
`Ins` track or the input Vocal score is broken. Stage 2 may repeat notes or
run into its `--max-section-tokens` limit; increasing that limit alone does not
fix repeated music.

## Short-group Stage 2 (new)

This alternative uses a counted Stage-1 structure model and trains a
**new** Stage-2 model to predict one 1–4-measure Vocal group at a time. Inputs
include the original chorus, entire structure, section position, section
occurrence, group index, and previous group. Outputs begin with `<MORE_GROUP>`
or `<END_SECTION>` followed by one complete Vocal group. Every generated group
is checked for valid meter and syntax before assembling a full native score;
the chorus is copied from the input, and `Ins` is restored as a silent track.
No existing training data or checkpoints are overwritten.

```sh
cd /data/cmq/YuE
.venv/bin/python -B tools/prepare_abc_chunk_groups.py
.venv/bin/python -B tools/prepare_abc_chunk_overfit.py
HF_HOME=/data/cmq/hf-cache CUDA_VISIBLE_DEVICES=7 \
  .venv/bin/python -B tools/train_abc_byt5.py \
  --data abc-two-stage/stage2-chunks-overfit-20/data \
  --model abc-chorus-pairs-vocal/checkpoints/byt5-small/best \
  --output abc-two-stage/stage2-chunks-overfit-20/checkpoints \
  --epochs 20 --gradient-accumulation-steps 1 \
  --learning-rate 2e-4 --eval-save-steps 1000
HF_HOME=/data/cmq/hf-cache CUDA_VISIBLE_DEVICES=7 \
  .venv/bin/python -B tools/evaluate_abc_chunk_overfit.py
```

The overfit diagnostic uses the **true structure** of each of its 20 training
songs, so it measures Stage 2 alone. `validation.json` duplicates training,
so its loss is not a generalization metric. Read
`abc-two-stage/stage2-chunks-overfit-20/evaluation/summary.json` and
`report.jsonl` for the actual full-song generation results; raw group output
and the assembled Vocal draft are retained in per-song `.debug/` directories.

Only after the diagnostic is satisfactory, train the full Stage 2 dataset:

```sh
HF_HOME=/data/cmq/hf-cache CUDA_VISIBLE_DEVICES=7 \
  .venv/bin/python -B tools/train_abc_byt5.py \
  --data abc-two-stage/stage2-chunks/hf \
  --model abc-chorus-pairs-vocal/checkpoints/byt5-small/best \
  --output abc-two-stage/checkpoints/stage2-chunks --epochs 5
HF_HOME=/data/cmq/hf-cache CUDA_VISIBLE_DEVICES=7 \
  .venv/bin/python -B tools/generate_abc_two_stage.py \
  --chunked --max-section-tokens 256 \
  --input abc-chorus-pairs-vocal/3780908535454470894/input.abc \
  --output abc-two-stage/evaluation/3780908535454470894.chunks.generated.abc \
  --structure-model abc-two-stage/checkpoints/stage1-counted/best \
  --section-model abc-two-stage/checkpoints/stage2-chunks/best
```

For the full-data checkpoints, use the counted Stage-1 model so inference
knows how many groups to generate per section. Evaluate on held-out songs
with **predicted** structure (no target score is read):

```sh
HF_HOME=/data/cmq/hf-cache CUDA_VISIBLE_DEVICES=7 \
  .venv/bin/python -B tools/evaluate_abc_chunk_overfit.py \
  --source abc-two-stage/stage1-counted/hf/test.json \
  --structure-model abc-two-stage/checkpoints/stage1-counted/best \
  --model abc-two-stage/checkpoints/stage2-chunks/best \
  --output abc-two-stage/evaluation/holdout-counted \
  --limit 20
```

The output directory must be empty. Inspect `summary.json`, `report.jsonl`,
and individual `.abc` files; successful parsing and exact input-section
copying do not guarantee that generated verses have musical variety.

To compare greedy decoding against four sampled candidates per Stage-2 group,
rerun the same holdout inputs with `--sample-candidates 4 --temperature 0.8
--top-p 0.9 --seed 20261029` and a different empty `--output` directory.
Each candidate is meter-validated; the most pitch-diverse valid group is
selected and all raw candidates are retained in per-song `.debug/` directories.
Stage 1 still uses greedy decoding. This is a diagnostic, not a guarantee of
musically useful output; check the summary for both validity and diversity.

## Untrained YuE2 prefix-continuation comparison

Use the same Stage-1 holdout IDs to see whether the original YuE2 model can
continue a chorus when its ABC is supplied as an **open** symbolic prefix.
Unlike `pipe.plan(abc=...)`, which accepts an already-finished score, this
experiment invokes the symbolic generator with `[ABC_START]` plus the input
ABC but without `[ABC_END]`; it does **not** train or change a checkpoint.

```sh
cd /data/cmq/YuE
HF_HOME=/data/cmq/hf-cache HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src \
  .venv/bin/python -B tools/evaluate_abc_yue2_continuation.py \
  --limit 5 --output abc-two-stage/evaluation/yue2-continuation-5-final
```

Compare `summary.json` and `report.jsonl` with the ByT5 holdout evaluation.
`*.raw.abc` contains every output even when invalid or truncated; `*.abc` is
written only for complete, parseable scores. Each input chorus is at the
**beginning** of the score, so `exact_chorus_repeated` measures whether YuE2
reproduces it later; this is not the same as conditioning on a mid-song chorus.
`song_like` also requires a new labeled section and at least eight newly
generated Vocal notes; it is a basic filter, not a musical-quality judgment.
Generic style/placeholder lyrics are fixed across songs, making this only a
symbolic-generation baseline rather than a full song-generation assessment.

## No-training YuE2 verse assembly

To use YuE2's new melody without allowing it to rewrite the input chorus,
generate several independent continuations for each chorus. Keep only valid
`% verse` sections with at least 16 Vocal notes, at least four distinct MIDI
pitches, and no single pitch dominating more than 70% of the notes. Assemble
two distinct accepted verses with two **exact copies** of the original chorus;
the final `Ins` voice is reconstructed as silent. A song is rejected if either
verse is missing or the rearranged score is invalid. This is a strict baseline,
not a model trained for middle-section conditioning or guaranteed musical form.

```sh
cd /data/cmq/YuE
HF_HOME=/data/cmq/hf-cache HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src \
  .venv/bin/python -B tools/evaluate_abc_yue2_assembled.py \
  --limit 20 --attempts 4 --max-tokens 2048 \
  --output abc-two-stage/evaluation/yue2-assembled-20
```

The script uses only test-split examples labeled `chorus`, never gold target
scores, and it leaves inputs and checkpoints untouched. Read `summary.json`
for the strict assembly success count, `report.jsonl` for per-song failures,
`*.abc` for successful scores, and `*.attempt-NN.txt` for raw outputs.
Requires a fresh empty output directory for each run.

## Listen to assembled songs

Render the eight successful ABC scores with the original YuE2 model. The
scores remain unchanged; `cot=melody` lets YuE2 produce accompaniment from
the Vocal melody. Default lyrics are neutral Chinese `啦` placeholders, so
these files are for melody/structure listening, not a vocal-lyrics benchmark.

```sh
cd /data/cmq/YuE
HF_HOME=/data/cmq/hf-cache HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src \
  .venv/bin/python -B tools/render_assembled_abc_audio.py \
  --scores abc-two-stage/evaluation/yue2-assembled-20 \
  --output abc-two-stage/audio/yue2-assembled-20
```

Open `<output>/index.html` in a browser to play previews or full songs.
`<output>/<song_id>/audio.flac` contains lossless audio;
`<output>/previews.m3u` plays one-minute samples of all completed songs and
`<output>/playlist.m3u` plays full tracks. Check
`<output>/summary.json` for failures or truncation. The script
skips completed songs on rerun and rejects an existing output whose score or
request differs. For your own lyrics, use `--lyrics-file PATH` and a **new**
`--output` directory; lyric sections should follow verse/chorus/verse/chorus.
