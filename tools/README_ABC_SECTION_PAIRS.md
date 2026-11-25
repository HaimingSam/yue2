# ABC 片段 → 完整歌曲训练对

用 `build_abc_section_pairs.py` 从 SheetSage2 转录的 ABC 构造数据集。每首歌优先取第一个可独立解析且含人声音符的 `% chorus` 段；完全没有 `% chorus` 的歌使用 `% verse`。遇到片段首尾跨段连音时，将连音裁剪为独立片段音符，不修改完整乐谱。片段保留原谱的 native 双声部 ABC 格式、变拍和原有调性；完整谱按字节复制。

```sh
python3 -B /data/cmq/YuE/tools/build_abc_section_pairs.py \
  --input-root /data2/cmq/abc-yue2 \
  --output-root /data/cmq/YuE/abc-chorus-pairs
```

先预检可加 `--dry-run`，小样本可加 `--limit 20`。为避免把旧结果和新结果混在一起，正式导出需要一个空目录。输出中的 `pairs.jsonl` 记录每条配对相对于输出目录的 `input` / `target` 路径、原始来源、选取的段落、原谱中段落位置和音符数量。每首歌在 `<歌曲 ID>/input.abc` 有一个独立片段，在 `<歌曲 ID>/target.abc` 有完整乐谱；无法选择有效段落的文件记录在 `failures.jsonl`。如存在失败，命令返回非零状态，但成功的数据对仍会保留。

训练时按**歌曲 ID**划分训练/验证集，勿把同一首歌的不同版本分到两边。`% chorus` / `% verse` 是转谱模型预测标签，可能不准确；`verse` 回退记录了 `section: "verse"`，可按需筛选。这个数据集只提供片段旋律与完整谱配对，不包含原始音频、不保证旋律标签准确，也不能证明目标的音乐结构是人工标注的。
