# RMVPE 辅助修谱（第一轮）

脚本：`audit_rmvpe.py`。不修改原转谱、模型文件或 MIDI-SAG 环境，不重新生成 cover。

## 本次结果

输入：`work/transcriptions/f8e2_hjf3`，音频：`download/f8e2_hjf3.mp3`。

最终结果位于 `work/f0_audit/f8e2_hjf3_round1/octave_aligned/`，可打开其中的 `listen.html` 对比试听和音高图。

- 原谱相对人声 F0 存在约一个八度的整体差异，RMVPE 与独立 pYIN 均支持这一观察。不能仅据此判断是转谱错误还是原转谱模型的音区处理。
- 为避免把八度差混同于旋律错音，先将检测 F0 在比较时提高 12 个半音；`corrected.abc` 保留原谱音区。
- 110 个音符中，33 个获得稳定的原音高支持，2 个通过 RMVPE 双输入及 pYIN 的修正检查，75 个未满足全部严格稳定条件，保持原样等待复核。
- 第 19 个音符，10.83–11.15 秒：MIDI 73 → 74（C♯5 → D5）。
- 第 102 个音符，54.61–54.93 秒：MIDI 70 → 69（A♯4 → A4）。
- 高稳定 RMVPE 音符上的一致数：33/35 → 35/35；pYIN 高稳定音符：39/41 → 41/41。这是用来选择改动的证据一致性，**不是独立测试集准确率**。
- 两检测器共同有效且 pYIN voiced probability ≥ 0.5 的 4055 帧中，音高差中位数约 4.33 音分，98.37% 相差不到半音。这支持 F0 的一致性，不等于绝对正确。
- 已检验 110 个音符的音高反解析、音符数量、起点、时值、小节和 BPM；原谱重新导出与输入 ABC 逐字一致。
- 第 8 小节的三拍疑点、歌词对齐、音符拆分/合并均未修改。未试听评价新 cover。

## 产物

| 文件 | 用途 |
|---|---|
| `listen.html` | 本地试听对照页面，定位到两处修改附近 |
| `corrected.abc` / `corrected.mid` | 保留原谱音区，只修正两个音 |
| `source_register.abc` / `source_register.mid` | 对照候选：修正版整体降八度，接近原唱音区；不是逐音符独立验证过的另一个版本 |
| `original_preview.wav` / `corrected_preview.wav` | 原始 LAB 时间轴、简单合成音色的修改前后对照；不是生成歌曲，也不是固定 BPM 的 ABC 演奏 |
| `source_register_preview.wav` | 降八度候选试听 |
| `pitch_comparison.png` | 原音高、修正音高、RMVPE 与 pYIN 曲线；图上 F0 显式加了 12 半音 |
| `note_audit.csv` / `note_audit.json` | 所有音符的候选、覆盖率、稳定度、处理结果 |
| `report.json` / `rhythm_review.md` | 结构化结果与节拍疑点 |

上一级目录保存分离人声、两路 RMVPE、独立 pYIN、模型校验哈希和未做八度归一化的初次结果。初次结果没有修改音符，属八度保护规则正常触发；以 `octave_aligned/` 为本次最终对照。

## 复现

在 YuE 根目录运行，使用本机现有 `midi-sag` 环境。分离使用本机 URV 的 Mel-Band Roformer Inst，通过混音减去估计伴奏得到人声。RMVPE 使用 MIDI-SAG 的 `rmvpe_model.pt`。

`beartype==0.18.5` 已安装到项目独立 `work/f0_audit/deps`，未修改任何外部环境。

```sh
PYTHON=/data/leyuan/miniforge3/envs/midi-sag/bin/python
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/data/cmq/YuE/yue2-cover-test/work/f0_audit/deps
export NUMBA_CACHE_DIR=/tmp/yue-rmvpe-numba
export MPLCONFIGDIR=/tmp/yue-rmvpe-mpl
export XDG_CACHE_HOME=/tmp/yue-rmvpe-cache
OUT=yue2-cover-test/work/f0_audit/f8e2_hjf3_new_run
CUDA_VISIBLE_DEVICES=1 "$PYTHON" yue2-cover-test/audit_rmvpe.py extract --output "$OUT"
"$PYTHON" yue2-cover-test/audit_rmvpe.py verify --output "$OUT"
"$PYTHON" yue2-cover-test/audit_rmvpe.py analyze --output "$OUT" --analysis-name octave_aligned --reference-octave 12
```

`--reference-octave 12` 是本首歌经独立核验后的显式选择，不应对所有歌曲默认使用。其他音频请同时指定 `--audio` 与 `--transcription`，并先使用默认零偏移检查。

校正规则：裁掉少量首尾滑音，至少 10 个有效帧、75% 覆盖率、80% 候选音高支持度、中位绝对偏差 ≤ 0.22 半音、中心偏离整数音高 ≤ 0.25 半音；普通自动修改最大两半音。多平台、八度跳变、低覆盖、检测分歧均不强制修正。

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -p test_rmvpe_audit.py -v
```
