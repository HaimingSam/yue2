# 两首歌的通用段落拼接

`make_mashup_generic.py` 使用两份已转录并标有 `% intro`、`% verse` 等段落的 YuE2 双声部 ABC 乐谱和两份带时间戳的歌词。最小配置只需要两份谱子和两份歌词；脚本会自动推断段落配方、目标调、速度和音符单位，并将音符移调、保留并隔离段落内的变拍，再拼接乐谱并生成歌词请求与时间映射；不运行 GPU、转录或音频拼接。

## 操作流程

1. 为两首歌准备无和弦标记的 native `score.abc`（最好带 `% intro`、`% verse`、`% chorus` 等段落）和 JSON 歌词（顶层数组或 `data` / `lyrics` 数组；每条有 `text`、`start`、`end`，有 `words` 时使用有效词的最早/最晚时间）。如无现成乐谱，可先用 `skills/yue2-music/scripts/transcribe.py` 转录。调性不必一致：脚本在两首来源调号中选择一个目标调，按最短半音距离实际移调 Vocal 与 Ins 音符，并检查人声音域；不是只手改 `K:`。`cot=melody` 不会自动移除 ABC 中的和弦标记，因此本脚本会拒绝含和弦的输入。
2. 最小配置只需复制 `mashup_config.minimal.json`，填写 `sources.A/B` 的四个路径。脚本会自动生成请求 ID、风格提示、目标调、平均速度、目标 `L:` 单位和 ABAB 段落顺序。默认 `compression` 为 `full`；要生成当前使用的浓缩版，可设置 `"compression": "compressed"`（也接受 `compact` / `short`）。浓缩版采用“前奏 → A 主歌/导歌 → B 副歌 → A 副歌 → B 主歌/副歌 → 尾奏”的短歌曲结构，保留两首歌的旋律和歌词。命令行也可用 `--compression compressed` 覆盖配置。若需要自定义，可再填写 `style`、`target_key`、`target_meter`、`target_bpm`、`target_denominator`、`recipe`、`seed`、`timestamp_unit` 或 `audio_offset_seconds`。自定义 `recipe` 后将覆盖自动段落顺序。可选 `vocal_pitch_range: [48, 88]` 是人声音高 MIDI 范围（默认值即 `[48, 88]`）；不适合该范围的音域可根据实际歌手调整，若两种来源调都无法满足范围则报错。可选 `max_tempo_ratio: 1.5` 限制原速度与统一目标速度相差的最大倍数，超限即报错；可选 `allow_meter_changes: true` 保留段落中的变拍号，若本地模型不适合变拍号则设为 `false`，一旦所选段落拍号与目标拍号不同即报错。
3. 在 `/data/cmq/YuE` 下执行：

   ```sh
   python3 -B yue2-cover-mashup/make_mashup_generic.py \
     --config yue2-cover-mashup/mashup_config.example.json \
     --output /tmp/my-mashup-score
   ```

   生成浓缩版：

   ```sh
   python3 -B yue2-cover-mashup/make_mashup_generic.py \
     --config yue2-cover-mashup/mashup_config.example.json \
     --compression compressed \
     --output /tmp/my-mashup-score-compressed
   ```

4. 检查输出的 `score.abc`（拼接的模型旋律条件）、`request.json`（歌手段落提示及歌词）、`manifest.json`（各来源的 `transposition_semitones`、目标音域、逐段四分音符拍时长、小节拍号序列、原/目标时间窗及速度比）、`aligned_lyrics.json`（源、目标句级时间戳）和 `syllable_alignment.json`（逐句音节—音符候选映射与待复核项）。非空输出目录不会被覆盖。重点检查音频与乐谱起点是否一致、有无漏词/跨段歌词、重复的段落是否确实对应第二次演唱。
5. 如需用 GPU 生成音频，在装有该项目的 Python 环境中执行（模型/VAE 路径换为实际本地目录）：

   ```sh
   PYTHONPATH=src /data/cmq/YuE/.venv/bin/python yue2-cover-mashup/run_mashup_generic.py \
     --input /tmp/my-mashup-score --output /tmp/my-mashup-audio \
     --model /path/to/YuE2-3B --vae /path/to/YuE2-Vae \
     --max-tokens 12000
   ```

   此脚本读取 `score.abc` 和 `request.json`，把谱子作为 `abc` 输入，设置 `cot=melody`，调用 `YuE2Pipeline` 并校验保存的音频结果。它不允许覆盖已有的非空输出目录；此步骤不会因构建乐谱而自动执行。

## 对齐算法与限制

每个段落从解析过的 ABC 小节求起止拍数；以源 BPM 换算到源音频时间，再加 `audio_offset_seconds`。按**歌词行时间中点**归属段落；行跨越段落边界时，只将其时间范围裁剪到该段，不自动拆分文字。输出时间按段落拼接的累计拍数和目标 BPM 换算：`target_start + (clipped_source_time - source_window_start) * source_bpm / target_bpm`。`aligned_lyrics.json` 仅供人工核对或后续工具消费；YuE2 当前生成请求读取的是不带时间码的 `request.json` 中的 `lyrics`，**不能保证模型逐字按这些时间戳演唱**。

脚本通过仓库现有 `abc_tools.parse_abc` 验证两份输入与最终输出，只支持其 native ABC 方言；支持正文中的 `M:` 变拍，不支持正文中的 `K:` 变调。每个来源段提取两个声部的四分音符拍内起点、音高、持续拍数和每小节的起点、时长、拍号；输出再次解析后逐段逐声部与原事件对照（音高只允许预定移调量），不允许无声丢拍、多拍、重拍或丢音。段落首个声部块缺拍号时补写该段原始拍号，后续声部块遵循原有明确变拍号，不会强制恢复段首拍号。自动段落识别依赖常见标签（`intro`、`verse`、`pre-chorus`、`chorus`、`bridge`、`interlude`、`outro` 及常见中文标签）；如果只有非标准标签，需显式提供 `recipe`。现有标签策略不分析是否完整乐句，只保证选中的是可解析、完整小节的片段。`target_denominator` 必须能整除两份输入的分母；若放大音符时值后超出 native 方言可接受的时值，将报错而不写输出。移调时按来源调号与目标调号根音的最短半音距离移动音高，分别检查 Vocal 配置音域与 Ins 的 MIDI 有效范围；遇到跨越段落边界的连音会拒绝该输入，不会凭空拆分音符。移调会生成相对于目标调性的显式升降号，避免源调号和小节内临时变音记号泄漏。默认整个重组谱只使用一个目标 BPM；不会在不同来源段写局部 Q:，超出速度倍数限制时直接拒绝，不能保证心理感知的节拍速度相同。当前不自动创作或插入过渡小节；若模型不接受变拍号，可禁用变拍号并拒绝该配方，而非无损强改原段拍号。上述检查不能保证两段和声相容或歌手实际可唱；歌词文件缺失或时间戳无效会直接报错；歌词行跨段时只会裁剪时间，不会自动拆字。段落切换、歌词校对和音色切换仍需人工试听和调整。

## 句内歌词—音符候选对齐

在句级选段之后，默认 `syllable_alignment: "report"` 另外生成可审查的 `syllable_alignment.json`。中文按单字拆分音节位置，英语按词拆分但**不猜测英文词内部音节**；有安装 `pypinyin` 时提供上下文拼音读音估计，未安装时 `pronunciation_estimate` 为 `null` 并标记人工复核。多音字和儿化的真实唱法仍需人工确认。优先用来源 JSON 中 `words` 的逐词/逐字时间戳估计音节起点，若文本与词时间戳对不上则用整句时间均匀分配并标记 `uniform_sentence_estimate`。映射从已算出的目标时间窗、Vocal 音符起点/持续时间中提取候选音符，以动态规划保持音节次序：每音节至少分配一个音符，允许同一音节延伸到后续多个音符，也允许没有歌词的音符与音符间休止；无法满足音节数、不足够精确或读音未知时输出 `review` 和修改歌词/选段的提示，不会杜撰音符。每音节记录文本、读音估计、预计时间、实际对应音符、起音误差（毫秒）；句子另记最大/平均绝对起音误差及 Vocal 空隙时间。配置 `max_alignment_error_ms` 默认为 700，仅决定是否应复核，**不是**生成音频与真实歌词的误差指标。

若设 `syllable_alignment: "strict"`，任何需要复核的入选歌词句都会在写文件之前失败；本机 `/data/cmq/YuE/.venv` 已安装 `pypinyin==0.55.0`，更换运行环境时需另装该库，且仍须人工复核多音字。设 `syllable_alignment: "off"` 可跳过句内映射（仍写空的报告）。此功能**只写候选对应表**：`run_mashup_generic.py` 仍只向模型传纯文本歌词和 ABC，不读取 `syllable_alignment.json`；它不能承诺模型逐字演唱或真实生成音频的对齐效果。
