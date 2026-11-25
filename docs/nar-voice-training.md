# NAR 音色 adapter：新手说明与改动记录

日期：2026-10-09。状态：实验性训练代码，不是已训练好的音色保持模型。

## 先看这几句话

- 原来怎么运行 YuE2，现在仍然怎么运行。没有修改原模型、原 CLI、原 pipeline、原 NAR 或 VAE 文件，也不会自动加载 adapter。
- 新功能只有显式运行 `python -m yue2.voice_training ...` 或导入独立 API 才启用。
- 训练只更新新增音色 adapter，不更新原 AR、NAR、VAE 权重。新增权重另存，原模型目录不写入。
- 你只有“原曲 + 分离人声”，还不能直接得到可信的完整训练对。缺少原曲对应的 semantic tokens 是真实限制，不是写代码就能解决的问题。
- 还需要一个**预训练音色/身份编码器**。本次没有随意下载模型、没有用随机网络或普通频谱统计冒充音色编码器。
- 可以启动受控实验，但不能承诺训练后音色一致，更不能承诺音质一定胜过 SVC。

## 为什么这样改？

把原 YuE2 想成“已经会唱歌的歌手”。我们没有重教它唱歌，而是给声音生成部分增加一个“参考声线控制器”。

```text
原有 cover：歌词/ABC/风格 → AR → semantic tokens c
                                             ↓
干声 → 预训练身份编码器 → 音色向量 → 新增 adapter → NAR → VAE → 完整歌曲
```

第一版用 **FiLM 残差调制**，不是 cross-attention：把音色向量变成各层的缩放和偏移，只调整 NAR 隐藏特征。对于一个全局身份向量，这比增加一整套注意力更小、更容易测试。默认只作用于后半部分 NAR 层。

调制输出初始化为零，所以刚创建时不改变输出。`--strength 0` 跳过调制，可在相同 token、噪声、步数下与原模型对照。增加强度不等于保证更像；过强可能损坏声音。

这版没有 LoRA，没有训练参考编码器，没有新增 AR 输入，没有改模型原始 config，也没有给原模型注册 hook。

## 纠正一个容易误解的地方

**冻结 AR，不代表可以把 AR 新生成的 c 当成原曲音频的真实标签。**

同歌词、同 ABC 重新生成的演唱，在发音时长、拖腔和节奏上仍可能不同。训练时将这种 c 配原曲 latent，可能损害咬字、旋律和音质。即使帧数相同，也不能证明内容对齐。

数据分为两种：

| alignment | 含义 | 是否默认允许 |
| --- | --- | --- |
| `paired` | 有依据确认 c 和目标音频对应同一次演唱，例如生成音频自己的 c，或内容/时序保持良好的转换目标 | 是 |
| `unverified_ar` | 用 AR 重新生成的 c 配原曲音频，没有真实 tokenizer 标签 | 否，必须显式 `--allow-unaligned-experiment` |

**不要为了通过校验把第二类改名为 `paired`。代码不能自动证明对齐，标记依赖数据的真实来源。**

长度不一致直接报错，不自动补零、截短、拉伸或重采样 semantic tokens。即使加实验开关，也不会跳过长度检查。大量样本长度不匹配时，应先解决训练数据路线，而不是强行凑齐形状。

同一次生成音频的自重建可以做代码检查，但不能证明学会了换声线。内容对齐的 SVC 目标可以验证控制能力，但有继承 SVC 伪影的风险。你目前的原曲数据没有因此变成已解决的监督数据。

## 数据和模型准备

需要现有本地 YuE2 模型目录和带 encoder 的完整 VAE 目录。原来的默认 decoder-only 加载不够用于制作目标 latent。本功能不自动联网下载。

建议先人工准备 10～20 秒目标片段，以及同一歌手另一个不重叠位置的干声参考。参考与目标不必同长度或逐样本同步：参考提供身份，不提供时间轴。若制作内容对齐的音色替换目标，则替换结果与 c 的演唱时序需要保持一致。

每位歌手只有一首歌时，可以分离后在同一首歌选不同片段，但仍可能泄漏歌曲特征。验证集必须按整首歌和歌手隔离，不能把同一歌曲不同切片分进训练和验证。代码强制检查 `song_id`、`singer_id` 不重叠，无法检查是否人为填错 ID。

### 1. 环境与入口

在仓库根目录使用已有环境。如果没有安装本仓库包，可显式加 `PYTHONPATH=src`：

```sh
PYTHONPATH=src .venv/bin/python -m yue2.voice_training --help
PYTHONPATH=src .venv/bin/python -m yue2.voice_training train --help
```

下面示例的 `/path/to/...` 都要替换成真实路径。示例命令不会自动准备歌词、ABC、分离人声或下载身份编码器。

### 2. 准备干声音色向量

推荐先拿确定可用于歌声的预训练身份编码器，验证“同一首歌不同唱段相似，不同歌手可区分”。这里暂时不指定未经本项目验证的权重。

有两种接入方式：

1. 用编码器自己的程序提取，保存为 float32 `[D]` 的 `.npy`。训练和推理必须使用相同模型、相同采样率和预处理；自行填写稳定的 `encoder_id`，记录模型版本和预处理。
2. 提供可信的 TorchScript 包装文件，包装内包含该模型的必要特征预处理。契约：输入 float32 单声道波形 `[1, samples]`，输出 `[1, D]`。不要把未训练的编码器拿来用。TorchScript 文件是可执行模型，只加载可信文件。

```sh
PYTHONPATH=src .venv/bin/python -m yue2.voice_training embed \
  --encoder /path/to/trusted_identity_encoder.ts \
  --audio /path/to/reference_mono.wav --sample-rate 16000 \
  --device cuda --output voice-data/song-a/reference.npy
```

`16000` 只是示例，必须改成你的编码器要求的采样率。命令不自动重采样，不自动平均立体声，不做峰值归一化。输出旁边的 `reference.npy.json` 记录 `encoder_id`，要复制到训练清单和渲染参数。

参考 embedding 在 adapter 内做 L2 归一化。只看 `.npy` 无法判断它是否来自正确编码器，使用外部提取程序时由你保证一致。

### 3. 制作目标 latent

目标是完整混音片段，不是参考干声。需要提前准备 48 kHz stereo；默认最多 20 秒，避免无意编码整首占满显存。

```sh
PYTHONPATH=src .venv/bin/python -m yue2.voice_training encode \
  --vae /path/to/YuE2-Vae --audio /path/to/target_stereo_48k.wav \
  --device cuda --output voice-data/song-a/target.npy
```

使用冻结 VAE 的 posterior mean（`sample=False`），保存 `[T,64]` float32。`target.npy.json` 记录 `vae_id`，将其填入清单。此选择是本实验的数据约定，不是声称复现了原论文训练配置。

### 4. 准备 semantic 条件

原来的 `SongResult.save_artifacts()` 保存 `prefix.npy` 和 `semantic.npy`，可以复用。这两个文件必须是同一次规划/生成的输出，不能只换 semantic 而保留不对应的 prefix。

如果只想保存 AR 输出而不先做 NAR，可使用原有 staged API：

```python
import numpy as np
from yue2 import YuE2Pipeline

with YuE2Pipeline.from_pretrained("/path/to/YuE2-3B", vae="/path/to/YuE2-Vae",
                                 local_files_only=True) as pipe:
    plan = pipe.plan(style="...", lyrics="...", abc="...", cot="melody", seed=42)
    semantic = pipe.generate_semantic(plan)
    plan.save("voice-data/song-a/semantic")
    np.save("voice-data/song-a/semantic/semantic.npy",
            np.asarray(semantic.tokens, dtype=np.int32))
```

这里生成的 c **不是原唱录音的 token 提取结果**。若配原曲音频，必须标为 `unverified_ar`。

### 5. 写训练/验证清单

取得底座模型指纹：

```sh
PYTHONPATH=src .venv/bin/python -m yue2.voice_training identify --model /path/to/YuE2-3B
```

文件组织示例：

```text
voice-data/
  train.jsonl
  validation.jsonl
  song-a/
    semantic/prefix.npy
    semantic/semantic.npy
    target.npy
    reference.npy
  song-b/
    ...
```

`train.jsonl` 每行一个 JSON 对象，路径相对清单文件所在目录：

```json
{"song_id":"song-a","singer_id":"singer-a","encoder_id":"填写编码器ID","base_id":"填写YuE2指纹","vae_id":"填写VAE指纹","alignment":"unverified_ar","semantic_dir":"song-a/semantic","target_latent":"song-a/target.npy","reference_embedding":"song-a/reference.npy"}
```

`validation.jsonl` 使用另外的歌手和歌曲。`base_id`、`vae_id`、`encoder_id` 在所有样本中必须一致。参考 embedding 维度必须一致。默认 c 最多 500 帧（25 Hz 下约 20 秒），目标 latent 必须恰好同帧数。

ID 是兼容性约定，不是标签真实性认证：manifest 中的 ID 应来自实际文件。不得随意填写来绕过校验。代码会比对模型指纹、adapter 指纹，并检查存在的 `.npy.json` 预处理记录；外部预计算向量和 latent 的真实来源仍由数据准备者负责。训练记录会保存输入数组的文件哈希。

## 开始训练

先用极小数据确认流程，不要一开始跑很多天：

```sh
PYTHONPATH=src .venv/bin/python -m yue2.voice_training train \
  --model /path/to/YuE2-3B \
  --manifest voice-data/train.jsonl --validation voice-data/validation.jsonl \
  --output voice-runs/experiment-001 --device cuda \
  --steps 20 --save-every 10 --rank 64 --lr 0.0001 \
  --checkpoint-layers --allow-unaligned-experiment
```

最后一个开关仅针对你目前的未验证原曲配对，是明确接受研究风险，不是让配对变正确。如果有真正配对的数据，标为 `paired` 并去掉该开关。

实现约定：

- batch size 为 1，不同样本可有不同长度，不做跨样本补零；不支持多卡训练和梯度累积。
- AR 预计算使用 `no_grad`，缓存是普通张量；NAR 路径支持反向传播。底座始终 eval 且不计算参数梯度。
- adapter 使用 FP32；CUDA 上底座使用 BF16，CPU 使用 FP32。未实测生产 GPU 显存占用，不承诺单卡容量。
- `--checkpoint-layers` 用计算换训练激活显存；不缓存整个数据集的 AR K/V。仍需保存单个短样本的 AR 条件缓存。
- 全部 `.npy` 数据目前一次读入 CPU 内存，适合小规模实验，不适合直接加载超大数据集。
- 优化器 AdamW，梯度裁剪 1.0，均匀采样时间；没有身份损失、ASR 损失、F0 损失或 CFG。
- 噪声约定：`z_t=(1-t)*target+t*noise`，目标速度 `noise-target`；沿用原 sampler 从 t=1 向 t=0 积分。网络时间 embedding 沿用原 timestep shift，不改变原 sampler。这是与当前推理接口匹配的实验目标，不是原论文训练时间分布的完整复现。
- 每次保存时评估固定时间、固定噪声的 validation loss。它只测速度预测误差，**不代表音色相似度或听感音质**。
- 只保存 adapter 和元数据，不保存 optimizer/RNG 状态，**暂不支持断点续训**。不要把重新运行当作续训。

输出：

```text
voice-runs/experiment-001/
  run.json
  metrics.jsonl
  step-000010/adapter.json
  step-000010/adapter.safetensors
  step-000020/adapter.json
  step-000020/adapter.safetensors
```

现有输出目录拒绝覆盖。若训练中断，最近已完整保存的 adapter 可以渲染，不能精确恢复优化器训练进度。

## 用训练后的 adapter 生成

使用已有 cover 的 `prefix.npy`、`semantic.npy`，不重跑 AR，也不影响原来的生成结果：

```sh
PYTHONPATH=src .venv/bin/python -m yue2.voice_training render \
  --model /path/to/YuE2-3B --vae /path/to/YuE2-Vae \
  --adapter voice-runs/experiment-001/step-000020 \
  --semantic-dir /path/to/saved-cover-artifacts \
  --reference-embedding voice-data/song-a/reference.npy \
  --encoder-id '与训练清单一致的编码器ID' \
  --seed 42 --steps 32 --strength 1 --device cuda \
  --output voice-results/reference-a.wav
```

改 `--strength 0` 并换一个输出文件名，得到关闭 adapter 的对照；再替换另一歌手的 embedding，固定 seed 和 c，对比声线、歌词、旋律、伴奏和伪影。

渲染依次执行 NAR 和 VAE，不保留两个完整模型同时占显存。它是独立的 PyTorch 路径，暂不支持原 pipeline 的 vLLM、FP8、AR CPU offload、取消回调或速度优化。整个底座仍需在 NAR 计算时放到选定设备，不保证比原优化推理入口快。

部署后不需要 SVC，但仍需身份编码器前处理和 adapter 计算。是否实际更快、更好听，要测量；不能仅凭省掉 SVC 就作保证。

## 怎么判断有没有用？

1. 刚初始化和强度为零时，不应改变原版结果。
2. 在同一 c、同一噪声下，更换参考应改变声线，而不是歌词和旋律。
3. 必须听训练未出现过的歌手/歌曲，不能只看训练歌曲重建。
4. 对比原版、adapter、原版+SVC 的声线与音质，计时包含身份编码前处理。
5. 若 loss 下降但咬字退化、伴奏变形、参考被忽略，应暂停扩大训练并检查数据配对。

每人一首歌并不从理论上禁止多歌手条件学习，但增加了歌曲/身份混淆，不能保证跨歌泛化。缺少同歌手第二首歌也意味着无法直接验证“同一歌手跨歌曲”的真实参考一致性。

## 改动记录

| 文件 | 改动 | 为什么 |
| --- | --- | --- |
| `src/yue2/voice_adapter.py` | 新增零初始化 FiLM adapter、冻结 AR 条件计算、可求导 NAR、独立生成 API | 把风险隔离在新模块，原推理无需变化 |
| `src/yue2/voice_training.py` | 新增 identify/embed/encode/train/render 命令、数据检查、训练记录、独立权重保存加载 | 给实验提供可执行入口，避免误覆盖或混用模型 |
| `tests/test_voice_adapter.py` | 新增原版等价、梯度、冻结、时间方向、分块、权重保存和短训练测试 | 检查“能训练”且“不改原来的声音” |
| `docs/nar-voice-training.md` | 本说明与改动记录 | 让新手知道怎么运行以及还缺什么 |

没有改动现有文件。删除/不导入新增模块即可停止使用实验功能；原 CLI 不依赖它。

## 测试与当前限制

2026-10-09 已在本地完成完整 CPU 测试：**206 passed、11 skipped、30 subtests passed**。包括新增 adapter 两步真实优化器训练、保存/加载、梯度检查和原有 NAR/模型回归测试。运行时关闭 GPU 可见性，跳过项不视为通过，尚未验证真实 3B 模型的 GPU 训练。

测试环境按确认安装了项目声明的 `pytest==9.0.3` 及其缺失依赖；没有修改项目依赖声明。现有 Hugging Face 默认模块缓存为只读，因此测试把模块缓存定向到了 `/tmp`。

CPU 小模型测试不需要下载真实模型：

```sh
HF_MODULES_CACHE=/tmp/yue2-voice-hf-modules PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_voice_adapter.py tests/test_nar.py tests/test_model.py
```

完整 CPU 回归命令：

```sh
CUDA_VISIBLE_DEVICES='' HF_MODULES_CACHE=/tmp/yue2-voice-hf-modules PYTHONPATH=src .venv/bin/python -m pytest -q
```

真实歌手训练、真实 GPU 显存/速度、真实音色改善均需另行验证。代码测试通过不等于音色保持能力训练成功。
