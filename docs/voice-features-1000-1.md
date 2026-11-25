# 1000-1：从音频素材到训练特征

## 最重要的结论

本步骤提取参考声线向量和目标 VAE latent，不更改原 YuE2 推理，也不启动使用错误标签的训练。

**之前“根据 ABC 重新生成 c，再配原曲训练”的建议不应作为可靠的正式训练方案。** 两次演唱可能不对齐，同样的 token 帧数不是对齐证明。缺少原 tokenizer 时，这不是只差一个预处理命令，而是还缺监督数据的来源。

因此本步骤清单命名为 `features_train.jsonl` 和 `features_validation.jsonl`，标为 `training_ready=false`。不能改成 `paired` 就认为可开训。

## 文件位置

- 原始音频素材：`voice-data/1000-1-prepared/`，保持不变。
- 特征结果：`voice-data/1000-1-features/`。
- 参考编码器权重和模型卡：`models/voice-speaker-ecapa/`。
- 独立编码器依赖环境：`.venv-voice-features/`；通过 `.pth` 只读复用原 `.venv` 中的 PyTorch 等依赖，不向原环境安装新包。

本次模型 snapshot 为 `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`，模型下载缓存元数据和哈希留在本地。依赖固定版本见 `tools/voice_features_requirements.txt`，使用 `--no-deps` 安装在隔离环境，避免替换已有 PyTorch。这个环境依赖本项目 `.venv`，不是可随意移到另一台机器的独立打包环境。

## 本次执行结果

已在 GPU 1 上完成（所有模型冻结，无参数更新）：

- **993 个参考音色向量**，每个 `[192]`，另提取 993 个同曲非重叠唱段向量作检查。
- **1,984 个目标 VAE latent**，每个 `[375,64]`，全部为有限 FP32 数值，零失败。
- 训练侧 1,784 条、验证侧 200 条；完整索引生成时重新核验数组哈希。
- 保存首条 VAE 重建试听；目标编码纯循环约 150 秒，参考和探测向量纯循环约 75 秒，不包括模型加载和权重哈希检查。
- 使用真实参考音频对比本地显式实现和官方 `EncoderClassifier.encode_batch(normalize=False)`，输出逐元素相同。
- 完整 CPU 回归：**215 passed、11 skipped、30 subtests passed**。这不代表训练效果通过。

身份向量检查结果：

| 项目 | 全量 993 个编号 | 验证侧 100 个编号 |
| --- | --- | --- |
| 同曲非重叠片段平均余弦相似度 | 0.6431 | 0.6446 |
| 不同编号配对平均余弦相似度 | 0.3035 | 0.3151 |
| 用目标唱段检索参考，Top-1 命中原编号 | 28.20% | 40.00% |

不同编号负对由固定循环错位构造，不是穷举全部不同歌手负对。Top-1 则在该分组全部参考中检索。编号不等于经验证的真实身份，若存在相同歌手的多个编号，检索结果还会受到这种标注误差影响。

**平均相似度有区分，不等于可以可靠克隆歌手。Top-1 表现有限，当前编码器只能作为原型，不能据此放行正式音色训练。** 应试听失败样本、核对编号身份、比较更适合歌声的编码器或改进参考聚合。不会根据验证集结果反复调参后把它仍当无偏测试集。

```text
1000-1-features/
  reference_config.json         身份编码器指纹和预处理规则
  reference_report.json         同曲不同段的身份向量检查
  latent_config.json            VAE 固定版本、指纹、posterior mean 约定
  latent_report.json            全量提取统计
  reconstruction_check.wav     第一条目标经过 VAE 的重建试听
  reconstruction_check.json    重建来源和长度等检查信息
  songs/<编号>/
    reference.npy              参考向量，192 维
    identity_probe.npy         同曲另一段人声的向量，只用于检查，不用于监督目标
    target_01.npy              原曲混音 latent，[375,64]
    target_02.npy              第二个目标（有该片段时）
    *.npy.json                 源音频哈希、模型指纹、数组形状和哈希
  features_train.jsonl         待补语义条件的特征清单
  features_validation.jsonl
  feature_summary.json
```

## 参考编码器的选择

选用 `speechbrain/spkrec-ecapa-voxceleb` 的预训练 ECAPA-TDNN。下载的模型卡标注 Apache-2.0，输入为 16 kHz 单声道，输出采用 `encode_batch(normalize=False)` 对应的 192 维向量。模型卡和 `hyperparams.yaml` 保存在本地。

实现只加载可信本地 checkpoint 的 tensor 字典，`torch.load(weights_only=True)`，不执行下载的 YAML 构造指令。80-bin Fbank、句级均值归一化、ECAPA 参数按下载的 recipe 显式构造，严格加载权重。没有使用 classifier 或全局 embedding normalization，因为本次只需原始 embedding，adapter 内会做 L2 归一化。

这是**说话人模型在歌声上的实验性使用**，不是已经证明适合任意唱法的歌手编码器。检查参考片段与同曲另一段目标人声的相似度，以及与不同编号的相似度。即使分数好，也可能受同曲录音条件影响，不能当成跨歌曲、跨唱法身份泛化的证明。

本地另有 ECAPA2 缓存，但未核验其使用说明与许可，因此本次不使用它提取训练数据。

## 提取命令

先在少量歌曲上检查，再去掉 `--limit` 全量提取。环境路径和模型路径必须与本次记录一致。

```sh
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src .venv-voice-features/bin/python tools/prepare_voice_features.py references \
  --audio-root voice-data/1000-1-prepared --output voice-data/1000-1-features \
  --encoder models/voice-speaker-ecapa --limit 16
```

VAE 使用已有固定 snapshot：

```sh
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src .venv/bin/python tools/prepare_voice_features.py latents \
  --audio-root voice-data/1000-1-prepared --output voice-data/1000-1-features \
  --vae /data/cmq/hf-cache/hub/models--m-a-p--YuE2-Vae/snapshots/9a94e1d0ea9f8087e98f77fa88df4a4068104d2a
```

所有特征齐全后核验数组哈希，生成“不含语义标签”的清单：

```sh
PYTHONPATH=src .venv/bin/python tools/prepare_voice_features.py index \
  --audio-root voice-data/1000-1-prepared --output voice-data/1000-1-features \
  --base-model /data/cmq/hf-cache/hub/models--m-a-p--YuE2-3B/snapshots/29b3558dd46954a0cd9021dc76d5c91864a0f1c7
```

相同配置重复运行时复用已校验的 `.npy`。模型、预处理或音频哈希变化会报错，不会悄悄混用特征。中断后若出现只有数组但没有 `.json` 的未完成文件，需先确认该文件确实是未完成的产物，再移走后重试；不会自动覆盖可能有价值的数据。

## VAE 结果如何解释？

15 秒、48 kHz 双声道目标编码成 375 个 64 维 latent。固定 `sample=False`，使用 posterior mean，编码器和解码器均保持 FP32。

首条重建输出 719,936 帧，相比输入 720,000 帧短 64 帧，是当前 VAE 解码长度约定。没有拉伸 latent 或伪造 semantic 对齐。`reconstruction_check.wav` 应人工与 JSON 指向的原目标音频比较：有限值检查不等于音质验收。

## 最后怎样才能开训练？

当前训练器需要 `(c, reference_embedding, target_latent)`。本步骤只能可靠地补齐后两项。

可选路线：

1. **获得原 tokenizer 或预提取的真实 c**：最直接使用本次原曲 latent。还需保留相符的歌词/ABC/prefix 和时序。
2. **离线制作内容对齐的转换训练对**：YuE2 先生成 c 和音频，仅在制作训练集时转换人声音色，以转换后的完整音频 latent 为目标。训练成功后推理不必再跑 SVC。但它会引入转换伪影，也不能再把本次原曲 latent 直接拿来当这套 c 的目标。必须先确认转换工具、质量和数据构造规则。
3. **研发兼容 token 估计器或不同的内容条件模型**：这是额外研究任务，不是目前 adapter 脚本的一个参数。不能保证效果或立即开始可靠训练。

**没有原 tokenizer，也没有质量合格的转换监督，就不能声称已经具备“直接从这批原曲训练音色保持 NAR”的可靠方案。** 原曲数据本身有用，但不是完整标签。

本次没有默认启动 `--allow-unaligned-experiment`，也没有将缺失 c 改成全零、随机 token 或按时长强行配对。

## 改动记录

| 文件/目录 | 内容 |
| --- | --- |
| `tools/prepare_voice_features.py` | 参考向量、VAE latent、校验与清单汇总，模型只加载一次 |
| `tools/voice_features_requirements.txt` | 固定隔离环境新增依赖版本，不修改原项目依赖 |
| `tests/test_prepare_voice_features.py` | 特征缓存哈希、配置冲突、相似度检查、分组互斥、清单不可误开训的测试 |
| `docs/voice-features-1000-1.md` | 本说明及未解决的开训条件 |
| `.venv-voice-features/` | 隔离安装参考编码器依赖，不更新原推理环境 |
| `models/voice-speaker-ecapa/` | 模型卡、recipe 和预训练权重 |
| `voice-data/1000-1-features/` | 预处理结果与报告 |

原来的 `src/yue2/` 推理和训练模块均未修改。
