# 管线参考

生产链路只有一条：HTDemucs 人声分离 → VAD 切句 → 双声纹嵌入 → 视觉真值校准
→ 音频判定 → 残留 BGM/SFX 门控。视觉只在预校准阶段使用，批量生产只读音频。

## 阶段

| 阶段 | 输入 | 输出 | extra |
| --- | --- | --- | --- |
| `separate` | `dataset/sources/src_NN.*` | `data/sep_out/htdemucs/src_NN/{vocals,no_vocals}.wav` | `sep` |
| `segment` | `vocals.wav` | `dataset/utterances/`、`utt_manifest.csv` | `spk` |
| `transcribe` | 单句 wav | `sv_all.csv` | `asr` |
| `embed-redimnet` | 人声轨 + 句清单 | `emb_redimnet.npy`，仅作审计旁证 | `spk` |
| `embed-eres2net` | 人声轨 + 句清单 | `emb_eres2netv2.npy`，生产主后端 | `asr` |
| `calibrate-visual` | 双嵌入 + `visual_labels.csv` | 四个声纹中心、阈值、`production_scores.csv` | — |
| `prepare-production` | 全量分数 + 两个 Demucs stem | accepted/quarantine 音频与生产清单 | — |
| `nailong-tts review/audit/pack` | accepted + 逐段复核 | 机器门禁与正式训练列表 | — |

## 决策规则

- 视觉真值：14 条 target、7 条 other speaker、1 条 non-speech。
- 主模型：ERes2NetV2；按视频留一 AUC 0.759。ReDimNet AUC 0.705，只作旁证。
- accept 阈值：取“按视频留一负例最高分之上”和最佳 balanced-accuracy 阈值中
  更严格者；当前为 0.2283。小样本上的观察 FPR=0 不是总体保证。
- review：落在 accept/reject 之间的句子不自动进入连续候选。
- 音质门控：`RMS(vocals)-RMS(no_vocals) >= 15dB` 才进入 accepted。
- 连续片段：3–10 秒，相邻句最大间隔 0.8 秒，导出为 24kHz 单声道 WAV。

阈值、归一化参数、模型和输入哈希均在
`dataset/calibration/calibration.json`，逐段结果在
`dataset/manifests/production_manifest.csv`。

## 完整复现

GPU 环境先运行 `./scripts/setup-gpu.ps1`，再使用该脚本输出的 Python 执行：

```powershell
data/gpu_env/Scripts/python.exe -m nailong.cli build-dataset --device cuda:0
```

`build-dataset` 把同一个设备参数传给 Demucs、SenseVoice、ReDimNet 和
ERes2NetV2。`--plan` 只列命令；缺少 CUDA 时正式运行会失败，不会静默使用 CPU。
`rank-sources` 也支持 `--device cuda:0`。已有源的分离和嵌入会增量跳过；
验证 GPU 实际计算需处理新源或指定单独的测试输出。

2026-09-23 在 RTX 4060 Ti 上以 `--batch 1 --device cuda:0` 验证：
`src_35` 经 Demucs GPU 分离（约 58.5s 轨道的模型进度约 2.6s）、
SenseVoice GPU 转写 11 句、ReDimNet 与 ERes2NetV2 GPU 嵌入各 11 句。
后续音频门控得到 1 段 / 4.84s；训练审计按预期因未复核和语料不足失败。

同日再对 5 条官方账号公开视频执行 `fetch-public` → GPU `rank-sources` →
`build-dataset --batch 5 --ranking data/eval_sep/official/candidate_scores.csv --device cuda:0`。
共新增 80 个切句，混音预筛约 80.7s，最终只新增 1 段 / 4.08s。
当前 40 个源约 1539.01s、427 个切句约 1090.60s、accepted 21 段 / 97.16s、
quarantine 31 段；该次运行后 accepted 均待复核，正式审计失败。

随后对 21 段 accepted 提取对应公开视频的逐时刻画面，逐段记录在
`dataset/manifests/visual_review.csv`。6 段 / 32.52s 可从角色交替和字幕确认
跨说话人，已在 `training_review.csv` 标为 `approved=no`；15 段 / 64.64s
保留待听音状态。画面确认只能排除明显混说话人，不能替代听音验收和逐字转写，
因此正式训练审计仍无获批准片段。

完成逐段复核并达到训练门槛后，运行 `nailong-tts pack` 生成训练列表。

当前 `nailong-tts pack` 只读取 `dataset/production/accepted/` 中经逐段批准的样本；
quarantine 永不自动消费。正式门禁要求至少 200 段 / 30 分钟。
`segment`、`transcribe` 与两套 `embed-*` 默认只追加新源/新句，保持已有视觉标签引用的
idx 不变。源溯源信息见 `source_manifest.csv`。

## 数据检查

```bash
nailong reel
nailong band dataset/production/accepted/*.wav
nailong-tts stats
```

模型权重的 ID、大小和 SHA-256 见 `dataset/manifests/model_manifest.csv`；源文件与
两个 Demucs stem 的 SHA-256 见 `dataset/manifests/separation_manifest.csv`。
