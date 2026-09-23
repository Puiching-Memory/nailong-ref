# 当前实验结论

## 1. 视觉只用于预校准

公开转载视频与本地音频通过指纹对齐，只在画面明确显示发声角色时标注：

- target：14 条奶龙对白
- other speaker：7 条其他角色对白
- non-speech：1 条歌曲/舞蹈

src_07~09 没有视觉硬标签。生产阶段不下载、不解码视频，只加载视觉校准产生的
声纹中心与阈值。

## 2. ERes2NetV2 是当前主模型

按视频留一验证，避免同一视频相邻句同时进入训练与测试：

| 判定器 | AUC | balanced accuracy |
| --- | ---: | ---: |
| ERes2NetV2 视觉正负中心 | **0.759** | **0.857** |
| ReDimNet 视觉正负中心 | 0.705 | 0.723 |
| 标准化均值融合 | 0.688 | 0.732 |

因此生产分数只使用 ERes2NetV2。ReDimNet 原型与分数保留为审计旁证，不参与自动
决策。不能仅凭“融合更稳”的直觉合并模型；本批真值上融合会变差。

## 3. Nomo-pVAD 不能区分相似卡通声线

Nomo-pVAD 能把歌曲压到约 0.083，但把其他角色对白打到 0.89–0.97；视觉真值上的
AUC 约 0.50。它可以提示非对白，却不能承担奶龙身份判定。

## 4. 说话人正确不代表音频干净

低频占比、停顿静音和高通滤波都不能证明 BGM/SFX 已去除。生产使用同窗口两个 stem
的差分：

```text
stem_margin_db = 20 * log10(RMS(vocals) / RMS(no_vocals))
```

门槛为 15dB。当前 52 个连续候选中有 21 段 / 97.16s 通过；31 段 / 159.50s
进入 quarantine。下游只消费 accepted。

## 5. 当前生产数据仍然很小

accepted 的 21 段中，两段是 `visual_confirmed`，19 段是 `calibrated_audio`。总长
97.16s，尚未逐段复核，离正式训练集很远。

自动接受线使用按视频留一的零观察误接受边界与 balanced-accuracy 阈值中更严格者，
当前为 0.2283。现有 8 个负例仍太少，不能解释成真实总体误接受率为零；下一轮应
优先增加与奶龙声线相似的负例。

## 6. 可复现性

- ERes2NetV2 CPU 全量嵌入重复运行 SHA-256 一致。
- 校准包、全量评分和生产 WAV 重复构建逐字节一致。
- `production_manifest.csv` 记录逐段 SHA-256、说话人依据和 stem margin。
- `separation_manifest.csv` 记录源文件及两个 Demucs stem 的 SHA-256。
- `model_manifest.csv` 记录模型 ID、权重大小、许可证与权重 SHA-256。
