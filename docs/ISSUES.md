# 当前限制（2026-09-23）

## 1. 干净素材量不足

| 层级 | 数量 | 时长 |
| --- | ---: | ---: |
| 源素材 | 40 个片段 | 1539.01s |
| VAD 单句 | 427 | 1090.60s |
| 视觉真值 | 22 | — |
| 连续生产候选 | 52 | 256.66s |
| accepted | 21 | **97.16s** |
| quarantine | 31 | 159.50s |

按源时长计自动产出率约 6.3%。accepted 已越过一分钟级实验下限，但还不足以覆盖
多情绪和多语速；放宽门槛会重新引入相似角色或残留 BGM/SFX。继续扩充应优先找
对白更干净的源，或改善对白增强。

## 2. 视觉真值规模仍小

目前只有 14 个正例和 8 个负例。它足够完成当前后端选型，但不足以稳定估计总体
误接受率。尤其 src_07~09 没有视觉硬标签，其中的 `calibrated_audio` 结果需要保留
来源标记。

优先补充：

1. 与奶龙音色最接近的其他角色完整句。
2. 奶龙的短语气词、受力声和高情绪对白。
3. 强 BGM、混响和密集音效下的正负例。

## 3. 音质是主要损失来源

当前仍有 31 段 / 159.50s 因未达到 15dB stem margin 留在
`dataset/production/quarantine/`，不能自动用于 TTS。改善对白增强或寻找更干净的
源视频，优先级仍高于继续放松声纹阈值。

逐段视觉复核又在 accepted 中发现 6 段 / 32.52s 跨角色发言，已明确拒绝。
剩余 15 段 / 64.64s 仍需听音核对，未批准进入正式训练集。

## 4. TTS 仍需要主观验收

自动声纹分数能发现明显跑音色，但不能替代听感。当前没有正式批准的默认参考；
新合成需显式提供已听音核对的音频和逐字文本。

2026-09-23 对候选缓存中预筛最高的 5 个新源做了完整增量运行：新增 36 句，
只有 1 段 / 3.40s 进入 accepted，另 3 段 / 19.42s 进入 quarantine。
新增 accepted 的 ASR 文本以“气，”开头，表明边界和转写仍可能出错。候选缓存中
原有转载源预筛接受的句子加起来约 1191s，仍低于正式 1800s 门槛；继续
同质扩源无法取代更干净、来源明确的对白素材与逐段复核。

同日又对 5 条官方账号视频完整运行：混音预筛约 80.7s，最终仅新增 1 段 / 4.08s；
另外 9 段 / 40.62s 因音质门控进入 quarantine。公开视频可访问不等于训练授权，
`source_manifest.csv` 保留原始链接和作者，正式使用前需要核对权利。

## 复核命令

```powershell
Import-Csv dataset/manifests/visual_labels.csv | Group-Object class
Import-Csv dataset/manifests/production_scores.csv | Group-Object decision
Import-Csv dataset/manifests/production_manifest.csv | Group-Object status,speaker_basis
nailong-tts stats
```
