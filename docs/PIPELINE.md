# 管线参考

每个阶段都是脚本式模块：`nailong <stage>` 等价于 `python -m nailong.stages.<stage>`。
阈值和路径都在 [src/nailong/config.py](../src/nailong/config.py)，改参数只改那一处。

## 阶段

| 阶段             | 模块                       | 输入                              | 输出                                                                               | 需要 extra |
| ---------------- | -------------------------- | --------------------------------- | ---------------------------------------------------------------------------------- | ---------- |
| `separate`       | `stages/separate.py`       | `dataset/sources/src_NN.mp4`      | `data/sep_in/src_NN.wav`、`data/sep_out/htdemucs/src_NN/vocals.wav`                | `sep`      |
| `segment`        | `stages/segment.py`        | `vocals.wav`                      | `dataset/utterances/utt_NNN.wav`、`utt_manifest.csv`、`dataset/reels/utt_reel.wav` | `spk`      |
| `transcribe`     | `stages/transcribe.py`     | `utt_NNN.wav`                     | `sv_all.csv`（事件/情绪/文本）                                                     | `asr`      |
| `embed-redimnet` | `stages/embed_redimnet.py` | `vocals.wav` + `utt_manifest.csv` | `emb_redimnet.npy`                                                                 | `spk`      |
| `embed-episode`  | `stages/embed_episode.py`  | `vocals.wav`                      | 整集/前后半段嵌入对照报告                                                          | `spk`      |
| `embed-chunk`    | `stages/embed_chunk.py`    | `vocals.wav` + `utt_manifest.csv` | `emb_chunks.npy`、`chunk_meta.csv`                                                 | `spk`      |
| `score-long-ref` | `stages/score_long_ref.py` | `vocals.wav` + `emb_redimnet.npy` | `utt_margin_long.csv`                                                              | `spk`      |
| `score-two-pass` | `stages/score_two_pass.py` | 同上 + `emb_episodes.npy`         | `utt_two_pass.csv`、`emb_episodes.npy`                                             | `spk`      |
| `verify`         | `stages/verify.py`         | `utt_two_pass.csv` + `sv_all.csv` | 三条证据报告（不写文件）                                                           | `spk`      |
| `finalize`       | `stages/finalize.py`       | `utt_two_pass.csv` + `sv_all.csv` | `dataset/final/*.wav`、`final_manifest.csv`                                        | `spk`      |

工具（不产出交付片段）：

| 工具            | 用途                                                              |
| --------------- | ----------------------------------------------------------------- |
| `annotate`      | 主动学习标注：`seed` / `status` / `query [N]` / `teach 12:1 47:0` |
| `inspect-group` | 列出 A/B 两组全部成员，判断 B 组是真实说话人还是语气词堆          |
| `sanity`        | 声纹模型自检：语音 vs 白噪声 vs 440Hz 正弦的余弦矩阵              |
| `reel`          | 把目录里片段串成试听带（段间 1s 静音）                            |
| `band`          | 频段/RMS 量化分析，无播放设备时判 BGM 残留                        |

## 关键参数

| 参数                      | 值            | 依据                                                             |
| ------------------------- | ------------- | ---------------------------------------------------------------- |
| `GAP_TOL`                 | 0.10s         | 偏松会把相邻不同说话人的台词并成一句，是聚类无簇结构的主因之一   |
| `MIN_UTT`                 | 0.30s         | 更短的句子人耳也分不出说话人                                     |
| `SEG_PAD`                 | 0.06s         | 切句前后各留，保住字头字尾                                       |
| `DB_THRESH` / `SP_THRESH` | -45 dB / 0.25 | 帧 RMS + 250-4000Hz 能量占比双阈值                               |
| `OUTLIER`                 | `src_02`      | 30s 级嵌入就确认是另一个说话人，全程排除                         |
| `MIN_A`                   | 0.62          | 组A 内部截尾；#2(0.464) 有人工反例，低分段还混有其他角色的敬语句 |
| `MIN_RUN` / `MAX_RUN`     | 3.0 / 10.0s   | 交付片段时长窗口                                                 |
| `HIGHPASS_HZ`             | 80            | 导出高通，压掉分离残留的低频伴奏                                 |

## 复现整条链路

```bash
uv sync --extra all

nailong separate                  # 必需：data/sep_out 是后面所有阶段的前提
nailong segment
nailong transcribe
nailong embed-redimnet
nailong embed-episode
nailong score-long-ref
nailong score-two-pass
nailong verify                    # 人工核对组A 确实是奶龙
nailong finalize
nailong-tts pack
```

只想核对切分而不碰音频（不读 `data/sep_out`）：

```bash
nailong finalize --manifest-only
```

## 旧脚本对照表

重构前根目录有 16 个扁平的 `*.py`。收编时内容改动较大，git 的相似度检测
没能把它们识别成重命名，因此记录为「删除 + 新增」。下表用于对照旧路径与当前
模块；旧脚本本身已不在本仓库历史中（提交已压缩为一次），其功能全部由右列
的模块承接。

| 旧脚本                    | 现在的位置                 |
| ------------------------- | -------------------------- |
| `export_utts.py`          | `stages/segment.py`        |
| `scan_all_sv.py`          | `stages/transcribe.py`     |
| `extract_emb_redimnet.py` | `stages/embed_redimnet.py` |
| `extract_emb.py`          | `stages/embed_funasr.py`   |
| `episode_emb.py`          | `stages/embed_episode.py`  |
| `chunk_emb.py`            | `stages/embed_chunk.py`    |
| `cluster.py`              | `stages/cluster.py`        |
| `long_ref.py`             | `stages/score_long_ref.py` |
| `two_pass.py`             | `stages/score_two_pass.py` |
| `verify_A.py`             | `stages/verify.py`         |
| `final_a.py`              | `stages/finalize.py`       |
| `annotate.py`             | `tools/annotate.py`        |
| `dump_group.py`           | `tools/inspect_group.py`   |
| `sanity.py`               | `tools/sanity.py`          |
| `make_reel.py`            | `tools/reel.py`            |
| `band.py`                 | `tools/band.py`            |

`separate` 是新补的阶段，旧脚本里没有对应物。

## 注意

- 阶段模块是脚本式的，**模块级就执行**，`import` 会真的跑起来。
  这是刻意的取舍：这些逻辑来自一次性研究脚本，塞进函数反而容易改错数值。
  所以别在测试里 import 它们，用 CLI 调用。
- `separate` 之前是手敲命令，没留脚本，导致 `data/sep_out/` 被删后
  8 个下游脚本一起静默失效。现在 `config.vocals_path()` 会在缺文件时
  直接报错并提示跑哪个命令。
