# nailong-ref

从 B 站奶龙短剧里抽取**干净的奶龙人声**，再把它变成能玩的东西：

- `dataset/` —— 交付语料：11 段连续干净奶龙片段，共 75.3s
- `tts/` —— 奶龙音色语音库：**离线预合成**成 wav，运行期只查表播放
- `game/` —— 配套小游戏；`nailong-tts bundle` 把语音内联成单文件 HTML
- `native/` —— 自研 C++ 推理引擎的调研与实测（**已停手，不阻塞交付**）

硬约束：**不接 LLM、不接 ASR**。TTS 只做单向输出，台词来自封闭词表、
全部离线预合成——这也正是 `native/` 停手的理由：运行期不需要实时合成。

## 仓库布局

| 目录                  | 内容                                                                                                                                                          |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/nailong/`        | 抽取管线：可导入包 + `nailong` CLI                                                                                                                            |
| `dataset/`            | `manifests/` 清单、`final/` 交付片段、`embeddings/` 声纹嵌入 **入库**；`sources/`、`utterances/`、`queries/`、`reels/` 体积大且可重建，**不入库**（本地保留） |
| `tts/`                | 封闭词表 + 离线合成 + 打包/内联（独立包 `nailong-tts`）；`assets/voice/` 的预合成 wav **入库**                                                                 |
| `native/`             | 自研 C++/TensorRT 推理的调研、ONNX 导出与实测脚本；`third_party/`、`gsv/`、`build/` **不入库**                                                                |
| `game/prototype/`     | 两个 canvas 原型（冻结）+ 已合并片头的 `archive.html`                                                                                                        |
| `tools/vid-analysis/` | 另一支视频结构分析（与主链路无关；截图与音轨不入库）                                                                                                          |
| `docs/`               | 管线说明、实验结论与当前待解问题                                                                                                                              |
| `data/`               | 运行时可再生的中间产物，**不入库**                                                                                                                            |

## 快速开始

```bash
uv venv
uv sync --extra all          # 或按阶段只装需要的 extra
```

只跑数据侧的检查不需要重依赖：

```bash
uv pip install numpy soundfile
uv pip install -e . --no-deps
nailong help
```

`nailong-tts` 的子命令里只有 `synth` 需要外部 GPT-SoVITS 环境（onnxruntime +
`GPT_SoVITS/` 文本前端），`corpus` / `bank` / `bundle` 在项目 venv 里就能跑。
细节见 [tts/README.md](tts/README.md)。

## 管线

链路是**单向**的，每一步只读上一步的产物：

```mermaid
flowchart TD
    A[dataset/sources/src_NN.mp4] -->|nailong separate| B[data/sep_out/htdemucs/src_NN/vocals.wav]
    B -->|nailong segment| C[dataset/utterances/utt_NNN.wav + utt_manifest.csv]
    C -->|nailong transcribe| D[sv_all.csv<br/>事件/情绪/文本]
    C -->|nailong embed-redimnet| E[emb_redimnet.npy]
    B -->|nailong embed-episode| F[emb_episodes.npy]
    E --> G[nailong score-long-ref]
    F --> G
    G --> H[utt_margin_long.csv]
    H --> I[nailong score-two-pass]
    I --> J[utt_two_pass.csv<br/>组A=奶龙]
    D --> K[nailong verify]
    J --> K
    K -->|nailong finalize| L[dataset/final/*.wav<br/>final_manifest.csv]
    L --> M[nailong-tts pack]
    L -.参考音频.- N
    O[corpus.py 封闭词表 34 条] -->|nailong-tts synth| N[tts/assets/voice/*.wav<br/>CPU 离线预合成，入库]
    N -->|nailong-tts bundle| P[game/dist/nailong.html<br/>单文件，双击即玩]
```

```bash
nailong separate            # mp4 -> demucs 人声轨
nailong segment             # 切句
nailong transcribe          # SenseVoice 事件/情绪/文本
nailong embed-redimnet      # 声纹嵌入
nailong score-long-ref      # 长参考初筛
nailong score-two-pass      # 两遍迭代细化
nailong verify              # 三条独立证据验证组A
nailong finalize            # 出交付片段
nailong-tts pack            # 打包成训练格式（filelist.txt / metadata.csv）
nailong-tts corpus          # 词表 -> tts/build/voice_lines.csv
nailong-tts synth           # 预合成语音（需外部 GPT-SoVITS 环境）
nailong-tts bank            # 校验语音资源齐全
nailong-tts bundle          # 内联语音 -> 单文件游戏
```

`nailong help` 列出全部阶段（含 `annotate` / `reel` / `band` / `sanity` 等工具）。

## 数据契约

清单是唯一的跨阶段接口，都在 `dataset/manifests/`，主键为 `idx`：

| 文件                  | 字段                                                     |
| --------------------- | -------------------------------------------------------- |
| `utt_manifest.csv`    | idx, src, t0, t1, dur, f0_med, centroid, cluster, reel_t |
| `sv_all.csv`          | idx, src, t0, dur, event, emotion, text                  |
| `utt_margin_long.csv` | idx, src, t0, dur, sim_pos, sim_neg, margin              |
| `utt_two_pass.csv`    | idx, src, t0, dur, simA, simB, margin, group             |
| `final_manifest.csv`  | file, src, t0, t1, dur, min_simA, text                   |
| `labels.csv`          | idx, label, source（human 优先于 weak）                  |

阈值与路径全部集中在 [config.py](src/nailong/config.py)，不再散落在各脚本里。

## 已知缺口

- `data/sep_out/` 与 `data/sep_in/` 曾被删除且无脚本可重建，现已补上 `nailong separate`。
  **重跑管线必须先执行它**（需 `--extra sep`）。
- 现有的 `dataset/final/` 11 段产生于 `final_manifest.csv` 加入之前，
  因此 `nailong-tts pack` 目前会报缺清单。跑 `nailong finalize --manifest-only`
  可只补清单（不读 `data/sep_out`），或完整重跑。
- `pretrained_models/` 里 5 个文件全是 0 字节（speechbrain 下载失败残骸），已忽略。
- `tts/assets/voice/` 只有 **14 条**（12 材质 + `form0`/`form1`），词表共 34 条；
  `voice_manifest.csv` 还没生成过，`nailong-tts bank` 会因此报缺清单 →
  补齐跑一次 `nailong-tts synth` 即可。
- `native/` 重建需要两样外部输入：TensorRT 11.3 Windows x64 zip（官方许可页，
  要接受许可）与 `native/scripts/fetch_gsv_models.py` 拉的 ~1.28GB 权重
  （HF 直连会断，得走 `HF_ENDPOINT=https://hf-mirror.com`）。
- **仓库只收代码、清单、交付片段、嵌入与语音库**（Git LFS）。以下内容存在本地但不入库，
  克隆后需要重建或另行取得：

  | 内容                                      | 如何拿回                                     |
  | ----------------------------------------- | -------------------------------------------- |
  | `dataset/sources/*.mp4`                   | 需从 B 站另行下载（唯一无法自动重建的一项）  |
  | `dataset/utterances/`                     | `nailong segment`（需先 `nailong separate`） |
  | `dataset/queries/`                        | `nailong annotate query`                     |
  | `dataset/reels/`                          | `nailong reel`                               |
  | `tools/vid-analysis/*.png`、`audio_*.wav` | 重跑该目录下的 `analyze.py`                  |
  | `native/third_party/`、`native/gsv/`      | 自行下载 SDK / 重新导出 ONNX 并建引擎        |
  | `tts/build/`、`game/dist/`                | `nailong-tts pack` / `nailong-tts bundle`    |

  **含这些文件的完整原始历史保留在本地分支 `backup/pre-slim`（未推送）**，
  需要时 `git switch backup/pre-slim` 取用。

## 结论备忘

声纹路线在这批素材上踩过的坑（换更强模型不等于能分开、短句嵌入不可靠、
低频占比不能当 BGM 判据）记在 [docs/FINDINGS.md](docs/FINDINGS.md)，
动手前先看，避免重复投入。

语音侧的决策（为什么放弃实时引擎、参考音频为什么只认 8.00s 那条、
哪些坑在 `synth.py` 里绕开了）记在 [tts/README.md](tts/README.md) 与
[docs/FINDINGS.md](docs/FINDINGS.md) §9/§10。

若要继续扩数据集或换 TTS 模型，先看 [docs/ISSUES.md](docs/ISSUES.md)——
源素材仅 4.53 分钟、产出率已达 27.7%、ground truth 只有 2 条人工标注，
在动手前这些是硬约束。
