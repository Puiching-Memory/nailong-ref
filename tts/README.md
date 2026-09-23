# tts —— 奶龙音色 TTS

分**构建期**与**运行期**两侧：

| 侧 | 模块 | 做什么 | 依赖 |
| --- | --- | --- | --- |
| 构建期 | `corpus.py` | 封闭词表 + 穷举式模板 | 本包 |
| 构建期 | `synth.py` | 词表 → wav，离线批量合成 | 外部 GPT-SoVITS 环境 |
| 运行期 | `bank.py` | 按 `line_id` 查预合成 wav | 本包 |
| 运行期 | `engine.py` | 引擎契约 + `NullEngine` 占位 | 本包 |

**运行期零模型加载**：不接 LLM、不接 ASR、也不跑 TTS。台词在设计期就穷举完了，
合成在打包时跑一次，游戏只读 wav。

## 选型：GPT-SoVITS + 离线预合成

原本的计划是自研实时推理引擎（ONNX Runtime → TensorRT-RTX → 裸 TensorRT，
C++20，见 `native/`）。调研到"社区先例全部可用、自研只剩工程风险"之后判断**收益不匹配**：
本项目的用法是封闭词表 + 离线打包，运行期压根不需要实时合成。

所以改为**用原始 PyTorch / ONNX 在 CPU 上离线批量生成**：

- 离线对速度不敏感 —— 实测 RTF ≈ 13，出 4s 音频约 60s，可接受
- 不需要 GPU、TensorRT、ONNX 图优化、C++ 扩展
- `native/` 保留但**不阻塞交付**（TRT-RTX 在 `gpt_step` 上有真实缺陷，经典 TensorRT 可用，
  结论与证据见 `docs/FINDINGS.md`）

音色走**零样本克隆**：`pengyichen/NaiLong-Voice-Clone` 的 GPT/SoVITS 权重
（微调自奶龙）已在本地。新合成必须显式提供已听音复核的参考音频与逐字文本。

## 离线预合成

```bash
nailong-tts corpus                  # 看词表，落 tts/build/voice_lines.csv
nailong-tts synth                   # -> tts/assets/voice/*.wav + voice_manifest.csv
nailong-tts synth --only mat00,rhy1 # 只重合成改动的那几条
nailong-tts synth --force           # 全量重渲染
nailong-tts bank                    # 检查资源是否齐全
nailong-tts bundle                  # 语音库内联进游戏 -> game/dist/nailong.html（单文件，双击即玩）
```

`synth` 需要外部 GPT-SoVITS 环境（onnxruntime / torch + `GPT_SoVITS/` 文本前端），
路径用环境变量给：

| 变量 | 默认 |
| --- | --- |
| `NAILONG_GSV_REPO` | `C:\workspace\github\GPT-SoVITS_minimal_inference` |
| `NAILONG_GSV_ONNX` | `native/gsv/onnx_out` |
| `NAILONG_GSV_BERT` | `native/gsv/chinese-roberta-wwm-ext-large` |
| `NAILONG_GSV_REF` | 无；必须显式设置已听音核对的参考音频 |
| `NAILONG_GSV_REF_TEXT` | 无；必须显式设置对应的逐字文本 |
| `NAILONG_GSV_SEED` | 不设；设了固定随机种子，韵律可复现 |

当前正式复核清单没有获批准片段，所以仓库不指定默认参考。
`src_07` 仅由音频模型接受，且视觉证据不完整，不能自动当作已核对参考。

## 封闭词表

`corpus.py` 是台词的**唯一来源**，名词直接取自 `game/prototype/archive.html`：

| 组 | 条数 | 内容 |
| --- | --- | --- |
| `material` | 12 | 线描 蚀刻 等高 点彩 星座 大理石 草木 电路 陶土 乐谱 木纹 数字 |
| `form` | 3 | 奶娃形 木形 云形 |
| `rhythm` | 3 | 静 息 游 |
| `stage` | 4 | 点 芽 幼体 生灵 |
| `ui` | 12 | 招呼 / 成功 / 正典 / 光不足 / 解锁 / 收齐 … |

`line_id` 用下标（`mat00..mat11` / `form0..form2` / `rhy0..rhy2` / `stage0..stage3`），
和 HTML 里 `MATCOST` / `CANON` 的下标一一对应，改名不会破坏对应关系。

**故意只做 34 条**：12×3×3=108 种组合逐条合成既贵又没意义。
粒度小的意义就在这儿——`mat00` + `form1` + `rhy2` 三条片段拼起来播放，
运行期就覆盖 108 种组合，而合成量只有 34 条。这是
`game/prototype/README.md` 里「词汇粒度要小，交互分支要少而深」的具体落实。

## 运行期消费方

| 消费方 | 取用方式 |
| --- | --- |
| `game/dist/nailong.html`（**给玩家**，`nailong-tts bundle` 产出） | wav 以 base64 内联成 `window.VOICE_BANK`，播放走 `data:` URL。单文件、无服务器、无联网，双击即玩 |
| `game/prototype/archive.html`（开发态） | 同一层播放代码，但没有 `VOICE_BANK` 时回退成相对路径 `../../tts/assets/voice/{line_id}.wav`，于是必须从仓库根起服务 |
| `nailong_tts/bank.py`（Python 运行期） | 读 `tts/assets/voice_manifest.csv` 建索引，`say()` 返回 float32 波形 |

**HTML 侧只认文件名约定，不读 manifest**（浏览器里读 csv 要被 CORS 管，也不利于
`file://` 直接打开）。所以新增台词只要按 `{line_id}.wav` 落盘就自动生效，
但反过来：**改了文件名约定两边都会断**。

打包版与开发态共用同一份播放代码，分支只在一处：`hasVoice()` / `voiceSrc()`。
发布版里缺条目是**构建期**就该知道的事（`bundle` 会打印静音清单），
所以运行期不再往控制台喊话、也不再发无意义的失败请求。

`tts/assets/voice/` 现只有 **14 条**（12 材质 + `form0`/`form1`），
`voice_manifest.csv` 还没生成过（`nailong-tts bank` 会因此报缺清单）。
补齐跑一次 `nailong-tts synth` 即可。`tts/assets/` 已入库（LFS，体积小，
不含机器相关路径，没装 ML 栈的机器也能直接跑）。

## 现有语料

`dataset/production/accepted/` 共 21 段、**97.16s**，仅是自动候选：

- 说话人：2 段由画面确认；19 段由视觉校准后的 ERes2NetV2 保守阈值接受。
- 音质：同窗口 `RMS(vocals)-RMS(no_vocals) ≥ 15dB`，并加 80Hz 高通。
- 另有 31 段 / 159.50s 因残留 BGM/SFX 进入 `quarantine/`，打包器不会读取。

视觉复核已拒绝 6 段 / 32.52s；另 15 段 / 64.64s 待听音，当前没有获批准训练片段。

## 打包

正式训练包先运行 `nailong-tts review`，逐段听音、修正文本、确认目标说话人和残留音效，
再在 `dataset/manifests/training_review.csv` 填 `approved=yes`。`nailong-tts audit`
对音频哈希、24kHz/单声道/PCM16、时长、静音、削波、复核状态及语料量作硬门禁。
不足 200 段或 30 分钟时 `pack` 会失败并清除旧的训练列表；要做小样本调试可用
`pack --experimental`，但逐段复核依旧必需。

```bash
nailong calibrate-visual
nailong prepare-production
nailong-tts pack              # -> tts/build/{filelist.txt, metadata.csv}
nailong-tts stats             # 只跑质检不写文件
```

`filelist.txt` 的文本字段已清洗：去掉 SenseVoice 附带的表情符号，
以及段内拼接用的 ` / ` 边界标记——两者都不是要读出来的字。
`metadata.csv` 同时保留原始 `text` 与清洗后的 `text_speakable`，便于回溯。

打包时会告警这些情况：清单与磁盘不一致、时长漂移（音频被改过）、
缺台词、总长低于克隆下限。**有告警时不要直接拿去训练。**

## 接引擎

```python
from nailong_tts import engine

@engine.register
class MyEngine:
    name = "my-engine"
    sample_rate = 32000
    def prepare(self): ...                      # 载权重
    def say(self, line: engine.Line): ...       # -> float32 波形
```

`engine.Line` 的 `line_id` 是**封闭词表里的键**，`text` 不允许运行期拼接——
硬约束是不接 LLM、不接 ASR，所以所有台词必须离线预合成、
以 wav 落盘打包进游戏，运行期零模型加载。

`engine.NullEngine` 是占位实现：按字数估时长产出静音，用来先把游戏侧的播放与字幕时序跑通。

真实现是 `bank.BankEngine`（引擎名 `bank`，已默认注册）：按 `line_id` 查
`tts/assets/voice/` 下的 wav，解码结果缓存在内存。找不到资源时**退化为静音而不是抛异常**——
资源没打全时游戏还能跑，只是没声音，不该因此崩掉。
