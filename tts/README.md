# tts —— 奶龙音色 TTS

本期**不选引擎**，只做两件事：把数据集打包到可用状态，把推理接口钉死。
这样等选型确定后，游戏侧不用改。

## 为什么先不选

候选路线各有明确代价，而当前语料规模不适合先押注：

| 路线 | 代价 |
| --- | --- |
| GPT-SoVITS | 中文小样本微调成熟，但需要额外的 WebUI/训练工程，且社区权重来源分散 |
| CosyVoice2 | 零样本音质好，但更依赖较长参考音频（理想 3-10s×多条，越多越稳） |
| fish-speech / OpenAudio | 中文表现好，文档与生态较薄，踩坑成本高 |
| 自训 VITS | 完全可控，但 75s 语料自训基本不可行 |

共同前提是**先把数据整理成标准格式**——这四种路线吃的都是
`wav|说话人|语言|文本` 这一类 filelist，所以打包先行、引擎后接。

## 现有语料

`dataset/final/` 共 11 段、**75.3s**，单段 4.28-8.00s，全部是连续干净片段：

- 来源：B 站奶龙短剧 9 集，demucs 分离人声轨后 VAD 切句，
  再用 ReDimNet 长参考打分 + 两遍迭代选出「组A = 奶龙」。
- 质量门槛：段内每句的 `simA ≥ 0.62`（低于此值句子里已确认混入其他角色）。
- 导出时加 `highpass=f=80` 压掉分离残留的低频伴奏；实测伴奏比人声低 8.7dB。

75s 处于音色克隆的经验下限附近：够做零样本/少样本微调，
但**不够自训**，且情感覆盖很窄（素材里主要是😡/😊/😮三种）。

## 打包

```bash
nailong finalize              # 若还没有 final_manifest.csv；完整跑需先 nailong separate
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

`engine.NullEngine` 是占位实现：按字数估时长产出静音，
用来先把游戏侧的播放与字幕时序跑通。
