# 实验结论（含大量否定结果）

动手换模型或改参数前先看这里。下面每条都是实测出来的，
重复投入同一方向已经浪费过时间。

## 1. 换更强的声纹模型，分不开还是分不开

在卡通/童声配音素材上实测（80 句 / 9 集，VAD 切句，约 2s/句）：

| 模型                              | 锚点内部一致性 | 最优 k | silhouette |
| --------------------------------- | -------------- | ------ | ---------- |
| cam++                             | 0.556          | 2      | 0.429      |
| ERes2NetV2                        | 0.615          | 2      | ~0.35      |
| ReDimNet(M, ft_mix, vb2+vox2+cnc) | 0.659          | 2      | 0.339      |

- 三者的最优 k 都是 2，silhouette 全在 0.34-0.43，**没有可用簇结构**。
- 唯一被分出来的簇不是说话人，而是「非语音噪音」或「某源录音异常」。
- **一致性提升 ≠ 区分度提升**：ReDimNet 锚点一致性最高（0.659 vs 0.556），
  silhouette 反而最低（0.339 vs 0.429）。
- 原始混音 vs demucs 人声轨提取嵌入，结果几乎一致 → **分离不是元凶**。
- 结论：无监督聚类在这类素材上不可用，别在「再换个模型」上继续投入。

模型档位（Vox1-O EER，越低越强）供参考：
ReDimNet2-B6 0.29% > ERes2Net-large 0.52% ≈ ReDimNet-B6 0.53% >
ERes2NetV2 0.61% > CAM++ 0.65% > ECAPA-TDNN 0.86%。

工具链限制：FunASR 只注册了 `CAMPPlus` 与 `ERes2NetV2`（ERes2Net-large 未注册）。
要上更强模型必须换工具链，ReDimNet 走
`torch.hub.load("IDRnD/ReDimNet", "ReDimNet", model_name="M", train_type="ft_mix", dataset="vb2+vox2+cnc")`，
输入 16k 单声道 `[N,T]`，输出 `[N,192]`，权重约 19MB。

## 2. 短句嵌入不可靠，长参考可靠 —— 这是本项目的突破口

| 对象                             | 两两余弦                                |
| -------------------------------- | --------------------------------------- |
| 2s 短句之间                      | 中位 **0.445**（无簇结构）              |
| 30s 整集嵌入，同集前后半段       | **0.81-0.92**                           |
| 人工锚点内部（4 个「奶龙锚点」） | 0.42-0.70（跨说话人，锚点本身就失效了） |

做法：**用长参考（整集均值）给短句打分**，而不是「2s vs 2s」。
留一法排除本集，避免自身泄漏。

**非循环验证法**：先用整集嵌入自动找出离群集（本例 `src_02`），
再看短句打分是否把它排到底部。实测该集 3 句排到 80 句的第 77/78/79 名 → 方法可信。

两遍迭代（长参考初筛 → 取高置信句自建参考 → 重打分）收敛快且稳定，3 遍内组A 50→53→54 句。

## 3. 陷阱：短句/语气词的嵌入会漂到「另一个团」

容易被误当成第二说话人。检查办法：看该组的高置信成员里有没有**完整句子**；
若全是「啊/嗯/哎呀」这类语气词，则该组不可信。
（`nailong inspect-group` 就是为此写的，它会按汉字数标注「语气词?」。）

## 4. 「停顿电平够静 ⇒ 无 BGM」是错的，且失效方向最坏

导出时的 `highpass=f=80` 会把 0-120Hz 压到 1-2%，低频占比看起来「很干净」。
所以**低频占比不能直接当 BGM 判据**。

要判断伴奏必须做**差分对比**：同窗口分别测 `original` / `vocals` / `no_vocals` 三个 stem。
正确判据是 `rms(vocals) - rms(no_vocals)`（伴奏比人声低多少 dB）：

| 差值     | 结论                         |
| -------- | ---------------------------- |
| < 15 dB  | 有明确 BGM/SFX，必须人声分离 |
| 15-25 dB | 轻度伴奏                     |
| > 25 dB  | 几乎没有可剔除内容           |

已验证反例：某片段伴奏仅低于人声 **8.7 dB**，却因停顿静被判定「可用原始」而绕过分离。
**越吵越容易通过判据**——这个方向的错误最危险。

`--two-stems=vocals` 近似能量守恒（`orig ≈ vocals ⊕ no_vocals`），可做一致性校验。

## 5. 人工锚点会骗人

早期 4 个「奶龙锚点」两两余弦仅 0.42-0.70（同一人通常 > 0.70），
说明锚点集本身就跨了多个说话人。以锚点做监督的假设从根上不成立。

## 6. 标注数据优先级

`labels.csv` 里 `human` 标签最高权威，永远覆盖 `weak`。
主动学习流程（`nailong annotate query`）**故意不显示模型预测**，避免锚定偏误，
保证人的判定是独立真值。只有 `dur ≥ 1.5s` 的句子才值得问人——更短的人耳也分不出。

## 7. 音频管线的验证方式

改了音频管线后，用 **SHA256 逐位对比**验证重构等价性：
预期不变的文件哈希必须完全相同。

## 8. 本次重构的等价性验证

`nailong finalize --manifest-only` 重跑切分逻辑，产出的 11 个文件名
与重构前 `dataset/final/` 里的**逐一吻合**（`src_NN_t0-t1.wav` 全部一致），
说明把散落脚本收编进包、把路径集中到 `config.py` 的过程中没有改动任何数值。

## 9. 实时 TTS 自研引擎：可以做，但不该做

本项目的用法是**封闭词表 + 离线打包**，运行期不需要实时合成。自研引擎
（ONNX Runtime → TensorRT-RTX → 裸 TensorRT，C++20）走到"社区先例全部可用、
只剩工程风险"之后停手，改用原始 PyTorch/ONNX 在 CPU 上离线批量生成。

**TensorRT-RTX 在 `gpt_step` 上有真实缺陷（实测）**

| 后端 | `topk_values` | `k_cache_new` | `v_cache_new` | `topk_indices` |
| --- | --- | --- | --- | --- |
| CPU | baseline | | | |
| **CUDA EP** | **rel 9.39e-07** | **4.51e-07** | **5.21e-07** | **逐位相同** |
| TRT-RTX | 溢出失败 | | | |

- TRT-RTX 报 `updateDeviceMemorySizeForShapes: shape calculation overflow (-36028788437417983)`,
  无 profile 时报 `The engine requires 4194187776 device memory`（基线 108 MB）
- **该溢出的常数与 dtype 无关**：FP16 版本复现出完全相同的 `-36028788437417983`
- 量级论证：TRT-RTX 的 `k_cache_new` rel = **8.2e-01**，而纯 FP16 舍入只有 **4.3e-04**,
  相差约 1900 倍 → 是真缺陷，不是精度
- 结论：**ONNX 导出是对的，缺陷在 TRT-RTX**。社区先例（`onnx2trt.py` /
  `run_trt_inference.py`）把全部 8 个模型都跑在经典 TensorRT 上、不排除 `gpt_step`,
  也印证了这点

**FP16 转换本身是可靠的**（排除掉整数索引输出的相对误差判据之后）

- `gpt_step` / `gpt_step_static`：`topk_values` 7.917e-04、`k_cache_new` 4.282e-04、
  `v_cache_new` 3.427e-04 → 忠实
- `gpt_encoder`：最差 rel 8.811e-04，`topk_indices` 逐位相同 → 忠实（早期把它判成
  "索引不稳"是错的，`topk_values` rel 只有 5.86e-04，索引差异纯粹是 top-50 边界并列）
- 但 FP16 **救不了 TRT-RTX**：静态 FP16 版本能跑却仍发散（rel 6.73e-01~8.75e-01）,
  且速度是 0.65×（比 FP32 的 0.33× 好，但仍慢于基线）

**两次自我修正，记下来避免重犯**

- 曾把 TRT-RTX 溢出归因于 KV cache 过大、以为减小 `max_len` 能修好。
  读 `onnx2trt.py` 后否掉：`gpt_step` 的 shape profile 在 small/fitted/medium/large
  四个档位下**完全相同**，`patch_gpt_step_shapes()` 是把静态 dim2 **同步**到 `max_len`,
  不是缩小它
- 整数索引输出（`topk_indices` 这类）**必须排除在相对误差判据之外**，否则会把
  正常的边界并列误判成"转换损坏"（本次因此误判了两回）

**环境坑（都已解）**

- 官方 `cmake/TensorRT-EnterpriseConfig.cmake` 是坏的：
  `find_package` 因引用不存在的 `bin/tensorrt_player.exe` 直接 FATAL_ERROR。
  改为自己写 `native/cmake/FindTensorRT.cmake`
- TensorRT 11 的 `BuilderFlag` **没有** `kFP16`/`kINT8`，`platformHasFastFp16()`
  已被删除，`NetworkDefinitionCreationFlag::kEXPLICIT_BATCH` 也没了；
  **TRT 11 网络永远是强类型**，精度由 ONNX 张量类型决定，没有 builder flag
- `getInferLib{Major,Minor,Patch,Build}Version()` 是 `extern "C"` 全局函数,
  不在 `nvinfer1` 命名空间里
- `pypi.org` 上**没有任何可用的 Windows TensorRT wheel**（唯一命中的
  `tensorrt-10.0.0b6` 是个 0.0 MB 的空包）；`pypi.nvidia.com` 有，但 pip wheel
  **只有运行期 DLL，不含头文件也不含 `trtexec`**。要 C++ 必须用官方 zip

## 10. 离线预合成的实测结论

**链路本身**：`GPT-SoVITS_minimal_inference` 的 ONNX 路径 + 我们自己导出的
`native/gsv/onnx_out/`，在 **CPU** 上跑通。19 字 → 4.00s 音频，峰值 0.9000、
RMS 0.2058、无削波，RTF ≈ 13（出 4s 音频约 60s）。离线批量完全可接受。

**参考音频的长度是关键变量（实测 5 条参考）**

同一句 probe（18 字）换参考重新合成，用 `sv_embedding` 打分。判据是**相对**的：
先把 11 段参考互相之间的余弦分布测出来当"同一说话人自相似区间"
（n=55，p5=0.6843 / 中位 0.7991 / p95=0.8482），再把合成片段放进去比。

| 参考 | 时长 | 平均余弦 | 同长度对照 p5 | 判定 |
| --- | --- | --- | --- | --- |
| `src_06` | 8.00s | **0.7562** | 0.6569 | OK |
| `src_01` | 6.56s | 0.7024 | 0.6363 | OK |
| `src_05`（原文） | 4.28s | 0.6696 | 0.7129 | 低于对照 |
| `src_05`（改错字） | 4.28s | 0.6714 | 0.7172 | 低于对照 |
| `src_03` | 7.22s | 0.6593 | 0.6766 | 低于对照 |

- **参考越长越好**。8.00s 的 `src_06` 是唯一一条"到 11 段参考的**最小**余弦
  0.6908"都高于自相似下沿 0.6843 的，即所有参考都认可它像奶龙本身
- **必须做同长度对照**。声纹嵌入稳定性随时长上升，短片段天然低分；
  不排除这个因素就会把"片段太短"误判成"音色不对"。
  对照做法：把每段参考截成与该目标等长的窗口，再与其余参考比
- 修参考文本里的 ASR 错字（"没血染"→"没血缘"）**几乎没影响**
  （0.6714 vs 0.6696，在随机采样噪声内）
- 单次判定不能只看均值：`src_05` 那两条的低分伴随"与自身 prompt 的相似度 0.8256"
  （argmax 正确命中），说明音色是对的、只是整体偏离心

**踩过的坑**

- **32kHz 音频喂给 16kHz 的 `sv_embedding`**：会让嵌入整体偏低
  （实测把 0.67 压到 0.53），排序因此完全不可信。必须重采样到 16k 再嵌入
- 上游 `sample_topk` **没有固定随机种子**，同一句每次韵律都不同
  （实测同文本 token 数在 82~121 间波动）。要复现就设 `NAILONG_GSV_SEED`
- `g2pw`（多音字消歧）把模型路径写成**相对路径** `GPT_SoVITS/text/G2PWModel`,
  必须把 cwd 切到参考仓库根，否则它连下载 zip 都打不开文件。
  模型从 **modelscope.cn** 拉（国内可达），只要目标目录已存在就跳过下载

**用了两个 shim / 依赖，都记一下**

- `LangSegmenter` 走 `fast_langdetect`，会去 `dl.fbaipublicfiles.com` 下 126MB 的
  `lid.176.bin`。台词来自**封闭词表**，语种在设计期就已知，不需要统计式检测 →
  用一个确定性的字符规则分段（连续 ASCII 字母算 en，其余算 zh）替掉，
  同时去掉了外网依赖。注意 `get_phones_and_bert` 里 zh 的每条分支都强制走
  `LangSegmenter`，只有 `"en"` 有直通分支
- `jieba_fast`（编译型）在本机用 MSVC 14.51 能正常编译安装；`opencc` 直接有 wheel
- `nltk` 数据（`cmudict` / `averaged_perceptron_tagger`）**下不下来**：
  环境里 `nltk_data` 的下载地址被解析到保留地址 198.18.0.58，被安全层按
  "SSRF attempt to restricted IP" 拦掉。这只影响**英文**前端，
  纯中文台词不受影响（实测 zh 合成功）
