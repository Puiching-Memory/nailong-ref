# game/prototype —— 冻结的 canvas 原型

单文件 canvas 原型，**本期冻结**：只做归档，不重构。
开场动画「奶娃 · BECOMING」已内嵌进 `archive.html`，作为进入世界前的片头；
台词走「构建期合成、运行期只读 wav」（见下方「与 TTS 的衔接约束」）。

| 文件 | 内容 | URL 参数 |
| --- | --- | --- |
| `archive.html` | 片头（BECOMING，38s / 120 BPM / 76 拍，逐镜头演出）＋「一形千面 · 奶娃档案馆」图鉴/合成玩法：形态 × 材质 × 韵律组合，材质有解锁光价，正典组合有额外标记；进度存 localStorage（键 `naiwa-archive-v1`）。片头放完点「进入世界」交接给正片；世界内左下「开 场 ↺」可重看（重看时游戏环境声自动让位） | `?fast` 跳过片头直入世界、`?reset` 清进度、`?shot=<秒>` 定格片头某一帧、`?size=<像素>` 改画布边长 |
| `becoming.html` | 片头的独立版（内容与内嵌版一致），保留作单独演示与截图用 | `?shot=<秒>`、`?size=<像素>` |

## 两种形态：给玩家 / 给自己

**给玩家** —— 一个文件，双击即玩（片头 + 正片 + 台词全在里面，不需要服务器、不需要联网）：

```bash
nailong-tts bundle        # -> game/dist/nailong.html（约 2.3 MB）
```

**给自己（开发）** —— 直接开 `archive.html`。它按相对路径
`../../tts/assets/voice/{line_id}.wav` 取语音，所以要从**仓库根目录**起服务；
只服务 `game/prototype/` 会够不到语音库：

```bash
npx serve .                              # 或 uv run python -m http.server 8123
# -> http://localhost:3000/game/prototype/archive.html
```

为什么玩家版非得把语音内联成 base64：浏览器的 `file://` 子资源策略毫无一致性——
Chrome/Edge 放行同盘媒体，fetch/XHR 一律拒绝，VS Code 内置浏览器连 `<img>` 都读不到。
只要 wav 还在 HTML 外面，玩家就得先架服务器，架不起来就是静音。
内联后播放路径退化成 `data:` URL，才真正做到"双击即玩"。代价见
`tts/nailong_tts/bundle.py` 的模块说明（体积 +1/3、全量常驻内存）。

## 美术基调（已定，别推翻）

文艺复兴高雅典雅一路：纸色 `#e9e1cd`、墨色 `#26241f`、Georgia/Times 衬线字、
纸纹与颗粒叠加、细规线与图鉴式小标签。用户明确不喜欢普通卡通风。

两个文件里的 `VARIANTS` 是同一套手绘变体，`blob()` / `sprout()` / `eyes()` /
`outline()` / `hatch()` 是共用绘制原语——真要合并成一个引擎时从这里抽公共层。

## 与 TTS 的衔接约束

- **不接 LLM、不接 ASR**：运行期不能有任何文本生成或语音识别。
  台词必须来自**封闭词表的穷举式模板**，全部离线预合成成 wav 打包进资源。
- 因此玩法设计要迁就台词预算：词汇粒度要小，交互分支要少而深。
- 离线预合成**已落地**：词表在 `tts/nailong_tts/corpus.py`（材质 12 / 形态 3 / 律动 3 /
  阶段 4 / 短句 12，共 34 条；名词直接取自本项目 `archive.html` 的
  `VARIANTS`/`FORMS`/`RHYS`/`STAGES`），跑 `nailong-tts synth` 出 wav 到
  `tts/assets/voice/`，运行期用 `bank` 引擎按 `line_id` 取，零模型加载。
- 名词**必须套框架读，不能裸读**：裸读 2 字（如"线描"）会让后端只吐出 1 个 token
  （直接 EOS）并在解码阶段崩掉，所以 `mat00` 实际说的是「材质，线描。」。
  拼播放时按整句处理，别把它当成可以插进任意句子的词。
- **已接**：`archive.html` 里那层「读 wav 的播放」已经落地（`VOICE_BASE` 指向
  `../../tts/assets/voice/`），只按 `{line_id}.wav` 取文件，不动演出逻辑与时间轴。
  触发点：

  | 操作 | 台词 |
  | --- | --- |
  | 蓝图世界点形态 | `form{i}` |
  | 蓝图世界点已解锁材质 / 成功解锁材质 | `mat{ii}` |
  | 成功解锁律动 | `rhy{i}` |
  | 注入世界 | `form{i}` 接 `mat{ii}`（片段拼接） |
  | 图鉴点已收录标本 | `mat{ii}` |

  缺哪条就静音跳过（`VOICE_MISS` 记下来不再重试，只 warn 一次），永不抛错。
- **当前只合成了 14 条**：12 个材质 + `form0`/`form1`（缺 `form2`、`rhy*`、`stage*`、
  `ui*`），所以上表里律动那条现在是静音的。补齐跑 `nailong-tts synth`。
  `tts/assets/` 目前**未入库**（`git status` 里是 `??`），要进版本库得显式 `git add`。
- `nailong-tts say` 仍是占位引擎（产静音），只用于验证播放与字幕时序链路。
