"""奶龙音色 TTS。

分两侧：

**构建期**（需要外部 GPT-SoVITS 环境，见 `synth.py` 模块头）
- `corpus.py` —— 封闭词表 + 穷举式模板，游戏台词的全部来源
- `synth.py`  —— 把词表离线批量合成成 wav（`tts/assets/voice/`）

**运行期**（只用本包依赖，不加载任何模型）
- `bank.py`   —— 按 line_id 查预合成 wav 的引擎，游戏唯一该用的实现
- `engine.py` —— 引擎契约与 `NullEngine` 占位

`dataset.py` 是把 `dataset/production/accepted/` 打包给训练器用的，与推理无关。
"""

from . import bank as _bank  # noqa: F401  导入即注册 "bank" 引擎

__version__ = "0.2.0"
