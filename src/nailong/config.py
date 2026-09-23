"""全局配置：路径推导与阈值集中处。

此前路径散落在 16 个脚本里各自硬编码，结果是 `sep_out/` 被删掉后没有
任何地方报错，整条链路静默失效。现在所有相对路径都从仓库根推导，
缺失的输入用 `vocals_path()` / `mix_path()` 立刻给出可操作的提示。
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("NAILONG_ROOT") or Path(__file__).resolve().parents[2])

# ---- 运行时可再生的中间产物（.gitignore，不入库）----
DATA = ROOT / "data"
SEP_IN = DATA / "sep_in"           # demucs 的输入：原始音轨转写
SEP_OUT = DATA / "sep_out"         # demucs 分离结果
PRETRAINED = DATA / "pretrained"   # 声纹 / ASR 权重缓存

# ---- 入库数据集 ----
DATASET = ROOT / "dataset"
SOURCES = DATASET / "sources"        # src_NN.*（mp4/m4a 等 ffmpeg 可读容器）
MANIFESTS = DATASET / "manifests"    # 跨阶段数据契约
UTTERANCES = DATASET / "utterances"  # 细粒度切句
QUERIES = DATASET / "queries"        # 主动学习提问批
PRODUCTION = DATASET / "production"  # 视觉真值 + 残留伴奏门控后的生产片段
CALIBRATION = DATASET / "calibration"  # 部署时直接加载的声纹原型与阈值
EMBEDDINGS = DATASET / "embeddings"
REELS = DATASET / "reels"

# ---- 清单（唯一的跨阶段契约）----
UTT_MANIFEST = MANIFESTS / "utt_manifest.csv"   # idx,src,t0,t1,dur
SV_ALL = MANIFESTS / "sv_all.csv"               # idx,src,t0,dur,event,emotion,text
LABELS = MANIFESTS / "labels.csv"               # idx,label,source
Q_BATCH = MANIFESTS / "q_batch.csv"             # order,idx,src,dur,p_nailong,text
VISUAL_LABELS = MANIFESTS / "visual_labels.csv"
PRODUCTION_SCORES = MANIFESTS / "production_scores.csv"
PRODUCTION_MANIFEST = MANIFESTS / "production_manifest.csv"

EMB_REDIMNET = EMBEDDINGS / "emb_redimnet.npy"
EMB_ERES2NETV2 = EMBEDDINGS / "emb_eres2netv2.npy"

# ---- 音频格式 ----
SR_MODEL = 16_000    # 声纹模型输入采样率
SR_EXPORT = 24_000   # 导出 / 试听采样率
SEG_PAD = 0.06       # 切句前后各留，保住字头字尾
HIGHPASS_HZ = 80     # 导出高通，压掉分离残留的低频伴奏

# ---- VAD 切句 ----
FRAME, HOP = 640, 320            # 40ms / 20ms @16k
GAP_TOL = 0.10                   # 段内允许的静音间隔
MIN_UTT = 0.30                   # 最短句长
MIN_SEG = 0.80                   # 提交给声纹的最短片段
DB_THRESH, SP_THRESH = -45.0, 0.25

# ---- 生产片段 ----
ADJ_GAP = 0.80                   # 相邻句合并成片段的最大间隔
MIN_RUN, MAX_RUN = 3.0, 10.0     # 交付片段时长窗口
SEG_SEP = " / "                  # 段内语句边界，不是要读出来的字

# ---- 主动学习 ----
HUMAN_SEED = {15: 1, 2: 0}       # 用户亲口判定：15=奶龙, 2=另一说话人
MIN_ASK_DUR = 1.5                # 短于此的句子人耳也分不出，不问


def rel(path) -> str:
    """转成相对仓库根的 posix 路径，用于日志。"""
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def mix_path(src: str) -> Path:
    """demucs 的输入音轨。"""
    p = SEP_IN / f"{src}.wav"
    if not p.exists():
        raise FileNotFoundError(f"缺少 {rel(p)}；先跑分离阶段：nailong separate")
    return p


def vocals_path(src: str) -> Path:
    """demucs 分离出的人声轨。"""
    p = SEP_OUT / "htdemucs" / src / "vocals.wav"
    if not p.exists():
        raise FileNotFoundError(f"缺少 {rel(p)}；先跑分离阶段：nailong separate")
    return p


def no_vocals_path(src: str) -> Path:
    """Demucs 的非人声轨；用于量化残留 BGM/SFX，不参与说话人识别。"""
    p = SEP_OUT / "htdemucs" / src / "no_vocals.wav"
    if not p.exists():
        raise FileNotFoundError(f"缺少 {rel(p)}；先跑分离阶段：nailong separate")
    return p


def source_path(src: str) -> Path:
    matches = sorted(SOURCES.glob(f"{src}.*"))
    if len(matches) != 1:
        raise FileNotFoundError(f"源文件应恰好有一个 {rel(SOURCES / src)}.*，实际 {len(matches)} 个")
    return matches[0]


def source_names() -> list[str]:
    """按编号返回当前本地源；容器格式可以是 mp4、m4a 或其他 ffmpeg 输入。"""
    return sorted({path.stem for path in SOURCES.glob("src_*.*")})


def ensure_dirs() -> None:
    """创建会写入的目录。各阶段入口调用一次即可，不必各自 makedirs。"""
    for d in (DATA, SEP_IN, SEP_OUT, PRETRAINED, MANIFESTS, UTTERANCES,
              QUERIES, PRODUCTION, CALIBRATION, EMBEDDINGS, REELS):
        d.mkdir(parents=True, exist_ok=True)
