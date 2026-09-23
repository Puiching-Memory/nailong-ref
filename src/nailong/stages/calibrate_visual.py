"""用视觉真值生成部署可直接加载的双模型声纹校准包。

输入是 22 条音视频对齐后人工确认的标签，以及 ReDimNet/ERes2NetV2 的全量
嵌入。视觉只用于这次离线校准；生产打分只读音频嵌入和这里生成的原型/阈值。
"""

from __future__ import annotations

import hashlib
import json
import sys

import numpy as np

from .. import config, manifests


def _norm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def _unit_mean(values: np.ndarray) -> np.ndarray:
    value = values.mean(axis=0)
    return value / max(float(np.linalg.norm(value)), 1e-12)


def _margin(values: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    return values @ pos - values @ neg


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives, negatives = scores[labels == 1], scores[labels == 0]
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in positives for n in negatives]))


def _best_threshold(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    best = None
    for threshold in np.r_[np.inf, np.unique(scores)[::-1], -np.inf]:
        predicted = scores >= threshold
        tp = int(np.sum(predicted & (labels == 1)))
        fp = int(np.sum(predicted & (labels == 0)))
        fn = int(np.sum(~predicted & (labels == 1)))
        tn = int(np.sum(~predicted & (labels == 0)))
        recall = tp / max(1, tp + fn)
        fpr = fp / max(1, fp + tn)
        balanced = 0.5 * (recall + tn / max(1, tn + fp))
        candidate = (balanced, recall - fpr, float(threshold), recall, fpr)
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    return {"threshold": best[2], "recall": best[3], "fpr": best[4],
            "balanced_accuracy": best[0]}


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    config.ensure_dirs()
    utterances = manifests.read(config.UTT_MANIFEST)
    ids = [int(row["idx"]) for row in utterances]
    position = {idx: offset for offset, idx in enumerate(ids)}
    texts = manifests.texts()
    labels = manifests.read(config.VISUAL_LABELS)
    label_by_id = {int(row["idx"]): row for row in labels}
    if len(label_by_id) != len(labels):
        raise SystemExit("visual_labels.csv 有重复 idx")
    missing = sorted(set(label_by_id) - set(position))
    if missing:
        raise SystemExit(f"视觉标签引用了不存在的 idx: {missing}")

    matrices = {
        "redimnet": _norm(np.load(config.EMB_REDIMNET)),
        "eres2netv2": _norm(np.load(config.EMB_ERES2NETV2)),
    }
    for name, matrix in matrices.items():
        if matrix.shape != (len(ids), 192):
            raise SystemExit(f"{name} 嵌入形状应为 ({len(ids)}, 192)，实际 {matrix.shape}")

    label_ids = np.array([int(row["idx"]) for row in labels])
    y = np.array([int(row["label"]) for row in labels])
    target_ids = [int(row["idx"]) for row in labels if row["class"] == "target"]
    other_ids = [int(row["idx"]) for row in labels if row["class"] == "other_speaker"]
    if not target_ids or not other_ids:
        raise SystemExit("校准至少需要 target 和 other_speaker 各一类")

    profiles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    full_scores: dict[str, np.ndarray] = {}
    scales = {}
    for name, matrix in matrices.items():
        pos = _unit_mean(matrix[[position[idx] for idx in target_ids]])
        neg = _unit_mean(matrix[[position[idx] for idx in other_ids]])
        profiles[name] = (pos, neg)
        full_scores[name] = _margin(matrix, pos, neg)
        calibration_values = full_scores[name][[position[idx] for idx in label_ids]]
        scales[name] = {
            "mean": float(calibration_values.mean()),
            "std": max(float(calibration_values.std()), 1e-9),
        }
        np.save(config.CALIBRATION / f"{name}_target_centroid.npy", pos.astype(np.float32))
        np.save(config.CALIBRATION / f"{name}_other_centroid.npy", neg.astype(np.float32))

    ensemble = np.mean([
        (full_scores[name] - scales[name]["mean"]) / scales[name]["std"]
        for name in matrices
    ], axis=0)

    # 留一视频评估：同一视频内的相邻句不会同时出现在训练和测试侧。
    loeo = {name: {} for name in matrices}
    for source in sorted({row["src"] for row in labels}):
        test = [row for row in labels if row["src"] == source]
        train_target = [int(row["idx"]) for row in labels
                        if row["src"] != source and row["class"] == "target"]
        train_other = [int(row["idx"]) for row in labels
                       if row["src"] != source and row["class"] == "other_speaker"]
        for name, matrix in matrices.items():
            pos = _unit_mean(matrix[[position[idx] for idx in train_target]])
            neg = _unit_mean(matrix[[position[idx] for idx in train_other]])
            for row in test:
                idx = int(row["idx"])
                loeo[name][idx] = float(_margin(matrix[position[idx]:position[idx] + 1], pos, neg)[0])

    loeo_arrays = {name: np.array([loeo[name][idx] for idx in label_ids]) for name in matrices}
    loeo_ensemble = np.mean([
        (loeo_arrays[name] - loeo_arrays[name].mean()) /
        max(float(loeo_arrays[name].std()), 1e-9)
        for name in matrices
    ], axis=0)

    metrics = {
        name: {"auc_leave_one_video_out": _auc(y, values),
               **_best_threshold(y, values)}
        for name, values in loeo_arrays.items()
    }
    metrics["ensemble_diagnostic"] = {
        "auc_leave_one_video_out": _auc(y, loeo_ensemble),
        **_best_threshold(y, loeo_ensemble),
    }
    primary = max(matrices, key=lambda name: metrics[name]["auc_leave_one_video_out"])
    production_score = ((full_scores[primary] - scales[primary]["mean"])
                        / scales[primary]["std"])
    primary_loeo = loeo_arrays[primary]
    zero_observed_fpr_raw = float(np.nextafter(primary_loeo[y == 0].max(), np.inf))
    reject_raw = float(np.nextafter(primary_loeo[y == 1].min(), -np.inf))
    balanced_raw = _best_threshold(y, primary_loeo)["threshold"]
    # 数据扩容时宁可少收，不让相似卡通声线污染生产集。平衡阈值若更严格，
    # 就覆盖仅“高于已见负例”的边界；两者都来自按视频留一预测。
    accept_raw = max(zero_observed_fpr_raw, balanced_raw)
    thresholds = {
        "accept": (accept_raw - scales[primary]["mean"]) / scales[primary]["std"],
        "reject": (reject_raw - scales[primary]["mean"]) / scales[primary]["std"],
        "balanced": (balanced_raw - scales[primary]["mean"]) / scales[primary]["std"],
        "zero_observed_fpr": ((zero_observed_fpr_raw - scales[primary]["mean"])
                              / scales[primary]["std"]),
        "source": "max(zero-observed-FPR boundary, balanced threshold) on leave-one-video-out predictions",
    }

    score_rows = []
    for offset, row in enumerate(utterances):
        idx = int(row["idx"])
        truth = label_by_id.get(idx)
        if truth:
            decision = "confirmed_target" if truth["label"] == "1" else "confirmed_reject"
        elif production_score[offset] >= thresholds["accept"]:
            decision = "auto_accept"
        elif production_score[offset] < thresholds["reject"]:
            decision = "auto_reject"
        else:
            decision = "review"
        score_rows.append([
            idx, row["src"], row["t0"], row["t1"], row["dur"],
            truth["label"] if truth else "", truth["class"] if truth else "",
            f"{full_scores['redimnet'][offset]:.6f}",
            f"{full_scores['eres2netv2'][offset]:.6f}", f"{ensemble[offset]:.6f}",
            f"{production_score[offset]:.6f}",
            decision, texts[idx],
        ])
    manifests.write(config.PRODUCTION_SCORES,
                    ["idx", "src", "t0", "t1", "dur", "visual_label", "visual_class",
                     "redimnet_margin", "eres2netv2_margin", "ensemble_score",
                     "production_score", "decision", "text"],
                    # ensemble_score 仅作诊断；production_score 才参与 decision。
                    score_rows)
    package = {
        "schema_version": 1,
        "policy": "visual calibration offline; audio-only inference; non_speech excluded from other-speaker centroid",
        "counts": {"target": len(target_ids), "other_speaker": len(other_ids),
                   "non_speech": sum(row["class"] == "non_speech" for row in labels)},
        "models": {
            "redimnet": {
                "id": "IDRnD/ReDimNet M ft_mix vb2+vox2+cnc",
                "weights_sha256": "1e0716c2f351e79df054554f17c525b899045d2d05e4c839208b975e87d94e6a",
            },
            "eres2netv2": {
                "id": "iic/speech_eres2netv2_sv_zh-cn_16k-common",
                "weights_sha256": "0eb4057106b2573dd7b132cf0c36273ab29afd192c1610f80baa9c556dbb963c",
            },
        },
        "scoring": {
            "primary": primary,
            "formula": "(primary_margin - mean) / std",
            "selection": "highest leave-one-video-out AUC",
            "normalization": scales[primary],
        },
        "ensemble_diagnostic": {
            "formula": "mean((margin - mean) / std)",
            "normalization": scales,
            "used_for_decision": False,
        },
        "thresholds": thresholds,
        "validation": metrics,
        "inputs": {
            config.rel(config.VISUAL_LABELS): _sha256(config.VISUAL_LABELS),
            config.rel(config.EMB_REDIMNET): _sha256(config.EMB_REDIMNET),
            config.rel(config.EMB_ERES2NETV2): _sha256(config.EMB_ERES2NETV2),
        },
        "artifacts": {},
    }
    for path in sorted(config.CALIBRATION.glob("*.npy")):
        package["artifacts"][config.rel(path)] = _sha256(path)
    output = config.CALIBRATION / "calibration.json"
    output.write_text(json.dumps(package, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"视觉标签: target={len(target_ids)}, other={len(other_ids)}, non_speech={package['counts']['non_speech']}")
    for name, value in metrics.items():
        print(f"{name:12s} LOEO AUC={value['auc_leave_one_video_out']:.3f} "
              f"balanced={value['balanced_accuracy']:.3f}")
    print(f"生产主模型: {primary}（按 LOEO AUC 自动选择；融合仅作诊断）")
    print(f"阈值: reject<{thresholds['reject']:.4f}, accept>={thresholds['accept']:.4f}")
    print(f"校准包 -> {config.rel(config.CALIBRATION)}/")
    print(f"全量分数 -> {config.rel(config.PRODUCTION_SCORES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
