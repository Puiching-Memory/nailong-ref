"""Training corpus audit. Approval is bound to the exact audio hash and transcript."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from nailong import config, manifests

REVIEW = config.MANIFESTS / "training_review.csv"
FIELDS = ["file", "sha256", "approved", "text", "note"]
MIN_TRAIN_SECONDS = 30 * 60
MIN_TRAIN_CLIPS = 200


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def accepted_rows(manifest: Path = config.PRODUCTION_MANIFEST) -> list[dict]:
    return [row for row in manifests.read(manifest) if row["status"] == "accepted"]


def make_review(manifest: Path = config.PRODUCTION_MANIFEST,
                review: Path = REVIEW) -> int:
    """Refresh the review queue without carrying approvals to changed audio/text."""
    from .dataset import speakable

    existing = {row["file"]: row for row in manifests.read(review)} if review.exists() else {}
    rows = []
    pending = 0
    for clip in accepted_rows(manifest):
        file = clip["file"]
        text = speakable(clip["text"])
        old = existing.get(file, {})
        same = old.get("sha256") == clip["sha256"]
        approved = old.get("approved", "") if same else ""
        note = old.get("note", "") if same else ""
        pending += approved.lower() not in {"yes", "no"}
        rows.append([file, clip["sha256"], approved, old.get("text", text) if same else text, note])
    manifests.write(review, FIELDS, rows)
    print(f"复核清单: {len(rows)} 段；待判定 {pending} 段 -> {config.rel(review)}")
    return pending


@dataclass
class Audit:
    usable: list[tuple[dict, str]]
    issues: list[str]
    accepted_seconds: float
    usable_seconds: float


def audit(manifest: Path = config.PRODUCTION_MANIFEST,
          audio_dir: Path = config.PRODUCTION, review: Path = REVIEW,
          min_seconds: float = MIN_TRAIN_SECONDS,
          min_clips: int = MIN_TRAIN_CLIPS) -> Audit:
    from .dataset import speakable

    rows = accepted_rows(manifest)
    reviews = {row["file"]: row for row in manifests.read(review)} if review.exists() else {}
    issues: list[str] = []
    usable: list[tuple[dict, str]] = []
    seen_hashes: set[str] = set()
    accepted_seconds = sum(float(row["dur"]) for row in rows)
    for row in rows:
        file = row["file"]
        path = audio_dir / file
        problems = []
        if not path.is_file():
            problems.append("missing_audio")
        else:
            actual_hash = sha256(path)
            if actual_hash != row["sha256"]:
                problems.append("hash_mismatch")
            if actual_hash in seen_hashes:
                problems.append("duplicate_audio")
            seen_hashes.add(actual_hash)
            try:
                info = sf.info(path)
                duration = info.frames / info.samplerate
                if info.samplerate != config.SR_EXPORT or info.channels != 1 or info.subtype != "PCM_16":
                    problems.append("wrong_audio_format")
                if abs(duration - float(row["dur"])) > 0.01:
                    problems.append("duration_mismatch")
                wave, _ = sf.read(path, dtype="float32")
                if not np.isfinite(wave).all() or len(wave) == 0:
                    problems.append("invalid_samples")
                elif np.max(np.abs(wave)) < 0.01:
                    problems.append("near_silence")
                elif np.mean(np.abs(wave) >= 0.999) > 0.001:
                    problems.append("clipping")
            except (OSError, RuntimeError, ValueError):
                problems.append("unreadable_audio")
        if float(row["stem_margin_db"]) < 15:
            problems.append("stem_margin_below_gate")
        reviewed = reviews.get(file)
        if reviewed is None:
            problems.append("unreviewed")
            text = ""
        else:
            text = reviewed["text"].strip()
            if reviewed["sha256"] != row["sha256"]:
                problems.append("stale_review")
            decision = reviewed["approved"].lower()
            if decision not in {"yes", "no"}:
                problems.append("unreviewed")
            if decision == "yes" and (not text or "|" in text or "\n" in text
                                       or text != speakable(text)):
                problems.append("invalid_text")
        if problems:
            issues.append(f"{file}: {', '.join(problems)}")
        elif reviewed is not None and reviewed["approved"].lower() == "yes":
            usable.append((row, text))
    usable_seconds = sum(float(row["dur"]) for row, _ in usable)
    if len(usable) < min_clips:
        issues.append(f"corpus: {len(usable)} clips < required {min_clips}")
    if usable_seconds < min_seconds:
        issues.append(f"corpus: {usable_seconds:.2f}s < required {min_seconds:.2f}s")
    return Audit(usable, issues, accepted_seconds, usable_seconds)


def print_audit(result: Audit) -> None:
    print(f"accepted {result.accepted_seconds:.2f}s; reviewed usable "
          f"{len(result.usable)} clips / {result.usable_seconds:.2f}s")
    for issue in result.issues:
        print(f"  {issue}")
    print("PASS" if not result.issues else f"FAIL: {len(result.issues)} issues")
