"""Promote ranked, locally cached source audio into the stable source manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

from .. import config, manifests

DEFAULT_RANKING = config.DATA / "eval_sep" / "visual" / "candidate_scores.csv"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ingest(ranking: Path, limit: int) -> list[str]:
    if limit < 1:
        raise ValueError("--limit 必须大于零")
    if not ranking.is_file():
        raise FileNotFoundError(f"缺少候选排名 {ranking}；先运行 nailong rank-sources")
    manifest = config.MANIFESTS / "source_manifest.csv"
    existing = manifests.read(manifest)
    known_bvid = {row["bvid"] for row in existing if row["bvid"]}
    known_hash = {row["sha256"] for row in existing}
    numbers = [int(row["src"].split("_")[-1]) for row in existing]
    next_number = max(numbers, default=0) + 1
    candidates = sorted(manifests.read(ranking),
                        key=lambda row: (-float(row["accept_seconds"]), row["bvid"]))
    fields = list(existing[0]) if existing else [
        "src", "bvid", "title", "owner", "owner_mid", "page_url", "role",
        "duration", "prescreen_accept_seconds", "prescreen_accept_segments", "file", "sha256",
    ]
    added: list[str] = []
    config.SOURCES.mkdir(parents=True, exist_ok=True)
    with manifest.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not existing:
            writer.writeheader()
        for row in candidates:
            if len(added) >= limit:
                break
            bvid = row["bvid"]
            if not bvid or bvid in known_bvid or float(row["accept_seconds"]) <= 0:
                continue
            raw = Path(row["local_file"])
            source = raw if raw.is_absolute() else config.ROOT / raw
            if not source.is_file():
                continue
            digest = _sha256(source)
            if digest in known_hash:
                continue
            src = f"src_{next_number:02d}"
            dest = config.SOURCES / f"{src}{source.suffix.lower()}"
            shutil.copy2(source, dest)
            if _sha256(dest) != digest:
                dest.unlink()
                raise OSError(f"复制校验失败: {source} -> {dest}")
            entry = {field: "" for field in fields}
            entry.update({
                "src": src, "bvid": bvid, "title": row["title"], "owner": row["owner"],
                "owner_mid": row.get("owner_mid", ""),
                "page_url": row["page_url"], "role": "ranked_audio_candidate",
                "duration": row["duration"],
                "prescreen_accept_seconds": row["accept_seconds"],
                "prescreen_accept_segments": row["accept_segments"],
                "file": dest.name, "sha256": digest,
            })
            writer.writerow(entry)
            stream.flush()
            known_bvid.add(bvid)
            known_hash.add(digest)
            next_number += 1
            added.append(src)
            print(f"{src} <- {bvid}: prescreen {row['accept_seconds']}s")
    print(f"新增 {len(added)} 个源；总计 {len(existing) + len(added)} -> {config.rel(manifest)}")
    return added


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking", type=Path, default=DEFAULT_RANKING)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args(argv)
    ingest(args.ranking, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
