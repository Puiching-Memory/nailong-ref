"""Cache audio from a bounded set of public Bilibili posts with provenance.

This stage only collects candidates. Rank and audit them before considering
them for the training set. Public access alone does not grant training rights.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .. import config, manifests

OFFICIAL_LIST = "https://www.bilibili.com/list/230481071?bvid=BV1Cr4y1u7Xu&oid=770678812"
OFFICIAL_MID = "230481071"
FIELDS = [
    "bvid", "title", "owner", "owner_mid", "duration", "pubdate", "view",
    "danmaku", "reply", "favorite", "coin", "share", "like", "cid",
    "local_file", "page_url",
]


def fetch(directory: Path, *, scan: int, limit: int, min_duration: float,
          max_duration: float) -> int:
    try:
        import yt_dlp
    except ImportError as exc:
        raise SystemExit("缺少 yt-dlp；先运行 uv pip install yt-dlp") from exc

    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory.parent / "candidate_metadata.csv"
    metadata = ({row["bvid"]: row for row in manifests.read(metadata_path)}
                if metadata_path.is_file() else {})
    known_sources = {row["bvid"] for row in manifests.read(
        config.MANIFESTS / "source_manifest.csv") if row.get("bvid")}
    options = {"quiet": True, "no_warnings": True, "noprogress": True, "noplaylist": True,
               "socket_timeout": 20, "retries": 3}
    with yt_dlp.YoutubeDL({**options, "noplaylist": False,
                           "extract_flat": "in_playlist", "playlistend": scan}) as ydl:
        listing = ydl.extract_info(OFFICIAL_LIST, download=False)
    entries = listing.get("entries") or []
    if not entries:
        raise RuntimeError("公开视频列表没有返回条目，请检查列表 URL 或站点限制")
    added = 0
    for entry in entries:
        if added >= limit:
            break
        bvid = entry.get("id")
        if (not bvid or bvid in known_sources or
                (bvid in metadata and (directory / f"{bvid}.m4a").is_file())):
            continue
        page_url = f"https://www.bilibili.com/video/{bvid}"
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(page_url, download=False)
            duration = float(info.get("duration") or 0)
            if (str(info.get("uploader_id")) != OFFICIAL_MID or
                    not min_duration <= duration <= max_duration):
                continue
            with yt_dlp.YoutubeDL({**options,
                    "format": "bestaudio[ext=m4a]",
                    "outtmpl": str(directory / "%(id)s.%(ext)s")}) as ydl:
                ydl.download([page_url])
            audio_path = directory / f"{bvid}.m4a"
            if not audio_path.is_file() or audio_path.stat().st_size == 0:
                raise RuntimeError(f"下载后找不到音频: {audio_path}")
        except Exception as exc:
            print(f"跳过 {bvid}: {exc}", flush=True)
            continue
        row = {
            "bvid": bvid, "title": info.get("title") or "",
            "owner": info.get("uploader") or "", "owner_mid": info.get("uploader_id") or "",
            "duration": f"{duration:.2f}", "pubdate": info.get("upload_date") or "",
            "view": info.get("view_count") or "", "danmaku": info.get("danmaku_count") or "",
            "reply": info.get("comment_count") or "", "favorite": info.get("favorite_count") or "",
            "coin": info.get("coin_count") or "", "share": info.get("share_count") or "",
            "like": info.get("like_count") or "", "cid": info.get("cid") or "",
            "local_file": config.rel(audio_path), "page_url": page_url,
        }
        metadata[bvid] = row
        # Persist each completed download so an interrupted run can resume.
        with metadata_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(metadata.values())
        added += 1
        print(f"缓存 {added}/{limit} {bvid}: {duration:.1f}s {row['title']}", flush=True)
    print(f"新增 {added} 个公开候选 -> {config.rel(metadata_path)}")
    return added


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path,
                        default=config.DATA / "eval_sep" / "official" / "audio_index")
    parser.add_argument("--scan", type=int, default=50, help="最多检查列表中的前 N 条")
    parser.add_argument("--limit", type=int, default=10, help="本次最多新增 N 条音频")
    parser.add_argument("--min-duration", type=float, default=20)
    parser.add_argument("--max-duration", type=float, default=180)
    args = parser.parse_args(argv)
    if args.scan < 1 or args.limit < 1 or args.min_duration < 0 or args.max_duration < args.min_duration:
        parser.error("检查 --scan、--limit 和时长范围")
    fetch(args.directory, scan=args.scan, limit=args.limit,
          min_duration=args.min_duration, max_duration=args.max_duration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
