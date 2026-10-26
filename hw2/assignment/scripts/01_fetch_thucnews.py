"""按固定位置从 THUCNews 镜像抽取 100 组 x 100 篇，作为词向量训练语料。

数据源与课程页面一致：HuggingFace 数据集 Tongjilibo/THUCNews
revision ae77e363396e90b54cdaa9eafdbc5e102a3d4019，共 836,075 条。

数据集行 API 有速率限制，因此按组串行下载并缓存到 batches/，中断后可续跑。
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
RAW_DIR = ROOT / "data" / "thucnews"
BATCH_DIR = RAW_DIR / "batches"
RAW_DIR.mkdir(parents=True, exist_ok=True)
BATCH_DIR.mkdir(parents=True, exist_ok=True)
JSONL = RAW_DIR / "raw.jsonl"
META = RAW_DIR / "fetch_meta.json"
API = "https://datasets-server.huggingface.co/rows"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LLM-HW/1.0"
SLEEP = 1.5


def offsets() -> list[int]:
    total = CONFIG["thucnews"]["total_rows"]
    groups = CONFIG["thucnews"]["groups"]
    return [round(index * total / groups) for index in range(groups)]


def fetch_batch(start: int, length: int, retries: int = 8) -> list[dict]:
    query = urllib.parse.urlencode(
        {
            "dataset": CONFIG["thucnews"]["dataset"],
            "config": "default",
            "split": "train",
            "offset": start,
            "length": length,
        }
    )
    url = f"{API}?{query}"
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return [item["row"] for item in payload["rows"]]
        except Exception as error:  # noqa: BLE001 - retry network failures
            last_error = error
            wait = 20 if "429" in str(error) else 3 * attempt
            print(f"  批次 {start} 第 {attempt} 次失败：{error}；等待 {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"批次 {start} 下载失败：{last_error}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if JSONL.exists() and META.exists():
        meta = json.loads(META.read_text(encoding="utf-8"))
        if meta.get("revision") == CONFIG["thucnews"]["revision"] and meta.get("rows") == 10_000:
            print(f"已存在 {JSONL}（{meta['rows']} 条），跳过下载。", flush=True)
            return

    per_group = CONFIG["thucnews"]["per_group"]
    starts = offsets()
    records: list[dict] = []
    begin = time.perf_counter()
    for group, start in enumerate(starts):
        batch_path = BATCH_DIR / f"group_{group:03d}.json"
        if batch_path.exists():
            rows = json.loads(batch_path.read_text(encoding="utf-8"))
        else:
            rows = fetch_batch(start, per_group)
            batch_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            time.sleep(SLEEP)
        for row in rows:
            records.append(
                {
                    "group": group,
                    "offset": start,
                    "row_idx": row.get("row_idx"),
                    "id": row.get("id"),
                    "title": (row.get("title") or "").strip(),
                    "content": (row.get("content") or "").strip(),
                }
            )
        print(f"[{group + 1}/{len(starts)}] offset={start} 累计 {len(records)} 条", flush=True)

    records.sort(key=lambda item: (item["group"], item["row_idx"]))
    if len(records) != 10_000:
        raise RuntimeError(f"应下载 10,000 条，实际 {len(records)} 条")
    if len({item["id"] for item in records}) != len(records):
        raise RuntimeError("抽样中出现重复 id")

    with JSONL.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    meta = {
        "dataset": CONFIG["thucnews"]["dataset"],
        "revision": CONFIG["thucnews"]["revision"],
        "total_rows": CONFIG["thucnews"]["total_rows"],
        "groups": CONFIG["thucnews"]["groups"],
        "per_group": per_group,
        "offsets": starts,
        "rows": len(records),
        "unique_ids": len({item["id"] for item in records}),
        "download_seconds": round(time.perf_counter() - begin, 2),
        "raw_sha256": sha256(JSONL),
    }
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成：{JSONL}（{len(records)} 条，{meta['download_seconds']} s）", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
