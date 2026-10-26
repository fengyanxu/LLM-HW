"""Clean THUCNews samples and build the shared segmented corpus for Word2Vec/FastText."""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import jieba

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hw2a.common import DATA, sha256_text, write_json  # noqa: E402

RAW = DATA / "thucnews" / "raw.jsonl"
OUT = DATA / "thucnews"
CLEAN = OUT / "clean.jsonl"
SEGMENTED = OUT / "segmented.txt"
STATS = OUT / "clean_stats.json"

MIN_CHARS = 100
MIN_SENTENCE_WORDS = 3
URL_RE = re.compile(r"https?://\S+|www\.\S+")
TAG_RE = re.compile(r"<[^>]{1,40}>")
SPACE_RE = re.compile(r"[\s\u3000\xa0]+")
SENT_RE = re.compile(r"(?<=[。！？!?])")
KEEP_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")


def normalize(text: str) -> str:
    text = TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = text.replace("\r", "\n").replace("\u3000", " ")
    return SPACE_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[list[str]]:
    sentences = []
    for piece in SENT_RE.split(text):
        piece = piece.strip()
        if not piece:
            continue
        words = []
        for word in jieba.cut(piece, cut_all=False, HMM=True):
            word = word.strip()
            if word and KEEP_RE.search(word):
                words.append(word)
        if len(words) >= MIN_SENTENCE_WORDS:
            sentences.append(words)
    return sentences


def main() -> None:
    if not RAW.exists():
        raise FileNotFoundError(f"缺少 {RAW}，请先运行 01_fetch_thucnews.py")
    records = [json.loads(line) for line in RAW.open(encoding="utf-8")]
    stats = Counter()
    stats["raw_articles"] = len(records)
    seen: dict[str, str] = {}
    kept = []
    begin = time.perf_counter()
    for index, record in enumerate(records, 1):
        text = normalize(record.get("content") or "")
        if not text:
            stats["dropped_empty"] += 1
            continue
        if len(text) < MIN_CHARS:
            stats["dropped_short"] += 1
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen:
            stats["dropped_duplicate"] += 1
            continue
        seen[digest] = record.get("id", str(index))
        sentences = tokenize(text)
        tokens = sum(len(sentence) for sentence in sentences)
        if not sentences or tokens < MIN_CHARS // 2:
            stats["dropped_no_sentence"] += 1
            continue
        kept.append(
            {
                "id": record.get("id"),
                "title": record.get("title") or "",
                "sentences": sentences,
                "sentences_count": len(sentences),
                "tokens": tokens,
                "sha256": digest,
            }
        )
        if index % 1000 == 0:
            print(f"已处理 {index}/{len(records)}，保留 {len(kept)}", flush=True)

    with CLEAN.open("w", encoding="utf-8", newline="\n") as handle:
        for article in kept:
            handle.write(json.dumps(article, ensure_ascii=False) + "\n")
    with SEGMENTED.open("w", encoding="utf-8", newline="\n") as handle:
        for article in kept:
            for sentence in article["sentences"]:
                handle.write(" ".join(sentence) + "\n")

    stats["kept_articles"] = len(kept)
    stats["sentences"] = sum(article["sentences_count"] for article in kept)
    stats["tokens"] = sum(article["tokens"] for article in kept)
    stats["avg_sentences"] = round(stats["sentences"] / max(1, len(kept)), 2)
    stats["avg_tokens"] = round(stats["tokens"] / max(1, len(kept)), 2)
    stats["min_chars"] = MIN_CHARS
    stats["min_sentence_words"] = MIN_SENTENCE_WORDS
    stats["jieba"] = jieba.__version__
    stats["segmented_sha256"] = sha256_text(SEGMENTED.read_text(encoding="utf-8"))
    stats["clean_seconds"] = round(time.perf_counter() - begin, 2)
    write_json(STATS, dict(stats))
    print(json.dumps(dict(stats), ensure_ascii=False, indent=2), flush=True)
    print(f"语料：{SEGMENTED}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
