"""Parse PeopleDaily1998, deduplicate, and count topic candidates by title keywords."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hw2a.common import CONFIG, DATA, TOPIC_KEYWORDS, write_json  # noqa: E402
from hw2a.peopledaily import parse_all  # noqa: E402

MIN_BODY = CONFIG["classification"]["min_body_words"]
OUT = DATA / "peopledaily"
OUT.mkdir(parents=True, exist_ok=True)


def hits(title: str) -> dict[str, list[str]]:
    return {topic: [kw for kw in kws if kw in title] for topic, kws in TOPIC_KEYWORDS.items() if any(kw in title for kw in kws)}


def main() -> None:
    records, stats, anomalies = parse_all()
    print("解析统计：", stats, flush=True)

    seen: dict[str, str] = {}
    kept = []
    duplicate = []
    short = []
    for record in records:
        if len(record["body"]) < MIN_BODY:
            short.append(record["id"])
            continue
        digest = record["body_sha256"]
        if digest in seen:
            duplicate.append({"id": record["id"], "duplicate_of": seen[digest]})
            continue
        seen[digest] = record["id"]
        kept.append(record)

    summary = {
        "parse": stats,
        "min_body_words": MIN_BODY,
        "duplicates_removed": len(duplicate),
        "short_removed": len(short),
        "articles_kept": len(kept),
        "anomalies": anomalies,
        "duplicate_examples": duplicate[:20],
    }

    with (OUT / "articles.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in kept:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    counts = Counter()
    examples: dict[str, list[dict]] = {topic: [] for topic in TOPIC_KEYWORDS}
    for record in kept:
        found = hits(record["title"])
        if not found:
            continue
        ranked = sorted(found.items(), key=lambda item: (-len(item[1]), item[0]))
        top_topic, top_hits = ranked[0]
        second = len(ranked[1][1]) if len(ranked) > 1 else 0
        if len(top_hits) >= 2 and second < 2:
            counts[top_topic] += 1
            if len(examples[top_topic]) < 12:
                examples[top_topic].append({"id": record["id"], "title": record["title"], "hits": top_hits[:5]})
        elif len(top_hits) == 1 and second == 0:
            counts[top_topic] += 0  # 单独命中留给后续放宽
    summary["strict_candidates"] = dict(counts)
    summary["examples"] = examples

    with (OUT / "candidate_counts.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"保留文章 {len(kept)} 篇；重复 {len(duplicate)}，过短 {len(short)}", flush=True)
    print("严格候选数（>=2 个关键词且第二主题 <2）：", dict(counts), flush=True)
    for topic in TOPIC_KEYWORDS:
        print(f"\n[{topic}]", flush=True)
        for item in examples[topic][:8]:
            print(f"  {item['id']}  {item['title']}  {item['hits']}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
