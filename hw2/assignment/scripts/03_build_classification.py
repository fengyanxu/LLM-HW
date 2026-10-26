"""Build the 8-topic PeopleDaily classification set and 2/4/8 nested splits."""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hw2a.common import CONFIG, DATA, TOPIC_KEYWORDS, write_json  # noqa: E402

PD = DATA / "peopledaily"
OUT = PD / "classification"
OUT.mkdir(parents=True, exist_ok=True)
CLASSES = CONFIG["classification"]["classes"]
PER_CLASS = CONFIG["classification"]["per_class"]
TRAIN = CONFIG["classification"]["train_per_class"]
TEST = CONFIG["classification"]["test_per_class"]
MAX_WORDS = CONFIG["classification"]["max_body_words"]
SEED = CONFIG["seed"]


BAD_TITLE = (
    "本报讯", "本报", "新华社", "记者", "图片", "摄影", "简历", "名单",
    "公告", "启事", "讣告", "征稿", "广告", "目录", "定价", "书讯",
    "连载", "答记者问", "发言摘要",
)
STRONG_EDU = (
    "教育", "学校", "教师", "老师", "高考", "中考", "招生", "课程", "教学",
    "校园", "幼儿园", "研究生", "博士", "留学", "考试", "教材", "校长",
    "课堂", "学费", "毕业", "教育部", "义务教育", "素质教育", "高校", "学院",
)
SPORTS_MARKERS = (
    "赛", "冠军", "运动员", "球队", "足球", "篮球", "排球", "乒乓球",
    "羽毛球", "举重", "游泳", "田径", "体操", "棋", "球迷", "运动会",
)
MILITARY_MARKERS = ("战士", "空军", "部队", "军营", "官兵", "参军", "军事", "国防")
MEDICAL_MARKERS = ("抢救", "患者", "病人", "医院", "疾病", "手术", "救治")


def topic_hits(title: str) -> dict[str, list[str]]:
    return {
        topic: [kw for kw in keywords if kw in title]
        for topic, keywords in TOPIC_KEYWORDS.items()
        if any(kw in title for kw in keywords)
    }


def candidates(records):
    strict = defaultdict(list)
    relaxed = defaultdict(list)
    for record in records:
        title = record["title"]
        if not (4 <= len(title) <= 45) or any(bad in title for bad in BAD_TITLE):
            continue
        if len(record["body"]) < 80:
            continue
        found = topic_hits(title)
        if not found or len(found) > 2:
            continue
        ranked = sorted(found.items(), key=lambda item: (-len(item[1]), item[0]))
        topic, hits = ranked[0]
        second = len(ranked[1][1]) if len(ranked) > 1 else 0
        if topic == "教育" and not any(word in title for word in STRONG_EDU):
            if any(word in title for word in SPORTS_MARKERS + MILITARY_MARKERS + MEDICAL_MARKERS):
                continue
        item = {
            "id": record["id"],
            "date": record["date"],
            "title": record["title"],
            "topic": topic,
            "hits": hits,
            "body_words": len(record["body"]),
        }
        if len(hits) >= 2 and second < 2:
            strict[topic].append(item)
        elif len(hits) == 1 and second == 0:
            relaxed[topic].append(item)
    return strict, relaxed


def pick(items, count: int) -> list[dict]:
    """Round-robin across months so a class is not dominated by one month."""
    by_month: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_month[item["date"][:6]].append(item)
    for month in by_month:
        by_month[month].sort(key=lambda x: (-len(x["hits"]), abs(len(x["title"]) - 18), x["id"]))
    selected: list[dict] = []
    while len(selected) < count:
        advanced = False
        for month in sorted(by_month):
            if by_month[month] and len(selected) < count:
                selected.append(by_month[month].pop(0))
                advanced = True
        if not advanced:
            break
    return selected


def main() -> None:
    records = [json.loads(line) for line in (PD / "articles.jsonl").open(encoding="utf-8")]
    strict, relaxed = candidates(records)
    print("严格候选：", {c: len(strict[c]) for c in CLASSES}, flush=True)
    print("放宽候选（仅 1 个关键词且无冲突）：", {c: len(relaxed[c]) for c in CLASSES}, flush=True)

    chosen: dict[str, list[dict]] = {}
    used: set[str] = set()
    for topic in CLASSES:
        pool = [x for x in strict[topic] if x["id"] not in used]
        selected = pick(pool, PER_CLASS)
        if len(selected) < PER_CLASS:
            extra = [x for x in relaxed[topic] if x["id"] not in used and x["id"] not in {y["id"] for y in selected}]
            selected += pick(extra, PER_CLASS - len(selected))
        if len(selected) < PER_CLASS:
            raise RuntimeError(f"{topic} 只有 {len(selected)} 篇候选，不足 {PER_CLASS}")
        chosen[topic] = selected
        used.update(x["id"] for x in selected)

    by_id = {record["id"]: record for record in records}
    split_rows = []
    for topic in CLASSES:
        rng = random.Random(SEED)
        ordered = sorted(chosen[topic], key=lambda x: x["id"])
        rng.shuffle(ordered)
        for index, item in enumerate(ordered):
            item["split"] = "train" if index < TRAIN else "test"
            item["label"] = topic
            split_rows.append(item)

    # 写出清单和带正文的记录
    fields = ["label", "split", "id", "date", "title", "hits", "body_words"]
    with (OUT / "selected.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in split_rows:
            row = dict(row)
            row["hits"] = "|".join(row["hits"])
            writer.writerow(row)

    with (OUT / "selected.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in split_rows:
            record = by_id[row["id"]]
            handle.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "label": row["label"],
                        "split": row["split"],
                        "title": row["title"],
                        "body": record["body"][:MAX_WORDS],
                        "body_words": len(record["body"]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    stage_info = {}
    for stage in CONFIG["classification"]["stages"]:
        active = CLASSES[:stage]
        for part in ("train", "test"):
            path = OUT / f"{part}_{stage}.txt"
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                for row in split_rows:
                    if row["label"] not in active or row["split"] != part:
                        continue
                    body = by_id[row["id"]]["body"][:MAX_WORDS]
                    handle.write(f"__label__{row['label']} {' '.join(body)}\n")
        counts = Counter(row["label"] for row in split_rows if row["label"] in active)
        stage_info[str(stage)] = {
            "classes": active,
            "per_class": dict(counts),
            "train_lines": sum(1 for _ in (OUT / f"train_{stage}.txt").open(encoding="utf-8")),
            "test_lines": sum(1 for _ in (OUT / f"test_{stage}.txt").open(encoding="utf-8")),
        }

    manifest = {
        "source": "PeopleDaily1998 1998-01..06",
        "selection": "title keyword rules; at least 2 hits and no rival with 2 hits; month round-robin",
        "classes": CLASSES,
        "per_class": PER_CLASS,
        "train_per_class": TRAIN,
        "test_per_class": TEST,
        "stages": stage_info,
        "split_seed": SEED,
        "max_body_words": MAX_WORDS,
        "selected_csv_sha256": hashlib.sha256((OUT / "selected.csv").read_bytes()).hexdigest(),
        "candidate_counts": {topic: len(strict[topic]) for topic in CLASSES},
        "relaxed_counts": {topic: len(relaxed[topic]) for topic in CLASSES},
    }
    write_json(OUT / "manifest.json", manifest)
    print("阶段：", json.dumps(stage_info, ensure_ascii=False), flush=True)
    for topic in CLASSES:
        rows = [row for row in split_rows if row["label"] == topic]
        print(f"\n[{topic}] {len(rows)} 篇 / 训练 {sum(r['split']=='train' for r in rows)} / 测试 {sum(r['split']=='test' for r in rows)}", flush=True)
        for row in rows[:6]:
            print(f"  {row['split']:5s} {row['id']} {row['title']}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
