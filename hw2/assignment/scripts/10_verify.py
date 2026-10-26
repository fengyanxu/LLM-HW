"""Check that the HW2 assignment artefacts are complete and internally consistent."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hw2a.common import CONFIG, DATA, FIGURES, MODELS, RESULTS, read_json  # noqa: E402

CLASSES = CONFIG["classification"]["classes"]
STAGES = CONFIG["classification"]["stages"]
errors: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def main() -> None:
    raw = DATA / "thucnews" / "raw.jsonl"
    meta = read_json(DATA / "thucnews" / "fetch_meta.json")
    check(raw.exists() and sum(1 for _ in raw.open(encoding="utf-8")) == 10_000, "THUCNews 原始抽样不是 10,000 条")
    check(meta.get("revision") == CONFIG["thucnews"]["revision"], "THUCNews revision 不一致")

    stats = read_json(DATA / "thucnews" / "clean_stats.json")
    check(stats["kept_articles"] > 8_000, "清洗后文章过少")
    check(stats["sentences"] > 150_000, "句子数异常")
    segmented_lines = sum(1 for _ in (DATA / "thucnews" / "segmented.txt").open(encoding="utf-8"))
    check(segmented_lines == stats["sentences"], "segmented.txt 行数与句子数不一致")

    training = read_json(RESULTS / "embedding_training.json")
    check(len(training) == 2, "应训练 Word2Vec 与 FastText 两个模型")
    check(training[0]["vocab_size"] == training[1]["vocab_size"], "两模型词表不一致")
    for name in ("word2vec", "fasttext"):
        check((MODELS / f"{name}.model").exists(), f"缺少 {name}.model")

    manifest = read_json(DATA / "peopledaily" / "classification" / "manifest.json")
    rows = list(csv.DictReader((DATA / "peopledaily" / "classification" / "selected.csv").open(encoding="utf-8-sig")))
    check(len(rows) == 480, "分类集不是 480 篇")
    for label in CLASSES:
        subset = [row for row in rows if row["label"] == label]
        check(len(subset) == 60, f"{label} 不是 60 篇")
        check(sum(row["split"] == "train" for row in subset) == 48, f"{label} 训练不是 48 篇")
        check(sum(row["split"] == "test" for row in subset) == 12, f"{label} 测试不是 12 篇")
    for stage in STAGES:
        check((DATA / "peopledaily" / "classification" / f"train_{stage}.txt").exists(), f"缺少 train_{stage}.txt")
        check((DATA / "peopledaily" / "classification" / f"test_{stage}.txt").exists(), f"缺少 test_{stage}.txt")
    check(manifest["stages"]["8"]["train_lines"] == 384, "8 类训练行数不是 384")
    check(manifest["stages"]["8"]["test_lines"] == 96, "8 类测试行数不是 96")

    classification = read_json(RESULTS / "classification.json")
    rows_c = [row for row in classification["results"] if row.get("test_n")]
    check(len(rows_c) == 6, "分类结果应有 6 组（3 个类别数 × 2 种条件）")
    for stage in STAGES:
        for condition in ("pretrained", "scratch"):
            row = next((item for item in rows_c if item["stage"] == stage and item["condition"] == condition), None)
            check(row is not None, f"缺少 {stage} 类 {condition} 结果")
            if row:
                check(0 <= row["accuracy"] <= 1 and 0 <= row["macro_f1"] <= 1, f"{stage}/{condition} 指标越界")
                check(row["test_n"] == 12 * stage, f"{stage} 类测试数不是 {12 * stage}")
    check(classification["deterministic_repeat_8_pretrained"] is True, "8 类预训练模型重复预测不一致")
    sensitivity = read_json(RESULTS / "epoch_sensitivity.json")
    check(len(sensitivity) == 6, "训练轮数敏感性应有 6 组")
    tfidf = read_json(RESULTS / "tfidf_baseline.json")
    check(len(tfidf) == 3, "TF-IDF 诊断基线应有 3 组")

    for name in ("01_data_overview", "02_embedding_similarity", "03_classification_curve", "04_confusion_8", "05_epoch_sensitivity"):
        check((FIGURES / f"{name}.png").exists(), f"缺少图 {name}.png")
    check((ROOT / "report.md").exists(), "缺少 report.md")
    check((ROOT / "README.md").exists(), "缺少 README.md")

    if errors:
        print("校验失败：", flush=True)
        for message in errors:
            print(f" - {message}", flush=True)
        raise SystemExit(1)
    print(f"校验通过：480 篇分类集，2/4/8 类结果完整，图表与报告齐全。", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
