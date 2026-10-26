"""Train fastText supervised classifiers: pretrained Gensim vectors vs from scratch."""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import fasttext
from gensim.models import FastText

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hw2a.common import CONFIG, DATA, MODELS, RESULTS, write_json  # noqa: E402

OUT = DATA / "peopledaily" / "classification"
CLASSES = CONFIG["classification"]["classes"]
STAGES = CONFIG["classification"]["stages"]
FT = CONFIG["classification"]["fasttext"]
VEC = MODELS / "fasttext_pretrained.vec"


def export_vectors() -> dict:
    model = FastText.load(str(MODELS / "fasttext.model"))
    model.wv.save_word2vec_format(str(VEC), binary=False)
    with VEC.open(encoding="utf-8") as handle:
        header = handle.readline().strip()
        count = sum(1 for _ in handle)
    return {"path": str(VEC.relative_to(ROOT)), "header": header, "vectors": count, "bytes": VEC.stat().st_size}


def read_split(stage: int, part: str):
    rows = [json.loads(line) for line in (OUT / "selected.jsonl").open(encoding="utf-8")]
    active = set(CLASSES[:stage])
    return [row for row in rows if row["split"] == part and row["label"] in active]


def macro_scores(true, pred, labels):
    per_class = {}
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(true, pred))
        fp = sum(t != label and p == label for t, p in zip(true, pred))
        fn = sum(t == label and p != label for t, p in zip(true, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4), "support": tp + fn}
    macro = sum(value["f1"] for value in per_class.values()) / len(per_class)
    confusion = {label: {other: 0 for other in labels} for label in labels}
    for t, p in zip(true, pred):
        confusion[t][p] += 1
    return per_class, round(macro, 4), confusion


def predict_rows(model, rows):
    texts = [" ".join(row["body"]) for row in rows]
    labels = model.predict(texts, k=1)
    predicted = [item[0].replace("__label__", "") for item in labels[0]]
    probs = [float(item[0]) for item in labels[1]]
    return predicted, probs


def run_condition(stage: int, condition: str):
    train_path = OUT / f"train_{stage}.txt"
    test_path = OUT / f"test_{stage}.txt"
    kwargs = dict(
        input=str(train_path),
        dim=FT["dim"],
        lr=FT["lr"],
        epoch=FT["epoch"],
        wordNgrams=FT["wordNgrams"],
        loss=FT["loss"],
        thread=FT["thread"],
        seed=FT["seed"],
        minn=FT["minn"],
        maxn=FT["maxn"],
        minCount=FT["minCount"],
        minCountLabel=FT["minCountLabel"],
        verbose=0,
    )
    if condition == "pretrained":
        kwargs["pretrainedVectors"] = str(VEC)
    begin = time.perf_counter()
    model = fasttext.train_supervised(**kwargs)
    seconds = time.perf_counter() - begin
    # fastText 二进制模型含 2,000,000 个哈希桶，单个约 800 MB；指标与预测已足够复核，故不保存。

    rows = read_split(stage, "test")
    predicted, probs = predict_rows(model, rows)
    true = [row["label"] for row in rows]
    labels = CLASSES[:stage]
    per_class, macro, confusion = macro_scores(true, predicted, labels)
    accuracy = sum(t == p for t, p in zip(true, predicted)) / len(true)
    predictions = [
        {
            "id": row["id"],
            "title": row["title"],
            "true": t,
            "predicted": p,
            "confidence": round(prob, 4),
        }
        for row, t, p, prob in zip(rows, true, predicted, probs)
    ]
    with (RESULTS / f"predictions_{stage}_{condition}.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "title", "true", "predicted", "confidence"])
        writer.writeheader()
        writer.writerows(predictions)
    return {
        "stage": stage,
        "condition": condition,
        "accuracy": round(accuracy, 4),
        "macro_f1": macro,
        "test_n": len(rows),
        "train_seconds": round(seconds, 3),
        "per_class": per_class,
        "confusion": confusion,
        "predictions": predictions,
    }


def main() -> None:
    export = export_vectors()
    print("导出词向量：", export, flush=True)
    results = []
    for stage in STAGES:
        for condition in ("pretrained", "scratch"):
            row = run_condition(stage, condition)
            results.append(row)
            print(f"stage={stage} {condition}: acc={row['accuracy']} macroF1={row['macro_f1']} ({row['train_seconds']}s)", flush=True)

    # 8 类预训练模型重复一次，检查同种子预测是否一致
    repeat = run_condition(8, "pretrained")
    base = next(row for row in results if row["stage"] == 8 and row["condition"] == "pretrained")
    same = [item["predicted"] for item in base["predictions"]] == [item["predicted"] for item in repeat["predictions"]]
    results.append({"stage": 8, "condition": "pretrained_repeat", "same_predictions": same, "accuracy": repeat["accuracy"]})

    summary = {
        "vector_export": export,
        "fasttext_params": FT,
        "deterministic_repeat_8_pretrained": same,
        "results": results,
    }
    write_json(RESULTS / "classification.json", summary)
    with (RESULTS / "classification.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "condition", "accuracy", "macro_f1", "test_n", "train_seconds"])
        for row in results:
            if "accuracy" in row and "test_n" in row:
                writer.writerow([row["stage"], row["condition"], row["accuracy"], row["macro_f1"], row["test_n"], row["train_seconds"]])
    print(json.dumps({row["stage"]: row.get("accuracy") for row in results if "test_n" in row}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
