"""Supplementary sensitivity check: fastText epochs at the 8-class setting."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import fasttext
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hw2a.common import CONFIG, DATA, MODELS, RESULTS, write_json  # noqa: E402

OUT = DATA / "peopledaily" / "classification"
FT = CONFIG["classification"]["fasttext"]
VEC = MODELS / "fasttext_pretrained.vec"


def load(path: Path):
    texts, labels = [], []
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        label, text = line.split(" ", 1)
        labels.append(label.replace("__label__", ""))
        texts.append(text)
    return texts, labels


def main() -> None:
    train_texts, train_labels = load(OUT / "train_8.txt")
    test_texts, test_labels = load(OUT / "test_8.txt")
    rows = []
    for condition in ("pretrained", "scratch"):
        for epoch in (25, 50, 100):
            kwargs = dict(
                input=str(OUT / "train_8.txt"),
                dim=FT["dim"],
                lr=FT["lr"],
                epoch=epoch,
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
            predicted = [item[0].replace("__label__", "") for item in model.predict(test_texts, k=1)[0]]
            rows.append(
                {
                    "condition": condition,
                    "epoch": epoch,
                    "accuracy": round(accuracy_score(test_labels, predicted), 4),
                    "macro_f1": round(f1_score(test_labels, predicted, average="macro"), 4),
                    "train_seconds": round(seconds, 3),
                }
            )
            print(condition, epoch, rows[-1], flush=True)
    write_json(RESULTS / "epoch_sensitivity.json", rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
