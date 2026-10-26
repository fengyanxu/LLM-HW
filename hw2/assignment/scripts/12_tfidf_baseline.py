"""Supplementary diagnostic: TF-IDF + LinearSVC on the same splits."""
from __future__ import annotations

import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hw2a.common import CONFIG, DATA, RESULTS, write_json  # noqa: E402

OUT = DATA / "peopledaily" / "classification"
STAGES = CONFIG["classification"]["stages"]


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
    rows = []
    for stage in STAGES:
        train_texts, train_labels = load(OUT / f"train_{stage}.txt")
        test_texts, test_labels = load(OUT / f"test_{stage}.txt")
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        train_matrix = vectorizer.fit_transform(train_texts)
        test_matrix = vectorizer.transform(test_texts)
        model = LinearSVC(C=1.0, max_iter=5000)
        model.fit(train_matrix, train_labels)
        predicted = model.predict(test_matrix)
        rows.append(
            {
                "stage": stage,
                "accuracy": round(accuracy_score(test_labels, predicted), 4),
                "macro_f1": round(f1_score(test_labels, predicted, average="macro"), 4),
                "features": train_matrix.shape[1],
            }
        )
        print(rows[-1], flush=True)
    write_json(RESULTS / "tfidf_baseline.json", rows)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
