"""Train Word2Vec and FastText on the same segmented THUCNews corpus."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from gensim.models import FastText, Word2Vec
from gensim.models.word2vec import LineSentence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hw2a.common import CONFIG, DATA, MODELS, RESULTS, sha256_text, write_json  # noqa: E402

SEGMENTED = DATA / "thucnews" / "segmented.txt"
EMB = CONFIG["embeddings"]


def main() -> None:
    if not SEGMENTED.exists():
        raise FileNotFoundError(f"缺少 {SEGMENTED}，请先运行 04_prepare_embeddings.py")
    common = dict(
        vector_size=EMB["vector_size"],
        window=EMB["window"],
        min_count=EMB["min_count"],
        sg=EMB["sg"],
        workers=EMB["workers"],
        epochs=EMB["epochs"],
        seed=EMB["seed"],
        negative=EMB["negative"],
    )
    rows = []
    for name, cls, extra in (
        ("word2vec", Word2Vec, {}),
        ("fasttext", FastText, EMB["fasttext"]),
    ):
        print(f"开始训练 {name}：{common} {extra}", flush=True)
        begin = time.perf_counter()
        model = cls(sentences=LineSentence(str(SEGMENTED)), **common, **extra)
        elapsed = time.perf_counter() - begin
        path = MODELS / f"{name}.model"
        model.save(str(path))
        row = {
            "name": name,
            "seconds": round(elapsed, 2),
            "vector_size": EMB["vector_size"],
            "vocab_size": len(model.wv),
            "params": {**common, **extra},
            "model": str(path.relative_to(ROOT)),
            "segmented_sha256": sha256_text(SEGMENTED.read_text(encoding="utf-8")),
            "segmented_bytes": SEGMENTED.stat().st_size,
        }
        rows.append(row)
        print(f"{name}: {row['seconds']} s, vocab={row['vocab_size']}", flush=True)
    write_json(RESULTS / "embedding_training.json", rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
