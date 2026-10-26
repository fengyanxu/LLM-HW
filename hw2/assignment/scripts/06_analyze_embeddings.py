"""Neighbour, OOV and similarity analyses for the two embedding models."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from gensim.models import FastText, Word2Vec

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hw2a.common import MODELS, RESULTS, write_json  # noqa: E402

NEIGHBOR_WORDS = ["银行", "学校", "电脑", "经济"]
OOV_WORDS = ["量子计算芯片", "区块链金融", "新能源汽车"]
GROUPS = {
    "体育": ["足球", "篮球", "排球", "乒乓球"],
    "金融": ["银行", "贷款", "利率", "股票"],
    "教育": ["学校", "学生", "教师", "高考"],
    "科技": ["电脑", "软件", "互联网", "科技"],
}


def load(name: str):
    cls = FastText if name == "fasttext" else Word2Vec
    return cls.load(str(MODELS / f"{name}.model"))


def neighbors(model, word: str, topn: int = 10):
    if word not in model.wv.key_to_index:
        return []
    return [{"word": item, "score": round(float(score), 4)} for item, score in model.wv.most_similar(word, topn=topn)]


def oov_record(model, word: str, is_fasttext: bool):
    in_vocab = word in model.wv.key_to_index
    record = {"in_vocab": in_vocab, "vector": None, "neighbors": []}
    if not in_vocab and not is_fasttext:
        record["note"] = "Word2Vec 无法查询词表外词"
        return record
    try:
        vector = np.asarray(model.wv[word], dtype=np.float32)
    except KeyError:
        record["note"] = "无法生成向量"
        return record
    record["vector"] = {"norm": round(float(np.linalg.norm(vector)), 4), "first5": [round(float(x), 4) for x in vector[:5]]}
    norms = np.linalg.norm(model.wv.vectors, axis=1)
    denom = norms * max(float(np.linalg.norm(vector)), 1e-12)
    sims = (model.wv.vectors @ vector) / np.maximum(denom, 1e-12)
    order = np.argsort(-sims)[:5]
    index_to_word = model.wv.index_to_key
    record["neighbors"] = [
        {"word": index_to_word[int(i)], "score": round(float(sims[int(i)]), 4)}
        for i in order
        if index_to_word[int(i)] != word
    ]
    return record


def similarity(model):
    vectors = {}
    missing = {}
    for topic, words in GROUPS.items():
        vectors[topic] = []
        missing[topic] = []
        for word in words:
            if word in model.wv.key_to_index:
                vectors[topic].append(word)
            else:
                missing[topic].append(word)
    within = []
    group_means = {}
    for topic, words in vectors.items():
        pairs = [
            float(model.wv.similarity(words[i], words[j]))
            for i in range(len(words))
            for j in range(i + 1, len(words))
        ]
        group_means[topic] = round(float(np.mean(pairs)), 4) if pairs else None
        within.extend(pairs)
    cross = []
    topics = [topic for topic in vectors if len(vectors[topic]) >= 2]
    for i in range(len(topics)):
        for j in range(i + 1, len(topics)):
            for a in vectors[topics[i]]:
                for b in vectors[topics[j]]:
                    cross.append(float(model.wv.similarity(a, b)))
    return {
        "within_mean": round(float(np.mean(within)), 4) if within else None,
        "cross_mean": round(float(np.mean(cross)), 4) if cross else None,
        "group_means": group_means,
        "missing_words": missing,
        "groups": vectors,
    }


def main() -> None:
    output = {}
    rows = []
    for name in ("word2vec", "fasttext"):
        model = load(name)
        output[name] = {
            "vocab_size": len(model.wv),
            "neighbors": {word: neighbors(model, word) for word in NEIGHBOR_WORDS},
            "oov": {word: oov_record(model, word, name == "fasttext") for word in OOV_WORDS},
            "similarity": similarity(model),
        }
        for word, items in output[name]["neighbors"].items():
            for rank, item in enumerate(items, 1):
                rows.append({"model": name, "query": word, "rank": rank, "neighbor": item["word"], "score": item["score"]})
        print(f"\n{name}") 
        for word in NEIGHBOR_WORDS:
            print(f"  {word}: {[item['word'] for item in output[name]['neighbors'][word][:6]]}", flush=True)
        print(f"  OOV: { {w: v['in_vocab'] for w, v in output[name]['oov'].items()} }", flush=True)
        print(f"  相似度: within={output[name]['similarity']['within_mean']} cross={output[name]['similarity']['cross_mean']}", flush=True)
        del model
    write_json(RESULTS / "embedding_analysis.json", output)
    with (RESULTS / "word_neighbors.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        handle.write("model,query,rank,neighbor,score\n")
        for row in rows:
            handle.write(f"{row['model']},{row['query']},{row['rank']},{row['neighbor']},{row['score']}\n")
    print(json.dumps({name: {"vocab": output[name]["vocab_size"], "similarity": output[name]["similarity"]["within_mean"], "cross": output[name]["similarity"]["cross_mean"]} for name in output}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
