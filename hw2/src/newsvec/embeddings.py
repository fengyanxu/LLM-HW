import gc
import time
from collections import Counter

import numpy as np
from gensim.models import FastText, Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.fasttext import ft_ngram_hashes
from gensim.models.fasttext_inner import compute_ngrams_bytes

from .common import ROOT, SentenceCorpus, read_docs, read_json, sha256, write_json


class EpochTrace(CallbackAny2Vec):
    def __init__(self):
        self.records = []
        self.started = None

    def on_epoch_begin(self, model):
        self.started = time.perf_counter()

    def on_epoch_end(self, model):
        self.records.append({"epoch": len(self.records) + 1,
                             "seconds": time.perf_counter() - self.started})


def train_embeddings(config, only=None):
    corpus = SentenceCorpus(ROOT / "data/processed/sentences.txt")
    corpus_hash = sha256(corpus.path)
    all_metrics = []
    for variant in config["variants"]:
        name, family = variant["name"], variant["family"]
        metric_file = ROOT / f"results/embedding_{name}.json"
        if only and name not in only:
            continue
        params = dict(config["embedding"])
        if family == "fasttext":
            params.update(config["fasttext"])
        params.update({k: v for k, v in variant.items() if k not in {"name", "family"}})
        if metric_file.exists() and (ROOT / f"models/{name}.model").exists():
            saved_metric = read_json(metric_file)
            if saved_metric["parameters"] == params and saved_metric.get("corpus_sha256") == corpus_hash:
                print("CACHED", name, flush=True)
                continue
        trace = EpochTrace()
        start = time.perf_counter()
        model = (FastText if family == "fasttext" else Word2Vec)(**params)
        model.build_vocab(corpus)
        build_s = time.perf_counter() - start
        begin = time.perf_counter()
        effective, raw = model.train(corpus, total_examples=model.corpus_count,
                                     epochs=params["epochs"], callbacks=[trace])
        train_s = time.perf_counter() - begin
        saved = ROOT / f"models/{name}.model"
        model.save(str(saved))
        restored = (FastText if family == "fasttext" else Word2Vec).load(str(saved))
        first = model.wv.index_to_key[0]
        roundtrip = float(np.max(np.abs(model.wv[first] - restored.wv[first])))
        del restored
        array_bytes = model.wv.vectors.nbytes + model.syn1neg.nbytes
        if family == "fasttext":
            array_bytes += model.wv.vectors_vocab.nbytes + model.wv.vectors_ngrams.nbytes
        metrics = {"name": name, "family": family, "parameters": params, "corpus_sha256": corpus_hash,
                   "vocab_size": len(model.wv), "corpus_sentences": model.corpus_count,
                   "raw_corpus_tokens": model.corpus_total_words, "effective_train_words": effective,
                   "raw_train_words": raw, "build_seconds": build_s, "train_seconds": train_s,
                   "array_mib": array_bytes / 2**20, "save_load_max_error": roundtrip,
                   "epochs": trace.records,
                   "neighbors": {q: model.wv.most_similar(q, topn=8) for q in
                                 ["中国", "经济", "教育", "科技", "市场", "银行", "北京", "美国"]
                                 if q in model.wv.key_to_index}}
        if family == "fasttext":
            unique = set()
            for w in model.wv.index_to_key:
                unique.update(compute_ngrams_bytes(w, model.wv.min_n, model.wv.max_n))
            # Mapping uses Gensim's own UTF-8-aware FNV hash; repeated word occurrences are deduplicated.
            from gensim.models.fasttext_inner import ft_hash_bytes
            buckets = {int(ft_hash_bytes(g)) % model.wv.bucket for g in unique}
            metrics.update(unique_subwords=len(unique), occupied_buckets=len(buckets),
                           collision_fraction=1 - len(buckets) / max(1, len(unique)),
                           occupied_fraction=len(buckets) / model.wv.bucket)
            metrics["oov_probes"] = []
            for word in ["金融科技", "农业现代化", "新能源产业", "数字经济", "科研创新"]:
                hashes = ft_ngram_hashes(word, model.wv.min_n, model.wv.max_n, model.wv.bucket)
                vec = model.wv.get_vector(word)
                metrics["oov_probes"].append({"word": word,
                    "in_vocabulary": word in model.wv.key_to_index,
                    "norm": float(np.linalg.norm(vec)), "ngram_count": len(hashes),
                    "occupied_bucket_fraction": sum(h in buckets for h in hashes) / max(1, len(hashes)),
                    "neighbors": model.wv.similar_by_vector(vec, topn=5)})
            idx = model.wv.key_to_index[first]
            composed = model.wv.vectors_vocab[idx].copy()
            for h in model.wv.buckets_word[idx]:
                composed += model.wv.vectors_ngrams[h]
            composed /= len(model.wv.buckets_word[idx]) + 1
            metrics["composition_max_error"] = float(np.max(np.abs(composed - model.wv[first])))
        write_json(metric_file, metrics)
        print("TRAINED", name, "seconds", round(train_s, 2), "vocab", len(model.wv), flush=True)
        all_metrics.append(metrics)
        del model
        gc.collect()
    return all_metrics


def document_vectors(docs, wv, idf=None):
    result = np.zeros((len(docs), wv.vector_size), dtype=np.float32)
    cache, raw_oov, token_count, represented = {}, 0, 0, 0
    is_fasttext = hasattr(wv, "vectors_ngrams") and wv.bucket > 0
    for i, doc in enumerate(docs):
        weights = 0.0
        for word, count in Counter(doc["tokens"]).items():
            token_count += count
            raw_oov += count * (word not in wv.key_to_index)
            if word not in cache:
                if word in wv.key_to_index or is_fasttext:
                    v = wv.get_vector(word)
                    cache[word] = v if np.isfinite(v).all() and np.linalg.norm(v) > 0 else None
                else:
                    cache[word] = None
            vec = cache[word]
            if vec is None:
                continue
            represented += count
            weight = count * (idf.get(word, 1.0) if idf else 1.0)
            result[i] += weight * vec
            weights += weight
        if weights:
            result[i] /= weights
    return result, {"vocab_oov_rate": raw_oov / max(1, token_count),
                    "represented_token_rate": represented / max(1, token_count),
                    "zero_document_count": int((np.linalg.norm(result, axis=1) == 0).sum())}
