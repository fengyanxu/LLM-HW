"""Validation-only model selection and fixed, article/date-isolated evaluation."""
import time
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from gensim import corpora, matutils
from gensim.models import FastText, Word2Vec, LsiModel, TfidfModel
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC

from .common import ROOT, read_docs, read_json, write_json
from .embeddings import document_vectors


def classifier(c, dense):
    # Dense document vectors have far fewer columns than rows: solve the primal problem.
    svm = LinearSVC(C=c, class_weight="balanced", random_state=42, max_iter=10000, dual=not dense)
    return make_pipeline(StandardScaler(), svm) if dense else svm


def bootstrap_by_day(y, pred, dates, labels, repeats, seed):
    rng = np.random.default_rng(seed)
    unique = sorted(set(dates))
    groups = {d: np.flatnonzero(np.array(dates) == d) for d in unique}
    scores = []
    for _ in range(repeats):
        idx = np.concatenate([groups[d] for d in rng.choice(unique, len(unique), replace=True)])
        scores.append(f1_score(y[idx], pred[idx], labels=labels, average="macro", zero_division=0))
    return list(np.quantile(scores, [0.025, 0.975]))


def fit_select(name, X, docs, classes, config, dense, tuning_only=False):
    y = {s: np.array([d["page"] for d in docs[s]]) for s in docs}
    best, best_score, best_c, trials = None, -1, None, []
    begin = time.perf_counter()
    for c in config["classifier_C"]:
        clf = classifier(c, dense)
        clf.fit(X["train"], y["train"])
        score = f1_score(y["validation"], clf.predict(X["validation"]), labels=classes,
                         average="macro", zero_division=0)
        trials.append({"C": c, "validation_macro_f1": score})
        if score > best_score:
            best, best_score, best_c = clf, score, c
    row = {"method": name, "n_classes": len(classes), "validation_macro_f1": best_score,
           "C": best_c, "fit_select_seconds": time.perf_counter() - begin,
           "n_train": len(y["train"]), "n_validation": len(y["validation"])}
    if tuning_only:
        return row, best
    begin = time.perf_counter()
    pred = best.predict(X["test"])
    inference_s = time.perf_counter() - begin
    ci = bootstrap_by_day(y["test"], pred, [d["date"] for d in docs["test"]], classes,
                          config["bootstrap_repeats"], config["seed"])
    row.update(accuracy=accuracy_score(y["test"], pred),
               macro_f1=f1_score(y["test"], pred, labels=classes, average="macro", zero_division=0),
               weighted_f1=f1_score(y["test"], pred, labels=classes, average="weighted", zero_division=0),
               macro_f1_ci_low=ci[0], macro_f1_ci_high=ci[1], n_test=len(pred),
               predict_seconds=inference_s)
    detailed = {"summary": row, "validation_trials": trials,
                "confusion_matrix": confusion_matrix(y["test"], pred, labels=classes).tolist(),
                "classes": classes,
                "classification_report": classification_report(y["test"], pred, labels=classes,
                                                                output_dict=True, zero_division=0)}
    write_json(ROOT / f"results/classification_{len(classes)}_{name}.json", detailed)
    write_json(ROOT / f"results/predictions/{len(classes)}_{name}.json", [
        {"id": d["id"], "date": d["date"], "gold": gold, "prediction": p,
         "title": d["title"], "tokens": len(d["tokens"])}
        for d, gold, p in zip(docs["test"], y["test"], pred)])
    joblib.dump(best, ROOT / f"models/classifier_{len(classes)}_{name}.joblib")
    return row, best


def evaluate(config):
    (ROOT / "results/features").mkdir(parents=True, exist_ok=True)
    all_docs = {s: read_docs(ROOT / f"data/processed/{s}.jsonl")
                for s in ["train", "validation", "test"]}
    # TF-IDF and document IDF are learned exclusively from January training articles.
    texts = {s: [" ".join(d["tokens"]) for d in docs] for s, docs in all_docs.items()}
    tfidf = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False,
                           min_df=2, sublinear_tf=True, ngram_range=(1, 2), max_features=60000)
    sparse = {"train": tfidf.fit_transform(texts["train"])}
    sparse.update({s: tfidf.transform(texts[s]) for s in ["validation", "test"]})
    joblib.dump(tfidf, ROOT / "models/tfidf.joblib")
    idf = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))
    dictionary = corpora.Dictionary(d["tokens"] for d in all_docs["train"])
    dictionary.filter_extremes(no_below=2, no_above=1.0, keep_n=40000)
    bows = {s: [dictionary.doc2bow(d["tokens"]) for d in ds] for s, ds in all_docs.items()}
    gtfidf = TfidfModel(bows["train"])
    lsi = LsiModel(gtfidf[bows["train"]], id2word=dictionary, num_topics=64, random_seed=config["seed"])
    lsi_x = {s: Normalizer().fit_transform(matutils.corpus2dense(lsi[gtfidf[corpus]], 64).T)
             for s, corpus in bows.items()}
    dictionary.save(str(ROOT / "models/dictionary.dict"))
    gtfidf.save(str(ROOT / "models/gensim_tfidf.model"))
    lsi.save(str(ROOT / "models/lsi.model"))
    write_json(ROOT / "results/lsi.json", {"components": 64,
        "singular_values": lsi.projection.s.tolist(), "features": len(dictionary),
        "topics": lsi.show_topics(num_topics=8, num_words=8, formatted=False)})
    all_features = {"tfidf": (sparse, False), "lsi": (lsi_x, True)}
    coverages = {}
    for variant in config["variants"]:
        name, family = variant["name"], variant["family"]
        model = (FastText if family == "fasttext" else Word2Vec).load(str(ROOT / f"models/{name}.model"))
        for weighted in ([False, True] if name in {"w2v_sg", "ft_sg"} else [False]):
            feature_name = name + ("_idf" if weighted else "")
            features = {}
            coverages[feature_name] = {}
            for split, docs in all_docs.items():
                features[split], coverages[feature_name][split] = document_vectors(
                    docs, model.wv, idf if weighted else None)
            all_features[feature_name] = (features, True)
        del model
    write_json(ROOT / "results/coverage.json", coverages)
    rows, sensitivity = [], []
    main_names = ["tfidf", "lsi", "w2v_cbow", "w2v_sg", "w2v_sg_idf", "ft_cbow", "ft_sg", "ft_sg_idf"]
    for n in config["stages"]:
        classes = config["classes"][:n]
        indices = {s: np.array([i for i, d in enumerate(ds) if d["page"] in classes])
                   for s, ds in all_docs.items()}
        docs = {s: [all_docs[s][i] for i in idx] for s, idx in indices.items()}
        for name in main_names:
            features, dense = all_features[name]
            X = {s: v[indices[s]] for s, v in features.items()}
            row, _ = fit_select(name, X, docs, classes, config, dense)
            rows.append(row)
            print("CLASSIFIED", n, name, "val", round(row["validation_macro_f1"], 4),
                  "test", round(row["macro_f1"], 4), flush=True)
        y = {s: np.array([d["page"] for d in ds]) for s, ds in docs.items()}
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(np.zeros((len(y["train"]), 1)), y["train"])
        p = dummy.predict(np.zeros((len(y["test"]), 1)))
        rows.append({"method": "majority", "n_classes": n,
                     "accuracy": accuracy_score(y["test"], p),
                     "macro_f1": f1_score(y["test"], p, labels=classes, average="macro", zero_division=0),
                     "n_train": len(y["train"]), "n_test": len(p)})
    classes = config["classes"]
    for name, (features, dense) in all_features.items():
        if name.startswith("ft_") and name not in {"ft_cbow", "ft_sg_idf"}:
            row, _ = fit_select(name, features, all_docs, classes, config, dense, tuning_only=True)
            metric = read_json(ROOT / f"results/embedding_{name}.json")
            row.update(train_seconds=metric["train_seconds"], array_mib=metric["array_mib"],
                       parameters=metric["parameters"], collision_fraction=metric["collision_fraction"])
            sensitivity.append(row)
    pd.DataFrame(rows).to_csv(ROOT / "results/classification.csv", index=False, encoding="utf-8-sig")
    write_json(ROOT / "results/sensitivity.json", sensitivity)
    selected = max(sensitivity, key=lambda r: r["validation_macro_f1"])
    if selected["method"] != "ft_sg":
        name = selected["method"]
        row, _ = fit_select("selected_" + name, all_features[name][0], all_docs, classes, config, True)
        pd.DataFrame([row]).to_csv(ROOT / "results/selected_fasttext.csv", index=False, encoding="utf-8-sig")
    write_json(ROOT / "results/selection.json", {"selected_fasttext": selected["method"],
        "selection_metric": "six-class validation Macro-F1", "selected": selected,
        "fixed_test_policy": "Sensitivity variants use validation scores; the selected variant is then evaluated on test."})
    # Locally saved numeric features support visualization without retraining models.
    np.savez_compressed(ROOT / "results/features/dense_projection.npz", X=all_features["ft_sg_idf"][0]["test"],
                        y=np.array([d["page"] for d in all_docs["test"]]))
    return rows
