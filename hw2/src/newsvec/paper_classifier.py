"""Educational implementation of Joulin et al.'s supervised averaged-feature architecture.

This is a PyTorch structural reproduction, with full softmax for 2/4/6 classes;
it does not claim to reproduce the original C++ implementation or its throughput.
"""
import copy
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn

from .common import ROOT, read_docs, write_json
from .evaluate import bootstrap_by_day


def fnv1a(value):
    h = 2166136261
    for b in value.encode("utf-8"):
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return h


class AveragedFeatureClassifier(nn.Module):
    def __init__(self, feature_count, dim, class_count):
        super().__init__()
        self.features = nn.EmbeddingBag(feature_count, dim, mode="mean")
        self.output = nn.Linear(dim, class_count, bias=False)
        nn.init.uniform_(self.features.weight, -1 / dim, 1 / dim)
        nn.init.zeros_(self.output.weight)

    def forward(self, ids, offsets):
        return self.output(self.features(ids, offsets))


def batch(ids, selected):
    rows = [ids[i] for i in selected]
    lengths = [len(r) for r in rows]
    flat = torch.tensor([x for r in rows for x in r], dtype=torch.long)
    offsets = torch.tensor(np.cumsum([0] + lengths[:-1]), dtype=torch.long)
    return flat, offsets


def predict(model, ids, size):
    model.eval()
    preds = []
    with torch.no_grad():
        for begin in range(0, len(ids), size):
            x, offsets = batch(ids, range(begin, min(begin + size, len(ids))))
            preds.extend(model(x, offsets).argmax(1).tolist())
    return np.array(preds)


def run_paper_classifier(config):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    params = config["paper_classifier"]
    all_docs = {s: read_docs(ROOT / f"data/processed/{s}.jsonl") for s in ["train", "validation", "test"]}
    freq = Counter(w for d in all_docs["train"] for w in d["tokens"])
    kept = sorted(w for w, count in freq.items() if count >= 2)
    vocab = {w: i + 1 for i, w in enumerate(kept)}
    word_count = max(vocab.values(), default=0) + 1
    rows, histories = [], []
    for n in config["stages"]:
        classes = config["classes"][:n]
        docs = {s: [d for d in ds if d["page"] in classes] for s, ds in all_docs.items()}
        y = {s: np.array([classes.index(d["page"]) for d in ds]) for s, ds in docs.items()}
        for bigrams in [False, True]:
            encoded = {}
            for s, ds in docs.items():
                encoded[s] = []
                for d in ds:
                    words = d["tokens"][:params["max_tokens"]]
                    features = [vocab.get(w, 0) for w in words]
                    if bigrams:
                        features += [word_count + fnv1a(a + "\x1f" + b) % params["bucket"]
                                     for a, b in zip(words, words[1:])]
                    encoded[s].append(features or [0])
            feature_count = word_count + (params["bucket"] if bigrams else 0)
            best = None
            trials = []
            start = time.perf_counter()
            for lr in params["lr_grid"]:
                torch.manual_seed(config["seed"])
                rng = np.random.default_rng(config["seed"])
                model = AveragedFeatureClassifier(feature_count, params["dim"], n)
                opt = torch.optim.SGD(model.parameters(), lr=lr)
                weights = torch.tensor([len(y["train"]) / (n * np.sum(y["train"] == i))
                                        for i in range(n)], dtype=torch.float32)
                loss_fn = nn.CrossEntropyLoss(weight=weights)
                steps = params["epochs"] * int(np.ceil(len(y["train"]) / params["batch_size"]))
                step = 0
                for epoch in range(params["epochs"]):
                    model.train()
                    order = rng.permutation(len(y["train"]))
                    losses = []
                    for begin in range(0, len(order), params["batch_size"]):
                        idx = order[begin:begin + params["batch_size"]]
                        x, offsets = batch(encoded["train"], idx)
                        opt.param_groups[0]["lr"] = lr * max(0.0001, 1 - step / steps)
                        opt.zero_grad()
                        loss = loss_fn(model(x, offsets), torch.tensor(y["train"][idx], dtype=torch.long))
                        # Sum document gradients to retain document-wise SGD's update scale.
                        (loss * len(idx)).backward()
                        opt.step()
                        losses.append(loss.item())
                        step += 1
                    p = predict(model, encoded["validation"], params["batch_size"])
                    score = f1_score(y["validation"], p, labels=list(range(n)), average="macro", zero_division=0)
                    trial = {"n_classes": n, "bigrams": bigrams, "lr": lr, "epoch": epoch + 1,
                             "train_cross_entropy": float(np.mean(losses)), "validation_macro_f1": score}
                    trials.append(trial)
                    if best is None or score > best["score"]:
                        best = {"score": score, "state": copy.deepcopy(model.state_dict()), "lr": lr, "epoch": epoch + 1}
            model.load_state_dict(best["state"])
            p = predict(model, encoded["test"], params["batch_size"])
            ci = bootstrap_by_day(y["test"], p, [d["date"] for d in docs["test"]], list(range(n)),
                                  config["bootstrap_repeats"], config["seed"])
            name = "paper_bigram" if bigrams else "paper_unigram"
            row = {"method": name, "n_classes": n, "validation_macro_f1": best["score"],
                   "accuracy": accuracy_score(y["test"], p),
                   "macro_f1": f1_score(y["test"], p, labels=list(range(n)), average="macro", zero_division=0),
                   "macro_f1_ci_low": ci[0], "macro_f1_ci_high": ci[1],
                   "selected_lr": best["lr"], "selected_epoch": best["epoch"],
                   "fit_select_seconds": time.perf_counter() - start, "feature_count": feature_count,
                   "n_train": len(y["train"]), "n_validation": len(y["validation"]), "n_test": len(p)}
            rows.append(row)
            histories.extend(trials)
            torch.save({"state": best["state"], "vocabulary": vocab, "parameters": params,
                        "classes": classes, "bigrams": bigrams, "word_count": word_count},
                       ROOT / f"models/{n}_{name}.pt")
            write_json(ROOT / f"results/classification_{n}_{name}.json", {
                "summary": row, "classes": classes,
                "confusion_matrix": confusion_matrix(y["test"], p, labels=list(range(n))).tolist()})
            print("PAPER CLASSIFIER", n, name, "val", round(best["score"], 4),
                  "test", round(row["macro_f1"], 4), flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results/paper_classification.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(histories).to_csv(ROOT / "results/paper_training.csv", index=False, encoding="utf-8-sig")
    return rows
