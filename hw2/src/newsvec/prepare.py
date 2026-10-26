"""Parse article IDs, remove pronunciation/layout metadata, and isolate data splits."""
import collections
import hashlib
import random
import re
import shutil
import subprocess
import unicodedata
from pathlib import Path

from .common import ROOT, read_json, sha256, write_json

ID = re.compile(r"^(\d{8})-(\d{2})-(\d{3})-(\d{3})\s+(.*)$")
PRONUNCIATION = re.compile(r"\{[^{}]*\}")
WORD = re.compile(r"[\u3400-\u9fffA-Za-z]")
END = set("。！？；.!?;")


def clean_line(text):
    text = unicodedata.normalize("NFKC", PRONUNCIATION.sub("", text))
    sentences, current = [], []
    for token in text.split():
        if token in END:
            if current:
                sentences.append(current)
                current = []
        elif WORD.search(token) and "\ufffd" not in token:
            # Legacy punctuation is sometimes attached to a presegmented word.
            word = re.sub(r"[^\u3400-\u9fffA-Za-z0-9]", "", token)
            if WORD.search(word):
                current.append(word)
    if current:
        sentences.append(current)
    return sentences


def decode_file(path):
    raw = Path(path).read_bytes()
    try:
        return raw.decode("utf-8"), "utf-8", 0
    except UnicodeDecodeError:
        text = raw.decode("gb18030", errors="replace")
        return text, "gb18030", text.count("\ufffd")


def parse_articles(path):
    text, encoding, replacement_count = decode_file(path)
    docs = {}
    malformed = 0
    for line in text.splitlines():
        m = ID.match(line)
        if not m:
            malformed += bool(line.strip())
            continue
        date, page, article, paragraph, body = m.groups()
        doc_id = f"{date}-{page}-{article}"
        doc = docs.setdefault(doc_id, {"id": doc_id, "date": date, "page": page,
                                      "sentences": [], "title": ""})
        cleaned = clean_line(body)
        if not doc["title"]:
            doc["title"] = "".join(w for sent in cleaned for w in sent)[:100]
        doc["sentences"].extend(cleaned)
    for doc in docs.values():
        doc["tokens"] = [w for sent in doc["sentences"] for w in sent]
        doc["content_hash"] = hashlib.sha256(" ".join(doc["tokens"]).encode()).hexdigest()
    return list(docs.values()), {"encoding": encoding, "replacement_chars": replacement_count,
                               "malformed_lines": malformed, "raw_articles": len(docs)}


def extract_archive(archive, raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)
    if len(list(raw_dir.rglob("*.txt"))) >= 6:
        return
    tar = shutil.which("tar")
    windows_tar = Path("C:/Windows/System32/tar.exe")
    if windows_tar.exists():
        tar = str(windows_tar)
    if tar is None:
        raise RuntimeError("Install a libarchive-based tar or extract the six text files into data/raw.")
    listing = subprocess.check_output([tar, "-tf", str(archive)], text=True, errors="replace")
    for member in listing.splitlines():
        if member.startswith(("/", "\\")) or ".." in Path(member).parts:
            raise ValueError("Archive contains an unsafe member path")
    subprocess.run([tar, "-xf", str(archive), "-C", str(raw_dir)], check=True)


def prepare(archive, config):
    raw_dir, out = ROOT / "data/raw", ROOT / "data/processed"
    extract_archive(Path(archive).resolve(), raw_dir)
    out.mkdir(parents=True, exist_ok=True)
    sources, by_month, seen = [], {}, set()
    duplicate_count, filtered_count = 0, 0
    for path in sorted(raw_dir.rglob("*.txt")):
        docs, info = parse_articles(path)
        info.update(name=path.name, sha256=sha256(path), bytes=path.stat().st_size)
        sources.append(info)
        for doc in docs:
            length = len(doc["tokens"])
            if not config["min_doc_tokens"] <= length <= config["max_doc_tokens"]:
                filtered_count += 1
                continue
            if doc["content_hash"] in seen:
                duplicate_count += 1
                continue
            seen.add(doc["content_hash"])
            by_month.setdefault(doc["date"][:6], []).append(doc)
    january = by_month["199801"]
    splits = {"train": [], "validation": [], "test": []}
    for doc in january:
        if doc["page"] not in config["classes"]:
            continue
        day = int(doc["date"][-2:])
        split = "train" if day <= config["train_last_day"] else (
            "validation" if day <= config["validation_last_day"] else "test")
        doc["split"] = split
        splits[split].append(doc)
    february = sorted(by_month["199802"], key=lambda d: d["id"])
    random.Random(config["seed"]).shuffle(february)
    external, tokens = [], 0
    for doc in february:
        if tokens >= config["external_token_budget"]:
            break
        external.append(doc)
        tokens += len(doc["tokens"])
    external.sort(key=lambda d: d["id"])
    for split, docs in splits.items():
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for doc in docs:
                f.write(__import__("json").dumps(doc, ensure_ascii=False) + "\n")
    embedding_docs = external + splits["train"]
    with (out / "sentences.txt").open("w", encoding="utf-8") as f:
        for doc in embedding_docs:
            for sent in doc["sentences"]:
                if len(sent) >= 2:
                    f.write(" ".join(sent) + "\n")
    hashes = {s: {d["content_hash"] for d in docs} for s, docs in splits.items()}
    assert not (hashes["train"] & hashes["validation"] or hashes["train"] & hashes["test"]
                or hashes["validation"] & hashes["test"])
    embedding_ids = {d["id"] for d in embedding_docs}
    assert not embedding_ids.intersection(d["id"] for d in splits["validation"] + splits["test"])
    counts = {s: dict(sorted(collections.Counter(d["page"] for d in docs).items()))
              for s, docs in splits.items()}
    stats = {"archive_sha256": sha256(archive), "sources": sources,
             "eligible_by_month": {m: len(d) for m, d in by_month.items()},
             "removed_duplicates": duplicate_count, "removed_by_length": filtered_count,
             "split_counts": counts, "external_articles": len(external), "external_tokens": tokens,
             "embedding_articles": len(embedding_docs),
             "embedding_tokens": sum(len(d["tokens"]) for d in embedding_docs),
             "january_lengths": [{"id": d["id"], "page": d["page"], "tokens": len(d["tokens"])}
                                 for d in january if d["page"] in config["classes"]],
             "split_fingerprints": {s: hashlib.sha256("\n".join(sorted(d["id"] for d in docs)).encode()).hexdigest()
                                    for s, docs in splits.items()},
             "evaluation_protocol": "January article classification, date-grouped split; February external unlabeled pretraining, not a future-news forecast."}
    write_json(ROOT / "results/data_stats.json", stats)
    # Publishing IDs and hashes permits exact split auditing without distributing articles.
    import csv
    with (ROOT / "results/split_manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "page", "date", "split", "tokens", "content_hash"])
        writer.writeheader()
        for split, docs in splits.items():
            for d in docs:
                writer.writerow({**{k: d[k] for k in ["id", "page", "date", "content_hash"]},
                                 "split": split, "tokens": len(d["tokens"])})
    write_json(ROOT / "results/external_manifest.json", [{"id": d["id"], "content_hash": d["content_hash"]}
                                                       for d in external])
    print("PREPARED", {s: len(d) for s, d in splits.items()}, "embedding tokens", stats["embedding_tokens"], flush=True)
    return stats
