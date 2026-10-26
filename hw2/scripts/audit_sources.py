"""Produce a versioned source map and local runtime snapshot, without vendoring Gensim."""
import ast
import hashlib
import inspect
import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import gensim
from gensim.models import FastText, Word2Vec
from gensim.models.word2vec_inner import FAST_VERSION
from threadpoolctl import threadpool_info

SOURCE = Path(gensim.__file__).parent
TARGETS = {
    "corpora/dictionary.py": ["doc2bow", "filter_extremes"],
    "models/tfidfmodel.py": ["initialize", "__getitem__"],
    "models/lsimodel.py": ["add_documents", "__getitem__", "stochastic_svd"],
    "interfaces.py": ["_apply", "__iter__"],
    "models/word2vec.py": ["build_vocab", "prepare_vocab", "make_cum_table", "create_binary_tree",
                           "train", "_job_producer", "_worker_loop", "_get_next_alpha", "_check_corpus_sanity"],
    "models/fasttext.py": ["__init__", "_do_train_job", "estimate_memory", "get_vector",
                          "adjust_vectors", "recalc_char_ngram_buckets", "ft_ngram_hashes"],
    "utils.py": ["save", "_smart_save", "_load_specials"],
}
records, markdown = [], ["# Gensim 4.3.0 源码位置与运行环境\n",
    "以下位置根据本地安装文件解析生成。链接指向相同版本标签，函数名用于定位。\n"]
for relative, functions in TARGETS.items():
    path = SOURCE / relative
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in functions:
            # Restrict generic __init__/save names to relevant classes where practical.
            if relative == "models/fasttext.py" and node.name == "__init__" and node.lineno != 274:
                continue
            record = {"file": relative, "function": node.name, "start": node.lineno,
                      "end": node.end_lineno, "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
                      "url": f"https://github.com/piskvorky/gensim/blob/4.3.0/gensim/{relative}#L{node.lineno}"}
            records.append(record)
            markdown.append(f"## {relative}：{node.name}\n\n"
                            f"本地行号：{node.lineno}—{node.end_lineno}。[版本源码]({record['url']})\n")
    markdown.append("")
for relative in ["models/fasttext_inner.pyx", "models/word2vec_inner.pyx", "models/fasttext_corpusfile.pyx"]:
    path = SOURCE / relative
    text = path.read_text(encoding="utf-8")
    import re
    for i, line in enumerate(text.splitlines(), 1):
        if re.search(r"^(?:cdef|def) .*?(?:fasttext_fast_sentence_sg_neg|train_batch_any|compute_ngrams_bytes|ft_hash_bytes|train_epoch_sg)\(", line):
            records.append({"file": relative, "declaration": line.strip(), "start": i,
                            "url": f"https://github.com/piskvorky/gensim/blob/4.3.0/gensim/{relative}#L{i}"})
            markdown.append(f"- [{relative}:{i}]({records[-1]['url']})：`{line.strip()}`\n")
packages = {name: metadata.version(name) for name in ["gensim", "numpy", "scipy", "scikit-learn",
             "matplotlib", "pandas", "torch", "threadpoolctl"]}
env = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
       "processor": platform.processor(), "logical_cpu_count": os.cpu_count(), "packages": packages,
       "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"), "gensim_FAST_VERSION": FAST_VERSION,
       "threadpools": threadpool_info(), "FastText_signature": str(inspect.signature(FastText.__init__)),
       "Word2Vec_signature": str(inspect.signature(Word2Vec.__init__))}
try:
    import psutil
    env["ram_gib"] = psutil.virtual_memory().total / 2**30
except ImportError:
    pass
(ROOT / "results/environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
(ROOT / "results/source_map.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
(ROOT / "docs/source-map.md").write_text("\n".join(markdown), encoding="utf-8")
print("Audited", len(records), "source entry points.")
