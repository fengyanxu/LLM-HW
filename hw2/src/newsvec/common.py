import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_docs(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class SentenceCorpus:
    """Fresh file traversal on every vocabulary/training pass."""

    def __init__(self, path):
        self.path = Path(path)

    def __iter__(self):
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                yield line.split()
