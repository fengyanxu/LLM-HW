"""Cross-check saved results, split isolation, plots and report links."""
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from newsvec.common import read_json, read_docs

config = read_json(ROOT / "configs/experiment.json")
manifest = pd.read_csv(ROOT / "results/split_manifest.csv", dtype={"id": str, "date": str, "page": str})
assert manifest.id.is_unique
assert manifest.content_hash.is_unique
for split, upper, lower in [("train", 20, 1), ("validation", 25, 21), ("test", 31, 26)]:
    days = manifest[manifest.split == split].date.str[-2:].astype(int)
    assert days.between(lower, upper).all()
records = pd.read_csv(ROOT / "results/combined_classification.csv")
assert records.groupby("n_classes").size().to_dict() == {2: 11, 4: 11, 6: 11}
assert records.macro_f1.between(0, 1).all()
assert records.accuracy.between(0, 1).all()
for v in config["variants"]:
    r = read_json(ROOT / f"results/embedding_{v['name']}.json")
    assert r["save_load_max_error"] == 0
    if v["family"] == "fasttext":
        assert r["composition_max_error"] < 1e-6
    assert r["parameters"]["workers"] == 1
for n in config["stages"]:
    for method in ["tfidf", "ft_sg_idf", "paper_bigram"]:
        r = read_json(ROOT / f"results/classification_{n}_{method}.json")
        cm = np.array(r["confusion_matrix"])
        assert int(cm.sum()) == r["summary"]["n_test"]
        assert np.isclose(np.trace(cm) / cm.sum(), r["summary"]["accuracy"])
report_path = ROOT / "docs/P137实验报告.md"
report = report_path.read_text(encoding="utf-8")
assert not re.search(r"(?<!\$)\$(?!\$)", report), "Use paired display-math delimiters"
assert report.count("$$") % 2 == 0
assert not re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", report), "Unexpected control characters"
assert not re.search(r"不是没有|并非没有|不能不|不可不|不得不|不能否认|不可否认", report)
for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", report):
    assert target.startswith("https://"), target
    parsed = urlparse(target)
    if parsed.netloc in {"github.com", "raw.githubusercontent.com"} and parsed.path.startswith("/Can1017/LLM-HW/"):
        if "/main/hw2/" in parsed.path:
            relative = unquote(parsed.path.split("/main/hw2/", 1)[1])
            assert (ROOT / relative).exists(), target
        else:
            assert parsed.path.rstrip("/") == "/Can1017/LLM-HW", target
assert len(list((ROOT / "figures").glob("*.png"))) == 12
assert len(list((ROOT / "figures").glob("*.svg"))) == 12
assert (ROOT / "docs/实验展示.html").exists()
result = {"status": "passed", "checks": ["article and content isolation", "date-group splits",
          "33 classification rows", "confusion matrix / accuracy consistency", "15 model roundtrips",
          "FastText vector composition", "math delimiters", "report links", "12 PNG/SVG figures"]}
(ROOT / "results/verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
