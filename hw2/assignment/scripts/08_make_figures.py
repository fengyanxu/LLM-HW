"""Generate the figures used by the HW2 report."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hw2a.common import CONFIG, DATA, FIGURES, RESULTS, read_json  # noqa: E402

CLASSES = CONFIG["classification"]["classes"]
STAGES = CONFIG["classification"]["stages"]
BLUE, ORANGE, GRAY = "#2f5d8a", "#d97742", "#6b7280"

for candidate in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
    if Path(candidate).exists():
        font_manager.fontManager.addfont(candidate)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=candidate).get_name()
        break
plt.rcParams.update({"axes.unicode_minus": False, "figure.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name: str, headers=None, rows=None) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    if headers is not None:
        with (FIGURES / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            writer.writerows(rows)
    fig.savefig(FIGURES / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def data_overview() -> None:
    stats = read_json(DATA / "thucnews" / "clean_stats.json")
    manifest = read_json(DATA / "peopledaily" / "classification" / "manifest.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    labels = ["原始抽样", "清洗后"]
    values = [stats["raw_articles"], stats["kept_articles"]]
    bars = ax.bar(labels, values, color=[GRAY, BLUE], width=0.55)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}", ha="center", va="bottom")
    ax.set_title(f"THUCNews 词向量语料（{stats['sentences']:,} 句 / {stats['tokens']:,} 词）")
    ax.set_ylabel("文章数")
    ax.set_ylim(0, max(values) * 1.15)

    ax = axes[1]
    train = [manifest["stages"]["8"]["per_class"][c] - 12 for c in CLASSES] if "stages" in manifest else [48] * 8
    test = [12] * 8
    x = np.arange(len(CLASSES))
    ax.bar(x, train, color=BLUE, label="训练 48")
    ax.bar(x, test, bottom=train, color=ORANGE, label="测试 12")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=0)
    ax.set_title("人民日报 8 主题分类集：每类 60 篇")
    ax.set_ylabel("文章数")
    ax.legend(frameon=False, ncol=2)
    save(fig, "01_data_overview", ["split", "articles"], [["raw", stats["raw_articles"]], ["clean", stats["kept_articles"]]])


def similarity_figure() -> None:
    analysis = read_json(RESULTS / "embedding_analysis.json")
    names = [("word2vec", "Word2Vec"), ("fasttext", "FastText")]
    within = [analysis[key]["similarity"]["within_mean"] for key, _ in names]
    cross = [analysis[key]["similarity"]["cross_mean"] for key, _ in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.18, within, width=0.34, color=BLUE, label="组内平均相似度")
    ax.bar(x + 0.18, cross, width=0.34, color=ORANGE, label="组间平均相似度")
    for i, (w, c) in enumerate(zip(within, cross)):
        ax.text(i - 0.18, w, f"{w:.4f}", ha="center", va="bottom")
        ax.text(i + 0.18, c, f"{c:.4f}", ha="center", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in names])
    ax.set_ylabel("余弦相似度")
    ax.set_title("四组主题词的组内 / 组间平均余弦相似度")
    ax.legend(frameon=False)
    ax.set_ylim(0, max(within + cross) * 1.2)
    save(fig, "02_embedding_similarity", ["model", "within", "cross"], [[label, w, c] for (_, label), w, c in zip(names, within, cross)])


def classification_curve() -> None:
    data = read_json(RESULTS / "classification.json")
    rows = {row["stage"]: {} for row in data["results"] if "test_n" in row}
    for row in data["results"]:
        if "test_n" in row:
            rows[row["stage"]][row["condition"]] = row
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, title in ((axes[0], "accuracy", "准确率"), (axes[1], "macro_f1", "Macro-F1")):
        for condition, color, label in (("pretrained", BLUE, "导入预训练词向量"), ("scratch", ORANGE, "从零训练")):
            values = [rows[stage][condition][metric] for stage in STAGES]
            ax.plot(STAGES, values, marker="o", color=color, label=label, linewidth=2)
            for stage, value in zip(STAGES, values):
                offset = (0, -14) if stage == STAGES[0] and condition == "scratch" else (0, 8)
                ax.annotate(f"{value:.3f}", (stage, value), textcoords="offset points", xytext=offset, ha="center", fontsize=9)
        ax.set_xticks(STAGES)
        ax.set_xlabel("类别数")
        ax.set_ylabel(title)
        ax.set_ylim(0, 1.08)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, loc="lower left")
    save(fig, "03_classification_curve", ["stage", "condition", "accuracy", "macro_f1"], [[stage, condition, rows[stage][condition]["accuracy"], rows[stage][condition]["macro_f1"]] for stage in STAGES for condition in ("pretrained", "scratch")])


def confusion_figure() -> None:
    data = read_json(RESULTS / "classification.json")
    row = next(item for item in data["results"] if item.get("stage") == 8 and item.get("condition") == "pretrained")
    matrix = np.array([[row["confusion"][t][p] for p in CLASSES] for t in CLASSES])
    fig, ax = plt.subplots(figsize=(8, 6.5))
    image = ax.imshow(matrix, cmap="Blues")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            value = matrix[i, j]
            ax.text(j, i, str(value), ha="center", va="center", color="white" if value > matrix.max() / 2 else "#1f2937")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=35, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("8 类 FastText（导入预训练词向量）混淆矩阵")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="文章数")
    save(fig, "04_confusion_8", ["true\\pred", *CLASSES], [[t, *matrix[i].tolist()] for i, t in enumerate(CLASSES)])


def sensitivity_figure() -> None:
    data = read_json(RESULTS / "epoch_sensitivity.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, title in ((axes[0], "accuracy", "准确率"), (axes[1], "macro_f1", "Macro-F1")):
        for condition, color, label in (("pretrained", BLUE, "导入预训练词向量"), ("scratch", ORANGE, "从零训练")):
            rows = [row for row in data if row["condition"] == condition]
            ax.plot([row["epoch"] for row in rows], [row[metric] for row in rows], marker="o", color=color, label=label, linewidth=2)
            for row in rows:
                ax.annotate(f"{row[metric]:.3f}", (row["epoch"], row[metric]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
        ax.set_xticks([25, 50, 100])
        ax.set_xlabel("训练轮数")
        ax.set_ylabel(title)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"8 类：{title}随训练轮数变化")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, loc="lower right")
    save(fig, "05_epoch_sensitivity", ["condition", "epoch", "accuracy", "macro_f1"], [[row["condition"], row["epoch"], row["accuracy"], row["macro_f1"]] for row in data])


def main() -> None:
    data_overview()
    similarity_figure()
    classification_curve()
    confusion_figure()
    sensitivity_figure()
    print("图表已生成：", [path.name for path in sorted(FIGURES.glob("*.png"))], flush=True)


if __name__ == "__main__":
    main()
