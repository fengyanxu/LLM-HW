"""Publication-style charts; every plotted value comes from saved experimental results."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .common import ROOT, read_json, write_json

LABELS = {"majority": "多数类", "tfidf": "TF-IDF", "lsi": "Gensim LSI",
          "w2v_cbow": "W2V CBOW", "w2v_sg": "W2V SG", "w2v_sg_idf": "W2V SG+IDF",
          "ft_cbow": "FT CBOW", "ft_sg": "FT SG", "ft_sg_idf": "FT SG+IDF",
          "paper_unigram": "监督 FT 单词", "paper_bigram": "监督 FT 双词"}
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]


def setup():
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    else:
        plt.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 15, "axes.labelsize": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.unicode_minus": False, "figure.facecolor": "white",
                         "savefig.facecolor": "white", "svg.fonttype": "path"})
    (ROOT / "figures").mkdir(exist_ok=True)


def save(fig, name):
    fig.savefig(ROOT / f"figures/{name}.png", dpi=180, bbox_inches="tight")
    svg_path = ROOT / f"figures/{name}.svg"
    fig.savefig(svg_path, bbox_inches="tight")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines())
                        + "\n", encoding="utf-8")
    plt.close(fig)


def plot_all(config):
    setup()
    stats = read_json(ROOT / "results/data_stats.json")
    base = pd.read_csv(ROOT / "results/classification.csv")
    paper = pd.read_csv(ROOT / "results/paper_classification.csv")
    combined = pd.concat([base, paper], ignore_index=True)
    combined.to_csv(ROOT / "results/combined_classification.csv", index=False, encoding="utf-8-sig")
    embeddings = [read_json(ROOT / f"results/embedding_{v['name']}.json") for v in config["variants"]]
    sensitivity = read_json(ROOT / "results/sensitivity.json")
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    bottom = np.zeros(6)
    pages = config["classes"]
    for split, label, color in zip(["train", "validation", "test"], ["训练", "验证", "测试"], COLORS):
        values = [stats["split_counts"][split].get(p, 0) for p in pages]
        ax.bar(pages, values, bottom=bottom, color=color, label=label, width=.65)
        for x, y, b in zip(range(6), values, bottom):
            ax.text(x, b + y / 2, str(y), ha="center", va="center", color="white", fontsize=10)
        bottom += values
    ax.set(xlabel="原始版面编号", ylabel="文章数", title="图 1  按日期划分后的类别与样本分布")
    ax.legend(frameon=False, ncol=3)
    save(fig, "01_data_split")

    lengths = pd.DataFrame(stats["january_lengths"])
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    ax.boxplot([lengths[lengths.page == p].tokens.to_numpy() for p in pages], labels=pages,
               patch_artist=True, boxprops={"facecolor": "#A0C4E8"},
               medianprops={"color": "#D55E00", "linewidth": 2}, showfliers=False)
    ax.set_yscale("log")
    ax.set(xlabel="版面编号", ylabel="清洗后词数（对数刻度）", title="图 2  各版面文章长度分布")
    save(fig, "02_document_lengths")

    fig, ax = plt.subplots(figsize=(10, 5.2), layout="constrained")
    methods = ["tfidf", "lsi", "w2v_sg_idf", "ft_sg_idf", "paper_bigram"]
    for method, color in zip(methods, COLORS):
        sub = combined[combined.method == method].sort_values("n_classes")
        ax.plot(sub.n_classes, sub.macro_f1, "o-", label=LABELS[method], color=color, linewidth=2)
    ax.set(xlabel="类别数量", ylabel="测试 Macro-F1", ylim=(0, 1), xticks=[2, 4, 6],
           title="图 3  类别逐步增加时的分类表现")
    ax.grid(axis="y", alpha=.18)
    ax.legend(frameon=False, ncol=2, loc="lower left")
    save(fig, "03_class_growth")

    sub = combined[(combined.n_classes == 6)].sort_values("macro_f1")
    fig, ax = plt.subplots(figsize=(10, 6.5), layout="constrained")
    ax.barh(range(len(sub)), sub.macro_f1, color=["#009E73" if str(x).startswith("ft_") else "#0072B2" for x in sub.method])
    for i, (_, row) in enumerate(sub.iterrows()):
        if np.isfinite(row.get("macro_f1_ci_low", np.nan)):
            ax.plot([row.macro_f1_ci_low, row.macro_f1_ci_high], [i, i], "k-", linewidth=1.4)
        ax.text(.99, i, f"{row.macro_f1:.3f}", va="center", ha="right", fontsize=10)
    ax.set(yticks=range(len(sub)), yticklabels=[LABELS.get(x, x) for x in sub.method], xlim=(0, 1.05),
           xlabel="测试 Macro-F1，黑线为按测试日期重采样的 95% 区间", title="图 4  六类版面分类主结果")
    save(fig, "04_main_comparison")

    fig, grid = plt.subplots(2, 2, figsize=(11, 10), layout="constrained")
    axes = list(grid.flat)[:3]
    grid.flat[3].axis("off")
    grid.flat[3].text(.1, .6, "每格第一行为文章数\n第二行为该真实类别中的比例\n\n对角线表示正确分类\n非对角线表示类别混淆", fontsize=13,
                      color="#456", va="center", linespacing=1.8)
    for ax, name in zip(axes, ["tfidf", "ft_sg_idf", "paper_bigram"]):
        r = read_json(ROOT / f"results/classification_6_{name}.json")
        cm = np.array(r["confusion_matrix"])
        normalized = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{cm[i,j]}\n{normalized[i,j]:.0%}", ha="center", va="center",
                        fontsize=10, color="white" if normalized[i,j] > .5 else "#263238")
        ax.set(xticks=range(6), yticks=range(6), xticklabels=pages, yticklabels=pages,
               xlabel="预测版面", ylabel="真实版面", title=LABELS[name])
    fig.colorbar(im, ax=axes, shrink=.7, label="按真实类别归一化的比例")
    fig.suptitle("图 5  六类混淆矩阵：同时展示文章数与行比例", fontsize=16)
    save(fig, "05_confusion")

    sensitivity = sorted(sensitivity, key=lambda r: r["validation_macro_f1"])
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    vals = [r["validation_macro_f1"] for r in sensitivity]
    for i, (v, r) in enumerate(zip(vals, sensitivity)):
        color = "#D55E00" if r["method"] == "ft_sg" else "#0072B2"
        ax.plot([0, v], [i, i], color=color, linewidth=2)
        ax.scatter(v, i, color=color, s=50)
        ax.text(v + .012, i, f"{v:.3f}", va="center", fontsize=10)
    ax.set(yticks=range(len(vals)), yticklabels=[r["method"] for r in sensitivity], xlim=(0, 1),
           xlabel="验证集 Macro-F1（测试集保持独立）", title="图 6  FastText 参数变化的验证结果")
    save(fig, "06_sensitivity")

    fig, axes = plt.subplots(1, 2, figsize=(12, 8), layout="constrained")
    names = [r["name"] for r in embeddings]
    for ax, field, title, xlabel in zip(axes, ["train_seconds", "array_mib"],
            ["向量训练耗时", "主要 NumPy 数组内存"], ["秒", "MiB"]):
        ax.barh(range(len(names)), [r[field] for r in embeddings], color="#0072B2")
        ax.set(yticks=range(len(names)), yticklabels=names, xlabel=xlabel, title=title)
    fig.suptitle("图 7  参数设置与计算资源", fontsize=16)
    save(fig, "07_resources")

    coverage = read_json(ROOT / "results/coverage.json")
    selected_names = ["w2v_sg", "ft_sg", "ft_char3_6"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8), layout="constrained")
    x = np.arange(3)
    ax.bar(x - .18, [1 - coverage[n]["test"]["vocab_oov_rate"] for n in selected_names], width=.36,
           label="训练词表内的词", color=COLORS[0])
    ax.bar(x + .18, [coverage[n]["test"]["represented_token_rate"] for n in selected_names], width=.36,
           label="可生成有效向量的词", color=COLORS[2])
    ax.set(xticks=x, xticklabels=selected_names, ylim=(0, 1.08), ylabel="测试 token 覆盖比例",
           title="图 8  词表覆盖与子词向量生成能力")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2)
    save(fig, "08_oov_coverage")

    packed = np.load(ROOT / "results/features/dense_projection.npz")
    pca = PCA(n_components=2, random_state=config["seed"])
    projected = pca.fit_transform(packed["X"])
    fig, ax = plt.subplots(figsize=(9, 6.3), layout="constrained")
    for page, color in zip(pages, COLORS):
        idx = packed["y"] == page
        ax.scatter(projected[idx, 0], projected[idx, 1], s=24, alpha=.7, label=f"第 {page} 版", color=color)
    ax.set(xlabel=f"主成分 1（{pca.explained_variance_ratio_[0]:.1%}）",
           ylabel=f"主成分 2（{pca.explained_variance_ratio_[1]:.1%}）",
           title="图 9  FastText+IDF 测试文档向量的 PCA 投影")
    ax.legend(frameon=False, ncol=3)
    write_json(ROOT / "results/projection.json", {"explained_variance_ratio": pca.explained_variance_ratio_.tolist()})
    save(fig, "09_document_projection")

    training = pd.read_csv(ROOT / "results/paper_training.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for bigram, color in zip([False, True], COLORS):
        subset = paper[(paper.n_classes == 6) & (paper.method == ("paper_bigram" if bigram else "paper_unigram"))].iloc[0]
        h = training[(training.n_classes == 6) & (training.bigrams == bigram) & (training.lr == subset.selected_lr)]
        label = "单词+双词特征" if bigram else "单词特征"
        axes[0].plot(h.epoch, h.train_cross_entropy, color=color, label=label)
        axes[1].plot(h.epoch, h.validation_macro_f1, color=color, label=label)
    axes[0].set(xlabel="训练轮数", ylabel="加权交叉熵", title="训练目标的变化")
    axes[1].set(xlabel="训练轮数", ylabel="验证 Macro-F1", ylim=(0, 1), title="泛化效果的变化")
    axes[1].legend(frameon=False)
    fig.suptitle("图 10  监督 FastText 结构复现的学习曲线", fontsize=16)
    save(fig, "10_supervised_learning")

    selected = [r for r in embeddings if r["name"] in ["ft_bucket10k", "ft_sg", "ft_bucket200k"]]
    selected.sort(key=lambda r: r["parameters"]["bucket"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    buckets = [str(r["parameters"]["bucket"]) for r in selected]
    axes[0].bar(buckets, [r["collision_fraction"] for r in selected], color=COLORS[1])
    axes[0].set(xlabel="哈希桶数量", ylabel="不同子词共享桶的比例", ylim=(0, 1), title="子词哈希碰撞")
    axes[1].bar(buckets, [r["array_mib"] for r in selected], color=COLORS[0])
    axes[1].set(xlabel="哈希桶数量", ylabel="主要数组 MiB", title="内存代价")
    fig.suptitle("图 11  桶大小的空间与碰撞权衡", fontsize=16)
    save(fig, "11_bucket_tradeoff")

    model = __import__("gensim.models", fromlist=["FastText"]).FastText.load(str(ROOT / "models/ft_sg.model"))
    words = [w for w in ["中国", "北京", "上海", "美国", "法国", "英国", "经济", "市场", "银行", "金融", "企业", "农业",
        "教育", "学校", "大学", "学生", "科技", "研究", "科学", "技术"] if w in model.wv.key_to_index]
    vectors = np.array([model.wv[w] / np.linalg.norm(model.wv[w]) for w in words])
    sim = vectors @ vectors.T
    pd.DataFrame(sim, index=words, columns=words).to_csv(ROOT / "results/word_similarity.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(10, 9), layout="constrained")
    im = ax.imshow(sim, cmap="viridis", vmin=0, vmax=1)
    ax.set(xticks=range(len(words)), yticks=range(len(words)), xticklabels=words, yticklabels=words,
           title="图 12  FastText 词向量余弦相似性")
    ax.tick_params(axis="x", labelrotation=60)
    fig.colorbar(im, ax=ax, shrink=.75)
    save(fig, "12_word_similarity")
    print("Saved 12 figures as PNG and SVG.", flush=True)
