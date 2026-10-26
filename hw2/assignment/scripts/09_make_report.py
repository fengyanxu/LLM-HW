"""Generate the HW2 experiment report from measured artefacts."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hw2a.common import CONFIG, DATA, RESULTS, read_json  # noqa: E402

CLASSES = CONFIG["classification"]["classes"]
STAGES = CONFIG["classification"]["stages"]


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def f(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def table(headers, rows) -> str:
    lines = ["| " + " | ".join(str(x) for x in headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def main() -> None:
    stats = read_json(DATA / "thucnews" / "clean_stats.json")
    training = read_json(RESULTS / "embedding_training.json")
    analysis = read_json(RESULTS / "embedding_analysis.json")
    classification = read_json(RESULTS / "classification.json")
    manifest = read_json(DATA / "peopledaily" / "classification" / "manifest.json")
    candidate = read_json(DATA / "peopledaily" / "candidate_counts.json")
    sensitivity = read_json(RESULTS / "epoch_sensitivity.json")
    tfidf = read_json(RESULTS / "tfidf_baseline.json")

    train_by_name = {row["name"]: row for row in training}
    res = {(row["stage"], row["condition"]): row for row in classification["results"] if "test_n" in row and row["condition"] in {"pretrained", "scratch"}}

    neighbor_rows = []
    for word in ("银行", "学校", "电脑", "经济"):
        w2v = analysis["word2vec"]["neighbors"].get(word, [])
        ft = analysis["fasttext"]["neighbors"].get(word, [])
        fmt = lambda items: "、".join(f"{item['word']}({item['score']:.2f})" for item in items[:5])  # noqa: E731
        neighbor_rows.append([word, fmt(w2v), fmt(ft)])

    oov_rows = []
    for word in analysis["fasttext"]["oov"]:
        ft = analysis["fasttext"]["oov"][word]
        w2v = analysis["word2vec"]["oov"][word]
        nearest = "、".join(f"{item['word']}({item['score']:.2f})" for item in ft.get("neighbors", [])[:3])
        oov_rows.append([
            word,
            "是" if ft["in_vocab"] else "否",
            "可查询" if w2v.get("vector") else "不可查询",
            f"norm={ft['vector']['norm']:.3f}" if ft.get("vector") else "无",
            nearest or "—",
        ])

    sim_rows = []
    for key, label in (("word2vec", "Word2Vec"), ("fasttext", "FastText")):
        item = analysis[key]["similarity"]
        sim_rows.append([label, f(item["within_mean"]), f(item["cross_mean"]), f(item["within_mean"] - item["cross_mean"])])

    class_rows = []
    for stage in STAGES:
        row = res[(stage, "pretrained")]
        scratch = res[(stage, "scratch")]
        class_rows.append([
            f"{stage} 类",
            pct(row["accuracy"]), f(row["macro_f1"]),
            pct(scratch["accuracy"]), f(scratch["macro_f1"]),
            row["train_seconds"], scratch["train_seconds"],
        ])

    eight = res[(8, "pretrained")]
    per_class_rows = [
        [label, pct(value["precision"]), pct(value["recall"]), pct(value["f1"]), value["support"]]
        for label, value in eight["per_class"].items()
    ]
    errors = [item for item in eight["predictions"] if item["true"] != item["predicted"]]
    error_rows = [[item["true"], item["predicted"], item["title"], f"{item['confidence']:.3f}"] for item in errors[:12]]
    sens_rows = []
    for epoch in (25, 50, 100):
        pre = next(row for row in sensitivity if row["condition"] == "pretrained" and row["epoch"] == epoch)
        scr = next(row for row in sensitivity if row["condition"] == "scratch" and row["epoch"] == epoch)
        sens_rows.append([epoch, pct(pre["accuracy"]), f(pre["macro_f1"]), pct(scr["accuracy"]), f(scr["macro_f1"])])
    tfidf_rows = [[f"{row['stage']} 类", pct(row["accuracy"]), f(row["macro_f1"]), f"{row['features']:,}"] for row in tfidf]

    data_rows = [
        ["原始抽样", f"{stats['raw_articles']:,}", "—", "—"],
        ["清洗后", f"{stats['kept_articles']:,}", f"{stats['sentences']:,}", f"{stats['tokens']:,}"],
    ]
    split_rows = [
        [label, manifest["stages"]["8"]["per_class"][label] - 12, 12, manifest["stages"]["8"]["per_class"][label]]
        for label in CLASSES
    ]

    report = f"""# HW2 实验报告：Gensim 表示学习与 FastText 新闻分类

> 课程页面 `question @34` · 本报告由 `scripts/09_make_report.py` 从 `results/` 与 `figures/` 自动生成

## 摘要

本实验按课程页面要求完成“两次学习”：先用 THUCNews 新闻训练 **Gensim Word2Vec** 与 **FastText** 词向量，比较近邻、词表外查询和主题词相似度；再把 FastText 词向量导出为 `.vec`，作为 **fastText 监督分类器**的 `pretrainedVectors`，在人民日报 1998 年语料上完成 **2 类 → 4 类 → 8 类** 的嵌套新闻主题分类，并设置“不导入预训练向量、从零训练”的对照组。

主要结果：词向量语料由 10,000 篇抽样新闻清洗为 **{stats['kept_articles']:,} 篇、{stats['sentences']:,} 句、{stats['tokens']:,} 词**；两种模型词表均为 **{train_by_name['word2vec']['vocab_size']:,}** 词。分类集每类 60 篇（48 训练 / 12 测试）。8 类任务中，导入预训练词向量的 FastText 测试准确率为 **{pct(eight['accuracy'])}**、Macro-F1 为 **{f(eight['macro_f1'])}**；从零训练组为 **{pct(res[(8, 'scratch')]['accuracy'])}** / **{f(res[(8, 'scratch')]['macro_f1'])}**。类别数增加时任务变难；在固定 25 轮的主设置下，预训练词向量的优势在 8 类最明显。4.5 节的 100 轮敏感性显示从零组可以追到 81.25%，因此差距中有一部分是收敛速度，而不只是表示质量。

## 1. 任务与实验设计

课程页面把作业拆成两次学习：

1. **词表示学习**：让模型从上下文里学习固定长度词向量。Word2Vec 只为词表内的词学习表示；FastText 还利用字符子词，因此可以为未登录词合成向量。
2. **文本分类**：给定“文章 + 类别”样本，学习从文章表示到类别的映射。词向量本身不是分类器，分类器必须用带标签文章另行训练。

本实验的研究问题有三个：

- Word2Vec 与 FastText 的近邻、OOV 和主题相似度有何差别？
- 导入预训练词向量与从零训练分类器，在 2/4/8 类任务上差别多大？
- 类别数增加时，任务难度和错误类型如何变化？

### 与仓库早期 P137 方案的区别

仓库原有的 `docs/P137实验报告.md` 采用人民日报 **版面字段** 做 2/4/6 类分类，分类器是线性 SVM，词向量也来自人民日报。本报告按课程页面改为：**THUCNews 训练词向量 + 人民日报 8 个新闻主题 + fastText 官方监督分类器 + 2/4/8 嵌套类别**。两套实验的标签、数据、分类器和类别数均不同，数值不可直接比较。旧方案保留为历史版本，本报告是课程页面版的提交结果。

## 2. 数据与预处理

### 2.1 词向量语料：THUCNews

数据来自课程页面指定的 HuggingFace 镜像 `Tongjilibo/THUCNews`，revision `{CONFIG['thucnews']['revision']}`，共 **{CONFIG['thucnews']['total_rows']:,}** 条。按固定位置取 **{CONFIG['thucnews']['groups']} 组 × {CONFIG['thucnews']['per_group']} 篇 = 10,000 篇**：第 g 组从 `round(g × 836075 / 100)` 开始连续取 100 篇，组内按原顺序保留。

清洗规则：去掉空正文、过短正文（少于 {stats['min_chars']} 字符）和重复正文；按 `。！？!?` 切句；用 jieba 分词；舍弃不足 {stats['min_sentence_words']} 个词的片段。两种模型读取完全相同的 `segmented.txt`。

{table(['阶段', '文章数', '句子数', '词数'], data_rows)}

清洗共丢弃空正文 {stats['dropped_empty']} 篇、过短 {stats['dropped_short']} 篇、重复 {stats['dropped_duplicate']} 篇、无法组成 3 词句 {stats['dropped_no_sentence']} 篇。平均每篇 {stats['avg_sentences']} 句、{stats['avg_tokens']} 词。

### 2.2 分类语料：人民日报 1998 与 8 个主题

输入为 1998 年 1—6 月人民日报切分标注语料。按文章编号前三个字段合并段落，共 **{candidate['parse']['articles'] + candidate['parse'].get('no_body', 0):,}** 篇；其中 {candidate['parse'].get('no_body', 0):,} 篇只有标题、没有正文，首段作标题、其余段落作正文后得到 **{candidate['parse']['articles']:,}** 篇有正文文章，再按正文长度 ≥ {CONFIG['classification']['min_body_words']} 词、正文 SHA-256 去重，保留 **{candidate['articles_kept']:,}** 篇作为候选池（精确值见 `data/peopledaily/candidate_counts.json`）。

课程页面特别说明：原语料没有“体育/金融”等主题标签，需要自行筛选。本实验用**标题关键词规则**挑选候选：标题命中某主题至少 2 个关键词，且没有第二个主题命中 2 个及以上；再按月份轮转抽取，避免某一月垄断。体育、金融、教育、文化、科技、医疗、农业、法治各取 **60 篇**，每类随机 **48 训练 / 12 测试**（seed 42）。2 类 = 体育 + 金融；4 类再加入教育 + 文化；8 类再加入科技 + 医疗 + 农业 + 法治。三个实验嵌套，已出现类别的文章和划分保持不变。

{table(['类别', '训练', '测试', '合计'], split_rows)}

![数据概览：THUCNews 清洗前后与 8 类分类集构成](figures/01_data_overview.png)

需要说明：标题规则不是人工金标准，仍会留下少量交叉主题（例如银行腐败案、带“大学生”的体育报道）。课程页面的参考实现也明确保留了这类歧义。分类器输入**只使用正文，不使用标题**，以避免“标题直接给出标签”的循环。

## 3. 词表示学习

### 3.1 参数与训练

两种模型共享：skip-gram、100 维、窗口 5、最低词频 3、训练 5 轮、单线程、随机种子 42、负采样 5。FastText 另设字符子词长度 2—4、200,000 个哈希桶。

{table(['模型', '训练秒数', '词表大小', '向量维度'], [[row['name'], row['seconds'], f"{row['vocab_size']:,}", row['vector_size']] for row in training])}

两者读取同一份分词文件，因此词表完全一致；训练时间只反映本机单次运行，不作为一般性的速度排名。

### 3.2 近邻

{table(['查询词', 'Word2Vec 前 5 近邻', 'FastText 前 5 近邻'], neighbor_rows)}

Word2Vec 的“银行”近邻更偏向金融机构；FastText 常把共享字符的词放在一起，因而会出现“非银行、抢银行、中小银行”这类字面相近但语义不同的词。这正好说明：**子词能提高词表外覆盖率，但不保证语义正确**。

### 3.3 词表外（OOV）查询

{table(['查询词', '在词表内', 'Word2Vec', 'FastText 向量', 'FastText 最近邻'], oov_rows)}

FastText 能为未登录词合成向量，但“能返回数值”不等于“理解了这个词”。OOV 向量的质量取决于字符片段在训练语料中的统计，仍需下游任务验证。

### 3.4 主题词相似度

四组词：体育（足球、篮球、排球、乒乓球）、金融（银行、贷款、利率、股票）、教育（学校、学生、教师、高考）、科技（电脑、软件、互联网、科技）。计算组内两两余弦相似度的平均值，以及不同组之间所有词对的平均值。

{table(['模型', '组内平均', '组间平均', '组内 − 组间'], sim_rows)}

![主题词组内/组间相似度](figures/02_embedding_similarity.png)

两种模型的组内相似度都高于组间，说明向量确实捕捉到了主题聚类；FastText 的组内值更高，部分来自字符子词带来的字面相似。该指标只反映这四组词，不能外推为“FastText 全面更好”。

## 4. 文本分类

### 4.1 训练设置

- 输入格式：`__label__体育 中国 队 比赛 ...`，正文最多取前 **{CONFIG['classification']['max_body_words']}** 个词，不含标题。
- 分类器：fastText 官方 `train_supervised`，100 维、学习率 0.3、25 轮、词 bigram、softmax 损失、单线程、种子 42；字符子词参数 `minn=maxn=0`（与课程页面“字符片段参数设为零”一致）。
- 预训练组：把 Gensim FastText 的词表内词向量导出为 `.vec`（{classification['vector_export']['vectors']:,} 个词），传给 `pretrainedVectors`。
- 从零组：只去掉 `pretrainedVectors`，其他数据、参数、划分完全相同。
- 8 类预训练模型重复训练一次，预测结果是否一致：**{classification['deterministic_repeat_8_pretrained']}**。

### 4.2 主结果

{table(['类别数', '预训练 准确率', '预训练 Macro-F1', '从零 准确率', '从零 Macro-F1', '预训练训练秒', '从零训练秒'], class_rows)}

![分类准确率与 Macro-F1](figures/03_classification_curve.png)

读法：2 类任务本身容易，两组都接近上限；4 类开始出现差距；在固定 25 轮的主设置下，8 类时预训练组明显优于从零组（4.5 节说明这部分差距也包含收敛速度）。原因不是“预训练向量更懂语义”这么简单，而是从零训练只有 384 篇带标签文章，100 维词向量和分类边界都要从很少的监督信号里学；导入的词向量已经把新闻语料的共现结构带了进来。

### 4.3 8 类逐类结果（预训练）

{table(['类别', '精确率', '召回率', 'F1', '测试支持'], per_class_rows)}

![8 类混淆矩阵](figures/04_confusion_8.png)

8 类测试集共 96 篇，模型错 {len(errors)} 篇。错误主要集中在主题交叉的报道，例如科技/农业（农业科技）、文化/教育（出版社、校园文化）、法治/金融（银行腐败案）。这说明“标题规则标签”本身存在噪声，不能把全部错误归因于模型。

### 4.4 错误样例

{table(['真实', '预测', '标题', '置信度'], error_rows) if error_rows else '本次没有分类错误。'}

### 4.5 补充：训练轮数敏感性

课程页面的固定设置是 25 轮。为判断 8 类从零组的低分是“数据不可分”还是“训练不足”，保持学习率、维度、bigram 等参数不变，只把轮数改为 50 和 100：

{table(['训练轮数', '预训练 准确率', '预训练 Macro-F1', '从零 准确率', '从零 Macro-F1'], sens_rows)}

![8 类训练轮数敏感性](figures/05_epoch_sensitivity.png)

25 轮时从零组明显欠拟合；100 轮时升到 {pct(sensitivity[5]['accuracy'])}，已经追平预训练组 25 轮的水平。预训练向量的作用更像是“把收敛所需轮数降下来”，而不是单纯提高上限：100 轮时预训练组仍达到 {pct(sensitivity[2]['accuracy'])}，说明表示质量也有贡献。

同一划分上，TF-IDF + LinearSVC 诊断基线如下：

{table(['类别数', '准确率', 'Macro-F1', '特征数'], tfidf_rows)}

这说明 480 篇标签本身是可学习的，8 类从零组的 37.5% 不能解释为“类别不可分”。该基线只用于诊断，不参与 fastText 主比较。

## 5. 讨论

**Word2Vec vs FastText。** 近邻表显示 Word2Vec 的“银行”更接近金融机构，FastText 更偏向共享字符的词。FastText 的优点是 OOV 可合成向量，缺点是字面相似会混入“抢银行”这类非同类词。课程页面给出的判断是“不能凭四个词判定哪个模型更好”，本实验同意这一点：词向量质量应看下游任务，而不是单看几个近邻。

**预训练向量 vs 从零训练。** 2 类时两者几乎相同，因为任务太容易；8 类时预训练组领先约 {(res[(8, 'pretrained')]['accuracy'] - res[(8, 'scratch')]['accuracy']) * 100:.2f} 个百分点。这与课程页面的参考结果方向一致：类别越多，监督样本越少，预训练表示的收益越大。但本实验每个类别只有 12 篇测试文章，单个样本就能造成 8.3 个百分点的波动，因此不应把差距精确化成“方法提升多少”。

**类别数。** 2 → 4 → 8 类时，类别边界越来越细，且测试集固定为每类 12 篇。2 类的 1.0 附近结果不能外推到 8 类；类别增加本身就是难度增加。

**与课程页面参考数值的差异。** 课程页面的参考实现报告 2 类 1.000、4 类 0.958、8 类 0.917（预训练向量）。本次为 {pct(res[(2, 'pretrained')]['accuracy'])}、{pct(res[(4, 'pretrained')]['accuracy'])}、{pct(eight['accuracy'])}。差异来自抽样位置、清洗阈值、标题规则、jieba 版本、单次划分和 fastText 版本；本报告只报告本目录可复现的实测值，不把参考数值当作本次结果。

**局限。** ①标题规则标签有噪声，且 8 类每类只有 60 篇；②只做了一次固定划分，没有置信区间；③fastText 的 `pretrainedVectors` 只导入词表内词向量，不含 Gensim 的完整子词参数；④分类器输入截断到 500 词，长文的尾部信息丢失；⑤Word2Vec/FastText 只训练 5 轮，未做参数搜索。

## 6. 复现

环境：Windows + Python 3.11.9，依赖见 `requirements.txt`；本目录使用 `hw2/.venv`。

| 组件 | 版本 / 说明 |
|---|---|
| Python | 3.11.9 |
| gensim | 4.3.3（4.3.0 的 Windows 轮子在当前机器缺少编译扩展，故升级到 4.3.3） |
| fasttext | fasttext-wheel 0.9.2 |
| jieba | 0.42.1 |
| numpy / scipy | 1.26.4 / 1.11.1 |
| scikit-learn | 1.2.2（仅诊断基线） |

```powershell
# 在仓库根目录
py -3.11 -m venv hw2\\.venv
hw2\\.venv\\Scripts\\python.exe -m pip install -r hw2\\assignment\\requirements.txt
$env:PYTHONUTF8='1'
hw2\\.venv\\Scripts\\python.exe hw2\\assignment\\scripts\\01_fetch_thucnews.py
hw2\\.venv\\Scripts\\python.exe hw2\\assignment\\scripts\\04_prepare_embeddings.py
hw2\\.venv\\Scripts\\python.exe hw2\\assignment\\scripts\\05_train_embeddings.py
hw2\\.venv\\Scripts\\python.exe hw2\\assignment\\scripts\\06_analyze_embeddings.py
hw2\\.venv\\Scripts\\python.exe hw2\\assignment\\scripts\\03_build_classification.py
hw2\\.venv\\Scripts\\python.exe hw2\\assignment\\scripts\\07_train_fasttext.py
hw2\\.venv\\Scripts\\python.exe hw2\\assignment\\scripts\\08_make_figures.py
hw2\\.venv\\Scripts\\python.exe hw2\\assignment\\scripts\\09_make_report.py
```

`01_fetch_thucnews.py` 会把 100 个批次缓存到 `data/thucnews/batches/`，中断后可续跑；`03_build_classification.py` 只依赖 `hw1/data/raw/199801/` 的原始语料。完整执行清单见 `CHECKLIST.md`。

### 文件索引

| 路径 | 内容 |
|---|---|
| `config.json` | 抽样、词向量、分类参数 |
| `scripts/01_fetch_thucnews.py` | THUCNews 100 组 × 100 篇抽样 |
| `scripts/03_build_classification.py` | 8 主题筛选与 2/4/8 嵌套划分 |
| `scripts/04_prepare_embeddings.py` | 清洗、切句、jieba 分词 |
| `scripts/05_train_embeddings.py` | Word2Vec / FastText 训练 |
| `scripts/06_analyze_embeddings.py` | 近邻、OOV、相似度 |
| `scripts/07_train_fasttext.py` | 预训练 vs 从零的 fastText 分类 |
| `scripts/08_make_figures.py` | 图表 |
| `scripts/09_make_report.py` | 本报告 |
| `scripts/11_sensitivity.py` | 8 类训练轮数敏感性 |
| `scripts/12_tfidf_baseline.py` | TF-IDF + SVM 诊断基线 |
| `data/thucnews/` | 抽样、清洗语料、分词语料 |
| `data/peopledaily/classification/` | 480 篇分类集与 train/test 文件 |
| `models/` | Gensim 词向量与 `.vec`；fastText 分类器二进制约 800 MB/个，不落盘 |
| `results/` | 训练、分析、分类指标与预测 |
| `figures/` | 报告图表 |

## 7. 参考

1. Řehůřek R, Sojka P. Software Framework for Topic Modelling with Large Corpora. LREC 2010.
2. Bojanowski P, Grave E, Joulin A, Mikolov T. Enriching Word Vectors with Subword Information. TACL 2017.
3. Joulin A, Grave E, Bojanowski P, Mikolov T. Bag of Tricks for Efficient Text Classification. EACL 2017.
4. 清华大学自然语言处理实验室. THUCNews 中文文本分类数据集. <http://thuctc.thunlp.org/>
5. 北京大学计算语言学研究所, 富士通研究开发中心. 1998 年人民日报切分标注语料.
"""
    path = ROOT / "report.md"
    path.write_text(report, encoding="utf-8")
    print(f"报告已生成：{path}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
