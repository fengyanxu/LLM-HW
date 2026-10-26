# HW2：Gensim 表示学习与 FastText 新闻分类（课程页面版）

> 对应课程页面 `question @34`。本目录是可按课程要求复现的完整实验：THUCNews 词向量 + 人民日报 8 主题 fastText 分类。
>
> - 实验报告：[report.md](report.md)
> - 执行清单：[CHECKLIST.md](CHECKLIST.md)
> - 早期按版面 2/4/6 类的 P137 方案保留在 [`../docs/P137实验报告.md`](../docs/P137实验报告.md)，不作为本课程页面要求的提交结果。

## 结果速览

| 类别数 | 预训练词向量 准确率 / Macro-F1 | 从零训练 准确率 / Macro-F1 |
|---:|---:|---:|
| 2 类（体育、金融） | 95.83% / 0.9583 | 91.67% / 0.9167 |
| 4 类（+教育、文化） | 87.50% / 0.8745 | 64.58% / 0.6374 |
| 8 类（+科技、医疗、农业、法治） | 81.25% / 0.8108 | 37.50% / 0.3333 |

补充：8 类从零组在 25 轮时欠拟合，100 轮升至 81.25%；同划分 TF-IDF + LinearSVC 诊断基线为 92.71%，说明标签本身可学习。详见报告 4.5。

## 数据与设置

- **词向量语料**：THUCNews 镜像 `Tongjilibo/THUCNews`（revision `ae77e36`，836,075 条），按固定位置取 100 组 × 100 篇 = 10,000 篇；清洗后 **9,334 篇、167,518 句、4,056,014 词**。
- **词向量模型**：Word2Vec / FastText，skip-gram、100 维、窗口 5、min_count 3、5 轮、单线程、seed 42；FastText 子词 2—4、200,000 桶。两模型词表均为 **67,863** 词。
- **分类语料**：人民日报 1998 年 1—6 月；按标题关键词筛出体育、金融、教育、文化、科技、医疗、农业、法治各 60 篇，48 训练 / 12 测试，2/4/8 嵌套。
- **分类器**：fastText 官方 `train_supervised`，100 维、lr 0.3、25 轮、word bigram、softmax、单线程、seed 42；预训练组把 Gensim FastText 词向量导出为 `.vec` 传入 `pretrainedVectors`，从零组只去掉该参数。

## 快速开始

```powershell
# 在仓库根目录
py -3.11 -m venv hw2\.venv
hw2\.venv\Scripts\python.exe -m pip install -r hw2\assignment\requirements.txt
.\hw2\assignment\run_all.ps1
```

也可以分步执行：

```powershell
$env:PYTHONUTF8='1'
hw2\.venv\Scripts\python.exe hw2\assignment\scripts\01_fetch_thucnews.py
hw2\.venv\Scripts\python.exe hw2\assignment\scripts\04_prepare_embeddings.py
hw2\.venv\Scripts\python.exe hw2\assignment\scripts\05_train_embeddings.py
hw2\.venv\Scripts\python.exe hw2\assignment\scripts\06_analyze_embeddings.py
hw2\.venv\Scripts\python.exe hw2\assignment\scripts\03_build_classification.py
hw2\.venv\Scripts\python.exe hw2\assignment\scripts\07_train_fasttext.py
hw2\.venv\Scripts\python.exe hw2\assignment\scripts\11_sensitivity.py
hw2\.venv\Scripts\python.exe hw2\assignment\scripts\12_tfidf_baseline.py
hw2\.venv\Scripts\python.exe hw2\assignment\scripts\08_make_figures.py
hw2\.venv\Scripts\python.exe hw2\assignment\scripts\10_verify.py
```

`report.md` 由作者根据实验结果手写定稿，`09_make_report.py` 仅作为内部整理草稿的辅助工具，不参与一键流水线。

首次运行需要下载 10,000 条 THUCNews 抽样（约 100 个批次，已缓存在 `data/thucnews/batches/`，可断点续跑）；词向量训练约 2—3 分钟，分类器训练数秒。

## 目录

```text
assignment/
  config.json                 抽样、词向量、分类参数
  scripts/                    01—12 数据、训练、分析、图表、报告、校验脚本
  data/thucnews/              抽样、清洗、分词语料（大文件不提交）
  data/peopledaily/           人民日报解析结果、480 篇分类集、train/test
  models/                     Word2Vec / FastText 词向量与 .vec（fastText 分类器二进制不保存）
  results/                    训练、分析、分类、敏感性、诊断指标与预测
  figures/                    5 张图及同名 CSV
  report.md                   实验报告
  CHECKLIST.md                执行清单与验收对照
```

大型模型、原始抽样与分词语料不提交；运行上面的命令即可重建。分类集清单 `data/peopledaily/classification/selected.csv`、结果 JSON 与图表会保留，便于直接核对。

## 与旧版 P137 的关系

`../docs/P137实验报告.md` 使用人民日报**版面字段**做 2/4/6 类分类，分类器是线性 SVM，词向量也来自人民日报，与本课程页面的“THUCNews 词向量 + 8 个新闻主题 + fastText 监督分类”不是同一套协议。旧版保留，避免覆盖已有工作；本次提交以 `assignment/report.md` 为准。

## 数据来源

- THUCNews：清华大学自然语言处理实验室，镜像 `Tongjilibo/THUCNews`（Apache-2.0），原始站点 <http://thuctc.thunlp.org/>。
- 人民日报 1998 切分标注语料：北京大学计算语言学研究所、富士通研究开发中心；使用请注明来源。
