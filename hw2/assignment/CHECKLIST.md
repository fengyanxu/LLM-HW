# HW2 执行清单

> 目标：按课程页面 `question @34` 完成“Gensim 词表示学习 + FastText 新闻分类”，并产出可复核的报告、图表与数据清单。  
> 所有命令在仓库根目录 `D:\Projects\LLM-HW` 执行；Python 使用 `hw2\.venv`。

## A. 环境

- [x] 创建虚拟环境：`py -3.11 -m venv hw2\.venv`
- [x] 安装依赖：`hw2\.venv\Scripts\python.exe -m pip install -r hw2\assignment\requirements.txt`
- [x] 固定 `PYTHONUTF8=1`，避免 Windows 控制台中文乱码
- [x] 记录版本：gensim 4.3.3、fasttext-wheel 0.9.2、jieba 0.42.1、numpy 1.26.4、scikit-learn 1.2.2

## B. 词表示学习（第一次学习）

- [x] **B1 抽样 THUCNews**：从 `Tongjilibo/THUCNews`（revision `ae77e36`，836,075 条）按固定位置取 100 组 × 100 篇 = 10,000 篇  
      命令：`hw2\.venv\Scripts\python.exe hw2\assignment\scripts\01_fetch_thucnews.py`  
      产物：`data/thucnews/raw.jsonl`、`fetch_meta.json`、`batches/`
- [x] **B2 清洗与分词**：去空/过短/重复正文，按句切分，jieba 分词，丢弃不足 3 词的片段  
      命令：`... 04_prepare_embeddings.py`  
      产物：`data/thucnews/clean.jsonl`、`segmented.txt`、`clean_stats.json`
- [x] **B3 训练两种模型**：skip-gram、100 维、窗口 5、min_count 3、5 轮、单线程、seed 42；FastText 子词 2—4、200k 桶  
      命令：`... 05_train_embeddings.py`  
      产物：`models/word2vec.model`、`models/fasttext.model`、`results/embedding_training.json`
- [x] **B4 表示分析**：银行/学校/电脑/经济近邻；3 个 OOV 查询；四组主题词组内/组间相似度  
      命令：`... 06_analyze_embeddings.py`  
      产物：`results/embedding_analysis.json`、`results/word_neighbors.csv`

## C. 文本分类（第二次学习）

- [x] **C1 解析人民日报 1998**：按文章编号合并段落，首段标题、其余正文，去空正文、短正文、重复正文  
      命令：`... 02_parse_peopledaily.py`  
      产物：`data/peopledaily/articles.jsonl`、`candidate_counts.json`
- [x] **C2 构造 8 主题集**：体育、金融、教育、文化、科技、医疗、农业、法治；标题关键词筛选 + 月份轮转；每类 60 篇（48 训练 / 12 测试），2/4/8 嵌套  
      命令：`... 03_build_classification.py`  
      产物：`data/peopledaily/classification/selected.csv`、`train_{2,4,8}.txt`、`test_{2,4,8}.txt`、`manifest.json`
- [x] **C3 导出预训练向量**：Gensim FastText → `models/fasttext_pretrained.vec`
- [x] **C4 训练 fastText 分类器**：官方 `train_supervised`，dim=100、lr=0.3、25 轮、wordNgrams=2、softmax、thread=1、seed=42、`minn=maxn=0`  
      预训练组传 `pretrainedVectors`；从零组只去掉该参数  
      命令：`... 07_train_fasttext.py`  
      产物：`results/classification.json`、`classification.csv`、`predictions_*.csv`（fastText 二进制模型约 800 MB/个，不保存）
- [x] **C5 复现性检查**：8 类预训练模型重复训练一次，比较 96 条预测是否一致
- [x] **C6 补充诊断**：8 类训练轮数 25/50/100 敏感性；同划分 TF-IDF + LinearSVC 诊断基线  
      命令：`... 11_sensitivity.py`、`... 12_tfidf_baseline.py`  
      产物：`results/epoch_sensitivity.json`、`results/tfidf_baseline.json`

## D. 图表与报告

- [x] **D1 图表**：数据概览、词向量相似度、分类曲线、8 类混淆矩阵、轮数敏感性  
      命令：`... 08_make_figures.py`  
      产物：`figures/01_data_overview.png`、`02_embedding_similarity.png`、`03_classification_curve.png`、`04_confusion_8.png`、`05_epoch_sensitivity.png`（均附 CSV）
- [x] **D2 报告**：作者根据实测结果手写定稿，正文不再由脚本自动生成  
      产物：`report.md`
- [x] **D3 一致性校验**：检查 10,000 条抽样、480 篇分类集、6 组分类结果、图表与报告  
      命令：`... 10_verify.py`

## E. 验收对照（课程页面要求 → 本目录证据）

| 课程页面要求 | 本目录证据 |
|---|---|
| 用 Gensim 训练 Word2Vec / FastText | `models/`、`results/embedding_training.json` |
| 检查近邻、词表外查询、词相似度 | `results/embedding_analysis.json`、报告 3.2–3.4 |
| FastText 词向量导出 `.vec` 并传给 `pretrainedVectors` | `models/fasttext_pretrained.vec`、`scripts/07_train_fasttext.py` |
| 监督分类 + 不导入预训练向量的对照 | `results/classification.json` 中 `pretrained` / `scratch` |
| 类别从 2 → 4 → 8 递增，沿用同一划分 | `train_{2,4,8}.txt`、`test_{2,4,8}.txt`、`manifest.json` |
| 报告、脚本、数据校验清单、结果和图 | `report.md`、`scripts/`、`results/`、`figures/`、本清单 |
| 补充：训练轮数敏感性与诊断基线 | `results/epoch_sensitivity.json`、`results/tfidf_baseline.json`、报告 4.5 |

## F. 一键复现

```powershell
.\hw2\assignment\run_all.ps1
```

脚本会依次执行 B1–D3；已存在的 THUCNews 批次和结果会复用（`01` 有批次缓存，其余步骤按产物覆盖）。
