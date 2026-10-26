# HW2 作业索引

> **课程页面版（当前提交）**：按 `question @34` 的要求，用 THUCNews 训练 Word2Vec/FastText，再用人民日报 8 个新闻主题做 2/4/8 类 fastText 监督分类。  
> 入口：[`assignment/README.md`](assignment/README.md) · 报告：[`assignment/report.md`](assignment/report.md) · 执行清单：[`assignment/CHECKLIST.md`](assignment/CHECKLIST.md)
>
> 早期按人民日报版面做 2/4/6 类、使用线性 SVM 的 P137 方案保留在下方，仅作历史版本，不作为本次课程页面要求的提交结果。

## 早期 P137 方案（保留）

### Gensim 表示学习与 FastText 新闻分类

完成 L2-p1 第 137 页的基础、进阶与高级要求。实验使用人民日报新闻语料，训练 Gensim Word2Vec/FastText，构建词和文档向量，并按真实版面开展 **2→4→6 类分类**。

- [详细实验报告](docs/P137实验报告.md)：环境、论文、源码、参数、结果与分析。
- [结果展示](docs/实验展示.html)：下载后用浏览器打开，可切换类别数量查看结果。
- [源码位置索引](docs/source-map.md)：Gensim 4.3.0 的函数、行号与版本链接。
- [论文核验记录](results/references_verified.json)：Gensim 框架、子词向量和监督分类论文的来源及用途。
- [主结果 CSV](results/combined_classification.csv)、[参数比较](results/sensitivity.json)和 [PNG/SVG 图表](figures/)。

## 实验内容

Gensim Dictionary、TF-IDF 与 LSI 提供词项和文档表示；Word2Vec 和 FastText 分别比较 CBOW、Skip-gram、平均聚合和 IDF 聚合。15 组词向量设置覆盖维度、窗口、字符子词、哈希桶、轮数、采样指数、降采样和学习率。

Gensim 框架论文为 Řehůřek 与 Sojka 的 [Software Framework for Topic Modelling with Large Corpora](https://radimrehurek.com/lrec2010_final.pdf)，用于对照流式语料与 Dictionary—TF-IDF—LSI 接口。FastText 算法论文为 Bojanowski 等人的 [Enriching Word Vectors with Subword Information](https://aclanthology.org/Q17-1010/)，用于对照字符子词实现。Joulin 等人的 [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759) 用于补充监督分类结构复现；项目通过 PyTorch 的平均特征嵌入与线性分类层完成该部分，并比较词 unigram/bigram。

## 数据与评价

输入文件为 `1998-上半年-带音.rar`，包含预分词新闻。解析器去除读音和布局标记，按文章合并，使用文章编号中的版面字段作为标签。1 月 1—20 日为训练集，21—25 日为验证集，26—31 日为测试集。文章 ID、日期、长度和清洗文本哈希见 [划分清单](results/split_manifest.csv)。

表示学习使用 2 月抽样的 675 篇新闻及 1 月训练文本，共约 62.7 万个清洗词。各阶段共享同一训练表示空间。验证和测试文章独立于词向量训练语料；IDF、LSI、标准化等统计量从 1 月训练集拟合。

2 月是外部无标签语料，本实验研究已有语料条件下的文档分类。历史部署预测应将外部语料限定在训练截止日期之前。版面标签描述报纸编辑位置，主题相近的文章可能分布在不同版面。

分类器 C 与 FastText 变体通过验证集选择。测试结果报告 Accuracy、Macro-F1 与按测试日期整组重采样的区间。图表由保存的实验结果生成。

## 环境与运行

本地验证环境为 Windows、Python 3.11.7、Gensim 4.3.0、NumPy 1.26.4、SciPy 1.11.1 和 scikit-learn 1.2.2。具体版本和数值库信息见 [environment.json](results/environment.json)。现有预分词文本直接提供词界。

在项目目录执行。以下为 Git Bash 的独立安装方式：

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
export PYTHONHASHSEED=0
python scripts/audit_sources.py
python scripts/run.py all --archive "/d/PostGraduate/LLM/hw2/1998-上半年-带音.rar"
python -m unittest discover -s tests -v
python scripts/verify_results.py
```

Linux/macOS 激活路径为 `.venv/bin/activate`。本次本地运行使用 `--system-site-packages` 创建独立虚拟环境，复用已验证的数值库依赖。若 `tar` 支持 RAR，prepare 会直接解压；也可以手动将六个月的 TXT 文件放到 `data/raw/`，再运行相同命令。

支持分阶段运行：

```bash
python scripts/run.py prepare --archive "/path/to/1998-上半年-带音.rar"
python scripts/run.py embeddings
python scripts/run.py evaluate
python scripts/run.py paper
python scripts/run.py plot
python scripts/run.py report
python scripts/verify_results.py
```

`embeddings --only ft_sg` 可以单独执行指定模型。已有模型经参数和语料哈希核对后复用。清洗、模型与结果使用固定相对目录，方便分阶段检查。

## 项目结构

```text
configs/experiment.json    实验配置及参数网格
src/newsvec/              清洗、训练、评价、绘图和报告生成
scripts/                  运行入口、源码审计、结果核对
tests/                    清洗与模型保存重载集成检查
docs/                     实验报告、源码索引、浏览器展示
results/                  指标、参数、划分与环境清单
figures/                  12 张 PNG 及对应 SVG 图表
data/                     本地原始与清洗语料
models/                   本地词向量和分类模型
logs/                     本地日志与论文阅读材料
```

公开仓库保留代码、配置、报告、图表及必要数值结果。原始语料、完整文章、论文 PDF、模型权重和虚拟环境保留本地，可依据语料哈希及划分清单复现。
