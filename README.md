# LLM-HW

大语言模型应用与实践课程作业仓库。

## 作业

| 作业 | 内容 | 报告 | 说明 |
|---|---|---|---|
| HW1 | 人民日报分词语料的 1—10 阶 n-gram 语言模型：KenLM MKN 与 Stupid Backoff、困惑度、续写 | [作业报告](hw1/作业报告.md) | [运行说明](hw1/README.md) |
| HW2 | Gensim Word2Vec/FastText 表示学习 + 人民日报 8 主题 fastText 分类（2/4/8 类） | [实验报告](hw2/assignment/report.md) | [执行清单](hw2/assignment/CHECKLIST.md) · [运行说明](hw2/assignment/README.md) |

HW2 的早期 P137 方案（人民日报版面 2/4/6 类、线性 SVM）保留在 [P137实验报告](hw2/docs/P137实验报告.md)，作为历史版本，不属于课程页面版提交结果。

## 数据与模型

仓库只保留代码、配置、报告、图表和必要的结果清单。原始语料、词向量/语言模型、虚拟环境、便携工具链等大文件由各作业的脚本重建，具体见各目录的 README 与 `.gitignore`。

## 环境

- Windows + Python 3.11
- HW1：WSL Ubuntu 22.04 + KenLM（`lmplz`、Python 绑定）
- HW2：`hw2/.venv`，依赖见 [requirements.txt](hw2/assignment/requirements.txt)

各作业目录内提供一键运行脚本：

```powershell
# HW1
.\hw1\run_hw1.ps1

# HW2（课程页面版）
.\hw2\assignment\run_all.ps1
```
