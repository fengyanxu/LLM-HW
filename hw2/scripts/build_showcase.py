"""Build a local, dependency-free HTML results viewer."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from newsvec.visualize import LABELS

df = pd.read_csv(ROOT / "results/combined_classification.csv")
rows = [{"method": LABELS.get(r.method, r.method), "classes": int(r.n_classes),
         "accuracy": float(r.accuracy), "f1": float(r.macro_f1),
         "validation": None if pd.isna(r.validation_macro_f1) else float(r.validation_macro_f1)}
        for _, r in df.iterrows()]
files = sorted((ROOT / "figures").glob("*.png"))
names = ["数据划分", "文章长度", "类别数量与效果", "六类方法比较", "混淆矩阵", "参数验证",
         "计算资源", "词表外覆盖", "文档向量投影", "监督模型学习曲线", "哈希桶权衡", "词向量相似性"]
gallery = "\n".join(f'<figure id="f{i+1}"><figcaption>{i+1:02d} · {name}</figcaption>'
                     f'<img src="../figures/{p.name}" alt="{name}" loading="lazy" onclick="zoom(this)"></figure>'
                     for i, (p, name) in enumerate(zip(files, names)))
html = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gensim 与 FastText 新闻分类实验</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='6' fill='%230072b2'/%3E%3Ctext x='16' y='23' text-anchor='middle' font-size='22' fill='white'%3EG%3C/text%3E%3C/svg%3E">
<style>
:root{--ink:#17324c;--blue:#0072b2;--green:#009e73;--paper:#f5f8fa}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:"Microsoft YaHei",sans-serif;line-height:1.75}
main{max-width:1180px;margin:auto;padding:48px 28px}header{padding-bottom:28px;border-bottom:2px solid var(--blue)}
h1{font-size:clamp(28px,4vw,46px);margin:0 0 12px;line-height:1.3}header p{max-width:850px;color:#456;margin:12px 0}
a{color:var(--blue)}.facts{display:flex;gap:36px;flex-wrap:wrap;margin:28px 0}.facts strong{font-size:30px;display:block}
section{margin:36px 0}h2{font-size:24px;margin:16px 0}select{font-size:16px;padding:8px 18px;border:1px solid #ccd7df;border-radius:4px;color:var(--ink);background:white}
table{border-collapse:collapse;width:100%;background:white;font-size:15px}th,td{padding:11px 15px;border-bottom:1px solid #e0e7ed;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#e8f2f8}
.tablewrap{overflow:auto}.best{background:#e7f5ee}nav{display:flex;gap:10px 20px;flex-wrap:wrap;font-size:14px;margin:18px 0}
figure{margin:32px 0;padding:20px;background:white;border:1px solid #e1e8ec}figcaption{font-size:22px;font-weight:600;margin-bottom:14px}
img{width:100%;height:auto;display:block;cursor:zoom-in}dialog{border:0;max-width:96vw;padding:12px;background:white}dialog img{width:auto;max-width:92vw;max-height:85vh;object-fit:contain}
dialog::backdrop{background:#000a}button{background:var(--blue);color:white;border:0;padding:9px 18px;cursor:pointer;float:right}small{color:#526d80}
</style>
<main><header><h1>Gensim 与 FastText<br>新闻分类实验</h1>
<p>从词语上下文学习表示，用文档向量预测人民日报版面。实验逐步增加到 2、4、6 类，并通过验证集比较参数、聚合方式与分类结构。</p>
<a href="https://github.com/Can1017/LLM-HW/blob/main/hw2/docs/P137%E5%AE%9E%E9%AA%8C%E6%8A%A5%E5%91%8A.md">阅读完整实验报告</a> · <a href="https://github.com/Can1017/LLM-HW/blob/main/hw2/docs/source-map.md">查看源码对照</a>
</header><div class="facts"><div><strong>2,083</strong>训练、验证与测试文章</div><div><strong>15</strong>词向量配置</div><div><strong>12</strong>结果图表</div></div>
<section><h2>分类结果</h2><label for="classes">类别数量 </label><select id="classes" onchange="update()"><option>2</option><option>4</option><option selected>6</option></select>
<p><small>行按验证 Macro-F1 排序，浅绿色标记该阶段验证分数最高的方法。准确率与 Macro-F1 列展示独立测试结果。</small></p>
<div class="tablewrap"><table><thead><tr><th>方法</th><th>验证 Macro-F1</th><th>测试 Macro-F1</th><th>测试准确率</th></tr></thead><tbody id="rows"></tbody></table></div></section>
<section><h2>图表浏览</h2><nav>GALLERY_NAV</nav>GALLERY</section>
<footer><small>标签来自原始文章编号的版面字段。外部预训练语料为 2 月新闻，实验评价已有语料条件下的文档分类。</small></footer></main>
<dialog id="zoom"><button onclick="this.parentElement.close()">关闭</button><img alt="放大图表"></dialog>
<script>
const data=DATA_JSON;
function update(){const n=Number(document.getElementById('classes').value);const rows=data.filter(x=>x.classes===n).sort((a,b)=>(b.validation??-1)-(a.validation??-1));document.getElementById('rows').innerHTML=rows.map((r,i)=>`<tr class="${i===0?'best':''}"><td>${r.method}</td><td>${r.validation===null?'—':r.validation.toFixed(4)}</td><td>${r.f1.toFixed(4)}</td><td>${(100*r.accuracy).toFixed(2)}%</td></tr>`).join('')}
function zoom(img){const d=document.getElementById('zoom');d.querySelector('img').src=img.src;d.showModal()}
update();
</script></html>'''
nav = "".join(f'<a href="#f{i+1}">{name}</a>' for i, name in enumerate(names))
html = html.replace("GALLERY_NAV", nav).replace("GALLERY", gallery).replace("DATA_JSON", json.dumps(rows, ensure_ascii=False))
stats = json.loads((ROOT / "results/data_stats.json").read_text(encoding="utf-8"))
total = sum(sum(counts.values()) for counts in stats["split_counts"].values())
html = html.replace("2,083", f"{total:,}")
(ROOT / "docs/实验展示.html").write_text(html, encoding="utf-8")
print("Saved offline results viewer.")
