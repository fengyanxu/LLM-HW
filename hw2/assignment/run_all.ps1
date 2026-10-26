# 在仓库根目录执行：.\hw2\assignment\run_all.ps1
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $root 'hw2\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "未找到 hw2\.venv。请先运行：py -3.11 -m venv hw2\.venv；然后安装 hw2\assignment\requirements.txt"
}
$env:PYTHONUTF8 = '1'
$scripts = @(
    '01_fetch_thucnews.py',
    '04_prepare_embeddings.py',
    '05_train_embeddings.py',
    '06_analyze_embeddings.py',
    '03_build_classification.py',
    '07_train_fasttext.py',
    '11_sensitivity.py',
    '12_tfidf_baseline.py',
    '08_make_figures.py',
    '09_make_report.py',
    '10_verify.py'
)
foreach ($name in $scripts) {
    Write-Host "=== $name ===" -ForegroundColor Cyan
    & $python (Join-Path $PSScriptRoot "scripts\$name")
    if ($LASTEXITCODE -ne 0) { throw "执行失败：$name" }
}
Write-Host "全部完成：$PSScriptRoot\report.md" -ForegroundColor Green
