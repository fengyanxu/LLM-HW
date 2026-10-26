"""Entry point: execute from the project directory."""
import argparse
import os
import sys
import subprocess
import time
from pathlib import Path

# Set before numerical imports; the invoking shell must additionally set PYTHONHASHSEED.
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(key, "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "embeddings", "evaluate", "paper", "plot", "report", "all"])
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    if args.stage == "all":
        subprocess.run([sys.executable, str(ROOT / "scripts/audit_sources.py")], check=True)
    from newsvec.common import read_json, write_json
    config = read_json(ROOT / "configs/experiment.json")
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        if args.stage in {"prepare", "all"}:
            if args.archive is None:
                parser.error("--archive is required for prepare/all")
            from newsvec.prepare import prepare
            prepare(args.archive, config)
        if args.stage in {"embeddings", "all"}:
            from newsvec.embeddings import train_embeddings
            train_embeddings(config, args.only)
        if args.stage in {"evaluate", "all"}:
            from newsvec.evaluate import evaluate
            evaluate(config)
        if args.stage in {"paper", "all"}:
            from newsvec.paper_classifier import run_paper_classifier
            run_paper_classifier(config)
        if args.stage in {"plot", "all"}:
            from newsvec.visualize import plot_all
            plot_all(config)
            subprocess.run([sys.executable, str(ROOT / "scripts/build_showcase.py")], check=True)
        if args.stage in {"report", "all"}:
            from newsvec.report import make_report
            make_report(config)


if __name__ == "__main__":
    main()
