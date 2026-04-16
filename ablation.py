"""Full ablation sweep.

Runs the two baselines and three QLoRA ranks end-to-end, producing a
single `results/bleu_scores.csv` with one row per run — the headline
table for the final report.

Each QLoRA training is launched as a fresh Python subprocess because
unloading a 4-bit model from VRAM mid-process is unreliable.
"""
import subprocess
import sys

from baselines.zero_shot import main as zero_shot_main
from baselines.tfidf_baseline import main as tfidf_main
from inference import evaluate_adapter

RANKS = [4, 16, 64]


def run_training(rank):
    print(f"\n========== TRAIN rank={rank} ==========")
    subprocess.run(
        [sys.executable, "finetune.py", "--rank", str(rank)],
        check=True,
    )


def run_eval(rank):
    print(f"\n========== EVAL rank={rank} ==========")
    evaluate_adapter(rank)


def main():
    print("\n########## BASELINE: zero-shot ##########")
    zero_shot_main()

    print("\n########## BASELINE: tf-idf ##########")
    tfidf_main()

    for r in RANKS:
        run_training(r)
        run_eval(r)

    print("\nDone. See results/bleu_scores.csv")


if __name__ == "__main__":
    main()
