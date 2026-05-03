"""Full ablation sweep.

Runs the two baselines and three QLoRA ranks end-to-end, producing a
single `results/evaluation_scores.csv` with one row per run — the headline
table for the final report.

Each QLoRA training is launched as a fresh Python subprocess because
unloading a 4-bit model from VRAM mid-process is unreliable.
"""
import argparse
import subprocess
import sys

from baselines.zero_shot import main as zero_shot_main
from baselines.tfidf_baseline import main as tfidf_main
from inference import evaluate_adapter

RANKS = [4, 16, 64]
# added flush = True to write the prints immediately. want to make sure these things are running
def run_training(rank, output_dir):
    print(f"\n========== TRAIN rank={rank} ==========", flush=True)
    subprocess.run(
        [sys.executable, "finetune.py", "--rank", str(rank), "--output_dir", output_dir],
        check=True,
    )
    # USE THIS IF YOU WANNA TRAIN FOR MORE EPOCHS, takes ~30min per epoch
    # subprocess.run(
    #     [sys.executable, "finetune.py", "--rank", str(rank), "--output_dir", output_dir, "--epochs", "3"],
    #     check=True,
    # )


def run_eval(rank, output_dir):
    print(f"\n========== EVAL rank={rank} ==========", flush=True)
    evaluate_adapter(rank, output_dir=output_dir)


def main():
    import torch
    print("\n========== HARDWARE CHECK ==========", flush=True)
    print(f"PyTorch version: {torch.__version__}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("WARNING: No GPU detected! Inference will run on CPU and be extremely slow.", flush=True)
    print("====================================\n", flush=True)

    parser = argparse.ArgumentParser(description="Run full ablation sweep.")
    parser.add_argument("--output_dir", type=str, default=".", help="Base directory for all outputs.")
    parser.add_argument("--skip_baselines", action="store_true", help="Skip baselines to only run QLoRA sweeps.")
    args = parser.parse_args()

    if not args.skip_baselines:
        print("\n########## BASELINE: zero-shot ##########", flush=True)
        zero_shot_main(output_dir=args.output_dir)

        print("\n########## BASELINE: tf-idf ##########", flush=True)
        tfidf_main(output_dir=args.output_dir)
    else:
        print("\n########## SKIPPING BASELINES ##########", flush=True)

    for r in RANKS:
        run_training(r, args.output_dir)
        run_eval(r, args.output_dir)

    print(f"\nDone. See results in {args.output_dir}/results/evaluation_scores.csv", flush=True)

if __name__ == "__main__":
    main()
