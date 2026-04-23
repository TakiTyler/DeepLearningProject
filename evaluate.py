"""Shared BLEU evaluator used by every run (zero-shot, TF-IDF, QLoRA).

The `generate_fn` callback decouples the evaluator from the model backend
so a single function compares all 5 runs head-to-head.
"""
import os
import csv
import sacrebleu

from inference import format_reference


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _append_score(row, scores_csv_path):
    is_new = not os.path.exists(scores_csv_path)
    with open(scores_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["run_name", "rank", "bleu", "mean_output_len", "num_eval"])
        writer.writerow(row)


def _write_samples(run_name, samples, samples_dir):
    path = os.path.join(samples_dir, f"{run_name}.txt")
    with open(path, "w", encoding="utf-8") as f:
        for i, s in enumerate(samples):
            f.write(f"=== sample {i+1} ===\n")
            f.write("INGREDIENTS: " + ", ".join(s["ingredients"]) + "\n\n")
            f.write("REFERENCE:\n" + s["reference"] + "\n\n")
            f.write("GENERATED:\n" + s["generated"] + "\n\n")


def evaluate_model(generate_fn, test_dataset, run_name, rank="-", num_eval=200, num_samples_to_log=5, output_dir="."):
    """Run `generate_fn` on the first `num_eval` examples and log BLEU.

    Args:
        generate_fn: callable `(ingredients: list[str]) -> str`
        test_dataset: HF Dataset with columns name, ingredients, steps
        run_name: identifier for the row in bleu_scores.csv
        rank: "-" for baselines, integer for QLoRA runs
        output_dir: Base directory for results.
    """
    results_dir = os.path.join(output_dir, "results")
    samples_dir = os.path.join(results_dir, "samples")
    scores_csv_path = os.path.join(results_dir, "bleu_scores.csv")
    _ensure_dir(samples_dir)

    n = min(num_eval, len(test_dataset))
    hypotheses = []
    references = []
    samples = []

    print(f"[{run_name}] evaluating on {n} examples...")
    for i in range(n):
        ex = test_dataset[i]
        hyp = generate_fn(ex["ingredients"])
        ref = format_reference(ex["name"], ex["steps"])
        hypotheses.append(hyp)
        references.append(ref)

        if i < num_samples_to_log:
            samples.append({
                "ingredients": ex["ingredients"],
                "reference": ref,
                "generated": hyp,
            })

        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{n}")

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    mean_len = sum(len(h.split()) for h in hypotheses) / max(1, len(hypotheses))

    print(f"[{run_name}] BLEU = {bleu.score:.2f}  mean_len = {mean_len:.1f}")
    _append_score([run_name, rank, f"{bleu.score:.4f}", f"{mean_len:.2f}", n], scores_csv_path)
    _write_samples(run_name, samples, samples_dir)

    return {"run_name": run_name, "bleu": bleu.score, "mean_len": mean_len}
