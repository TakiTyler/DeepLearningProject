# Recipe Generation with Gemma-2B: A QLoRA Ablation Study

This project fine-tunes Google's **Gemma-2B** large language model to generate structured food recipes based on a list of input ingredients.

To evaluate the optimal training configuration, the project conducts an **ablation study** over the intrinsic rank ($r$) of the Low-Rank Adaptation (LoRA) matrices ($r \in \{4, 16, 64\}$). It compares these fine-tuned adapters against two baselines: a Zero-Shot Gemma-2B baseline and a TF-IDF K-Nearest Neighbors retrieval baseline.

## Table of Contents
- [Environment Setup](#environment-setup)
- [Hugging Face Authentication](#hugging-face-authentication)
- [Running on a SLURM Cluster](#running-on-a-slurm-cluster)
- [Project Structure & Outputs](#project-structure--outputs)
- [Interactive Demo](#interactive-demo)

---

## Environment Setup

Due to library version issues, this project relies on strictly pinned dependency versions to ensure compatibility between `transformers`, `accelerate`, 4-bit quantization, and `liger-kernel`.

**This portion was our setup script for usage on the UCF Newton GPU Cluster.**

1. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install the exact required dependencies:**
   ```bash
   pip install -U torch --index-url https://download.pytorch.org/whl/cu121
   pip install -U "transformers<4.45.0" "accelerate<0.34.0" peft "trl<0.12.0" bitsandbytes datasets pandas scikit-learn rouge-score "liger-kernel<0.3.0"
   ```

---

## Hugging Face Authentication

The base model (`google/gemma-2b-it`) is a gated model. You must accept the usage license on the Hugging Face website before downloading it.

1. Go to the Gemma-2B-it Hugging Face page and accept the terms.
2. Generate a Hugging Face Access Token (Read/Write) in your account settings.
3. Update the `run_experiments.slurm` script with your token:
   ```bash
   export HF_TOKEN="your_token_here"
   ```

---

## Running on a SLURM Cluster

The entire experimental pipeline—including data splitting, baselines, training all three QLoRA ranks, and ROUGE evaluation—is orchestrated by `ablation.py` and submitted via SLURM.

1. Ensure your dataset CSV files are placed in the `data/` directory (e.g., `data/RAW_recipes.csv`).
2. Review `run_experiments.slurm` to ensure the loaded modules match your cluster's specific Python and CUDA installations:
   ```bash
   module load python/python-3.11.4-gcc-12.2.0
   module load cuda/cuda-12.1.0
   ```
3. **Submit the job:**
   ```bash
   sbatch run_experiments.slurm
   ```

### Monitoring Progress
You can monitor the live output of your job by tailing the generated `.out` file in the `slurm_logs/` directory:
```bash
tail -f slurm_logs/recipe_generation_ablation-<job_id>.out
```

---

## Project Structure & Outputs

Once the SLURM job completes, all generated artifacts are saved in the `ablation_outputs/` directory.

```text
ablation_outputs/
├── gemma-2b-recipe-adapter-r4/     # Final LoRA adapter weights (Rank 4)
├── gemma-2b-recipe-adapter-r16/    # Final LoRA adapter weights (Rank 16)
├── gemma-2b-recipe-adapter-r64/    # Final LoRA adapter weights (Rank 64)
└── results/
    ├── evaluation_scores.csv       # Headline results table (ROUGE-1, ROUGE-2, ROUGE-L)
    ├── training_logs_r4.csv        # Epoch loss/eval_loss history for Rank 4
    ├── training_logs_r16.csv       # Epoch loss/eval_loss history for Rank 16
    ├── training_logs_r64.csv       # Epoch loss/eval_loss history for Rank 64
    └── samples/
        ├── zero_shot.txt           # Sample generations from the baseline
        ├── tfidf.txt               # Sample retrievals from the TF-IDF baseline
        ├── qlora_r4.txt            # Sample generations from the r=4 adapter
        └── ...
```

---

## Interactive Demo

After the adapters have been successfully trained, you can interact with the fine-tuned model directly in your terminal to generate recipes.

Ensure your virtual environment is activated and run the inference script. Specify which rank you want to load (default is 16):

```bash
python inference.py --rank 16 --output_dir ablation_outputs
```

**Example Usage:**
```text
Loading adapter r=16...
Ready. Enter comma-separated ingredients (or 'quit').
> chicken, garlic, heavy cream, parmesan, pasta
```