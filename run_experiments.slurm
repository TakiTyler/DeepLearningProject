#!/bin/bash
#SBATCH --job-name=recipe_generation_ablation
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Load necessary modules (uncomment and adjust according to your cluster's environment)
module load python/python-3.11.4-gcc-12.2.0
module load cuda/cuda-12.1.0

# Activate your virtual environment or conda environment (adjust path as needed)
source venv/bin/activate
# conda activate myenv

# Create directories for slurm logs and our experiment outputs
mkdir -p slurm_logs
mkdir -p ablation_outputs

# Authenticate w/ hugging face access
export HF_TOKEN="getyourowntoken"
echo "Starting ablation study..."

# Run the complete ablation script which handles baselines, training, evaluation, and saving models
python3 ablation.py --output_dir ablation_outputs

echo "!!!! ==== Ablation study finished. ==== !!!!"