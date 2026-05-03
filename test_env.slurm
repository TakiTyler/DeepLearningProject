#!/bin/bash
#SBATCH --job-name=test_environment
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# Load necessary modules (uncomment and adjust according to your cluster's environment)
module load python/python-3.11.4-gcc-12.2.0

# Activate your virtual environment or conda environment (adjust path as needed)
# source venv/bin/activate
# OR
# conda activate base

mkdir -p slurm_logs

echo "Running environment and write permission test..."
python3 test_env.py
echo "Test finished."
