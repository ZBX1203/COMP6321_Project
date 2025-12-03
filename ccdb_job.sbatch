#!/bin/bash
#SBATCH --job-name=comp6321_dnn
#SBATCH --account=def-gzhang-ab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/bingxu97/logs/ccdb_%j.out
#SBATCH --error=/scratch/bingxu97/logs/ccdb_%j.err

set -euo pipefail

# Load required modules (check with 'module avail' on Trillium)
module purge
module load StdEnv/2023
module load gcc/12.3
module load python/3.10
module load cuda/12.9

PROJECT_ROOT=${SLURM_SUBMIT_DIR:-$PWD}
VENV_DIR="/scratch/bingxu97/comp6321_env"

# Create necessary directories in /scratch
mkdir -p "/scratch/bingxu97/logs"
mkdir -p "/scratch/bingxu97/models"
mkdir -p "/scratch/bingxu97/results/plots"

# Activate pre-installed virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export TORCH_HOME=${TORCH_HOME:-$PROJECT_ROOT/.torch_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$PROJECT_ROOT/.hf_cache}

srun python "$PROJECT_ROOT/main.py"
