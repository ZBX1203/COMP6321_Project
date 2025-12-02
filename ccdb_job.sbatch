#!/bin/bash
#SBATCH --job-name=comp6321_dnn
#SBATCH --account=def-gzhang-ab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=~/scratch/logs/ccdb_%j.out
#SBATCH --error=~/scratch/logs/ccdb_%j.err

set -euo pipefail

# Load required modules (check with 'module avail' on Trillium)
module purge
module load python/3.10
module load cuda/12

PROJECT_ROOT=${SLURM_SUBMIT_DIR:-$PWD}
VENV_DIR="$HOME/comp6321_env"

mkdir -p "$HOME/scratch/logs"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Install dependencies
echo "Installing dependencies..."
python -m pip install --upgrade pip
pip install --upgrade -r "$PROJECT_ROOT/requirements.txt"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export TORCH_HOME=${TORCH_HOME:-$PROJECT_ROOT/.torch_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$PROJECT_ROOT/.hf_cache}

srun python "$PROJECT_ROOT/main.py"
