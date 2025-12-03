#!/bin/bash
#SBATCH --job-name=comp6321_test
#SBATCH --account=def-gzhang-ab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --partition=debug
#SBATCH --output=/scratch/bingxu97/logs/test_%j.out
#SBATCH --error=/scratch/bingxu97/logs/test_%j.err

set -euo pipefail

echo "=========================================="
echo "QUICK TEST - Code Validation"
echo "=========================================="
echo "Start time: $(date)"

# Load modules
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

# Set environment variables for data caching
export SCIKIT_LEARN_DATA="/scratch/bingxu97/scikit_learn_data"
export TORCH_HOME="/scratch/bingxu97/torch_cache"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# Verify GPU
echo ""
echo "=========================================="
echo "GPU Verification"
echo "=========================================="
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Run quick test
echo ""
echo "=========================================="
echo "Running Quick Test"
echo "=========================================="
srun python "$PROJECT_ROOT/test_quick.py"

echo ""
echo "=========================================="
echo "Test completed at: $(date)"
echo "=========================================="
