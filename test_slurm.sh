#!/bin/bash
#SBATCH --job-name=pl_ddp_test
#SBATCH --output=pl_ddp_test.out
#SBATCH --error=pl_ddp_test.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --partition=normal
#SBATCH --account=s83

# Load necessary modules
source ${CONDA_PREFIX}/etc/profile.d/conda.sh
conda activate neural-ddp

NUM_GPUS=$(echo $SLURM_JOB_GPUS | tr ',' '\n' | wc -l)

# Run the script
srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=$NUM_GPUS test_ddp.py
