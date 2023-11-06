#!/bin/bash
#SBATCH --job-name=pl_ddp_test
#SBATCH --output=pl_ddp_test.out
#SBATCH --error=pl_ddp_test.err
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=01:00:00
#SBATCH --partition=normal
#SBATCH --account=s83

# Load necessary modules
source ${CONDA_PREFIX}/etc/profile.d/conda.sh
conda activate ddp_starter

# Run the script
srun -ul --gpus-per-task=1 python test_ddp.py
