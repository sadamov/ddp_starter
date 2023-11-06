#!/bin/bash -l
#SBATCH --job-name=pl_ddp_test
#SBATCH --output=lightning_logs/pl_ddp_test.out
#SBATCH --error=lightning_logs/pl_ddp_test.err
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=01:00:00
#SBATCH --partition=a100-80gb
#SBATCH --account=s83

# Load necessary modules
conda activate neural-ddp

# Run the script
srun -ul --gpus-per-task=1 python test_ddp.py
