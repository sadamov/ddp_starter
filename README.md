## DDP Starter

This is a starter project for distributed deep learning with PyTorch and Slurm.

### Pre-requisites

- [Mamba](https://mamba.readthedocs.io/en/latest/installation.html)
- [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- Access to a Slurm cluster (e.g. balfrin@cscs.ch)

### Installation
Copy latest code from github:
```
git clone git@github.com:sadamov/ddp_starter.git
cd ddp_starter
```
Create a new conda environment and install dependencies:
```
mamba env create -f environment.yml
```

### Usage
```
sbatch test_slurm.sh
```
Then check out the logs in `./lightning_logs` to see if the run was successful. The
`metrics.csv` contains the training and validation losses across all epochs.
```
`Trainer.fit` stopped: `max_epochs=10` reached.
```
means that the run was successful.

For real case trainings, you will need to modify the `batch_size` and `num_workers` in
the dataloader of `class IrisDataModule` to best utilize the available GPU and CPU
resources.
