#!/bin/bash
set -e

#SBATCH -J simclr                       # Job name
#SBATCH --ntasks=2                      # Number of tasks
#SBATCH --nodes=1                       # Ensure that all cores are on the same machine with nodes=1
#SBATCH --partition=<PARTITION_NAME>    # Which partition will run your job
#SBATCH --time=0-04:00                  # Allowed runtime in D-HH:MM
#SBATCH --cpus-per-task=4               # Number of CPU cores per task
#SBATCH --gres=gpu:2                    # (optional) Requesting type and number of GPUs
#SBATCH --output=<RESULTS_PATH>.out     # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=<RESULTS_PATH>.err      # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=ALL                 # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=<YOUR_EMAIL>        # Email to which notifications will be sent

# Diagnostic and Analysis Phase - please leave these in.
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi # only if you requested gpus

# Setup Phase
source ~/.bashrc
source .venv/bin/activate

echo Training simclr on cifar10 for 500 epochs...
torchrun --standalone --nproc_per_node=2 -m dual_ifm.simclr.train_multigpu --config-name simclr train.batch_size=4096 train.epochs=500

echo Evaluate simclr on cifar10...
python -m dual_ifm.simclr.eval --config-name simclr train.epochs=500


deactivate
