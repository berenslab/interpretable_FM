#!/bin/bash
set -e

#SBATCH -J pretrain                     # Job name
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --nodes=1                       # Ensure that all cores are on the same machine with nodes=1
#SBATCH --partition=<PARTITION_NAME>    # Which partition will run your job
#SBATCH --time=0-16:00                  # Allowed runtime in D-HH:MM
#SBATCH --cpus-per-task=64              # Number of CPU cores per task (8 GPUs x 8 workers)
#SBATCH --gres=gpu:8                    # (optional) Requesting type and number of GPUs
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

echo Training simclr with cosine loss and bagnet33 on all eye datasets for 1000 epochs...
torchrun --standalone --nproc_per_node=8 -m dual_ifm.simclr.train_multigpu --config-name simclr dataset.name=all dataset.image_size=[256,256] backbone.name=bagnet33 train.batch_size=1024 train.epochs=1000 train.num_workers=8

echo Training tsimcne with euclidean loss and bagnet33 on all eye datasets for 1225 epochs...
torchrun --standalone --nproc_per_node=8 -m dual_ifm.tsimcne.train2d_multigpu --config-name tsimcne2d dataset.name=all dataset.image_size=[256,256] backbone.name=bagnet33 train.batch_size=1024 train.epochs=[1000,25,200] train.num_workers=8

echo Evaluate dual-ifm on areds...
python -m dual_ifm.tsimcne.eval --config-name tsimcne2d dataset.name=all dataset.name_eval=areds dataset.image_size=[256,256] dataset.sample_size=20000 backbone.name=bagnet33 train.batch_size=256 train.epochs=[1000,25,200] train.num_workers=8

deactivate