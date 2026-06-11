#!/bin/bash

#SBATCH -J finetune                     # Job name
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --nodes=1                       # Ensure that all cores are on the same machine with nodes=1
#SBATCH --partition=<PARTITION_NAME>    # Which partition will run your job
#SBATCH --time=0-01:00                  # Allowed runtime in D-HH:MM
#SBATCH --gres=gpu:1                    # (optional) Requesting type and number of GPUs
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

DATASETS=(idrid aptos messidor deepdrid glaucoma papila fives)

for dataset in "${DATASETS[@]}"; do

    backbone="tsimcne_bagnet33_all_256_1000/tsimcne_bagnet33_all_256_1000.pt"
    python -m dual_ifm.classification.finetune --multirun --config-name finetune_5fold experiment_prefix=tsimcnexb dataset="${dataset}" model.name="${backbone}"
    python -m dual_ifm.classification.eval_finetune --multirun --config-name finetune_5fold experiment_prefix=tsimcnexb dataset="${dataset}" model.name="${backbone}"
    python -m dual_ifm.classification.eval_embeddings --multirun --config-name finetune_5fold experiment_prefix=tsimcnexb dataset="${dataset}" model.name="${backbone}"

    backbone="tsimcne2d_bagnet33_all_256_1225/tsimcne2d_bagnet33_all_256_1225.pt"
    python -m dual_ifm.classification.finetune --multirun --config-name finetune_5fold experiment_prefix=tsimcneb dataset="${dataset}" model.name="${backbone}"
    python -m dual_ifm.classification.eval_finetune --multirun --config-name finetune_5fold experiment_prefix=tsimcneb dataset="${dataset}" model.name="${backbone}"
    python -m dual_ifm.classification.eval_embeddings --multirun --config-name finetune_5fold experiment_prefix=tsimcneb dataset="${dataset}" model.name="${backbone}"

    backbone="simclr_bagnet33_all_256_1000/simclr_bagnet33_all_256_1000.pt"
    python -m dual_ifm.classification.finetune --multirun --config-name finetune_5fold experiment_prefix=simclrb dataset="${dataset}" model.name="${backbone}"
    python -m dual_ifm.classification.eval_finetune --multirun --config-name finetune_5fold experiment_prefix=simclrb dataset="${dataset}" model.name="${backbone}"

    backbone="bagnet33"
    python -m dual_ifm.classification.finetune --multirun --config-name finetune_5fold experiment_prefix=imagenetb dataset="${dataset}" model.name="${backbone}"
    python -m dual_ifm.classification.eval_finetune --multirun --config-name finetune_5fold experiment_prefix=imagenetb dataset="${dataset}" model.name="${backbone}"

done

deactivate