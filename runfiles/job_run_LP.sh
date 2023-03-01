#!/bin/env bash

#SBATCH -A SNIC2022-22-744      # find your project with the "projinfo" command
#SBATCH -p alvis               # what partition to use (usually not necessary)
#SBATCH -t 0-06:00:00          # how long time it will take to run
#SBATCH --gpus-per-node=A100:4  # choosing no. GPUs and their type
#SBATCH -J LP             # the jobname (not necessary)

# Load PyTorch using the module tree
module purge

# Load container 

CONTAINER=~/container/paradox.sif

apptainer exec $CONTAINER python ~/paradox_alvis/train_BERT_LP.py