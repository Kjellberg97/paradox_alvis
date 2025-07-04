#!/bin/env bash

#SBATCH -A SNIC2022-22-744      # find your project with the "projinfo" command
#SBATCH -p alvis               # what partition to use (usually not necessary)
#SBATCH -t 0-20:00:00          # how long time it will take to run
#SBATCH --gpus-per-node=A100fat:1  # choosing no. GPUs and their type
#SBATCH -J RP_10X_ALL_test           # the jobname (not necessary)

# Load PyTorch using the module tree
module purge

# Load container 

CONTAINER=~/container/paradox.sif

apptainer exec $CONTAINER python ~/paradox_alvis/code/generative_models/eval_step_by_step.py