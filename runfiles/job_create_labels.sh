#!/bin/env bash

#SBATCH -A SNIC2022-22-744      # find your project with the "projinfo" command
#SBATCH -t 0-09:00:00          # how long time it will take to run
#SBATCH --gpus-per-node=A40:2  # choosing no. GPUs and their type
#SBATCH -J create_labels             # the jobname (not necessary)

# Load PyTorch using the module tree
module purge


# Load container 

CONTAINER=~/container/paradox.sif

apptainer exec $CONTAINER python ~/paradox_alvis/code/generate_target_labels/run_label_gen.py