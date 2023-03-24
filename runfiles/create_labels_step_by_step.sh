#!/bin/env bash

#SBATCH -A SNIC2022-22-744      # find your project with the "projinfo" command
#SBATCH -t 0-09:00:00          # how long time it will take to run
#SBATCH --gpus-per-node=A40:4  # choosing no. GPUs and their type
#SBATCH -J step_by_step            # the jobname (not necessary)

# Load PyTorch using the module tree
module purge


# Load container 

CONTAINER=~/container/paradox.sif

apptainer exec $CONTAINER python ~/paradox_alvis/code/gen_labels_step_by_step/run_label_generator_step_by_step.py