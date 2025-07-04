#!/bin/env bash

#SBATCH -A SNIC2022-22-744      # find your project with the "projinfo" command
#SBATCH -p alvis               # what partition to use (usually not necessary)
#SBATCH -t 0-00:05:00          # how long time it will take to run
#SBATCH --gpus-per-node=A40:1 # --gpus-per-node=T4:1 -C NOGPU # choosing no. GPUs and their type
#SBATCH -J ProofCheckerSBS           # the jobname (not necessary)

# Load PyTorch using the module tree
module purge

# Load container 

CONTAINER=~/container/paradox.sif

apptainer exec $CONTAINER python ~/paradox_alvis/code/evaluation/proof_checker_step_by_step.py