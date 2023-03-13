#!/bin/env bash

#SBATCH -A SNIC2022-22-744      # find your project with the "projinfo" command
#SBATCH -t 0-00:02:00          # how long time it will take to run
#SBATCH -C NOGPU  # choosing no. GPUs and their type
#SBATCH -J run_proof_checker             # the jobname (not necessary)

# Load PyTorch using the module tree
module purge


# Load container 

CONTAINER=~/container/paradox.sif

apptainer exec $CONTAINER python ~/paradox_alvis/code/evaluation/run_proof_checker.py
