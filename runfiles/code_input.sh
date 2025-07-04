#!/bin/env bash

#SBATCH -A SNIC2022-22-744      # find your project with the "projinfo" command
#SBATCH -t 0-00:10:00          # how long time it will take to run
#SBATCH -C NOGPU  # choosing no. GPUs and their type
#SBATCH -J reformat             # the jobname (not necessary)

# Load PyTorch using the module tree
module purge

# Load container 

CONTAINER=~/container/paradox.sif

apptainer exec $CONTAINER python ~/paradox_alvis/code/word_coding/coding_of_words.py