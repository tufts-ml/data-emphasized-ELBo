#!/bin/bash
#SBATCH --array=0-0
#SBATCH --error=/cluster/tufts/hugheslab/eharve06/slurmlog/err/log_%j.err
#SBATCH --mem=64g
#SBATCH --ntasks=4
#SBATCH --output=/cluster/tufts/hugheslab/eharve06/slurmlog/out/log_%j.out
#SBATCH --partition=batch
#SBATCH --time=48:00:00

source ~/.bashrc
conda activate l3d_2024f_cuda12_1

# Define an array of commands
experiments=(
    "python ../notebooks/subsample_hyperparameters.py"
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate