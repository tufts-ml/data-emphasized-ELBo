#!/bin/bash
#SBATCH --array=0-0%10
#SBATCH --error=slurmlog/err/log_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH --ntasks=4
#SBATCH --output=slurmlog/out/log_%j.out
#SBATCH --partition=ccgpu
#SBATCH --time=48:00:00

source ~/.bashrc
conda activate environment

# Define an array of commands
experiments=(
    TODO
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate
