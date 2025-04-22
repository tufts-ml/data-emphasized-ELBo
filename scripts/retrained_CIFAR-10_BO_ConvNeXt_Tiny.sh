#!/bin/bash
#SBATCH --array=0-0%20
#SBATCH --error=/cluster/tufts/hugheslab/eharve06/slurmlog/err/log_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH --ntasks=4
#SBATCH --output=/cluster/tufts/hugheslab/eharve06/slurmlog/out/log_%j.out
#SBATCH --partition=ccgpu
#SBATCH --time=48:00:00

source ~/.bashrc
conda activate l3d_2024f_cuda12_1

# Define an array of commands
experiments=(
    "python ../src/main_CIFAR-10.py --alpha=0.01 --beta=0.01 --batch_size=128 --criterion='l2-sp' --dataset_directory='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_CIFAR-10_BO_ConvNeXt_Tiny' --lr_0=0.04967567598569534 --model_arch='ConvNeXt_Tiny' --model_name='l2-sp_alpha=0.01_beta=0.01_lr_0=0.04967567598569534_n=50000_random_state=3001' --n=50000 --num_workers=0 --prior_directory='/cluster/tufts/hugheslab/eharve06/convnext_tiny_torchvision' --prior_type='convnext_tiny_torchvision' --random_state=3001 --save"
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate