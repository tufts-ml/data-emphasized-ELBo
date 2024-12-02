#!/bin/bash
#SBATCH --array=0-5%30
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
    "python ../src/main_Flowers_102.py --beta=1e-05 --batch_size=128 --criterion='ptyl' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Flowers_102' --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Flowers_102' --lambd=10000.0 --lr_0=0.1 --model_name='ptyl_beta=1e-05_lambd=10000.0_lr_0=0.1_n=510_random_state=1001' --n=510 --num_workers=0 --random_state=1001 --save"
    "python ../src/main_Flowers_102.py --beta=0.0 --batch_size=128 --criterion='ptyl' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Flowers_102' --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Flowers_102' --lambd=100000.0 --lr_0=0.1 --model_name='ptyl_beta=0.0_lambd=100000.0_lr_0=0.1_n=510_random_state=2001' --n=510 --num_workers=0 --random_state=2001 --save"
    "python ../src/main_Flowers_102.py --beta=1e-06 --batch_size=128 --criterion='ptyl' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Flowers_102' --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Flowers_102' --lambd=1000.0 --lr_0=0.1 --model_name='ptyl_beta=1e-06_lambd=1000.0_lr_0=0.1_n=510_random_state=3001' --n=510 --num_workers=0 --random_state=3001 --save"
    "python ../src/main_Flowers_102.py --beta=0.0 --batch_size=128 --criterion='ptyl' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Flowers_102' --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Flowers_102' --lambd=100000000.0 --lr_0=0.1 --model_name='ptyl_beta=0.0_lambd=100000000.0_lr_0=0.1_n=1020_random_state=1001' --n=1020 --num_workers=0 --random_state=1001 --save"
    "python ../src/main_Flowers_102.py --beta=1e-05 --batch_size=128 --criterion='ptyl' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Flowers_102' --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Flowers_102' --lambd=100000000.0 --lr_0=0.1 --model_name='ptyl_beta=1e-05_lambd=100000000.0_lr_0=0.1_n=1020_random_state=2001' --n=1020 --num_workers=0 --random_state=2001 --save"
    "python ../src/main_Flowers_102.py --beta=0.0001 --batch_size=128 --criterion='ptyl' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Flowers_102' --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Flowers_102' --lambd=100000.0 --lr_0=0.1 --model_name='ptyl_beta=0.0001_lambd=100000.0_lr_0=0.1_n=1020_random_state=3001' --n=1020 --num_workers=0 --random_state=3001 --save"
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate