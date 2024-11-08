#!/bin/bash
#SBATCH --array=0-11%20
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
    "python ../src/main_CIFAR-10.py --beta=0.0 --batch_size=128 --criterion='ptyl' --dataset_directory='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_CIFAR-10' --lambd=10.0 --lr_0=0.001 --model_name='ptyl_beta=0.0_lambd=10.0_lr_0=0.001_n=100_random_state=1001' --n=100 --num_workers=0 --random_state=1001 --save"
    "python ../src/main_CIFAR-10.py --beta=0.01 --batch_size=128 --criterion='ptyl' --dataset_directory='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_CIFAR-10' --lambd=1.0 --lr_0=0.01 --model_name='ptyl_beta=0.01_lambd=1.0_lr_0=0.01_n=100_random_state=2001' --n=100 --num_workers=0 --random_state=2001 --save"
    "python ../src/main_CIFAR-10.py --beta=0.001 --batch_size=128 --criterion='ptyl' --dataset_directory='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_CIFAR-10' --lambd=10.0 --lr_0=0.1 --model_name='ptyl_beta=0.001_lambd=10.0_lr_0=0.1_n=100_random_state=3001' --n=100 --num_workers=0 --random_state=3001 --save"
    "python ../src/main_CIFAR-10.py --beta=0.001 --batch_size=128 --criterion='ptyl' --dataset_directory='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_CIFAR-10' --lambd=10.0 --lr_0=0.1 --model_name='ptyl_beta=0.001_lambd=10.0_lr_0=0.1_n=1000_random_state=1001' --n=1000 --num_workers=0 --random_state=1001 --save"
    "python ../src/main_CIFAR-10.py --beta=0.001 --batch_size=128 --criterion='ptyl' --dataset_directory='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_CIFAR-10' --lambd=10.0 --lr_0=0.1 --model_name='ptyl_beta=0.001_lambd=10.0_lr_0=0.1_n=1000_random_state=2001' --n=1000 --num_workers=0 --random_state=2001 --save"
    "python ../src/main_CIFAR-10.py --beta=0.001 --batch_size=128 --criterion='ptyl' --dataset_directory='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_CIFAR-10' --lambd=10.0 --lr_0=0.1 --model_name='ptyl_beta=0.001_lambd=10.0_lr_0=0.1_n=1000_random_state=3001' --n=1000 --num_workers=0 --random_state=3001 --save"
    "python ../src/main_CIFAR-10.py --beta=0.01 --batch_size=128 --criterion='ptyl' --dataset_directory='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_CIFAR-10' --lambd=1.0 --lr_0=0.1 --model_name='ptyl_beta=0.01_lambd=1.0_lr_0=0.1_n=10000_random_state=1001' --n=10000 --num_workers=0 --random_state=1001 --save"
    "python ../src/main_CIFAR-10.py --beta=0.01 --batch_size=128 --criterion='ptyl' --dataset_directory='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_CIFAR-10' --lambd=1.0 --lr_0=0.1 --model_name='ptyl_beta=0.01_lambd=1.0_lr_0=0.1_n=10000_random_state=2001' --n=10000 --num_workers=0 --random_state=2001 --save"
    "python ../src/main_CIFAR-10.py --beta=0.01 --batch_size=128 --criterion='ptyl' --dataset_directory='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_CIFAR-10' --lambd=1.0 --lr_0=0.1 --model_name='ptyl_beta=0.01_lambd=1.0_lr_0=0.1_n=10000_random_state=3001' --n=10000 --num_workers=0 --random_state=3001 --save"
    "python ../src/main_CIFAR-10.py --beta=0.01 --batch_size=128 --criterion='ptyl' --dataset_directory='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_CIFAR-10' --lambd=1.0 --lr_0=0.1 --model_name='ptyl_beta=0.01_lambd=1.0_lr_0=0.1_n=50000_random_state=1001' --n=50000 --num_workers=0 --random_state=1001 --save"
    "python ../src/main_CIFAR-10.py --beta=0.01 --batch_size=128 --criterion='ptyl' --dataset_directory='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_CIFAR-10' --lambd=1.0 --lr_0=0.1 --model_name='ptyl_beta=0.01_lambd=1.0_lr_0=0.1_n=50000_random_state=2001' --n=50000 --num_workers=0 --random_state=2001 --save"
    "python ../src/main_CIFAR-10.py --beta=0.01 --batch_size=128 --criterion='ptyl' --dataset_directory='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_CIFAR-10' --lambd=1.0 --lr_0=0.1 --model_name='ptyl_beta=0.01_lambd=1.0_lr_0=0.1_n=50000_random_state=3001' --n=50000 --num_workers=0 --random_state=3001 --save"
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate
