#!/bin/bash
#SBATCH --array=0-7%30
#SBATCH --error=/cluster/tufts/hugheslab/eharve06/slurmlog/err/log_%j.err
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --mem=64g
#SBATCH --ntasks=4
#SBATCH --output=/cluster/tufts/hugheslab/eharve06/slurmlog/out/log_%j.out
#SBATCH --partition=hugheslab
#SBATCH --time=48:00:00

source ~/.bashrc
conda activate l3d_2024f_cuda12_1

# Define an array of commands
experiments=(
    "python ../src/main_CIFAR-10-Copy1.py --batch_size=128 --criterion='l2-sp' --dataset_directory='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --ELBo --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/train_eval_comparison' --kappa=1.0 --lr_0=0.1 --model_name='l2-sp_kappa=1.0_lr_0=0.1_n=10000_random_state=1001' --n=10000 --num_workers=0 --random_state=1001 --save"
    "python ../src/main_CIFAR-10-Copy1.py --batch_size=128 --criterion='l2-sp' --dataset_directory='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --ELBo --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/train_eval_comparison' --kappa=1.0 --lr_0=0.01 --model_name='l2-sp_kappa=1.0_lr_0=0.01_n=10000_random_state=1001' --n=10000 --num_workers=0 --random_state=1001 --save"
    "python ../src/main_CIFAR-10-Copy1.py --batch_size=128 --criterion='l2-sp' --dataset_directory='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --ELBo --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/train_eval_comparison' --kappa=1.0 --lr_0=0.001 --model_name='l2-sp_kappa=1.0_lr_0=0.001_n=10000_random_state=1001' --n=10000 --num_workers=0 --random_state=1001 --save"
    "python ../src/main_CIFAR-10-Copy1.py --batch_size=128 --criterion='l2-sp' --dataset_directory='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --ELBo --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/train_eval_comparison' --kappa=1.0 --lr_0=0.0001 --model_name='l2-sp_kappa=1.0_lr_0=0.0001_n=10000_random_state=1001' --n=10000 --num_workers=0 --random_state=1001 --save"

    "python ../src/main_CIFAR-10-Copy1.py --batch_size=128 --criterion='l2-sp' --dataset_directory='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --ELBo --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/train_eval_comparison' --kappa=23528.522 --lr_0=0.1 --model_name='l2-sp_kappa=23528.522_lr_0=0.1_n=10000_random_state=1001' --n=10000 --num_workers=0 --random_state=1001 --save"
    "python ../src/main_CIFAR-10-Copy1.py --batch_size=128 --criterion='l2-sp' --dataset_directory='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --ELBo --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/train_eval_comparison' --kappa=23528.522 --lr_0=0.01 --model_name='l2-sp_kappa=23528.522_lr_0=0.01_n=10000_random_state=1001' --n=10000 --num_workers=0 --random_state=1001 --save"
    "python ../src/main_CIFAR-10-Copy1.py --batch_size=128 --criterion='l2-sp' --dataset_directory='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --ELBo --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/train_eval_comparison' --kappa=23528.522 --lr_0=0.001 --model_name='l2-sp_kappa=23528.522_lr_0=0.001_n=10000_random_state=1001' --n=10000 --num_workers=0 --random_state=1001 --save"
    "python ../src/main_CIFAR-10-Copy1.py --batch_size=128 --criterion='l2-sp' --dataset_directory='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --ELBo --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/train_eval_comparison' --kappa=23528.522 --lr_0=0.0001 --model_name='l2-sp_kappa=23528.522_lr_0=0.0001_n=10000_random_state=1001' --n=10000 --num_workers=0 --random_state=1001 --save"
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate
