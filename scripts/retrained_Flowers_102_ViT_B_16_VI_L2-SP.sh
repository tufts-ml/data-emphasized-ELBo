#!/bin/bash
#SBATCH --array=0-23%30
#SBATCH --error=/cluster/tufts/hugheslab/eharve06/slurmlog/err/log_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --nodelist=cc1gpu[001,003,004,005]
#SBATCH --ntasks=4
#SBATCH --output=/cluster/tufts/hugheslab/eharve06/slurmlog/out/log_%j.out
#SBATCH --partition=ccgpu
#SBATCH --time=48:00:00

source ~/.bashrc
conda activate l3d_2024f_cuda12_1

# Define an array of commands
experiments=(
    "python ../src/main_Flowers_102.py --batch_size=128 --criterion='l2-sp' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Flowers_102' --ELBo --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Flowers_102_ViT_B_16_VI' --kappa=1.0 --lr_0=0.1 --model_arch='ViT-B/16' --model_name='l2-sp_kappa=1.0_lr_0=0.1_n=510_random_state=1001' --n=510 --num_workers=0 --prior_directory='/cluster/tufts/hugheslab/eharve06/vit_b_16_torchvision' --prior_type='vit_b_16_torchvision' --random_state=1001 --save"
    "python ../src/main_Flowers_102.py --batch_size=128 --criterion='l2-sp' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Flowers_102' --ELBo --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Flowers_102_ViT_B_16_VI' --kappa=1.0 --lr_0=0.1 --model_arch='ViT-B/16' --model_name='l2-sp_kappa=1.0_lr_0=0.1_n=510_random_state=2001' --n=510 --num_workers=0 --prior_directory='/cluster/tufts/hugheslab/eharve06/vit_b_16_torchvision' --prior_type='vit_b_16_torchvision' --random_state=2001 --save"
    "python ../src/main_Flowers_102.py --batch_size=128 --criterion='l2-sp' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Flowers_102' --ELBo --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Flowers_102_ViT_B_16_VI' --kappa=1.0 --lr_0=0.1 --model_arch='ViT-B/16' --model_name='l2-sp_kappa=1.0_lr_0=0.1_n=510_random_state=3001' --n=510 --num_workers=0 --prior_directory='/cluster/tufts/hugheslab/eharve06/vit_b_16_torchvision' --prior_type='vit_b_16_torchvision' --random_state=3001 --save"
    "python ../src/main_Flowers_102.py --batch_size=128 --criterion='l2-sp' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Flowers_102' --ELBo --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Flowers_102_ViT_B_16_VI' --kappa=1.0 --lr_0=0.1 --model_arch='ViT-B/16' --model_name='l2-sp_kappa=1.0_lr_0=0.1_n=1020_random_state=1001' --n=1020 --num_workers=0 --prior_directory='/cluster/tufts/hugheslab/eharve06/vit_b_16_torchvision' --prior_type='vit_b_16_torchvision' --random_state=1001 --save"
    "python ../src/main_Flowers_102.py --batch_size=128 --criterion='l2-sp' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Flowers_102' --ELBo --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Flowers_102_ViT_B_16_VI' --kappa=1.0 --lr_0=0.1 --model_arch='ViT-B/16' --model_name='l2-sp_kappa=1.0_lr_0=0.1_n=1020_random_state=2001' --n=1020 --num_workers=0 --prior_directory='/cluster/tufts/hugheslab/eharve06/vit_b_16_torchvision' --prior_type='vit_b_16_torchvision' --random_state=2001 --save"
    "python ../src/main_Flowers_102.py --batch_size=128 --criterion='l2-sp' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Flowers_102' --ELBo --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Flowers_102_ViT_B_16_VI' --kappa=1.0 --lr_0=0.1 --model_arch='ViT-B/16' --model_name='l2-sp_kappa=1.0_lr_0=0.1_n=1020_random_state=3001' --n=1020 --num_workers=0 --prior_directory='/cluster/tufts/hugheslab/eharve06/vit_b_16_torchvision' --prior_type='vit_b_16_torchvision' --random_state=3001 --save"
    "python ../src/main_Flowers_102.py --batch_size=128 --criterion='l2-sp' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Flowers_102' --ELBo --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Flowers_102_ViT_B_16_VI' --kappa=1.0 --lr_0=0.01 --model_arch='ViT-B/16' --model_name='l2-sp_kappa=1.0_lr_0=0.01_n=510_random_state=1001' --n=510 --num_workers=0 --prior_directory='/cluster/tufts/hugheslab/eharve06/vit_b_16_torchvision' --prior_type='vit_b_16_torchvision' --random_state=1001 --save"
    "python ../src/main_Flowers_102.py --batch_size=128 --criterion='l2-sp' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Flowers_102' --ELBo --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Flowers_102_ViT_B_16_VI' --kappa=1.0 --lr_0=0.01 --model_arch='ViT-B/16' --model_name='l2-sp_kappa=1.0_lr_0=0.01_n=510_random_state=2001' --n=510 --num_workers=0 --prior_directory='/cluster/tufts/hugheslab/eharve06/vit_b_16_torchvision' --prior_type='vit_b_16_torchvision' --random_state=2001 --save"
    "python ../src/main_Flowers_102.py --batch_size=128 --criterion='l2-sp' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Flowers_102' --ELBo --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Flowers_102_ViT_B_16_VI' --kappa=1.0 --lr_0=0.01 --model_arch='ViT-B/16' --model_name='l2-sp_kappa=1.0_lr_0=0.01_n=510_random_state=3001' --n=510 --num_workers=0 --prior_directory='/cluster/tufts/hugheslab/eharve06/vit_b_16_torchvision' --prior_type='vit_b_16_torchvision' --random_state=3001 --save"
    "python ../src/main_Flowers_102.py --batch_size=128 --criterion='l2-sp' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Flowers_102' --ELBo --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Flowers_102_ViT_B_16_VI' --kappa=1.0 --lr_0=0.01 --model_arch='ViT-B/16' --model_name='l2-sp_kappa=1.0_lr_0=0.01_n=1020_random_state=1001' --n=1020 --num_workers=0 --prior_directory='/cluster/tufts/hugheslab/eharve06/vit_b_16_torchvision' --prior_type='vit_b_16_torchvision' --random_state=1001 --save"
    "python ../src/main_Flowers_102.py --batch_size=128 --criterion='l2-sp' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Flowers_102' --ELBo --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Flowers_102_ViT_B_16_VI' --kappa=1.0 --lr_0=0.01 --model_arch='ViT-B/16' --model_name='l2-sp_kappa=1.0_lr_0=0.01_n=1020_random_state=2001' --n=1020 --num_workers=0 --prior_directory='/cluster/tufts/hugheslab/eharve06/vit_b_16_torchvision' --prior_type='vit_b_16_torchvision' --random_state=2001 --save"
    "python ../src/main_Flowers_102.py --batch_size=128 --criterion='l2-sp' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Flowers_102' --ELBo --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Flowers_102_ViT_B_16_VI' --kappa=1.0 --lr_0=0.01 --model_arch='ViT-B/16' --model_name='l2-sp_kappa=1.0_lr_0=0.01_n=1020_random_state=3001' --n=1020 --num_workers=0 --prior_directory='/cluster/tufts/hugheslab/eharve06/vit_b_16_torchvision' --prior_type='vit_b_16_torchvision' --random_state=3001 --save"
    "python ../src/main_Flowers_102.py --batch_size=128 --criterion='l2-sp' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Flowers_102' --ELBo --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Flowers_102_ViT_B_16_VI' --kappa=1.0 --lr_0=0.001 --model_arch='ViT-B/16' --model_name='l2-sp_kappa=1.0_lr_0=0.001_n=510_random_state=1001' --n=510 --num_workers=0 --prior_directory='/cluster/tufts/hugheslab/eharve06/vit_b_16_torchvision' --prior_type='vit_b_16_torchvision' --random_state=1001 --save"
    "python ../src/main_Flowers_102.py --batch_size=128 --criterion='l2-sp' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Flowers_102' --ELBo --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Flowers_102_ViT_B_16_VI' --kappa=1.0 --lr_0=0.001 --model_arch='ViT-B/16' --model_name='l2-sp_kappa=1.0_lr_0=0.001_n=510_random_state=2001' --n=510 --num_workers=0 --prior_directory='/cluster/tufts/hugheslab/eharve06/vit_b_16_torchvision' --prior_type='vit_b_16_torchvision' --random_state=2001 --save"
    "python ../src/main_Flowers_102.py --batch_size=128 --criterion='l2-sp' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Flowers_102' --ELBo --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Flowers_102_ViT_B_16_VI' --kappa=1.0 --lr_0=0.001 --model_arch='ViT-B/16' --model_name='l2-sp_kappa=1.0_lr_0=0.001_n=510_random_state=3001' --n=510 --num_workers=0 --prior_directory='/cluster/tufts/hugheslab/eharve06/vit_b_16_torchvision' --prior_type='vit_b_16_torchvision' --random_state=3001 --save"
    "python ../src/main_Flowers_102.py --batch_size=128 --criterion='l2-sp' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Flowers_102' --ELBo --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Flowers_102_ViT_B_16_VI' --kappa=1.0 --lr_0=0.001 --model_arch='ViT-B/16' --model_name='l2-sp_kappa=1.0_lr_0=0.001_n=1020_random_state=1001' --n=1020 --num_workers=0 --prior_directory='/cluster/tufts/hugheslab/eharve06/vit_b_16_torchvision' --prior_type='vit_b_16_torchvision' --random_state=1001 --save"
    "python ../src/main_Flowers_102.py --batch_size=128 --criterion='l2-sp' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Flowers_102' --ELBo --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Flowers_102_ViT_B_16_VI' --kappa=1.0 --lr_0=0.001 --model_arch='ViT-B/16' --model_name='l2-sp_kappa=1.0_lr_0=0.001_n=1020_random_state=2001' --n=1020 --num_workers=0 --prior_directory='/cluster/tufts/hugheslab/eharve06/vit_b_16_torchvision' --prior_type='vit_b_16_torchvision' --random_state=2001 --save"
    "python ../src/main_Flowers_102.py --batch_size=128 --criterion='l2-sp' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Flowers_102' --ELBo --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Flowers_102_ViT_B_16_VI' --kappa=1.0 --lr_0=0.001 --model_arch='ViT-B/16' --model_name='l2-sp_kappa=1.0_lr_0=0.001_n=1020_random_state=3001' --n=1020 --num_workers=0 --prior_directory='/cluster/tufts/hugheslab/eharve06/vit_b_16_torchvision' --prior_type='vit_b_16_torchvision' --random_state=3001 --save"
    "python ../src/main_Flowers_102.py --batch_size=128 --criterion='l2-sp' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Flowers_102' --ELBo --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Flowers_102_ViT_B_16_VI' --kappa=1.0 --lr_0=0.0001 --model_arch='ViT-B/16' --model_name='l2-sp_kappa=1.0_lr_0=0.0001_n=510_random_state=1001' --n=510 --num_workers=0 --prior_directory='/cluster/tufts/hugheslab/eharve06/vit_b_16_torchvision' --prior_type='vit_b_16_torchvision' --random_state=1001 --save"
    "python ../src/main_Flowers_102.py --batch_size=128 --criterion='l2-sp' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Flowers_102' --ELBo --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Flowers_102_ViT_B_16_VI' --kappa=1.0 --lr_0=0.0001 --model_arch='ViT-B/16' --model_name='l2-sp_kappa=1.0_lr_0=0.0001_n=510_random_state=2001' --n=510 --num_workers=0 --prior_directory='/cluster/tufts/hugheslab/eharve06/vit_b_16_torchvision' --prior_type='vit_b_16_torchvision' --random_state=2001 --save"
    "python ../src/main_Flowers_102.py --batch_size=128 --criterion='l2-sp' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Flowers_102' --ELBo --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Flowers_102_ViT_B_16_VI' --kappa=1.0 --lr_0=0.0001 --model_arch='ViT-B/16' --model_name='l2-sp_kappa=1.0_lr_0=0.0001_n=510_random_state=3001' --n=510 --num_workers=0 --prior_directory='/cluster/tufts/hugheslab/eharve06/vit_b_16_torchvision' --prior_type='vit_b_16_torchvision' --random_state=3001 --save"
    "python ../src/main_Flowers_102.py --batch_size=128 --criterion='l2-sp' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Flowers_102' --ELBo --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Flowers_102_ViT_B_16_VI' --kappa=1.0 --lr_0=0.0001 --model_arch='ViT-B/16' --model_name='l2-sp_kappa=1.0_lr_0=0.0001_n=1020_random_state=1001' --n=1020 --num_workers=0 --prior_directory='/cluster/tufts/hugheslab/eharve06/vit_b_16_torchvision' --prior_type='vit_b_16_torchvision' --random_state=1001 --save"
    "python ../src/main_Flowers_102.py --batch_size=128 --criterion='l2-sp' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Flowers_102' --ELBo --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Flowers_102_ViT_B_16_VI' --kappa=1.0 --lr_0=0.0001 --model_arch='ViT-B/16' --model_name='l2-sp_kappa=1.0_lr_0=0.0001_n=1020_random_state=2001' --n=1020 --num_workers=0 --prior_directory='/cluster/tufts/hugheslab/eharve06/vit_b_16_torchvision' --prior_type='vit_b_16_torchvision' --random_state=2001 --save"
    "python ../src/main_Flowers_102.py --batch_size=128 --criterion='l2-sp' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Flowers_102' --ELBo --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Flowers_102_ViT_B_16_VI' --kappa=1.0 --lr_0=0.0001 --model_arch='ViT-B/16' --model_name='l2-sp_kappa=1.0_lr_0=0.0001_n=1020_random_state=3001' --n=1020 --num_workers=0 --prior_directory='/cluster/tufts/hugheslab/eharve06/vit_b_16_torchvision' --prior_type='vit_b_16_torchvision' --random_state=3001 --save"
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate