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
    "python ../src/main_Oxford-IIIT_Pet.py --alpha=0.0 --batch_size=128 --criterion='l2-zero' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Oxford-IIIT_Pet' --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Oxford-IIIT_Pet' --lr_0=0.01 --model_name='linear_probing_alpha=0.0_lr_0=0.01_n=370_random_state=1001' --n=370 --num_workers=0 --random_state=1001 --save"
    "python ../src/main_Oxford-IIIT_Pet.py --alpha=0.0001 --batch_size=128 --criterion='l2-zero' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Oxford-IIIT_Pet' --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Oxford-IIIT_Pet' --lr_0=0.1 --model_name='linear_probing_alpha=0.0001_lr_0=0.1_n=370_random_state=2001' --n=370 --num_workers=0 --random_state=2001 --save"
    "python ../src/main_Oxford-IIIT_Pet.py --alpha=0.0001 --batch_size=128 --criterion='l2-zero' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Oxford-IIIT_Pet' --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Oxford-IIIT_Pet' --lr_0=0.01 --model_name='linear_probing_alpha=0.0001_lr_0=0.01_n=370_random_state=3001' --n=370 --num_workers=0 --random_state=3001 --save"
    "python ../src/main_Oxford-IIIT_Pet.py --alpha=0.0 --batch_size=128 --criterion='l2-zero' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Oxford-IIIT_Pet' --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Oxford-IIIT_Pet' --lr_0=0.01 --model_name='linear_probing_alpha=0.0_lr_0=0.01_n=3441_random_state=1001' --n=3441 --num_workers=0 --random_state=1001 --save"
    "python ../src/main_Oxford-IIIT_Pet.py --alpha=0.0001 --batch_size=128 --criterion='l2-zero' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Oxford-IIIT_Pet' --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Oxford-IIIT_Pet' --lr_0=0.01 --model_name='linear_probing_alpha=0.0001_lr_0=0.01_n=3441_random_state=2001' --n=3441 --num_workers=0 --random_state=2001 --save"
    "python ../src/main_Oxford-IIIT_Pet.py --alpha=0.0 --batch_size=128 --criterion='l2-zero' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Oxford-IIIT_Pet' --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Oxford-IIIT_Pet' --lr_0=0.01 --model_name='linear_probing_alpha=0.0_lr_0=0.01_n=3441_random_state=3001' --n=3441 --num_workers=0 --random_state=3001 --save"
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate