#!/bin/bash
#SBATCH --array=0-23%10
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
    "python ../src/main_Oxford-IIIT_Pet.py --beta=0.01 --batch_size=128 --criterion='ptyl' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Oxford-IIIT_Pet' --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Oxford-IIIT_Pet_SSL' --lambd=1000000000.0 --lr_0=0.001 --model_name='ptyl_beta=0.01_lambd=1000000000.0_lr_0=0.001_n=370_random_state=1001' --n=370 --num_workers=0 --prior_directory='/cluster/tufts/hugheslab/eharve06/resnet50_ssl_prior' --prior_type='resnet50_ssl_prior' --random_state=1001 --save"
    "python ../src/main_Oxford-IIIT_Pet.py --beta=0.0001 --batch_size=128 --criterion='ptyl' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Oxford-IIIT_Pet' --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Oxford-IIIT_Pet_SSL' --lambd=100.0 --lr_0=0.1 --model_name='ptyl_beta=0.0001_lambd=100.0_lr_0=0.1_n=370_random_state=2001' --n=370 --num_workers=0 --prior_directory='/cluster/tufts/hugheslab/eharve06/resnet50_ssl_prior' --prior_type='resnet50_ssl_prior' --random_state=2001 --save"
    "python ../src/main_Oxford-IIIT_Pet.py --beta=0.0001 --batch_size=128 --criterion='ptyl' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Oxford-IIIT_Pet' --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Oxford-IIIT_Pet_SSL' --lambd=10.0 --lr_0=0.1 --model_name='ptyl_beta=0.0001_lambd=10.0_lr_0=0.1_n=370_random_state=3001' --n=370 --num_workers=0 --prior_directory='/cluster/tufts/hugheslab/eharve06/resnet50_ssl_prior' --prior_type='resnet50_ssl_prior' --random_state=3001 --save"
    "python ../src/main_Oxford-IIIT_Pet.py --beta=0.0001 --batch_size=128 --criterion='ptyl' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Oxford-IIIT_Pet' --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Oxford-IIIT_Pet_SSL' --lambd=1.0 --lr_0=0.1 --model_name='ptyl_beta=0.0001_lambd=1.0_lr_0=0.1_n=3441_random_state=1001' --n=3441 --num_workers=0 --prior_directory='/cluster/tufts/hugheslab/eharve06/resnet50_ssl_prior' --prior_type='resnet50_ssl_prior' --random_state=1001 --save"
    "python ../src/main_Oxford-IIIT_Pet.py --beta=0.001 --batch_size=128 --criterion='ptyl' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Oxford-IIIT_Pet' --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Oxford-IIIT_Pet_SSL' --lambd=1.0 --lr_0=0.1 --model_name='ptyl_beta=0.001_lambd=1.0_lr_0=0.1_n=3441_random_state=2001' --n=3441 --num_workers=0 --prior_directory='/cluster/tufts/hugheslab/eharve06/resnet50_ssl_prior' --prior_type='resnet50_ssl_prior' --random_state=2001 --save"
    "python ../src/main_Oxford-IIIT_Pet.py --beta=1e-06 --batch_size=128 --criterion='ptyl' --dataset_directory='/cluster/tufts/hugheslab/eharve06/Oxford-IIIT_Pet' --experiments_directory='/cluster/tufts/hugheslab/eharve06/data-emphasized-ELBo/experiments/retrained_Oxford-IIIT_Pet_SSL' --lambd=1.0 --lr_0=0.1 --model_name='ptyl_beta=1e-06_lambd=1.0_lr_0=0.1_n=3441_random_state=3001' --n=3441 --num_workers=0 --prior_directory='/cluster/tufts/hugheslab/eharve06/resnet50_ssl_prior' --prior_type='resnet50_ssl_prior' --random_state=3001 --save"
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate