#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=vit_test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00
#SBATCH --output=aoutputs/vit_test_%A.out


module purge
module load 2025

source /gpfs/home1/bsood/group_project/ai4mi/ai4mi/bin/activate

python main.py --dataset SEGTHOR_CLEAN --arch vit --mode full --epochs 50 --dest results/segthor_vit --gpu