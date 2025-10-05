#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=tra_2p5d_e25_c2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00
#SBATCH --output=AAA_SLURM_OUTs/train_segthor_2p5d_e50_c2_w3%A.out


module purge
module load 2025

source ai4mi/bin/activate

# python -O main.py --dataset SEGTHOR_CLEAN --mode full --epoch 25 --dest result_final --gpu

# python -O main.py --dataset SEGTHOR_CLEAN --mode full --epochs 25 --context 2 --dest runs/segthor_2p5d_e25_c2 --gpu

python -O main.py --dataset SEGTHOR_CLEAN --mode full --epochs 50 --context 2 --lr 0.00025 --warmup-epochs 3 --dest runs/segthor_2p5d_e50_c2_w3 --gpu