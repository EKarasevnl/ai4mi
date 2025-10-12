#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=metrics
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=0:45:00
#SBATCH --output=AAA_SLURM_OUTs/metrics_%A.out


module purge


module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# python3 -m venv distorch_env

# source distorch_env/bin/activate

source ai4mi/bin/activate

# pip install git+https://github.com/jeromerony/distorch.git

# pip install nibabel numpy torch tqdm pillow torchvision


python compute_metrics.py \
  --ref_folder ref \
  --pred_folder runs/segthor_2p5d_e50_c2_k16_a-m_lr_0d00025_w3_dec_1e-4_cd_0d2/best_epoch/val_stitched \
  --ref_extension .nii.gz \
  --pred_extension .nii.gz \
  -C 5 \
  --background_class 0 \
  --metrics 3d_dice 3d_hd95 \
  --save_folder metrics \
  --overwrite



