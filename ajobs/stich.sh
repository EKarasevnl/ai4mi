#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=stich
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=0:45:00
#SBATCH --output=AAA_SLURM_OUTs/stich_%A.out


module purge


module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source ai4mi/bin/activate

python stitch.py --data_folder runs/segthor_2p5d_e50_c2_w3/best_epoch/val --dest_folder runs/segthor_2p5d_e50_c2_w3/best_epoch/val_stitched --num_classes 5 --grp_regex "(Patient_\d\d)_\d\d\d\d" --source_scan_pattern "data/segthor_train/train/{id_}/GT.nii.gz"