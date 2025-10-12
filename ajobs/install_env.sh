#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=install_env
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=0:45:00
#SBATCH --output=AAA_SLURM_OUTs/install_env_%A.out


module purge

module load 2023
module load Python/3.11.3-GCCcore-12.3.0


python3 -m venv ai4mi


source ai4mi/bin/activate

python -m pip install -r requirements.txt

pip install git+https://github.com/jeromerony/distorch.git


python --version



