# AI for medical imaging — Fall 2025 group 13

## Project overview

This project focuses on automated segmentation of thoracic organs using the SegTHOR dataset, which contains CT scans from patients with lung cancer or Hodgkin’s lymphoma. The goal is to accurately identify and segment four key Organs at Risk (OARs) (the Esophagus, Heart, Trachea, and Aorta) to support radiation therapy planning. By leveraging deep learning models such as TransUNet, this work aims to improve segmentation performance on a baseline ENet, for the broader goal of streamlining the therapy planning process.

Code base use
```
$ git clone https://github.com/EKarasevnl/ai4mi
$ cd ai4mi_project
$ git submodule init
$ git submodule update
```
Download requirements
```
$ python -m venv ai4mi
$ source ai4mi/bin/activate
$ which python  # ensure this is not your system's python anymore
$ python -m pip install -r requirements.txt
$ pip install git+https://github.com/jeromerony/distorch.git
```
 
Final Model (Imporvements from Baseline) - 
* TransUNet + Ranger Optimizer (Best Results)
  - Other improvements tried - Preprocessing with Intensity Normalization + Resolution Standardization (0.98, 0.98, 2.5), Data Augmentation, A wide range of loss functions, 2.5D networks

* Run command

```
$ python  slice_segthor.py --source_dir data --dest_dir data/SEGTHOR_test --shape 256 256 --retains 5

```
#

* To slice the files with preprocessing, run slicing with slice_segthor_preproc.py, and use resampled NIfTI files for stiching.
* To train with data augmentation use ```--augment``` when running main.py
* To run the code with different loss funcitons, use the ``` --mode "$loss"``` when runnign main.py
```
$ python -O main.py --dataset SEGTHOR_CLEAN \
 --mode full \
 --epochs 25 \
 --backbone TransUNet \
 --dest "path_to_folder/transunet_TransUNet_dice_ce_loss" \
 --run_name "transunet_TransUNet_dice_ce_loss" \
 --gpu \
 --lr 0.001 \
 --optimizer ranger \
 --weight_decay 0.001 \
 --patch_size 8 \
 --embed_dim 1024 \
 --depth 12 \
 --num_heads 16
```
Compute Additional Metrics with 
```
$ python compute_metrics.py --ref_folder "path_to_folder" --pred_folder "path_to_folder" --ref_extension ".nii.gz"\
    --pred_extension ".nii.gz" --metrics "3d_dice" "3d_hd95" "$other_metrics" -C 5 --background_class 0 --save_folder "val" --overwrite
```
