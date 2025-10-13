import albumentations as A

def get_augmentations(shift_scale_rotate = True,
                    random_brightness_contrast = True,
                    random_crop = True,
                    clahe = True,
                    shift = 0.10,
                    scale = 0.10,
                    rotate = 10,
                    crop_scale = (0.8, 1.0),
                    crop_size = (256, 256),
                    clahe_clip = 2.0,
                    clahe_grid = (8, 8),
                    ssr_p = 0.5,
                    bright_contrast_p = 0.2,
                    crop_p = 0.5,
                    clahe_p = 0.25
                    ):

    augmentations = []
    
    if shift_scale_rotate:
        augmentations.append(A.ShiftScaleRotate(shift_limit=shift, scale_limit=scale, rotate_limit=rotate, p=ssr_p))
    if random_brightness_contrast:
        augmentations.append(A.RandomBrightnessContrast(p=bright_contrast_p))
    if random_crop:
        augmentations.append(A.RandomResizedCrop(size=crop_size, scale=crop_scale, p=crop_p))
    if clahe:
        augmentations.append(A.CLAHE(clip_limit=clahe_clip, tile_grid_size=clahe_grid, p=clahe_p))
    return A.Compose(augmentations, is_check_shapes=False)
