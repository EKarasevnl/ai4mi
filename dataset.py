#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from pathlib import Path
from typing import Callable, Union

import torch
from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset
from utils import (class2one_hot)
import numpy as np

def make_dataset(root, subset) -> list[tuple[Path, Path | None]]:
    assert subset in ['train', 'val', 'test']

    root = Path(root)
    print(f"> {root=}")

    img_path = root / subset / 'img'
    full_path = root / subset / 'gt'

    images: list[Path] = sorted(img_path.glob("*.png"))
    full_labels: list[Path | None]
    if subset != 'test':
        full_labels = sorted(full_path.glob("*.png"))
    else:
        full_labels = [None] * len(images)

    return list(zip(images, full_labels))


class SliceDataset(Dataset):
    def __init__(self, subset, root_dir, K, img_transform=None,
                 gt_transform=None, equalize=False, debug=False,
                augmentations = None):
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.K: int = K
        # self.augmentation: bool = augment
        self.equalize: bool = equalize
        self.augmentations = augmentations
        self.test_mode: bool = subset == 'test'

        self.files = make_dataset(root_dir, subset)
        if debug:
            self.files = self.files[:10]
        print(f">> Augmentations {self.augmentations is not None}, test mode: {self.test_mode}")
        print(f">> Created {subset} dataset with {len(self)} images...")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]

        img: Tensor = self.img_transform(Image.open(img_path))
        data_dict  ={}

        if not self.test_mode:
            gt: Tensor = self.gt_transform(Image.open(gt_path))


            if self.augmentations is not None:
                img = img.transpose((1, 2, 0))  # C, W, H -> W, H, C
                augmented = self.augmentations(image = img, mask = gt)
                img = augmented['image']
                gt = augmented['mask']
                img = img.transpose((2, 0, 1))  # W, H, C -> C, W, H

            gt = gt / (255 / (self.K - 1)) if self.K != 5 else gt / 63  # max <= 1
            gt = torch.tensor(gt, dtype=torch.int64)[None, ...]  # Add one dimension to simulate batch
            gt = class2one_hot(gt, K=self.K)
            gt = gt[0]

            _, W, H = img.shape
            K, _, _ = gt.shape
            assert gt.shape == (K, W, H)

            data_dict["gts"] = gt
            
        img = img / 255  # max <= 1
        img = torch.tensor(img, dtype = torch.float32)

        data_dict["images"] = img
        data_dict["stems"] = img_path.stem
        # data_dict = {"images": img,
                    #  "stems": img_path.stem}
        return data_dict
