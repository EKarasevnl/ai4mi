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

from collections import defaultdict
from pathlib import Path
from typing import Callable, Union

import torch
from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset


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
    def __init__(self, subset, root_dir, img_transform=None,
                 gt_transform=None, augment=False, equalize=False, debug=False,
                 context_slices: int = 0, slice_dropout_prob: float = 0.0):
        assert context_slices >= 0, context_slices
        assert 0.0 <= slice_dropout_prob <= 1.0, slice_dropout_prob

        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentation: bool = augment
        self.equalize: bool = equalize
        self.context_slices: int = context_slices
        self.slice_dropout_prob: float = slice_dropout_prob

        self.test_mode: bool = subset == 'test'

        self.files = make_dataset(root_dir, subset)
        if debug:
            self.files = self.files[:10]

        if not self.files:
            msg = (f"No input slices found under '{Path(root_dir) / subset / 'img'}'. "
                   "Make sure the dataset is downloaded/extracted and matches the expected layout")
            raise FileNotFoundError(msg)

        self._neighbors: dict[int, list[int]] | None = None
        if self.context_slices:
            by_patient: dict[str, list[int]] = defaultdict(list)
            for idx, (img_path, _) in enumerate(self.files):
                stem = img_path.stem
                patient_id = stem.rsplit('_', 1)[0]
                by_patient[patient_id].append(idx)

            neighbors: dict[int, list[int]] = {}
            radius: int = self.context_slices
            for indices in by_patient.values():
                count = len(indices)
                for pos, idx in enumerate(indices):
                    window: list[int] = []
                    for offset in range(-radius, radius + 1):
                        neighbor_pos = min(max(pos + offset, 0), count - 1)
                        window.append(indices[neighbor_pos])
                    neighbors[idx] = window
            self._neighbors = neighbors

        print(f">> Created {subset} dataset with {len(self)} images...")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]

        img: Tensor
        if self.context_slices and self._neighbors is not None:
            slice_indices = self._neighbors[index]
            slices: list[Tensor] = []
            radius: int = self.context_slices
            for pos, neighbor_idx in enumerate(slice_indices):
                neighbor_img_path, _ = self.files[neighbor_idx]
                with Image.open(neighbor_img_path) as pil_img:
                    tensor = self.img_transform(pil_img)
                if self.slice_dropout_prob and pos != radius:
                    if torch.rand(1).item() < self.slice_dropout_prob:
                        tensor = torch.zeros_like(tensor)
                slices.append(tensor)
            if slices:
                img = torch.cat(slices, dim=0)
            else:
                with Image.open(img_path) as pil_img:
                    img = self.img_transform(pil_img)
        else:
            with Image.open(img_path) as pil_img:
                img = self.img_transform(pil_img)

        data_dict = {"images": img,
                     "stems": img_path.stem}

        if not self.test_mode:
            with Image.open(gt_path) as pil_img:
                gt: Tensor = self.gt_transform(pil_img)

            _, W, H = img.shape
            K, _, _ = gt.shape
            assert gt.shape == (K, W, H)

            data_dict["gts"] = gt

        return data_dict
