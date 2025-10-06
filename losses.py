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


from torch import einsum
import torch
import torch.nn.functional as F
from utils import simplex, sset


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        loss = - einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)


class DiceLoss():
    def __init__(self, smooth=1e-10, idk=None):
        self.smooth = smooth
        self.idk = idk
        print(f"Initialized {self.__class__.__name__} with idk={idk}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])
        assert len(self.idk) > 0

        weak_target = weak_target.float()

        dice_losses = []
        for c in self.idk:
            p = pred_softmax[:, c, ...]
            t = weak_target[:, c, ...]
            # elementwise multiply pred and target, sum over batch and dims
            intersect = einsum("b... , b... ->", p, t)
            union = p.sum() + t.sum()
            dice_score = (2 * intersect + self.smooth) / (union + self.smooth)
            dice_losses.append(1 - dice_score)

        return sum(dice_losses) / len(dice_losses)


class Hausdorff2DLoss:
    def __init__(self, idk=None, beta=2.0, eps=1e-10):
        self.idk = idk
        self.beta = beta
        self.eps = eps

    def _soft_distance_map(self, mask):
        dm = 1 - mask
        for _ in range(4):
            dm = F.max_pool2d(dm, kernel_size=3, stride=1, padding=1)
        return dm

    def __call__(self, pred_softmax, target):
        _, C, _, _ = pred_softmax.shape
        target = target.float()

        classes = self.idk if self.idk is not None else list(range(C))
        loss_per_class = []

        for c in classes:
            p = pred_softmax[:, c:c+1, ...]
            t = target[:, c:c+1, ...]

            dt_fore = self._soft_distance_map(t)
            dt_back = self._soft_distance_map(1 - t)

            hd = (p * dt_fore + (1 - p) * dt_back).mean()
            loss_per_class.append(hd)

        return torch.stack(loss_per_class).mean()


class CombinedLoss:
    def __init__(self, idk=None, alpha=0.5):
        self.dice_loss = DiceLoss(idk=idk)
        self.hd_loss = Hausdorff2DLoss(idk=idk)
        self.alpha = alpha

    def __call__(self, pred_softmax, target):
        dice = self.dice_loss(pred_softmax, target)
        hd = self.hd_loss(pred_softmax, target)
        return self.alpha * dice + (1 - self.alpha) * hd