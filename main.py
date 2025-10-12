#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec, Caroline Magg

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

import argparse
import warnings
from typing import Any
from pathlib import Path
from pprint import pprint
from operator import itemgetter
from shutil import copytree, rmtree

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader

from functools import partial 

from dataset import SliceDataset
from PIL import Image
from ShallowNet import shallowCNN
from ENet import ENet
from utils import (Dcm,
                   class2one_hot,
                   probs2one_hot,
                   probs2class,
                   tqdm_,
                   dice_coef,
                   save_images)

from losses import (CrossEntropy)

datasets_params: dict[str, dict[str, Any]] = {}
# K for the number of classes
# Avoids the classes with C (often used for the number of Channel)
datasets_params["TOY2"] = {'K': 2, 'net': shallowCNN, 'B': 2, 'kernels': 8, 'factor': 2, 'context': 0}
datasets_params["SEGTHOR"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2, 'context': 0}
datasets_params["SEGTHOR_CLEAN"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2, 'context': 0}

def build_img_transform(norm_mean: float | None = None,
            norm_std: float | None = None):
    if (norm_mean is None) != (norm_std is None):
        raise ValueError("Both norm_mean and norm_std must be provided together or left None")

    def _transform(img):
        tensor = torch.tensor(np.array(img.convert('L'))[np.newaxis, ...], dtype=torch.float32) / 255
        if norm_mean is not None and norm_std is not None:
                        tensor = (tensor - norm_mean) / max(abs(norm_std), 1e-8)
        return tensor

    return _transform


def compute_dataset_normalization(root_dir: Path) -> tuple[float, float]:
    img_dir = root_dir / 'train' / 'img'
    img_paths = sorted(img_dir.glob("*.png"))
    if not img_paths:
        raise FileNotFoundError(f"No PNG slices found under {img_dir}")

    total_mean: float = 0.0
    total_sq_mean: float = 0.0
    count: int = 0

    for path in img_paths:
        with Image.open(path) as pil_img:
            arr = np.array(pil_img.convert('L'), dtype=np.float32) / 255.0
        total_mean += float(arr.mean())
        total_sq_mean += float((arr ** 2).mean())
        count += 1

    dataset_mean = total_mean / count
    dataset_var = max(total_sq_mean / count - dataset_mean ** 2, 1e-8)
    dataset_std = float(np.sqrt(dataset_var))

    return float(dataset_mean), dataset_std

def gt_transform(K, img):
        img = np.array(img)[...]
        # The idea is that the classes are mapped to {0, 255} for binary cases
        # {0, 85, 170, 255} for 4 classes
        # {0, 51, 102, 153, 204, 255} for 6 classes
        # Very sketchy but that works here and that simplifies visualization
        img = img / (255 / (K - 1)) if K != 5 else img / 63  # max <= 1
        img = torch.tensor(img, dtype=torch.int64)[None, ...]  # Add one dimension to simulate batch
        img = class2one_hot(img, K=K)
        return img[0]

def setup(args) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int]:
    # Networks and scheduler
    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    params = datasets_params[args.dataset]
    K: int = params['K']
    kernels: int = args.kernels if args.kernels is not None else params.get('kernels', 8)
    factor: int = params['factor'] if 'factor' in params else 2
    context_slices: int = args.context if args.context is not None else params.get('context', 0)
    args.kernels_effective = kernels
    in_channels: int = 2 * context_slices + 1
    net = params['net'](in_channels, K, kernels=kernels, factor=factor)
    net.init_weights()
    net.to(device)

    lr = args.lr
    weight_decay: float = args.weight_decay
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999),
                                 weight_decay=weight_decay)

    # Dataset part
    B: int = params['B']
    root_dir = Path("data") / args.dataset

    norm_mean: float | None = args.norm_mean
    norm_std: float | None = args.norm_std
    if args.auto_norm:
        norm_mean, norm_std = compute_dataset_normalization(root_dir)

    img_transform = build_img_transform(norm_mean, norm_std)
    args.norm_mean_effective = norm_mean
    args.norm_std_effective = norm_std



    train_set = SliceDataset('train',
                             root_dir,
                             img_transform=img_transform,
                             gt_transform= partial(gt_transform, K),
                             debug=args.debug,
                             context_slices=context_slices,
                             slice_dropout_prob=args.context_drop_prob)
    train_loader = DataLoader(train_set,
                              batch_size=B,
                              num_workers=5,
                              shuffle=True)

    val_set = SliceDataset('val',
                           root_dir,
                           img_transform=img_transform,
                           gt_transform=partial(gt_transform, K),
                           debug=args.debug,
                           context_slices=context_slices,
                           slice_dropout_prob=0.0)
    val_loader = DataLoader(val_set,
                            batch_size=B,
                            num_workers=5,
                            shuffle=False)

    args.dest.mkdir(parents=True, exist_ok=True)

    return (net, optimizer, device, train_loader, val_loader, K)


def runTraining(args):
    print(f">>> Setting up to train on {args.dataset} with {args.mode}")
    net, optimizer, device, train_loader, val_loader, K = setup(args)

    base_lr: float = args.lr
    warmup_epochs: int = max(args.warmup_epochs, 0)
    warmup_steps: int = warmup_epochs * len(train_loader) if warmup_epochs else 0
    global_step: int = 0

    if warmup_steps:
        print(f">> Using base LR {base_lr:.3e} with {warmup_epochs} warmup epoch(s) (~{warmup_steps} steps)")
    else:
        print(f">> Using base LR {base_lr:.3e} with no warmup")

    if args.weight_decay:
        print(f">> Applying weight decay {args.weight_decay:.3e}")
    else:
        print(f">> Weight decay disabled")

    if args.context_drop_prob:
        print(f">> Context slice dropout probability {args.context_drop_prob:.2f}")
    else:
        print(">> Context slice dropout disabled")

    effective_kernels = getattr(args, "kernels_effective", None)
    if effective_kernels is not None:
        print(f">> ENet stem kernels: {effective_kernels}")

    norm_mean = getattr(args, "norm_mean_effective", None)
    norm_std = getattr(args, "norm_std_effective", None)
    if norm_mean is not None and norm_std is not None:
        print(f">> Normalizing inputs with mean={norm_mean:.3f}, std={norm_std:.3f}")
    else:
        print(f">> Input normalization: none (raw /255 scale)")

    if args.mode == "full":
        loss_fn = CrossEntropy(idk=list(range(K)))  # Supervise both background and foreground
    elif args.mode in ["partial"] and args.dataset == 'SEGTHOR':
        loss_fn = CrossEntropy(idk=[0, 1, 3, 4])  # Do not supervise the heart (class 2)
    else:
        raise ValueError(args.mode, args.dataset)

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))

    best_dice: float = 0

    for e in range(args.epochs):
        for m in ['train', 'val']:
            match m:
                case 'train':
                    net.train()
                    opt = optimizer
                    cm = Dcm
                    desc = f">> Training   ({e: 4d})"
                    loader = train_loader
                    log_loss = log_loss_tra
                    log_dice = log_dice_tra
                case 'val':
                    net.eval()
                    opt = None
                    cm = torch.no_grad
                    desc = f">> Validation ({e: 4d})"
                    loader = val_loader
                    log_loss = log_loss_val
                    log_dice = log_dice_val

            with cm():  # Either dummy context manager, or the torch.no_grad for validation
                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    img = data['images'].to(device)
                    gt = data['gts'].to(device)

                    if opt is not None and warmup_steps:
                        if global_step < warmup_steps:
                            warmup_lr = base_lr * float(global_step + 1) / warmup_steps
                        else:
                            warmup_lr = base_lr
                        for param_group in opt.param_groups:
                            param_group['lr'] = warmup_lr

                    if opt:  # So only for training
                        opt.zero_grad()

                    # Sanity tests to see we loaded and encoded the data correctly
                    assert 0 <= img.min() and img.max() <= 1
                    B, _, W, H = img.shape

                    pred_logits = net(img)
                    pred_probs = F.softmax(1 * pred_logits, dim=1)  # 1 is the temperature parameter

                    # Metrics computation, not used for training
                    pred_seg = probs2one_hot(pred_probs)
                    log_dice[e, j:j + B, :] = dice_coef(pred_seg, gt)  # One DSC value per sample and per class

                    loss = loss_fn(pred_probs, gt)
                    log_loss[e, i] = loss.item()  # One loss value per batch (averaged in the loss)

                    if opt:  # Only for training
                        loss.backward()
                        opt.step()
                        global_step += 1

                    if m == 'val':
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=UserWarning)
                            predicted_class: Tensor = probs2class(pred_probs)
                            mult: int = 63 if K == 5 else (255 / (K - 1))
                            save_images(predicted_class * mult,
                                        data['stems'],
                                        args.dest / f"iter{e:03d}" / m)

                    j += B  # Keep in mind that _in theory_, each batch might have a different size
                    # For the DSC average: do not take the background class (0) into account:
                    postfix_dict: dict[str, str] = {"Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                                                    "Loss": f"{log_loss[e, :i + 1].mean():5.2e}"}
                    if K > 2:
                        postfix_dict |= {f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}"
                                         for k in range(1, K)}
                    tq_iter.set_postfix(postfix_dict)

        # I save it at each epochs, in case the code crashes or I decide to stop it early
        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        np.save(args.dest / "dice_tra.npy", log_dice_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        np.save(args.dest / "dice_val.npy", log_dice_val)

        current_dice: float = log_dice_val[e, :, 1:].mean().item()
        if current_dice > best_dice:
            message = f">>> Improved dice at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC"
            print(message)
            best_dice = current_dice
            with open(args.dest / "best_epoch.txt", 'w') as f:
                f.write(message)

            best_folder = args.dest / "best_epoch"
            if best_folder.exists():
                rmtree(best_folder)
            copytree(args.dest / f"iter{e:03d}", Path(best_folder))

            torch.save(net, args.dest / "bestmodel.pkl")
            torch.save(net.state_dict(), args.dest / "bestweights.pt")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', default=0.00025, type=float,
                        help="Base learning rate for Adam (default: 2.5e-4)")
    parser.add_argument('--warmup-epochs', default=0, type=int,
                        help="Number of initial epochs for linear LR warmup")
    parser.add_argument('--kernels', default=None, type=int,
                        help="Override ENet stem width (kernels). Defaults to dataset preset")
    parser.add_argument('--weight-decay', default=0.0, type=float,
                        help="L2 weight decay for Adam (default: 0, keeps current behaviour)")
    parser.add_argument('--context-drop-prob', default=0.0, type=float,
                        help="Probability of zeroing each neighbour slice during training")
    parser.add_argument('--dataset', default='TOY2', choices=datasets_params.keys())
    parser.add_argument('--mode', default='full', choices=['partial', 'full'])
    parser.add_argument('--dest', type=Path, required=True,
                        help="Destination directory to save the results (predictions and weights).")

    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help="Keep only a fraction (10 samples) of the datasets, "
                             "to test the logics around epochs and logging easily.")
    parser.add_argument('--context', type=int, default=None,
                        help="Number of neighboring slices to add on each side (0 keeps pure 2D).")
    parser.add_argument('--norm-mean', type=float, default=None,
                        help="Mean used to normalize input slices after /255 scaling")
    parser.add_argument('--norm-std', type=float, default=None,
                        help="Std used to normalize input slices after /255 scaling")
    parser.add_argument('--auto-norm', action='store_true',
                        help="Compute dataset-wide mean/std (train split) automatically")

    args = parser.parse_args()

    pprint(args)

    runTraining(args)


if __name__ == '__main__':
    main()
