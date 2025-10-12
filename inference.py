import argparse
import os
import warnings
from typing import Any, List, Tuple
from pathlib import Path
from pprint import pprint

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import nibabel as nib

from functools import partial 

from dataset import SliceDataset
from ShallowNet import shallowCNN
from ENet import ENet
from TransUNet import TransUNet
from utils import (Dcm,
                   class2one_hot,
                   probs2one_hot,
                   probs2class,
                   tqdm_,
                   dice_coef,
                   save_images)


datasets_params: dict[str, dict[str, Any]] = {}
datasets_params["TOY2"] = {'K': 2, 'net': shallowCNN, 'B': 2, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_CLEAN"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_test"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}

def img_transform(img):
    """Transform input image for inference (same as training)"""
    img = img.convert('L')
    img = np.array(img)[np.newaxis, ...]
    img = img / 255 
    img = torch.tensor(img, dtype=torch.float32)
    return img

def gt_transform(K, img):
    """Transform ground truth for evaluation (same as training)"""
    img = np.array(img)[...]
    img = img / (255 / (K - 1)) if K != 5 else img / 63 
    img = torch.tensor(img, dtype=torch.int64)[None, ...]  # Add one dimension to simulate batch
    img = class2one_hot(img, K=K)
    return img[0]

def load_model(model_path: Path, args) -> nn.Module:
    """Load trained model from checkpoint"""
    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Using {device} for inference")
    
    K: int = datasets_params[args.dataset]['K']
    
    # Initialize model architecture
    if args.backbone == 'ENet':
        kernels: int = datasets_params[args.dataset]['kernels'] if 'kernels' in datasets_params[args.dataset] else 8
        factor: int = datasets_params[args.dataset]['factor'] if 'factor' in datasets_params[args.dataset] else 2
        net = ENet(1, K, kernels=kernels, factor=factor)
    elif args.backbone == 'TransUNet':
        img_size = getattr(args, 'img_size', 224)
        patch_size = getattr(args, 'patch_size', 16)
        embed_dim = getattr(args, 'embed_dim', 768)
        depth = getattr(args, 'depth', 12)
        num_heads = getattr(args, 'num_heads', 12)
        net = TransUNet(1, K, img_size=img_size, patch_size=patch_size, 
                       embed_dim=embed_dim, depth=depth, num_heads=num_heads)
    else:
        kernels: int = datasets_params[args.dataset]['kernels'] if 'kernels' in datasets_params[args.dataset] else 8
        factor: int = datasets_params[args.dataset]['factor'] if 'factor' in datasets_params[args.dataset] else 2
        net = datasets_params[args.dataset]['net'](1, K, kernels=kernels, factor=factor)
    
    # Load model weights
    if model_path.suffix == '.pkl':
        net = torch.load(model_path, map_location=device, weights_only=False)
    elif model_path.suffix == '.pt':
        net.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    else:
        raise ValueError(f"Unsupported model file format: {model_path.suffix}")
    
    net.to(device)
    net.eval()
    
    print(f">> Loaded {args.backbone} model from {model_path}")
    return net, device, K

def run_inference_on_dataset(args):
    """Run inference on a dataset (validation set)"""
    print(f">>> Running inference on {args.dataset} with {args.backbone}")
    

    net, device, K = load_model(args.model_path, args)
    

    B: int = datasets_params[args.dataset]['B']
    
    # For test data, use SEGTHOR_test structure
    if args.dataset == "SEGTHOR_test":
        root_dir = Path("data") / "SEGTHOR_test"

        inference_set = SliceDataset('test',
                                    root_dir,
                                    img_transform=img_transform,
                                    gt_transform=None,  # No ground truth for test data
                                    debug=False)
    else:
        root_dir = Path("data") / args.dataset

        inference_set = SliceDataset('val',
                                    root_dir,
                                    img_transform=img_transform,
                                    gt_transform=None,  # No ground truth for inference
                                    debug=False)
    inference_loader = DataLoader(inference_set,
                                 batch_size=B,
                                 num_workers=5,
                                 shuffle=False)
    

    args.output_dir.mkdir(parents=True, exist_ok=True)
    

    all_predictions = []
    all_stems = []
    
    with torch.no_grad():
        tq_iter = tqdm_(enumerate(inference_loader), total=len(inference_loader), desc=">> Inference")
        for i, data in tq_iter:
            img = data['images'].to(device)
            

            pred_logits = net(img)
            pred_probs = F.softmax(pred_logits, dim=1)
            pred_seg = probs2one_hot(pred_probs)
            predicted_class = probs2class(pred_probs)
            

            mult: int = 63 if K == 5 else (255 / (K - 1))
            save_images(predicted_class * mult,
                       data['stems'],
                       args.output_dir / "predictions")
            
            all_predictions.append(pred_seg.cpu())
            all_stems.extend(data['stems'])

            tq_iter.set_postfix({"Batch": f"{i+1}/{len(inference_loader)}"})
    
    print(f">> Inference completed. Results saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained models")
    

    parser.add_argument('--model_path', type=Path, required=True,
                       help="Path to trained model (.pkl or .pt file)")
    parser.add_argument('--dataset', default='SEGTHOR_test', choices=datasets_params.keys(),
                       help="Dataset used for training (determines number of classes)")
    parser.add_argument('--backbone', default='ENet', choices=['ENet', 'TransUNet'],
                       help="Model backbone architecture")
    parser.add_argument('--output_dir', type=Path, required=True,
                       help="Output directory for predictions")
    parser.add_argument('--gpu', action='store_true')

    # Default arguments are the ones used for the final model
    parser.add_argument('--img_size', default=512, type=int,
                       help='Input image size for TransUNet')
    parser.add_argument('--patch_size', default=16, type=int,
                       help='Patch size for Vision Transformer')
    parser.add_argument('--embed_dim', default=768, type=int,
                       help='Embedding dimension for Vision Transformer')
    parser.add_argument('--depth', default=12, type=int,
                       help='Number of transformer blocks')
    parser.add_argument('--num_heads', default=12, type=int,
                       help='Number of attention heads')
    
    
    
    
    args = parser.parse_args()
    
    if not args.model_path.exists():
        parser.error(f"Model file not found: {args.model_path}")
    
    pprint(args)
    
    run_inference_on_dataset(args)
    
if __name__ == '__main__':
    main()
