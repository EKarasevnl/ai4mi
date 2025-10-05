# ai4mi

## 2.5D slice context

The training script now supports optional axial context stacking. Use the `--context` flag to add `N` neighbouring slices on both sides of the centre slice;
the network input channel count is updated automatically.

```bash
python main.py --dataset SEGTHOR --mode full --epochs 1 --context 2 --dest runs/segthor_2p5d
```

Boundary slices are padded by repeating the closest available slice, so volume coverage stays consistent without extra preprocessing.