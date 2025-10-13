#!/usr/bin/env python3
import argparse, re
from pathlib import Path
from collections import defaultdict
import numpy as np
import nibabel as nib
from PIL import Image

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_folder", required=True)
    p.add_argument("--dest_folder", required=True)
    p.add_argument("--num_classes", required=True, type=int)
    p.add_argument("--grp_regex", required=True)
    p.add_argument("--source_scan_pattern", required=True)
    return p.parse_args()

def png_to_segmentation(img_array, num_classes):
    bins = np.linspace(0, 256, num_classes + 1)
    class_indices = np.digitize(img_array, bins) - 1
    return np.clip(class_indices, 0, num_classes - 1).astype(np.uint8)

def main():
    args = parse_args()
    grp_re = re.compile(args.grp_regex)
    
    dest = Path(args.dest_folder)
    dest.mkdir(parents=True, exist_ok=True)
    
    groups = defaultdict(list)
    for png_file in Path(args.data_folder).rglob("*.png"):
        match = grp_re.search(png_file.stem)
        if match:
            patient_id = match.group(1)
            slice_num = int(re.findall(r'\d+', png_file.stem)[-1])
            groups[patient_id].append((slice_num, png_file))
    
    for patient_id, files in groups.items():
        print(f"Processing {patient_id}...")
        
        source_scan_path = Path(args.source_scan_pattern.format(id_=patient_id))
        ref_scan = nib.load(str(source_scan_path))
        X, Y, Z = ref_scan.shape[:3]
        
        volume = np.zeros((X, Y, Z), dtype=np.uint8)
        
        sorted_files = sorted(files)
        
        for z, (slice_num, file_path) in enumerate(sorted_files[:Z]):
            img = Image.open(file_path).convert('L')
            img_array = np.array(img, dtype=np.uint8)
            seg_array = png_to_segmentation(img_array, args.num_classes)
            
            if seg_array.shape != (X, Y):
                seg_img = Image.fromarray(seg_array)
                seg_resized = seg_img.resize((Y, X), Image.NEAREST)
                seg_array = np.array(seg_resized, dtype=np.uint8)
            
            volume[:, :, z] = seg_array
        
        output_file = dest / f"{patient_id}.nii.gz"
        nifti_img = nib.Nifti1Image(volume, ref_scan.affine, ref_scan.header)
        nib.save(nifti_img, str(output_file))
        print(f"Saved {patient_id}: {volume.shape} -> {output_file}")

if __name__ == "__main__":
    main()
