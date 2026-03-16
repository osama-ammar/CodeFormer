#!/usr/bin/env python3
"""
Simple script to copy 50 random matching images from input and GT folders
"""

import os
import shutil
import random
from pathlib import Path


# ===== HARDCODE YOUR PATHS HERE =====
INPUT_FOLDER = r"/data1/hs_denoising/codeformer_dataset/codeformer_val/lq_extra"
GT_FOLDER =  r"/data1/hs_denoising/codeformer_dataset/codeformer_val/gt_extra"
OUTPUT_INPUT =  r"/data1/hs_denoising/codeformer_dataset/codeformer_val/lq_overfit"
OUTPUT_GT =r"/data1/hs_denoising/codeformer_dataset/codeformer_val/gt_overfit"
NUM_IMAGES = 50
# ====================================

# Image extensions to consider
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

def main():
    print("Starting random image pair copy...")
    print(f"Input folder: {INPUT_FOLDER}")
    print(f"GT folder: {GT_FOLDER}")
    print(f"Output input folder: {OUTPUT_INPUT}")
    print(f"Output GT folder: {OUTPUT_GT}")
    print(f"Number of images: {NUM_IMAGES}")
    print("-" * 50)
    
    # Check if folders exist
    for folder in [INPUT_FOLDER, GT_FOLDER]:
        if not os.path.exists(folder):
            print(f"ERROR: Folder '{folder}' does not exist!")
            return
    
    # Get common files
    print("Finding common image files...")
    input_files = {f.name for f in Path(INPUT_FOLDER).iterdir() 
                   if f.is_file() and f.suffix.lower() in EXTENSIONS}
    gt_files = {f.name for f in Path(GT_FOLDER).iterdir() 
                if f.is_file() and f.suffix.lower() in EXTENSIONS}
    
    common_files = sorted(list(input_files.intersection(gt_files)))
    
    if not common_files:
        print("ERROR: No common image files found between folders!")
        return
    
    print(f"Found {len(common_files)} common image pairs")
    
    # Check if we have enough
    available = len(common_files)
    if available < NUM_IMAGES:
        print(f"WARNING: Only {available} images available. Copying all.")
        num_to_copy = available
    else:
        num_to_copy = NUM_IMAGES
    
    # Select random images
    selected = random.sample(common_files, num_to_copy)
    
    # Create output folders
    Path(OUTPUT_INPUT).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_GT).mkdir(parents=True, exist_ok=True)
    
    # Copy files
    print(f"\nCopying {num_to_copy} image pairs...")
    for i, filename in enumerate(selected, 1):
        # Copy input
        shutil.copy2(
            os.path.join(INPUT_FOLDER, filename),
            os.path.join(OUTPUT_INPUT, filename)
        )
        # Copy GT
        shutil.copy2(
            os.path.join(GT_FOLDER, filename),
            os.path.join(OUTPUT_GT, filename)
        )
        print(f"[{i}/{num_to_copy}] Copied: {filename}")
    
    print("-" * 50)
    print(f"DONE! Copied {num_to_copy} image pairs")
    print(f"Input images: {OUTPUT_INPUT}")
    print(f"GT images: {OUTPUT_GT}")

if __name__ == "__main__":
    main()