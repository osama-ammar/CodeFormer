import json
import os
import shutil
from pathlib import Path

def parse_and_copy_images(json_file_path, output_base_dir):
    """
    Parse JSON file and copy images to noisy and gt folders
    
    Args:
        json_file_path: Path to the JSON file
        output_base_dir: Base directory where 'noisy' and 'gt' folders will be created
    """
    
    # Create output directories
    noisy_dir = os.path.join(output_base_dir, 'noisy')
    gt_dir = os.path.join(output_base_dir, 'gt')
    
    os.makedirs(noisy_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    
    # Read JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Process each entry in the "train" section
    n=0
    if 'train' in data:
        for key, value in data['train'].items():
            try:
                if n==100: break
                # Get image paths
                image_path = value.get('image')
                target_path = value.get('target_image')
                
                if not image_path or not target_path:
                    print(f"Warning: Missing image or target_image for key {key}")
                    continue
                
                # Generate output filenames (use original filename with key prefix to avoid conflicts)
                image_filename = f"{key}_{os.path.basename(image_path)}"
                target_filename = f"{key}_{os.path.basename(target_path)}"
                
                # Change extension for target if needed (to maintain consistency)
                target_filename = target_filename.replace('.avif', '.jpg').replace('.png', '.jpg')
                
                # Destination paths
                noisy_dest = os.path.join(noisy_dir, image_filename)
                gt_dest = os.path.join(gt_dir, target_filename)
                
                # Copy files
                if os.path.exists(image_path):
                    shutil.copy2(image_path, noisy_dest)
                    print(f"Copied noisy image: {image_path} -> {noisy_dest}")
                else:
                    print(f"Warning: Noisy image not found: {image_path}")
                
                if os.path.exists(target_path):
                    shutil.copy2(target_path, gt_dest)
                    print(f"Copied GT image: {target_path} -> {gt_dest}")
                else:
                    print(f"Warning: Target image not found: {target_path}")
                n+=1
            except Exception as e:
                print(f"Error processing key {key}: {str(e)}")
    
    print(f"\nDone! Files copied to:")
    print(f"  Noisy images: {noisy_dir}")
    print(f"  GT images: {gt_dir}")

def main():
    # Configuration - modify these paths as needed
    json_file = '/home/osama/datasets/datasets_paths_jsons/nersemble_ava_ref_filtered.json'  # Path to your JSON file
    output_directory = './sample_finetune_data'  # Directory where 'noisy' and 'gt' folders will be created
    
    # Run the script
    parse_and_copy_images(json_file, output_directory)

if __name__ == "__main__":
    main()