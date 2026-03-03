import json

# 1. Load your JSON file
with open('/home/osama/datasets/datasets_paths_jsons/nersemble_ava_ref_filtered.json', 'r') as f:
    data = json.load(f)

# 2. Open the new text file to save the pairs
with open('val_pairs.txt', 'w') as f:
    # We look inside the "val" section of your JSON
    for key, value in data['test'].items():
        # Get the two paths we need
        gt_path = value['target_image']  # Your clean image
        lq_path = value['image']         # Your noisy image
        
        # 3. Write them to the file separated by a space
        f.write(f"{gt_path} {lq_path}\n")

print("Done! Your 'val_pairs.txt' is ready.")