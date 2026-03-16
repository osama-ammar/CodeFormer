# import os

# # 1. Update these paths
# folder1 = r"/data1/hs_denoising/codeformer_dataset/codeformer_val/gt"
# folder2 = r"/data1/hs_denoising/codeformer_dataset/codeformer_val/lq"

# # 2. Get names from Folder 1 (the "Master" list)
# print("Reading Folder 1...")
# master_set = set(os.listdir(folder1))

# # 3. Scan Folder 2 and delete extras immediately
# print("Checking Folder 2 for extra files...")
# deleted_count = 0

# with os.scandir(folder2) as entries:
#     for entry in entries:
#         # If the file is NOT in the master list, delete it
#         if entry.is_file() and entry.name not in master_set:
#             try:
#                 os.remove(entry.path)
#                 deleted_count += 1
#                 if deleted_count % 1000 == 0:
#                     print(f"Deleted {deleted_count} files...")
#             except Exception as e:
#                 print(f"Could not delete {entry.name}: {e}")

# print(f"Finished! Total extra files removed: {deleted_count}")



import os
import shutil

# 1. Update your paths
folder1 = r"/data1/hs_denoising/codeformer_dataset/codeformer_val/gt"
folder2 = r"/data1/hs_denoising/codeformer_dataset/codeformer_val/lq"
N = 50  # Number of matching pairs to keep

# 2. Define where the extras go
extra1_dir = r"/data1/hs_denoising/codeformer_dataset/codeformer_val/gt_extra"
extra2_dir = r"/data1/hs_denoising/codeformer_dataset/codeformer_val/lq_extra"

os.makedirs(extra1_dir, exist_ok=True)
os.makedirs(extra2_dir, exist_ok=True)

# 3. Get lists and find the common "Keep" set
print("Scanning folders...")
files1 = set(os.listdir(folder1))
files2 = set(os.listdir(folder2))

# Find files that are in BOTH
common = sorted(list(files1.intersection(files2)))
keep_set = set(common[:N]) 

print(f"Keeping {len(keep_set)} pairs.")

# 4. Move everything else from Folder 1
print("Moving extras from Folder 1...")
for f1 in files1:
    if f1 not in keep_set:
        shutil.move(os.path.join(folder1, f1), os.path.join(extra1_dir, f1))

# 5. Move everything else from Folder 2
print("Moving extras from Folder 2...")
for f2 in files2:
    if f2 not in keep_set:
        shutil.move(os.path.join(folder2, f2), os.path.join(extra2_dir, f2))

print("Done! Folders are now synced and limited to N files.")