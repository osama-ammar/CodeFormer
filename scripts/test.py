import torch
# Load your generated map
data = torch.load('/home/osama/code_store/CodeFormer/experiments/my_identity_codes/latent_gt_code1024.pth')

# Print the first 5 names found in the map
print("Keys in your .pth file:")
print(list(data['orig'].keys())[:5])