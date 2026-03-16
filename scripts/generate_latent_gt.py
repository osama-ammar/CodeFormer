import argparse
import glob
import os
import cv2
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import img2tensor
from basicsr.utils.registry import ARCH_REGISTRY

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--test_path', type=str, default='/data1/hs_denoising/codeformer_dataset/codeformer_val/gt_overfit')
    parser.add_argument('-o', '--save_root', type=str, default='../experiments/my_identity_codes')
    parser.add_argument('--codebook_size', type=int, default=1024)
    parser.add_argument('--ckpt_path', type=str, default='../experiments/20260310_152715_VQGAN_Stage1_Personalized_50/models/net_g_latest.pth')
    args = parser.parse_args()

    os.makedirs(args.save_root, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize VQGAN
    vqgan = ARCH_REGISTRY.get('VQAutoEncoder')(
        img_size=512, nf=64, ch_mult=[1, 2, 2, 4, 4, 8], 
        quantizer='nearest', codebook_size=args.codebook_size
    ).to(device)
    
    checkpoint = torch.load(args.ckpt_path, map_location=device)['params_ema']
    vqgan.load_state_dict(checkpoint)
    vqgan.eval()

    # Create the nested structure required by FFHQBlindDataset
    latent_map = {'orig': {}, 'hflip': {}}
    
    img_list = sorted(glob.glob(os.path.join(args.test_path, '*.[jp][pn]g')))
    print(f"Found {len(img_list)} images. Generating codes for original and flipped versions...")

    for img_path in img_list:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        raw_img = cv2.imread(img_path)
        
        # We process both the original image and the horizontally flipped version
        for mode in ['orig', 'hflip']:
            img = raw_img.copy()
            if mode == 'hflip':
                img = cv2.flip(img, 1)
            
            # Preprocessing
            img = img2tensor(img / 255., bgr2rgb=True, float32=True)
            normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            img = img.unsqueeze(0).to(device)

            with torch.no_grad():
                # Extract indices from your Stage 1 VQGAN
                x = vqgan.encoder(img)
                _, _, log = vqgan.quantize(x)
                
                # Save as LongTensor (cpu) for compatibility with CrossEntropyLoss
                indices = log['min_encoding_indices'].squeeze().cpu()
                latent_map[mode][img_name] = indices

        print(f"Processed: {img_name}")

    # Save the nested dictionary
    save_path = os.path.join(args.save_root, f'latent_gt_code{args.codebook_size}.pth')
    torch.save(latent_map, save_path)
    print(f'\nSuccess! Latent GT codes saved to: {save_path}')