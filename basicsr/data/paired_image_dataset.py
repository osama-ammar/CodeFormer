from torch.utils import data as data
from torchvision.transforms.functional import normalize
from PIL import Image
import pillow_avif
import os 
import torch

import numpy as np
from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


# @DATASET_REGISTRY.register()
# class PairedImageDataset(data.Dataset):
#     """Paired image dataset for image restoration.

#     Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
#     GT image pairs.

#     There are three modes:
#     1. 'lmdb': Use lmdb files.
#         If opt['io_backend'] == lmdb.
#     2. 'meta_info_file': Use meta information file to generate paths.
#         If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
#     3. 'folder': Scan folders to generate paths.
#         The rest.

#     Args:
#         opt (dict): Config for train datasets. It contains the following keys:
#             dataroot_gt (str): Data root path for gt.
#             dataroot_lq (str): Data root path for lq.
#             meta_info_file (str): Path for meta information file.
#             io_backend (dict): IO backend type and other kwarg.
#             filename_tmpl (str): Template for each filename. Note that the
#                 template excludes the file extension. Default: '{}'.
#             gt_size (int): Cropped patched size for gt patches.
#             use_flip (bool): Use horizontal flips.
#             use_rot (bool): Use rotation (use vertical flip and transposing h
#                 and w for implementation).

#             scale (bool): Scale, which will be added automatically.
#             phase (str): 'train' or 'val'.
#     """

#     def __init__(self, opt):
#         super(PairedImageDataset, self).__init__()
#         self.opt = opt
#         # file client (io backend)
#         self.file_client = None
#         self.io_backend_opt = opt['io_backend']
#         self.mean = opt['mean'] if 'mean' in opt else None
#         self.std = opt['std'] if 'std' in opt else None

#         self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
#         if 'filename_tmpl' in opt:
#             self.filename_tmpl = opt['filename_tmpl']
#         else:
#             self.filename_tmpl = '{}'

#         if self.io_backend_opt['type'] == 'lmdb':
#             self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
#             self.io_backend_opt['client_keys'] = ['lq', 'gt']
#             self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
#         elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
#             self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
#                                                           self.opt['meta_info_file'], self.filename_tmpl)
#             print(self.lq_folder, self.gt_folder)
#             print(f"self.paths[0] : {self.paths[0]} ")

            
#         else:
#             self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)


#     def _load_img_as_np(self, path):
#         return np.array(Image.open(path).convert("RGB"))

#     def _load_mask_as_np(self, path):
#         # Load as grayscale (1 channel)
#         return np.array(Image.open(path).convert("L"))

#     def _apply_mask_tensor(self, img_t, mask_t):
#         """Apply white background where mask < 0.5."""
#         # img_t is [C, H, W], mask_t is [1, H, W]
#         mask_bool = (mask_t > 0.5)
#         return img_t.masked_fill(~mask_bool, 1.0) # 1.0 is white in normalized [0, 1] or [-1, 1]

#     def _normalize_to_tanh(self, img_np):
#         """Converts 0-255 uint8 to -1 to 1 float tensor."""
#         img_t = torch.from_numpy(img_np).permute(2, 0, 1).float()
#         return (img_t / 127.5) - 1.0

#     def __len__(self):
#         return len(self.img_ids)

#     def _resolve_mask_path(self, target_path):
#         if target_path.endswith(".avif"):
#             mask_path = target_path.replace("images", "masks").replace(".avif", ".png")
#         else:
#             mask_path = target_path.replace("images-2fps", "masks")
#         if not os.path.exists(mask_path):
#             mask_path = mask_path.replace(".jpg", ".png")
#         return mask_path
    
#     def __getitem__(self, index):
#         if self.file_client is None:
#             self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

#         scale = self.opt['scale']

#         # Load gt and lq images. Dimension order: HWC; channel order: BGR;
#         # image range: [0, 1], float32.
#         gt_path = self.paths[index]['gt_path']
#         img_bytes = self.file_client.get(gt_path, 'gt')
#         img_gt = imfrombytes(img_bytes, float32=True)
#         lq_path = self.paths[index]['lq_path']
        
#         img_bytes = self.file_client.get(lq_path, 'lq')
#         img_lq = imfrombytes(img_bytes, float32=True)

#         # augmentation for training
#         if self.opt['phase'] == 'train':
#             gt_size = self.opt['gt_size']
#             # random crop
#             img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
#             # flip, rotation
#             img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'], self.opt['use_rot'])

#         # TODO: color space transform
#         # BGR to RGB, HWC to CHW, numpy to tensor
#         img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
#         # normalize
#         if self.mean is not None or self.std is not None:
#             normalize(img_lq, self.mean, self.std, inplace=True)
#             normalize(img_gt, self.mean, self.std, inplace=True)

#         return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

#     def __len__(self):
#         return len(self.paths)



###########################################################################
#######################################################################3###

from typing import Union

def ava256_linear2srgb(img: Union[np.ndarray, torch.Tensor], dim: int = -1) -> Union[np.ndarray, torch.Tensor]:
    """
    Parameters
    ----------
        img: Image in linear sRGB space. Values should be in [0, 1]
        dim: Which dimension the color channel is

    Returns
    -------
        image in non-linear sRGB space with white balancing and gamma correction applied
    """

    if dim == -1:
        dim = len(img.shape) - 1
    # print(f"ava256_linear2srgb input shape: {img.shape}")
    assert img.shape[dim] == 3

    is_numpy = isinstance(img, np.ndarray)

    shape = [3 if i == dim else 1 for i in range(len(img.shape))]
    
    scale = 1.0 / 1.1059
    
    gamma = 1.5254
    black = [4.4 / 255, 3.1 / 255, 4.2 / 255]
    color_scale = [1.279545, 1.1059, 1.6]

    # color_scale = [1.05, 1.0, 1.1]   # reduced
    # black = [2/255, 2/255, 2/255]    # smaller shift
    # gamma = 1.9                      # closer to sRGB

    if is_numpy:
        color_scale = np.array(color_scale, dtype=np.float32).reshape(shape)
        black = np.array(black, dtype=np.float32).reshape(shape)
    else:
        color_scale = torch.tensor(color_scale, dtype=torch.float32, device=img.device).reshape(shape)
        black = torch.tensor(black, dtype=torch.float32, device=img.device).reshape(shape)

    img = (img * (color_scale * (scale / (1 - black))) - (black * (scale / (1 - black))))

    # img = img * color_scale
    # img = (scale / (1 - black)) * (img - black)

    if is_numpy:
        return np.clip(np.power(np.clip(img, a_min=1e-6, a_max=None), 1.0 / gamma), a_min=0.0, a_max=1.0)
    else:
        return torch.clamp(img.clamp(min=1e-6).pow(1.0 / gamma), min=0.0, max=1.0)
    



    
    
import cv2
import numpy as np

def crop_resize_with_white_padding(image, mask, target_size=512):
    try:
        # ---- 0. Match Mask size to Image size ----
        img_h, img_w = image.shape[:2]
        mask_h, mask_w = mask.shape[:2]
        
        if (mask_h != img_h) or (mask_w != img_w):
            mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        # ---- 1. Fix Mask Shape ----
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        mask_binary = (mask > 0).astype(np.uint8)
        ys, xs = np.where(mask_binary > 0)
        
        if len(xs) == 0 or len(ys) == 0:
            return np.ones((target_size, target_size, 3), dtype=np.uint8) * 255

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        cropped_img = image[y_min:y_max+1, x_min:x_max+1]
    
        cropped_mask = mask[y_min:y_max+1, x_min:x_max+1]

        # ---- 3. Resize Math ----
        h, w = cropped_img.shape[:2]
        scale = target_size / max(h, w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))

        resized_img = cv2.resize(cropped_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        resized_mask = cv2.resize(cropped_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # ---- 4. Clean and Paste ----
        if len(resized_mask.shape) == 3:
            resized_mask = resized_mask[:, :, 0]

        resized_img[resized_mask == 0] = 255
        
        output = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
        y_off = (target_size - new_h) // 2
        x_off = (target_size - new_w) // 2
        output[y_off:y_off+resized_img.shape[0], x_off:x_off+resized_img.shape[1]] = resized_img
        output.astype(np.uint8)
        
        return output

    except Exception as e:
        print(f"ERROR: {e}")
        return np.ones((target_size, target_size, 3), dtype=np.uint8) * 255

@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
            print(self.lq_folder, self.gt_folder)
            print(f"self.paths[0] : {self.paths[0]} ")

            
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)


    def _load_img_as_np(self, path):
        return np.array(Image.open(path).convert("RGB"))

    def _load_mask_as_np(self, path):
        # Load as grayscale (1 channel)
        return np.array(Image.open(path).convert("L"))

    def _apply_mask_tensor(self, img_t, mask_t):
        """Apply white background where mask < 0.5."""
        # img_t is [C, H, W], mask_t is [1, H, W]
        mask_bool = (mask_t > 0.5)
        return img_t.masked_fill(~mask_bool, 1.0) # 1.0 is white in normalized [0, 1] or [-1, 1]

    def _normalize_to_tanh(self, img_np):
        """Converts 0-255 uint8 to -1 to 1 float tensor."""
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float()
        return (img_t / 127.5) - 1.0
    def _resolve_mask_path(self, target_path):
            # Your logic to find the mask file based on the image folder
            if target_path.endswith(".avif"):
                mask_path = target_path.replace("images", "masks").replace(".avif", ".png")
            else:
                mask_path = target_path.replace("images-2fps", "masks")
            
            if not os.path.exists(mask_path):
                mask_path = mask_path.replace(".jpg", ".png")
            return mask_path

    def __getitem__(self, index):
            gt_path = self.paths[index]['gt_path']
            lq_path = self.paths[index]['lq_path']

            # 1. Load images (BGR format)
            img_gt = cv2.imread(gt_path)
            img_lq = cv2.imread(lq_path)
            mask_path = self._resolve_mask_path(gt_path)
            mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)


            gt_size = self.opt.get('gt_size', 512)
            img_gt = crop_resize_with_white_padding(img_gt, mask_np, target_size=gt_size)
            img_lq = crop_resize_with_white_padding(img_lq, mask_np, target_size=gt_size)
            
            # if index % 10 == 0:
            #     cv2.imwrite(f'debug_step_A_crop_{index}.png', img_gt)


            # We force float32=True and bgr2rgb=True
            img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
            
            # IMPORTANT: If img2tensor didn't divide by 255, we do it manually
            if img_gt.max() > 1.1:
                img_gt /= 255.0
                img_lq /= 255.0

            # if index % 10 == 0:
            #     from torchvision.utils import save_image
            #     # This should look perfect (Range 0-1)
            #     save_image(img_gt, f'debug_step_B_tensor_{index}.png')

            # -----------------------------------------
            # POINT C: Color Correction (AVIF only)
            # -----------------------------------------
            if gt_path.endswith(".avif"):
                img_gt = ava256_linear2srgb(img_gt, dim=0)
                # No need to correct LQ usually, but you can if needed

            # -----------------------------------------
            # POINT D: Normalization (Final CodeFormer Range)
            # -----------------------------------------
            # Shift from [0, 1] to [-1, 1]
            # Using [0.5], [0.5] is the standard way to get -1 to 1
            normalize(img_gt, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
            normalize(img_lq, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)

            # if index % 10 == 0:
            #     # To view [-1, 1] we MUST do (* 0.5 + 0.5)
            #     save_image(img_gt * 0.5 + 0.5, f'debug_step_D_final_{index}.png')
            #     print(f"Final Tensor {index} | Min: {img_gt.min():.2f} | Max: {img_gt.max():.2f}")

            return {'in': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}


    def __len__(self):
        return len(self.paths)
