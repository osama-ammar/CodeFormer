import os
import json
import numpy as np
import PIL.Image
import pillow_avif
import scipy
import scipy.ndimage
import argparse
import dlib
from basicsr.utils.download_util import load_file_from_url
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
# --- Setup ---
shape_predictor_url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/shape_predictor_68_face_landmarks-fbdc2cb8.dat'
ckpt_path = load_file_from_url(url=shape_predictor_url, model_dir='weights/dlib', progress=True)
predictor = dlib.shape_predictor('weights/dlib/shape_predictor_68_face_landmarks-fbdc2cb8.dat')
def get_smart_name(path, idx):
    """ Extracts ID, Expression, and Cam from the file path """
    # Standardize slashes for Windows/Linux compatibility
    path = path.replace('\\', '/')
    parts = path.split('/')
    
    # 1. Try to find the Camera ID (usually the last part of the filename)
    # Example: 'cam_222200036.jpg' -> 'cam_222200036'
    full_filename = parts[-1].split('.')[0]
    
    # If it's a NeRSemble path with the 'val_step' prefix, clean it
    # Example: 'val_step4999_222200036' -> '222200036'
    cam_id = full_filename.split('_')[-1] 
    if 'cam' not in cam_id.lower() and '222' in cam_id:
        cam_id = f"cam_{cam_id}"

    # 2. Logic for NeRSemble
    if 'NeRSemble' in path:
        try:
            # Structure: .../NeRSemble/data/108/extra_sequences/EXP-4-lips/...
            person_id = parts[parts.index('108')] if '108' in parts else "P" + parts[-5]
            expression = parts[parts.index('extra_sequences') + 1]
            return f"{person_id}_{expression}_{cam_id}.png"
        except: pass

    # 3. Logic for AVA
    if 'ava_reduced_v2' in path:
        try:
            # Structure: .../data/20220218--0839--XIT043/045032/images/cam401292.avif
            person_id = parts[parts.index('data') + 1].split('--')[-1] # XIT043
            expression = parts[parts.index('data') + 2]
            return f"{person_id}_{expression}_{cam_id}.png"
        except: pass

    # Fallback
    return f"sample_{idx}_{full_filename}.png"


def ava256_linear2srgb(img, dim):
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
        return

    img = (img * (color_scale * (scale / (1 - black))) - (black * (scale / (1 - black))))

    # img = img * color_scale
    # img = (scale / (1 - black)) * (img - black)

    if is_numpy:
        return np.clip(np.power(np.clip(img, a_min=1e-6, a_max=None), 1.0 / gamma), a_min=0.0, a_max=1.0)
    else:
        return 

def _resolve_mask_path(target_path):
    """ Finds the matching mask for the GT image """
    target_path = target_path.replace('\\', '/')
    if target_path.endswith(".avif"):
        mask_path = target_path.replace("images", "masks").replace(".avif", ".png")
    else:
        mask_path = target_path.replace("images-2fps", "masks")
    
    if not os.path.exists(mask_path):
        mask_path = mask_path.replace(".jpg", ".png")
    return mask_path

def apply_mask(img, mask_path):
    """ Multiplies the image by the mask to remove background """
    if os.path.exists(mask_path):
        mask = PIL.Image.open(mask_path).convert('L') # Convert to Grayscale
        # Resize mask to match image if they differ
        if mask.size != img.size:
            mask = mask.resize(img.size, PIL.Image.Resampling.LANCZOS)
        

        white_bg = PIL.Image.new("RGB", img.size, (255, 255, 255))
        # Paste the face onto the black background using the mask
        white_bg.paste(img, (0, 0), mask)
        return white_bg
    return img

def get_landmark(filepath, only_keep_largest=True):
    detector = dlib.get_frontal_face_detector()
    
    # --- FIX: Use Pillow + Numpy instead of dlib.load_rgb_image ---
    try:
        # This handles .avif, .jpg, and .png correctly
        with PIL.Image.open(filepath) as temp_img:
            img = np.array(temp_img.convert('RGB'))
    except Exception as e:
        print(f"\tError loading {filepath}: {e}")
        return None

    dets = detector(img, 1)
    if len(dets) == 0: return None

    if only_keep_largest:
        face_areas = [(d.right() - d.left()) * (d.bottom() - d.top()) for d in dets]
        d = dets[face_areas.index(max(face_areas))]
    else:
        d = dets[0]
    
    shape = predictor(img, d)
    return np.array([[p.x, p.y] for p in shape.parts()])

def align_face(filepath, out_path,mask_path):
    lm = get_landmark(filepath)
    if lm is None:
        print(f'\tNo landmark found for {filepath}')
        return None

    lm_chin = lm[0:17]  # left-right
    lm_eyebrow_left = lm[17:22]  # left-right
    lm_eyebrow_right = lm[22:27]  # left-right
    lm_nose = lm[27:31]  # top-down
    lm_nostrils = lm[31:36]  # top-down
    lm_eye_left = lm[36:42]  # left-clockwise
    lm_eye_right = lm[42:48]  # left-clockwise
    lm_mouth_outer = lm[48:60]  # left-clockwise
    lm_mouth_inner = lm[60:68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    img = PIL.Image.open(filepath).convert('RGB')
    
    # 2. Apply AVA Color Correction if it's an AVIF
    if filepath.lower().endswith('.avif'):
        img_np = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
        img_np = ava256_linear2srgb(img_np,dim=2)                   # Correct Colors
        img = PIL.Image.fromarray((img_np * 255).astype(np.uint8))

    # 3. Apply Mask
    if mask_path:
        img = apply_mask(img, mask_path)
    output_size = 512
    transform_size = 4096
    enable_padding = False

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)),
                 int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.Resampling.LANCZOS)
        quad /= shrink
        qsize /= shrink
 
    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0),
            min(crop[2] + border,
                img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
           int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border,
               0), max(-pad[1] + border,
                       0), max(pad[2] - img.size[0] + border,
                               0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(
            np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)),
            'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(
            1.0 -
            np.minimum(np.float32(x) / pad[0],
                       np.float32(w - 1 - x) / pad[2]), 1.0 -
            np.minimum(np.float32(y) / pad[1],
                       np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) -
                img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(
            np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    img = img.transform((transform_size, transform_size), PIL.Image.QUAD,
                        (quad + 0.5).flatten(), PIL.Image.Resampling.BILINEAR)

    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.Resampling.LANCZOS)

    # Save aligned image.
    # print('saveing: ', out_path)
    img.save(out_path)

    return img, np.max(quad[:, 0]) - np.min(quad[:, 0])

def process_single_sample(args_tuple):
    # Unpack the arguments
    idx, content, gt_folder, lq_folder, skip_log_path = args_tuple
    
    lq_src = content['image']
    gt_src = content['target_image']
    mask_src = _resolve_mask_path(gt_src)
    file_name = get_smart_name(gt_src, idx)
    
    # 1. Align LQ
    if not align_face(lq_src, os.path.join(lq_folder, file_name), mask_path=mask_src):
        return False, lq_src # Failed
        
    # 2. Align GT
    align_face(gt_src, os.path.join(gt_folder, file_name), mask_path=mask_src)
    return True, None # Success

def process_json_data(json_path, out_dir):
    lq_folder = os.path.join(out_dir, 'lq')
    gt_folder = os.path.join(out_dir, 'gt')
    os.makedirs(lq_folder, exist_ok=True)
    os.makedirs(gt_folder, exist_ok=True)
    
    skip_log_path = os.path.join(out_dir, 'skipped_images.txt')
    with open(json_path, 'r') as f:
            samples = json.load(f).get('test', {})

    # Prepare arguments for multiprocessing
    tasks = [(idx, content, gt_folder, lq_folder, skip_log_path) 
            for idx, content in samples.items()]

    skipped_count = 0
    
    # Use ProcessPoolExecutor for speed
    # 'max_workers' defaults to your CPU core count
    with ProcessPoolExecutor() as executor:
        # tqdm creates the progress bar
        results = list(tqdm(executor.map(process_single_sample, tasks), 
                        total=len(tasks), 
                        desc="Processing Images"))

    # Count skips and write to log
    with open(skip_log_path, 'w') as log:
        for success, path in results:
            if not success:
                skipped_count += 1
                log.write(f"Skip: {path}\n")

    print(f"\nFinished! Total: {len(tasks)} | Skipped: {skipped_count}")
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--json_path', type=str, default='/home/osama/datasets/datasets_paths_jsons/nersemble_ava_ref_filtered.json')
    parser.add_argument('-o', '--out_dir', type=str, default='/data1/hs_denoising/codeformer_dataset/codeformer_val')
    args = parser.parse_args()
    
    process_json_data(args.json_path, args.out_dir)
    print("\nDone! Check your 'lq' and 'gt' folders.")