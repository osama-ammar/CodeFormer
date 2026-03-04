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

def get_landmark(filepath, only_keep_largest=True):
    detector = dlib.get_frontal_face_detector()
    img = dlib.load_rgb_image(filepath)
    dets = detector(img, 1)
    if len(dets) == 0: return None

    if only_keep_largest:
        face_areas = [(d.right() - d.left()) * (d.bottom() - d.top()) for d in dets]
        d = dets[face_areas.index(max(face_areas))]
    else:
        d = dets[0]
    
    shape = predictor(img, d)
    return np.array([[p.x, p.y] for p in shape.parts()])

def align_face(filepath, out_path):
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

    # read image
    img = PIL.Image.open(filepath)

    output_size = 512
    transform_size = 4096
    enable_padding = False

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)),
                 int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.LANCZOS)
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
                        (quad + 0.5).flatten(), PIL.Image.BILINEAR)

    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.LANCZOS)

    # Save aligned image.
    # print('saveing: ', out_path)
    img.save(out_path)

    return img, np.max(quad[:, 0]) - np.min(quad[:, 0])

def process_json_data(json_path, out_dir):
    lq_folder = os.path.join(out_dir, 'lq')
    gt_folder = os.path.join(out_dir, 'gt')
    os.makedirs(lq_folder, exist_ok=True)
    os.makedirs(gt_folder, exist_ok=True)
    
    # Path for the skip log
    skip_log_path = os.path.join(out_dir, 'skipped_images.txt')

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    samples = data.get('train', {})
    total = len(samples)
    skipped_count = 0

    # Open the log file in 'w' (write) mode to start fresh
    with open(skip_log_path, 'w') as skip_log:
        for idx, content in samples.items():
            lq_src = content['image']
            gt_src = content['target_image']
            file_name = get_smart_name(gt_src, idx)
            
            print(f"[{int(idx)+1}/{total}] Processing: {file_name}")
            
            # Try to align GT first (if we can't find the face in GT, we skip both)
            result = align_face(gt_src, os.path.join(gt_folder, file_name))
            
            if result is None:
                print(f"  (!) Skipping: No face in {gt_src}")
                skip_log.write(f"Index {idx}: {gt_src}\n")
                skipped_count += 1
                continue
            
            # If GT worked, align LQ
            align_face(lq_src, os.path.join(lq_folder, file_name))

    print(f"\nDone! Processed: {total - skipped_count} | Skipped: {skipped_count}")
    print(f"See skipped cases here: {skip_log_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--json_path', type=str, required=True)
    parser.add_argument('-o', '--out_dir', type=str, default='./datasets/prepared_data')
    args = parser.parse_args()
    
    process_json_data(args.json_path, args.out_dir)
    print("\nDone! Check your 'lq' and 'gt' folders.")