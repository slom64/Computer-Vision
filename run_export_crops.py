import os, sys, time, math
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# Setup paths
DATA_DIR = Path('Real-Data')
CHECKPOINT_DIR = Path('preprocessed_checkpoint')
CROPS_DIR = CHECKPOINT_DIR / 'crops'
CROPS_SINGLE = CROPS_DIR / 'single'
CROPS_CONNECTED = CROPS_DIR / 'connected'
CROPS_OVERLAPPED = CROPS_DIR / 'overlapped'
CROPS_CLEAN_ALL = CROPS_DIR / 'all_clean_infocus'
CROPS_BLURRED = CROPS_DIR / 'blurred_rejected'
METADATA_CSV = CHECKPOINT_DIR / 'particles_metadata.csv'

for p in [CHECKPOINT_DIR, CROPS_DIR, CROPS_SINGLE, CROPS_CONNECTED, CROPS_OVERLAPPED, CROPS_CLEAN_ALL, CROPS_BLURRED]:
    p.mkdir(parents=True, exist_ok=True)

# 1. Build Static Fixture Mask from Empty Chamber Reference
bg_files = sorted(list((DATA_DIR / 'exp 1000 bak light only no pellets').glob('*.bmp')))
bg_stack = [cv2.imread(str(f), cv2.IMREAD_UNCHANGED) for f in bg_files]
I_bg_ref = np.median(np.stack(bg_stack, axis=0), axis=0).astype(np.uint8)

se_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
tophat_bg = cv2.morphologyEx(I_bg_ref, cv2.MORPH_TOPHAT, se_tophat)
_, binary_bg_fixtures = cv2.threshold(tophat_bg, 38, 255, cv2.THRESH_BINARY)
se_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
M_static = cv2.dilate(binary_bg_fixtures, se_dilate)

# 2. Filter function
def filter_particles(contours, raw_img, static_mask, margin=25):
    H, W = raw_img.shape[:2]
    accepted_particles = []
    rejected_blurred = []
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        roi_mask = static_mask[y:y+h, x:x+w]
        c_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(c_mask, [c - [x, y]], -1, 255, -1)
        if np.sum(cv2.bitwise_and(roi_mask, c_mask)) > 0.25 * np.sum(c_mask):
            continue
        if x <= margin or y <= margin or (x + w) >= (W - margin) or (y + h) >= (H - margin):
            continue
        area = cv2.contourArea(c)
        if area < 100 or area > 20000:
            continue
        aspect_ratio = max(w / max(1, h), h / max(1, w))
        if aspect_ratio > 4.0:
            continue
            
        crop = raw_img[y:y+h, x:x+w]
        lap_var = cv2.Laplacian(crop, cv2.CV_64F).var()
        gx = cv2.Sobel(crop, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(crop, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = np.mean(gx**2 + gy**2)
        
        p_dict = {
            'contour': c,
            'box': (x, y, w, h),
            'area': area,
            'aspect_ratio': aspect_ratio,
            'lap_var': lap_var,
            'tenengrad': tenengrad,
            'crop': crop
        }
        if lap_var >= 70.0 and tenengrad >= 1500.0:
            accepted_particles.append(p_dict)
        else:
            rejected_blurred.append(p_dict)
            
    return accepted_particles, rejected_blurred

# 3. Process all 22 images
se_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
master_records = []
crop_save_size = 96
crop_id = 0

all_bmp_files = sorted(list(DATA_DIR.glob('*/*.bmp')))
print(f"Processing {len(all_bmp_files)} images across Real-Data...")

for fpath in all_bmp_files:
    img_raw = cv2.imread(str(fpath), cv2.IMREAD_UNCHANGED)
    if img_raw is None:
        continue
    H, W = img_raw.shape[:2]
    folder_name = fpath.parent.name
    
    tophat_img = cv2.morphologyEx(img_raw, cv2.MORPH_TOPHAT, se_tophat)
    denoised_img = cv2.bilateralFilter(tophat_img, d=5, sigmaColor=30, sigmaSpace=30)
    _, th_bin = cv2.threshold(denoised_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_clean = cv2.bitwise_and(th_bin, cv2.bitwise_not(M_static))
    bin_closed = cv2.morphologyEx(bin_clean, cv2.MORPH_CLOSE, se_close)
    cnts, _ = cv2.findContours(bin_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    clean_parts, blurred_parts = filter_particles(cnts, img_raw, M_static, margin=25)
    frame_med_area = float(np.median([p['area'] for p in clean_parts])) if len(clean_parts) > 0 else 1000.0
    
    for p in clean_parts:
        c = p['contour']
        area = p['area']
        x, y, w, h = p['box']
        perim = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (perim**2) if perim > 0 else 0
        rect = cv2.minAreaRect(c)
        l_max = max(rect[1]) if max(rect[1]) > 0 else max(w, h)
        w_min = min(rect[1]) if min(rect[1]) > 0 else min(w, h)
        aspect_ratio = l_max / max(1.0, w_min)
        hull = cv2.convexHull(c)
        solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
        eq_diam = 2.0 * math.sqrt(area / np.pi)
        
        if circ >= 0.70 and aspect_ratio <= 1.30 and area <= 1.6 * frame_med_area:
            label = 'single'
        elif solidity < 0.85 or aspect_ratio > 1.50 or area > 2.2 * frame_med_area:
            label = 'connected'
        else:
            label = 'overlapped'
            
        cx, cy = x + w / 2.0, y + h / 2.0
        crop_dim = max(w, h) * 1.35
        x1 = int(max(0, cx - crop_dim / 2.0))
        x2 = int(min(W, cx + crop_dim / 2.0))
        y1 = int(max(0, cy - crop_dim / 2.0))
        y2 = int(min(H, cy + crop_dim / 2.0))
        
        crop_img = img_raw[y1:y2, x1:x2]
        if crop_img.size > 0:
            crop_resized = cv2.resize(crop_img, (crop_save_size, crop_save_size), interpolation=cv2.INTER_LINEAR)
            clean_folder_tag = "".join(c for c in folder_name[:10] if c.isalnum() or c in (' ', '_')).strip()
            crop_filename = f"p{crop_id:05d}_{clean_folder_tag}_{label}.png"
            
            cv2.imwrite(str(CROPS_DIR / label / crop_filename), crop_resized)
            cv2.imwrite(str(CROPS_CLEAN_ALL / crop_filename), crop_resized)
            
            master_records.append({
                'particle_id': crop_id,
                'frame_file': fpath.name,
                'folder': folder_name,
                'crop_filename': crop_filename,
                'bbox_x': x,
                'bbox_y': y,
                'bbox_w': w,
                'bbox_h': h,
                'area_px': area,
                'eq_diam_px': round(eq_diam, 2),
                'circularity': round(circ, 3),
                'aspect_ratio': round(aspect_ratio, 3),
                'solidity': round(solidity, 3),
                'F_Lap': round(p['lap_var'], 2),
                'F_Ten': round(p['tenengrad'], 2),
                'is_infocus': True,
                'initial_label': label
            })
            crop_id += 1

    for p in blurred_parts[:5]:
        x, y, w, h = p['box']
        crop_img = p['crop']
        if crop_img.size > 0:
            crop_resized = cv2.resize(crop_img, (crop_save_size, crop_save_size), interpolation=cv2.INTER_LINEAR)
            b_filename = f"blur_{crop_id:05d}_{fpath.name}.png"
            cv2.imwrite(str(CROPS_BLURRED / b_filename), crop_resized)

df_master = pd.DataFrame(master_records)
df_master.to_csv(METADATA_CSV, index=False)
print(f"[Done] Total In-Focus Particles Exported: {len(df_master)}")
print(f"Single Crops: {len(list(CROPS_SINGLE.glob('*.png')))}")
print(f"Connected Crops: {len(list(CROPS_CONNECTED.glob('*.png')))}")
print(f"Overlapped Crops: {len(list(CROPS_OVERLAPPED.glob('*.png')))}")
print(f"Clean All Crops: {len(list(CROPS_CLEAN_ALL.glob('*.png')))}")
print(f"Saved Metadata CSV: {METADATA_CSV.resolve()}")

