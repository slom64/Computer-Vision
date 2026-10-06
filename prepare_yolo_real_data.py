import os
import sys
import shutil
import random
import time
import math
from pathlib import Path
import cv2
import pandas as pd
import numpy as np

# Ensure UTF-8 stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def build_yolo_and_checkpoint_dataset():
    data_dir = Path('Real-Data')
    checkpoint_dir = Path('preprocessed_checkpoint')
    crops_dir = checkpoint_dir / 'crops'
    crops_single = crops_dir / 'single'
    crops_connected = crops_dir / 'connected'
    crops_overlapped = crops_dir / 'overlapped'
    crops_all = crops_dir / 'all_clean_infocus'
    crops_blur = crops_dir / 'blurred_rejected'
    meta_csv = checkpoint_dir / 'particles_metadata.csv'

    yolo_dir = Path('yolo_real_dataset')

    for p in [checkpoint_dir, crops_dir, crops_single, crops_connected, crops_overlapped, crops_all, crops_blur]:
        p.mkdir(parents=True, exist_ok=True)

    for split in ['train', 'val']:
        (yolo_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # 1. Build Static Chamber Fixture Mask
    bg_files = sorted(list((data_dir / 'exp 1000 bak light only no pellets').glob('*.bmp')))
    if bg_files:
        bg_stack = [cv2.imread(str(f), cv2.IMREAD_UNCHANGED) for f in bg_files]
        I_bg_ref = np.median(np.stack(bg_stack, axis=0), axis=0).astype(np.uint8)
        se_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
        tophat_bg = cv2.morphologyEx(I_bg_ref, cv2.MORPH_TOPHAT, se_tophat)
        _, binary_bg_fixtures = cv2.threshold(tophat_bg, 38, 255, cv2.THRESH_BINARY)
        se_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        M_static = cv2.dilate(binary_bg_fixtures, se_dilate)
    else:
        M_static = np.zeros((1536, 2048), dtype=np.uint8)

    # 2. Extract Particles across all 22 frames
    all_bmp_files = sorted(list(data_dir.glob('*/*.bmp')))
    print(f"Processing {len(all_bmp_files)} images across Real-Data...")

    se_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
    se_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    margin = 25
    crop_save_size = 96
    crop_id = 0
    master_records = []
    frames_annotations = {}

    for fpath in all_bmp_files:
        img_raw = cv2.imread(str(fpath), cv2.IMREAD_UNCHANGED)
        if img_raw is None:
            continue
        H, W = img_raw.shape[:2]
        folder_name = fpath.parent.name
        frames_annotations[fpath.name] = {'path': fpath, 'shape': (H, W), 'boxes': []}

        # Illumination normalization & denoising
        tophat_img = cv2.morphologyEx(img_raw, cv2.MORPH_TOPHAT, se_tophat)
        denoised_img = cv2.bilateralFilter(tophat_img, d=5, sigmaColor=30, sigmaSpace=30)
        _, th_bin = cv2.threshold(denoised_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bin_clean = cv2.bitwise_and(th_bin, cv2.bitwise_not(M_static))
        bin_closed = cv2.morphologyEx(bin_clean, cv2.MORPH_CLOSE, se_close)
        cnts, _ = cv2.findContours(bin_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        clean_parts = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            # Static mask check
            roi_mask = M_static[y:y+h, x:x+w]
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

            crop = img_raw[y:y+h, x:x+w]
            lap_var = cv2.Laplacian(crop, cv2.CV_64F).var()
            gx = cv2.Sobel(crop, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(crop, cv2.CV_64F, 0, 1, ksize=3)
            tenengrad = np.mean(gx**2 + gy**2)

            if lap_var >= 70.0 and tenengrad >= 1500.0:
                clean_parts.append({
                    'contour': c, 'box': (x, y, w, h), 'area': area,
                    'aspect_ratio': aspect_ratio, 'lap_var': lap_var, 'tenengrad': tenengrad
                })

        frame_med_area = float(np.median([p['area'] for p in clean_parts])) if clean_parts else 1000.0

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
                class_id = 0
            elif solidity < 0.85 or aspect_ratio > 1.50 or area > 2.2 * frame_med_area:
                label = 'connected'
                class_id = 1
            else:
                label = 'overlapped'
                class_id = 1 # In 2-class YOLO: single vs agglomerate/cluster

            # Add to YOLO frame annotations
            frames_annotations[fpath.name]['boxes'].append({
                'class_id': class_id, 'x': x, 'y': y, 'w': w, 'h': h
            })

            # Save square crop
            cx, cy = x + w / 2.0, y + h / 2.0
            crop_dim = max(w, h) * 1.35
            x1 = int(max(0, cx - crop_dim / 2.0))
            x2 = int(min(W, cx + crop_dim / 2.0))
            y1 = int(max(0, cy - crop_dim / 2.0))
            y2 = int(min(H, cy + crop_dim / 2.0))
            crop_img = img_raw[y1:y2, x1:x2]

            if crop_img.size > 0:
                crop_resized = cv2.resize(crop_img, (crop_save_size, crop_save_size), interpolation=cv2.INTER_LINEAR)
                clean_tag = "".join(c for c in folder_name[:10] if c.isalnum() or c in (' ', '_')).strip()
                crop_filename = f"p{crop_id:05d}_{clean_tag}_{label}.png"

                cv2.imwrite(str(crops_dir / label / crop_filename), crop_resized)
                cv2.imwrite(str(crops_all / crop_filename), crop_resized)

                master_records.append({
                    'particle_id': crop_id,
                    'frame_file': fpath.name,
                    'folder': folder_name,
                    'crop_filename': crop_filename,
                    'bbox_x': x, 'bbox_y': y, 'bbox_w': w, 'bbox_h': h,
                    'area_px': area, 'eq_diam_px': round(eq_diam, 2),
                    'circularity': round(circ, 3), 'aspect_ratio': round(aspect_ratio, 3),
                    'solidity': round(solidity, 3), 'F_Lap': round(p['lap_var'], 2),
                    'F_Ten': round(p['tenengrad'], 2), 'is_infocus': True,
                    'initial_label': label
                })
                crop_id += 1

    df_master = pd.DataFrame(master_records)
    df_master.to_csv(meta_csv, index=False)
    print(f"[Done] Checkpoint Exported: {len(df_master)} in-focus particles to {meta_csv.resolve()}")

    # 3. Format YOLO Dataset (Stratified Train / Val Split across conditions)
    val_f = [
        'Pic_20260914110617219-246.bmp', # early agg (frame 246)
        'Pic_20260914104818880-33.bmp',  # back + ext light (frame 33)
        'Pic_20260914104358980-24.bmp',  # exp 2000
        'Pic_20260914105825164-39.bmp',  # exp 1500
        'Pic_20260914115335231-0.bmp'    # empty chamber baseline
    ]
    train_f = [f for f in frames_annotations.keys() if f not in val_f]

    def write_yolo_split(f_list, split):
        img_out = yolo_dir / 'images' / split
        lbl_out = yolo_dir / 'labels' / split
        box_count = 0

        for fn in f_list:
            item = frames_annotations[fn]
            fpath = item['path']
            H, W = item['shape']
            img = cv2.imread(str(fpath), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            # Save JPG image
            out_img = img_out / f"{fpath.stem}.jpg"
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img
            cv2.imwrite(str(out_img), img_bgr)

            # Save label TXT
            out_lbl = lbl_out / f"{fpath.stem}.txt"
            lines = []
            for b in item['boxes']:
                cx_norm = (b['x'] + b['w'] / 2.0) / W
                cy_norm = (b['y'] + b['h'] / 2.0) / H
                w_norm = b['w'] / W
                h_norm = b['h'] / H
                lines.append(f"{b['class_id']} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
                box_count += 1

            with open(out_lbl, 'w', encoding='utf-8') as f:
                f.write("\n".join(lines))

        print(f"[{split.upper()}] Written {len(f_list)} frames with {box_count} annotations.")

    write_yolo_split(train_f, 'train')
    write_yolo_split(val_f, 'val')

    # Create dataset_real.yaml
    yaml_text = f"""# Portable YOLO Dataset Configuration (Relative paths)
path: {Path('yolo_real_dataset').resolve().as_posix()}
train: images/train
val: images/val

names:
  0: single
  1: agglomerate
"""
    yaml_file = Path('dataset_real.yaml')
    with open(yaml_file, 'w', encoding='utf-8') as f:
        f.write(yaml_text)

    print(f"[OK] Generated {yaml_file.resolve()} successfully!")

if __name__ == '__main__':
    build_yolo_and_checkpoint_dataset()
