import os
import sys
import glob
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_mehle_cnn import MehleCNN

# Ensure UTF-8 stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

class MehleCNNPipeline:
    """
    End-to-end inference pipeline:
    1. Illumination normalization & static background filtering
    2. Out-of-focus blur rejection
    3. Crop extraction
    4. MehleCNN deep learning classification
    5. In-line Agglomeration Degree calculation & visual overlay
    """
    def __init__(self, model_path='mehle_cnn_best.pt', device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found at {model_path}! Run train_mehle_cnn.py first.")

        checkpoint = torch.load(str(model_path), map_location=self.device)
        self.num_classes = checkpoint.get('num_classes', 2)
        self.class_names = checkpoint.get('class_names', ['Single Pellet', 'Agglomerate'])

        self.model = MehleCNN(in_channels=1, num_classes=self.num_classes).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"[INFO] MehleCNN Pipeline Initialized on {self.device}")
        print(f"* Model Checkpoint : {model_path.resolve()}")
        print(f"* Classes ({self.num_classes})    : {self.class_names}")

        # Build Static Fixture Mask from Empty Chamber Reference
        self._build_static_mask()

    def _build_static_mask(self):
        bg_files = sorted(list(Path('Real-Data/exp 1000 bak light only no pellets').glob('*.bmp')))
        if bg_files:
            bg_stack = [cv2.imread(str(f), cv2.IMREAD_UNCHANGED) for f in bg_files]
            I_bg_ref = np.median(np.stack(bg_stack, axis=0), axis=0).astype(np.uint8)
            se_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
            tophat_bg = cv2.morphologyEx(I_bg_ref, cv2.MORPH_TOPHAT, se_tophat)
            _, binary_bg = cv2.threshold(tophat_bg, 38, 255, cv2.THRESH_BINARY)
            se_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            self.M_static = cv2.dilate(binary_bg, se_dilate)
        else:
            self.M_static = np.zeros((1536, 2048), dtype=np.uint8)

    def process_frame(self, image_input, margin=25, crop_size=96):
        if isinstance(image_input, (str, Path)):
            img_raw = cv2.imread(str(image_input), cv2.IMREAD_UNCHANGED)
        else:
            img_raw = image_input

        H, W = img_raw.shape[:2]
        se_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
        tophat = cv2.morphologyEx(img_raw, cv2.MORPH_TOPHAT, se_tophat)
        denoised = cv2.bilateralFilter(tophat, d=5, sigmaColor=30, sigmaSpace=30)
        _, th_bin = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        bin_clean = cv2.bitwise_and(th_bin, cv2.bitwise_not(self.M_static))
        se_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bin_closed = cv2.morphologyEx(bin_clean, cv2.MORPH_CLOSE, se_close)
        cnts, _ = cv2.findContours(bin_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        batch_tensors = []

        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            # 1. Static mask check
            roi_mask = self.M_static[y:y+h, x:x+w]
            c_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(c_mask, [c - [x, y]], -1, 255, -1)
            if np.sum(cv2.bitwise_and(roi_mask, c_mask)) > 0.25 * np.sum(c_mask):
                continue
            # 2. Border margin check
            if x <= margin or y <= margin or (x + w) >= (W - margin) or (y + h) >= (H - margin):
                continue
            # 3. Area filter
            area = cv2.contourArea(c)
            if area < 100 or area > 20000:
                continue
            # 4. Aspect ratio check
            aspect_ratio = max(w / max(1, h), h / max(1, w))
            if aspect_ratio > 4.0:
                continue

            # 5. Defocus blur check
            crop = img_raw[y:y+h, x:x+w]
            lap_var = cv2.Laplacian(crop, cv2.CV_64F).var()
            gx = cv2.Sobel(crop, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(crop, cv2.CV_64F, 0, 1, ksize=3)
            tenengrad = np.mean(gx**2 + gy**2)

            if lap_var < 70.0 or tenengrad < 1500.0:
                continue

            # Padded square crop centered at centroid
            cx, cy = x + w / 2.0, y + h / 2.0
            crop_dim = max(w, h) * 1.35
            x1 = int(max(0, cx - crop_dim / 2.0))
            x2 = int(min(W, cx + crop_dim / 2.0))
            y1 = int(max(0, cy - crop_dim / 2.0))
            y2 = int(min(H, cy + crop_dim / 2.0))

            crop_patch = img_raw[y1:y2, x1:x2]
            if crop_patch.size == 0:
                continue

            crop_resized = cv2.resize(crop_patch, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
            img_float = crop_resized.astype(np.float32) / 255.0
            mean_val = float(img_float.mean())
            std_val = float(img_float.std()) + 1e-6
            img_norm = (img_float - mean_val) / std_val

            tensor_input = torch.tensor(img_norm, dtype=torch.float32).unsqueeze(0) # (1, H, W)
            batch_tensors.append(tensor_input)

            candidates.append({
                'contour': c,
                'box': (x, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'lap_var': lap_var,
                'tenengrad': tenengrad
            })

        if not batch_tensors:
            return {'candidates': [], 'vis': img_raw, 'agg_degree': 0.0, 'counts': {}}

        # Batch inference through MehleCNN
        batch_x = torch.stack(batch_tensors, dim=0).to(self.device)
        with torch.no_grad():
            logits = self.model(batch_x)
            probs = F.softmax(logits, dim=1).cpu().numpy()

        vis_img = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR) if len(img_raw.shape) == 2 else img_raw.copy()

        counts = {'single': 0, 'agglomerate': 0, 'overlapped': 0}
        color_palette = [
            (0, 220, 60),    # Single: Green (BGR)
            (0, 0, 230),     # Agglomerate: Red (BGR)
            (0, 160, 255)    # Overlapped: Orange (BGR)
        ]

        for i, c_data in enumerate(candidates):
            pred_class = int(np.argmax(probs[i]))
            conf = float(probs[i, pred_class])
            c_data['pred_class'] = pred_class
            c_data['confidence'] = conf

            if self.num_classes == 2:
                label_name = 'Single' if pred_class == 0 else 'Agglomerate'
                color = color_palette[pred_class]
                if pred_class == 0:
                    counts['single'] += 1
                else:
                    counts['agglomerate'] += 1
            else:
                names = ['Single', 'Connected', 'Overlapped']
                label_name = names[pred_class]
                color = color_palette[pred_class]
                counts[names[pred_class].lower()] += 1

            x, y, w, h = c_data['box']
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 2)
            caption = f"{label_name} {conf*100:.0f}%"
            cv2.putText(vis_img, caption, (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        total_pellets = len(candidates)
        agg_pellets = counts['agglomerate'] if self.num_classes == 2 else (counts['connected'] + counts['overlapped'])
        agg_degree = (agg_pellets / total_pellets * 100.0) if total_pellets > 0 else 0.0

        # Draw HUD Summary banner
        cv2.rectangle(vis_img, (0, 0), (W, 45), (20, 20, 20), -1)
        hud_text = (f"In-Line MehleCNN | Total In-Focus: {total_pellets} | "
                    f"Agglomeration Degree: {agg_degree:.1f}% | "
                    f"Single: {counts['single']} | Agglomerates: {agg_pellets}")
        cv2.putText(vis_img, hud_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        return {
            'candidates': candidates,
            'vis': vis_img,
            'agg_degree': agg_degree,
            'counts': counts,
            'total_infocus': total_pellets
        }

if __name__ == '__main__':
    pipeline = MehleCNNPipeline('mehle_cnn_best.pt')
    test_img = Path('Real-Data/exp 1000 back+ext light early agg/Pic_20260914110617188-243.bmp')
    res = pipeline.process_frame(test_img)

    out_dir = Path('preprocessed_checkpoint/inferred_frames')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"inferred_{test_img.name}"
    cv2.imwrite(str(out_path), res['vis'])

    print(f"Successfully processed {test_img.name}:")
    print(f"* Total In-Focus Pellets : {res['total_infocus']}")
    print(f"* In-Line Agglomeration  : {res['agg_degree']:.2f}%")
    print(f"* Category Breakdown    : {res['counts']}")
    print(f"* Annotated Output Saved : {out_path.resolve()}")

