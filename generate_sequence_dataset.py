import os
import argparse
import json
import math
import random
import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
import torch

from simulation.config import SimulationConfig
from simulation.background_engine import BackgroundEngine
from simulation.particle_physics import FluidBedKinematics
from simulation.optical_renderer import OpticalRenderer

def extract_safe_crop(image: np.ndarray, bbox: List[float], crop_size: int = 96) -> np.ndarray:
    """Extracts a centered square patch around the particle with reflection border padding."""
    H, W = image.shape[:2]
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half_s = crop_size / 2.0
    
    src_x1 = int(round(cx - half_s))
    src_y1 = int(round(cy - half_s))
    src_x2 = src_x1 + crop_size
    src_y2 = src_y1 + crop_size
    
    pad_left = max(0, -src_x1)
    pad_top = max(0, -src_y1)
    pad_right = max(0, src_x2 - W)
    pad_bottom = max(0, src_y2 - H)
    
    clamped_x1 = max(0, src_x1)
    clamped_y1 = max(0, src_y1)
    clamped_x2 = min(W, src_x2)
    clamped_y2 = min(H, src_y2)
    
    sub = image[clamped_y1:clamped_y2, clamped_x1:clamped_x2]
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        sub = cv2.copyMakeBorder(sub, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101)
        
    if sub.shape[:2] != (crop_size, crop_size):
        sub = cv2.resize(sub, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
        
    if len(sub.shape) == 2:
        sub = cv2.cvtColor(sub, cv2.COLOR_GRAY2BGR)
    return sub

def check_optical_overlap(target_annot: Dict[str, Any], all_annots: List[Dict[str, Any]], radius_thresh: float = 28.0) -> bool:
    """
    Checks if another independent particle passes within 2D optical overlap range
    of this particle at a different focal depth.
    """
    t_id = target_annot['id']
    t_pos = target_annot['position_3d']
    
    for other in all_annots:
        if other['id'] == t_id:
            continue
        o_pos = other['position_3d']
        dist_2d = math.hypot(t_pos[0] - o_pos[0], t_pos[1] - o_pos[1])
        diff_z = abs(t_pos[2] - o_pos[2])
        
        # 2D optical pass-by: within 28px in 2D projection, but separated in depth (diff_z > 2.0)
        if dist_2d <= radius_thresh and diff_z >= 1.5:
            return True
    return False

def generate_sequences(args):
    print(f"=== Generating Spatio-Temporal Sequence Dataset for Pellet Tracking ===")
    print(f"Sequences: {args.num_sequences} runs | Video length: {args.seq_length} frames | Window: {args.window_size} frames")
    print(f"Active particles per frame: {args.particles} | Output: {args.output_dir}")
    
    cfg = SimulationConfig(num_particles_active=args.particles)
    bg_engine = BackgroundEngine(cfg)
    renderer = OpticalRenderer(cfg)
    
    train_dir = os.path.join(args.output_dir, 'train')
    val_dir = os.path.join(args.output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs("outputs/previews", exist_ok=True)
    
    total_saved = 0
    class_counts = {0: 0, 1: 0} # 0: Single, 1: Agglomerate
    overlap_count = 0
    sample_previews = {'single': None, 'agglomerate': None, 'overlap': None}
    
    for seq_idx in range(args.num_sequences):
        print(f"\n--- Simulating Video Sequence {seq_idx + 1}/{args.num_sequences} ---")
        t_seq_start = time.time()
        
        # New fluid-bed physical state
        kinematics = FluidBedKinematics(cfg)
        
        # History dict: entity_id -> list of frame observations
        # {frame_idx, crop, bbox, position_3d, count, is_overlap}
        entity_tracks: Dict[int, List[Dict[str, Any]]] = {}
        
        for frame_idx in range(args.seq_length):
            # Advance physics
            kinematics.step(dt=1.0)
            
            # Fetch dynamic background plate
            bg = bg_engine.get_background(jitter_brightness=True)
            
            # Render frame
            frame_img, annotations = renderer.render_scene(bg, kinematics.entities)
            
            # Record per-particle in-focus observations
            for annot in annotations:
                eid = annot['id']
                crop = extract_safe_crop(frame_img, annot['bbox_xyxy'], crop_size=args.crop_size)
                is_overlap = check_optical_overlap(annot, annotations)
                
                # Normalized kinematics: [vx, vy, speed, w, h, aspect_ratio, area]
                vx, vy = annot['velocity'][0], annot['velocity'][1]
                speed = math.hypot(vx, vy)
                bx = annot['bbox_xyxy']
                w = max(1.0, bx[2] - bx[0])
                h = max(1.0, bx[3] - bx[1])
                ar = w / float(h)
                area = w * h
                
                kin_vec = [
                    vx / 15.0,
                    vy / 15.0,
                    speed / 20.0,
                    (w - 28.0) / 15.0,
                    (h - 28.0) / 15.0,
                    ar - 1.0,
                    (area - 780.0) / 500.0
                ]
                
                if eid not in entity_tracks:
                    entity_tracks[eid] = []
                    
                entity_tracks[eid].append({
                    'frame_idx': frame_idx,
                    'crop': crop,
                    'bbox': annot['bbox_xyxy'],
                    'kinematics': kin_vec,
                    'count': annot['count'],
                    'is_overlap': is_overlap
                })
                
        # Now extract sliding windows of length T = args.window_size
        T = args.window_size
        seq_samples = []
        
        for eid, track in entity_tracks.items():
            if len(track) < T:
                continue
                
            # Sample continuous sub-trajectories
            stride = max(1, T // 2)
            for start_i in range(0, len(track) - T + 1, stride):
                window = track[start_i : start_i + T]
                
                crops = np.stack([item['crop'] for item in window], axis=0) # (T, H, W, 3)
                kinematics_mat = np.array([item['kinematics'] for item in window], dtype=np.float32) # (T, 7)
                
                # Ground truth count & class
                count = window[len(window) // 2]['count']
                is_agg = 1 if count > 1 else 0
                has_overlap = any(item['is_overlap'] for item in window)
                
                sample = {
                    'crops': crops,
                    'kinematics': kinematics_mat,
                    'is_agglomerate': is_agg,
                    'pellet_count': count,
                    'has_overlap': 1 if has_overlap else 0,
                    'eid': eid
                }
                seq_samples.append(sample)
                
                # Save preview examples for montage
                if is_agg == 0 and not has_overlap and sample_previews['single'] is None:
                    sample_previews['single'] = crops
                elif is_agg == 1 and sample_previews['agglomerate'] is None:
                    sample_previews['agglomerate'] = crops
                elif is_agg == 0 and has_overlap and sample_previews['overlap'] is None:
                    sample_previews['overlap'] = crops
                    
        # Shuffle and split into train/val
        random.shuffle(seq_samples)
        num_val = int(len(seq_samples) * (1.0 - args.split_ratio))
        val_samples = seq_samples[:num_val]
        train_samples = seq_samples[num_val:]
        
        # Save samples as compressed numpy arrays (.npz)
        for s_idx, sample in enumerate(train_samples):
            fname = f"seq_tr_{seq_idx:03d}_{s_idx:04d}.npz"
            fpath = os.path.join(train_dir, fname)
            np.savez_compressed(
                fpath,
                crops=sample['crops'],
                kinematics=sample['kinematics'],
                is_agglomerate=sample['is_agglomerate'],
                pellet_count=sample['pellet_count'],
                has_overlap=sample['has_overlap']
            )
            class_counts[sample['is_agglomerate']] += 1
            if sample['has_overlap']:
                overlap_count += 1
            total_saved += 1
            
        for s_idx, sample in enumerate(val_samples):
            fname = f"seq_val_{seq_idx:03d}_{s_idx:04d}.npz"
            fpath = os.path.join(val_dir, fname)
            np.savez_compressed(
                fpath,
                crops=sample['crops'],
                kinematics=sample['kinematics'],
                is_agglomerate=sample['is_agglomerate'],
                pellet_count=sample['pellet_count'],
                has_overlap=sample['has_overlap']
            )
            class_counts[sample['is_agglomerate']] += 1
            if sample['has_overlap']:
                overlap_count += 1
            total_saved += 1
            
        t_seq_end = time.time()
        print(f"Sequence {seq_idx + 1} finished in {t_seq_end - t_seq_start:.1f}s. Extracted {len(seq_samples)} tracklets.")

    # Save summary metadata
    meta = {
        'total_samples': total_saved,
        'window_size': args.window_size,
        'crop_size': args.crop_size,
        'class_distribution': {
            'single_pellet': class_counts[0],
            'agglomerate': class_counts[1]
        },
        'transient_optical_overlaps': overlap_count,
        'train_count': len(os.listdir(train_dir)),
        'val_count': len(os.listdir(val_dir))
    }
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
        
    print("\n================ DATASET SUMMARY ================")
    print(f"Total tracklet sequences: {total_saved}")
    print(f"Train samples: {meta['train_count']} | Val samples: {meta['val_count']}")
    print(f"Single Pellets: {class_counts[0]} | Agglomerates: {class_counts[1]}")
    print(f"Transient 2D Optical Overlaps: {overlap_count}")
    print(f"Saved to: {args.output_dir}")
    
    # Save preview montage of sequences
    save_sequence_montage(sample_previews, "outputs/previews/sequence_trajectories_preview.jpg")

def save_sequence_montage(previews: Dict[str, Optional[np.ndarray]], out_path: str):
    """Draws a comparison grid of consecutive frames for Single, Agglomerate, and Overlap."""
    rows = []
    labels = [
        ('single', 'Single Pellet (Stable Trajectory)'),
        ('agglomerate', 'Agglomerate (3D Tumbling Rotation)'),
        ('overlap', 'Optical Overlap (Transient 2D Pass-by)')
    ]
    
    for key, title in labels:
        crops = previews.get(key)
        if crops is None:
            continue
        T, H, W, _ = crops.shape
        # Create horizontal strip
        strip = np.hstack([crops[t] for t in range(T)])
        # Add label header
        header = np.zeros((30, strip.shape[1], 3), dtype=np.uint8)
        cv2.putText(header, f"{title} [Frames 1..{T}]", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.vstack([header, strip]))
        
    if rows:
        montage = np.vstack(rows)
        cv2.imwrite(out_path, montage)
        print(f"Saved sequence comparison montage to: {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Spatio-Temporal Sequence Dataset for Pellet Tracking")
    parser.add_argument('--num_sequences', type=int, default=8, help='Number of video simulation sequences')
    parser.add_argument('--seq_length', type=int, default=20, help='Number of frames per video sequence')
    parser.add_argument('--particles', type=int, default=240, help='Active particles per frame in fluid bed')
    parser.add_argument('--window_size', type=int, default=8, help='Temporal window length T (number of frames)')
    parser.add_argument('--crop_size', type=int, default=96, help='Spatial crop resolution (H=W)')
    parser.add_argument('--output_dir', type=str, default='outputs/sequence_dataset', help='Output directory')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='Train split ratio')
    
    args = parser.parse_args()
    generate_sequences(args)
