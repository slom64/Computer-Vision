import os
import sys
import glob
import math
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

# Ensure UTF-8 stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

class FluidBedByteTracker:
    """
    Two-Stage ByteTrack adapted for High-Speed Fluidized Bed Particle Tracking (PTV).
    Uses physical velocity gating and 2-stage association to match fast-moving
    particles (where inter-frame displacement exceeds particle diameter).
    """
    def __init__(self, high_conf=0.12, low_conf=0.04, gate_dist=120.0, max_lost_age=3):
        self.high_conf = high_conf
        self.low_conf = low_conf
        self.gate_dist = gate_dist
        self.max_lost_age = max_lost_age
        
        self.tracks = {} # tid -> track dict
        self.next_id = 1
        self.frame_count = 0
        
    def update(self, detections, dt=0.0105):
        """
        detections: list of dicts: {'box': [x1,y1,x2,y2], 'conf': float, 'cls': int}
        Returns: list of active tracks with assigned IDs and computed velocities.
        """
        self.frame_count += 1
        
        # 1. Split into high and low confidence detections (ByteTrack principle)
        dets_high = []
        dets_low = []
        for d in detections:
            x1, y1, x2, y2 = d['box']
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
            ar = max(w / h, h / w)
            area = float(w * h)
            d_item = {
                'box': d['box'], 'cx': cx, 'cy': cy, 'w': w, 'h': h,
                'area': area, 'ar': ar, 'conf': d['conf'], 'cls': d['cls']
            }
            if d['conf'] >= self.high_conf:
                dets_high.append(d_item)
            elif d['conf'] >= self.low_conf:
                dets_low.append(d_item)
                
        # If first frame, initialize all high-conf detections
        if not self.tracks:
            for d in dets_high:
                self.tracks[self.next_id] = {
                    'id': self.next_id,
                    'cx': d['cx'], 'cy': d['cy'],
                    'vx': 0.0, 'vy': 0.0, 'speed': 0.0,
                    'box': d['box'], 'w': d['w'], 'h': d['h'],
                    'area': d['area'], 'ar': d['ar'], 'cls': d['cls'],
                    'trail': [(int(d['cx']), int(d['cy']))],
                    'ar_history': [d['ar']],
                    'lost_age': 0, 'total_hits': 1
                }
                self.next_id += 1
            return list(self.tracks.values())
            
        # 2. Predict positions using constant velocity model
        track_ids = list(self.tracks.keys())
        predicted_positions = {}
        for tid in track_ids:
            t = self.tracks[tid]
            pred_x = t['cx'] + t['vx'] * dt
            pred_y = t['cy'] + t['vy'] * dt
            predicted_positions[tid] = (pred_x, pred_y)
            
        # Helper for matching
        def associate(t_ids, det_list, max_dist):
            if not t_ids or not det_list:
                return set(), set(), []
            cost = np.zeros((len(t_ids), len(det_list)))
            for i, tid in enumerate(t_ids):
                px, py = predicted_positions[tid]
                for j, d in enumerate(det_list):
                    dist = math.sqrt((px - d['cx'])**2 + (py - d['cy'])**2)
                    cost[i, j] = dist
            row_idx, col_idx = linear_sum_assignment(cost)
            matched_t, matched_d, matches = set(), set(), []
            for r, c in zip(row_idx, col_idx):
                if cost[r, c] <= max_dist:
                    matched_t.add(t_ids[r])
                    matched_d.add(c)
                    matches.append((t_ids[r], det_list[c], cost[r, c]))
            return matched_t, matched_d, matches
            
        # 3. Stage 1: Match active tracks with HIGH confidence detections
        matched_t1, matched_d1, matches1 = associate(track_ids, dets_high, self.gate_dist)
        
        # 4. Stage 2: Match remaining unmatched tracks with LOW confidence detections (ByteTrack)
        unmatched_t1 = [tid for tid in track_ids if tid not in matched_t1]
        matched_t2, matched_d2, matches2 = associate(unmatched_t1, dets_low, self.gate_dist * 0.8)
        
        all_matches = matches1 + matches2
        
        # 5. Update matched tracks
        for tid, d, dist in all_matches:
            t = self.tracks[tid]
            dx = d['cx'] - t['cx']
            dy = d['cy'] - t['cy']
            vx = dx / dt
            vy = dy / dt
            speed = math.sqrt(vx**2 + vy**2)
            
            # Smooth velocity filter (alpha-beta filter)
            t['vx'] = 0.6 * vx + 0.4 * t['vx']
            t['vy'] = 0.6 * vy + 0.4 * t['vy']
            t['speed'] = math.sqrt(t['vx']**2 + t['vy']**2)
            
            t['cx'] = d['cx']
            t['cy'] = d['cy']
            t['box'] = d['box']
            t['w'] = d['w']
            t['h'] = d['h']
            t['area'] = d['area']
            t['ar'] = d['ar']
            t['cls'] = d['cls'] # Update class (or lock if agglomerate)
            t['trail'].append((int(d['cx']), int(d['cy'])))
            t['ar_history'].append(d['ar'])
            t['lost_age'] = 0
            t['total_hits'] += 1
            
        # 6. Mark unmatched tracks as lost
        all_matched_t = matched_t1.union(matched_t2)
        for tid in track_ids:
            if tid not in all_matched_t:
                self.tracks[tid]['lost_age'] += 1
                
        # 7. Initialize new tracks from unmatched high-confidence detections
        for j, d in enumerate(dets_high):
            if j not in matched_d1:
                self.tracks[self.next_id] = {
                    'id': self.next_id,
                    'cx': d['cx'], 'cy': d['cy'],
                    'vx': 0.0, 'vy': 0.0, 'speed': 0.0,
                    'box': d['box'], 'w': d['w'], 'h': d['h'],
                    'area': d['area'], 'ar': d['ar'], 'cls': d['cls'],
                    'trail': [(int(d['cx']), int(d['cy']))],
                    'ar_history': [d['ar']],
                    'lost_age': 0, 'total_hits': 1
                }
                self.next_id += 1
                
        # 8. Remove tracks lost for too long
        self.tracks = {tid: t for tid, t in self.tracks.items() if t['lost_age'] <= self.max_lost_age}
        
        # Return currently visible tracks
        return [t for t in self.tracks.values() if t['lost_age'] == 0]


def run_bytetrack_on_sequence(
    weights_path="runs/detect/yolo11s_real/weights/best.pt",
    seq_folder="exp 1000 back+ext light early agg",
    high_conf=0.12,
    low_conf=0.04,
    gate_dist=130.0,
    output_dir="tracking_results"
):
    """
    Executes end-to-end YOLOv11s detection + FluidBed ByteTrack tracking on sequential
    burst frames from Real-Data. Computes velocities, trajectories, and Agglomeration Degree.
    """
    out_path = Path(output_dir) / seq_folder
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Model Initialization
    if not os.path.exists(weights_path):
        fallback = "runs/detect/runs/detect/yolo11s_real/weights/best.pt"
        if os.path.exists(fallback):
            weights_path = fallback
        else:
            weights_path = "yolo11s.pt"
    print(f"[INFO] Initializing YOLO11s detector from: {weights_path}")
    model = YOLO(weights_path)
    
    # 2. Sequence Files
    seq_path = Path("Real-Data") / seq_folder
    img_files = sorted(list(seq_path.glob("*.bmp")))
    if not img_files:
        raise FileNotFoundError(f"No frames in: {seq_path}")
    print(f"[INFO] Loaded {len(img_files)} frames from sequence: {seq_folder}")
    
    # 3. Initialize Tracker
    tracker = FluidBedByteTracker(
        high_conf=high_conf,
        low_conf=low_conf,
        gate_dist=gate_dist,
        max_lost_age=2
    )
    dt = 0.0105 # ~10.5 ms per frame (100 FPS)
    
    annotated_frames = []
    telemetry_log = []
    
    for f_idx, fpath in enumerate(img_files):
        img_raw = cv2.imread(str(fpath))
        if img_raw is None:
            continue
        H, W = img_raw.shape[:2]
        
        # YOLOv11s Inference
        res = model.predict(img_raw, conf=low_conf, imgsz=1024, verbose=False)[0]
        
        detections = []
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            for b, c, conf in zip(boxes, clss, confs):
                detections.append({'box': b, 'cls': c, 'conf': float(conf)})
                
        # Update Tracker
        active_tracks = tracker.update(detections, dt=dt)
        
        # Frame rendering
        vis_frame = img_raw.copy()
        n_single = 0
        n_agg = 0
        sum_area_single = 0.0
        sum_area_agg = 0.0
        speeds = []
        
        for t in active_tracks:
            tid = t['id']
            x1, y1, x2, y2 = [int(v) for v in t['box']]
            cid = t['cls']
            speed = t['speed']
            ar = t['ar']
            area = t['area']
            
            # Metrics accumulation
            if cid == 0:
                n_single += 1
                sum_area_single += area
                color = (0, 220, 0) # Green for Single
                label_name = "Single"
            else:
                n_agg += 1
                sum_area_agg += area
                color = (30, 80, 240) # Bright Red/Orange for Agglomerate
                label_name = "Agglomerate"
                
            if t['total_hits'] >= 2:
                speeds.append(speed)
                
            # Draw Trajectory Trail
            trail = t['trail'][-10:]
            for p_i in range(1, len(trail)):
                alpha = p_i / len(trail)
                trail_col = (int(color[0] * alpha), int(color[1] * alpha), int(color[2] * alpha))
                cv2.line(vis_frame, trail[p_i - 1], trail[p_i], trail_col, 2)
                
            # Draw Bounding Box & Label Badge
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            badge_text = f"ID:{tid} {label_name}"
            if t['total_hits'] >= 2:
                badge_text += f" | {speed:.0f}px/s"
                
            (tw, th), baseline = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(vis_frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(vis_frame, badge_text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            
        # Compute Agglomeration Metrics
        n_total = n_single + n_agg
        d_agg_num = (n_agg / n_total * 100.0) if n_total > 0 else 0.0
        tot_area = sum_area_single + sum_area_agg
        d_agg_area = (sum_area_agg / tot_area * 100.0) if tot_area > 0 else 0.0
        mean_speed = float(np.mean(speeds)) if speeds else 0.0
        
        # Real-Time HUD Dashboard Banner at Top
        hud_h = 75
        overlay = vis_frame.copy()
        cv2.rectangle(overlay, (0, 0), (W, hud_h), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.85, vis_frame, 0.15, 0, vis_frame)
        cv2.line(vis_frame, (0, hud_h), (W, hud_h), (100, 100, 100), 2)
        
        title_str = f"BURST SEQUENCE: {seq_folder} | FRAME {f_idx+1}/{len(img_files)}: {fpath.name}"
        cv2.putText(vis_frame, title_str, (20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)
        
        stat_1 = f"Active Particles: {n_total} (Singles: {n_single} | Agglomerates: {n_agg})"
        stat_2 = f"Agglomeration Degree D_agg: {d_agg_num:.1f}% (Count) / {d_agg_area:.1f}% (Area) | Mean Speed: {mean_speed:.0f} px/s"
        cv2.putText(vis_frame, stat_1, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(vis_frame, stat_2, (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 130), 1, cv2.LINE_AA)
        
        # Save frame
        out_file = out_path / f"tracked_{fpath.stem}.jpg"
        cv2.imwrite(str(out_file), vis_frame)
        annotated_frames.append(vis_frame)
        
        telemetry_log.append({
            'frame_idx': f_idx,
            'frame_name': fpath.name,
            'n_total': n_total,
            'n_single': n_single,
            'n_agg': n_agg,
            'd_agg_num_pct': round(d_agg_num, 2),
            'd_agg_area_pct': round(d_agg_area, 2),
            'mean_speed_px_s': round(mean_speed, 1),
            'tracked_pellets_count': len(active_tracks)
        })
        print(f"[{f_idx+1}/{len(img_files)}] {fpath.name}: {n_total} pellets ({n_single} single, {n_agg} agg) | D_agg={d_agg_num:.1f}% | Speed={mean_speed:.0f} px/s | Saved: {out_file.name}")
        
    print(f"\n[SUCCESS] Tracking finished! {len(annotated_frames)} frames saved to {out_path.resolve()}")
    
    # Generate Multi-Frame Burst Montage
    n_f = len(annotated_frames)
    cols = min(4, n_f)
    rows = int(math.ceil(n_f / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5.2 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()
    
    for i in range(n_f):
        rgb = cv2.cvtColor(annotated_frames[i], cv2.COLOR_BGR2RGB)
        axes[i].imshow(rgb)
        axes[i].set_title(f"Burst Frame #{i+1} ({img_files[i].name})", fontsize=11, fontweight='bold')
        axes[i].axis('off')
    for j in range(n_f, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    montage_path = out_path / "tracking_burst_montage.png"
    plt.savefig(str(montage_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Multi-frame burst montage saved to: {montage_path.resolve()}")
    
    df_telemetry = pd.DataFrame(telemetry_log)
    telemetry_csv = out_path / "tracking_telemetry.csv"
    df_telemetry.to_csv(telemetry_csv, index=False)
    print(f"[SUCCESS] Telemetry CSV exported: {telemetry_csv.resolve()}")
    
    return df_telemetry, annotated_frames, tracker

if __name__ == '__main__':
    run_bytetrack_on_sequence()

