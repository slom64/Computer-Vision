import os
import glob
import argparse
import time
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
import torch

from simulation.pellet_tracker import PelletTracker, TrackState
from simulation.sequence_model import PelletSequenceModel
from ultralytics import YOLO

# Palette for clear visualization
COLOR_SINGLE = (0, 220, 0)        # Green: Confirmed Single
COLOR_AGGLOMERATE = (0, 0, 255)    # Red: Confirmed Agglomerate
COLOR_OVERLAP = (255, 220, 0)      # Cyan/Yellow: Optical Overlap resolved to Single
COLOR_TENTATIVE = (180, 180, 180)  # Gray: Tentative tracklet

def draw_tracked_scene(
    image: np.ndarray,
    active_tracks: List[Any],
    frame_idx: int,
    stats: Dict[str, int]
) -> np.ndarray:
    """
    Renders bounding boxes, persistent Track IDs, motion trails, and temporal classifications.
    """
    canvas = image.copy()
    if len(canvas.shape) == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        
    H, W = canvas.shape[:2]
    
    # Draw trails first (underneath boxes)
    for trk in active_tracks:
        if len(trk.history_boxes) < 2:
            continue
            
        color = COLOR_SINGLE
        if trk.sequence_class_id > 0:
            color = COLOR_AGGLOMERATE
        elif trk.is_optical_overlap:
            color = COLOR_OVERLAP
            
        # Draw past centroid line trail
        points = []
        for box in trk.history_boxes[-18:]:
            cx = int((box[0] + box[2]) / 2.0)
            cy = int((box[1] + box[3]) / 2.0)
            points.append((cx, cy))
            
        for i in range(1, len(points)):
            alpha = i / float(len(points))
            thick = max(1, int(2.5 * alpha))
            cv2.line(canvas, points[i - 1], points[i], color, thick)
            
    # Draw bounding boxes and text labels
    for trk in active_tracks:
        box = trk.get_state_xyxy()
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)
        
        # Decide status and color
        if trk.state == TrackState.TENTATIVE:
            color = COLOR_TENTATIVE
            label = f"ID #{trk.id} [Init]"
        elif trk.is_optical_overlap:
            color = COLOR_OVERLAP
            label = f"ID #{trk.id} [Overlap->Single] {trk.sequence_confidence:.2f}"
        elif trk.sequence_class_id > 0:
            color = COLOR_AGGLOMERATE
            label = f"ID #{trk.id} [Agglomerate] {trk.sequence_confidence:.2f}"
        else:
            color = COLOR_SINGLE
            label = f"ID #{trk.id} [Single] {trk.sequence_confidence:.2f}"
            
        # Box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        
        # Center marker
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(canvas, (cx, cy), 2, color, -1)
        
        # Label badge
        font_scale = 0.40
        thick = 1
        (txt_w, txt_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
        bg_y1 = max(0, y1 - txt_h - 4)
        bg_y2 = y1
        cv2.rectangle(canvas, (x1, bg_y1), (x1 + txt_w + 4, bg_y2), (20, 20, 20), -1)
        cv2.putText(canvas, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thick, cv2.LINE_AA)
        
    # Header dashboard overlay
    dash = np.zeros((46, W, 3), dtype=np.uint8)
    cv2.putText(
        dash,
        f"Frame: {frame_idx:04d} | Active Pellets Tracked: {len(active_tracks)} | "
        f"Singles: {stats['singles']} | Agglomerates: {stats['agglomerates']} | "
        f"Resolved Overlaps: {stats['resolved_overlaps']}",
        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA
    )
    canvas = np.vstack([dash, canvas])
    return canvas


def prepare_sequence_tensor(crops: List[np.ndarray], target_len: int = 8) -> torch.Tensor:
    """Formats and pads crop history list to (1, T, 3, H, W) normalized tensor."""
    if len(crops) >= target_len:
        selected = crops[-target_len:]
    else:
        # Pad with first crop
        pad_needed = target_len - len(crops)
        selected = [crops[0]] * pad_needed + crops
        
    tensor_list = []
    for c in selected:
        if len(c.shape) == 2:
            c = cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)
        c_rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        c_norm = (c_rgb - mean) / std
        tensor_list.append(np.transpose(c_norm, (2, 0, 1))) # (3, H, W)
        
    tensor_arr = np.stack(tensor_list, axis=0) # (T, 3, H, W)
    return torch.from_numpy(tensor_arr).unsqueeze(0).float() # (1, T, 3, H, W)


def run_video_tracking(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    
    print(f"=== Running Fluid Bed Pellet Multi-Object Tracking & Spatio-Temporal Disentanglement ===")
    print(f"Detector Model: {args.detector}")
    print(f"Sequence Model: {args.sequence_model}")
    print(f"Input Directory: {args.input_dir}")
    print(f"Device: {device}")
    
    # 1. Load Detector (YOLO)
    detector = YOLO(args.detector)
    
    # 2. Load Spatio-Temporal Sequence Model (Transformer / LSTM)
    seq_model = None
    if os.path.exists(args.sequence_model):
        try:
            checkpoint = torch.load(args.sequence_model, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                arch = checkpoint.get('arch', 'transformer')
                d_model = checkpoint.get('d_model', 256)
                num_layers = checkpoint.get('num_layers', 4)
                nhead = checkpoint.get('nhead', 8)
                
                seq_model = PelletSequenceModel(
                    arch=arch,
                    num_classes=2,
                    d_model=d_model,
                    num_layers=num_layers,
                    nhead=nhead,
                    crop_size=96
                ).to(device)
                seq_model.load_state_dict(checkpoint['model_state_dict'])
                seq_model.eval()
                print(f"Loaded {arch.upper()} Sequence Model (d_model={d_model}, layers={num_layers})")
            else:
                print("Sequence model file format is legacy. Falling back to ByteTrack kinematics.")
        except Exception as e:
            print(f"Notice: Could not load sequence model ({e}). Using ByteTrack kinematics.")
    else:
        print(f"Notice: Sequence model {args.sequence_model} not found yet. Using heuristic tracker state.")
        
    # 3. Initialize Tracker
    tracker = PelletTracker(
        high_conf_thresh=args.conf_high,
        low_conf_thresh=args.conf_low,
        iou_thresh=0.20,
        max_lost_frames=6,
        crop_size=96,
        max_crop_history=16
    )
    
    # 4. Find Real Input Frames
    image_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.bmp")))
    if not image_paths:
        image_paths = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.bmp"), recursive=True))
        
    if not image_paths:
        print(f"No .bmp files found in {args.input_dir}. Checking for .jpg / .png...")
        image_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.jpg"))) + sorted(glob.glob(os.path.join(args.input_dir, "*.png")))
        
    if not image_paths:
        raise FileNotFoundError(f"No image files found in {args.input_dir}")
        
    selected_paths = image_paths[:args.max_frames]
    print(f"Processing sequence of {len(selected_paths)} consecutive frames...")
    
    # Initialize VideoWriter
    first_frame = cv2.imread(selected_paths[0])
    H_img, W_img = first_frame.shape[:2]
    out_video_path = os.path.join(args.output_dir, "fluid_bed_tracked_sequence.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Extra 46px height for header dashboard
    video_writer = cv2.VideoWriter(out_video_path, fourcc, 10.0, (W_img, H_img + 46))
    
    unique_tracks_seen = set()
    total_singles = set()
    total_agglomerates = set()
    total_overlaps = set()
    
    for f_idx, img_path in enumerate(selected_paths):
        frame = cv2.imread(img_path)
        if frame is None:
            continue
            
        t0 = time.time()
        
        # Step A: YOLO Candidate Detection
        # Low confidence threshold captures faint/occluded pellets
        yolo_res = detector.predict(
            source=frame,
            conf=args.conf_low,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False
        )
        
        detections = []
        for box in yolo_res[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            detections.append({
                'bbox_xyxy': xyxy,
                'confidence': conf,
                'class_id': cls_id
            })
            
        # Step B: ByteTrack Multi-Object Association & Kalman Update
        active_tracks = tracker.update(detections, image=frame)
        
        # Step C: Spatio-Temporal Sequence Classification (Transformer / LSTM)
        if seq_model is not None:
            for trk in active_tracks:
                if len(trk.history_crops) >= 3:
                    seq_tensor = prepare_sequence_tensor(trk.history_crops, target_len=8).to(device)
                    with torch.no_grad():
                        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                            out = seq_model(seq_tensor)
                            
                    pred_class = int(torch.argmax(out['class_logits'], dim=1).item())
                    pred_conf = float(out['class_probs'][0, pred_class].item())
                    overlap_prob = float(out['overlap_prob'][0, 0].item())
                    
                    trk.sequence_class_id = pred_class
                    trk.sequence_confidence = pred_conf
                    
                    # Disentangle optical overlap: if overlap_prob is high and class was 1,
                    # it indicates two single pellets crossed paths in 2D
                    if overlap_prob > 0.45:
                        trk.is_optical_overlap = True
                        trk.sequence_class_id = 0 # Correct to Single!
                    else:
                        trk.is_optical_overlap = False
                        
        # Update statistics
        stats = {'singles': 0, 'agglomerates': 0, 'resolved_overlaps': 0}
        for trk in active_tracks:
            unique_tracks_seen.add(trk.id)
            if trk.sequence_class_id > 0:
                stats['agglomerates'] += 1
                total_agglomerates.add(trk.id)
            elif trk.is_optical_overlap:
                stats['resolved_overlaps'] += 1
                total_overlaps.add(trk.id)
            else:
                stats['singles'] += 1
                total_singles.add(trk.id)
                
        # Step D: Render visual overlay
        rendered = draw_tracked_scene(frame, active_tracks, f_idx + 1, stats)
        video_writer.write(rendered)
        
        # Save snapshot for key frames
        if f_idx in [0, 4, 9, 14, 19]:
            snap_path = os.path.join(args.output_dir, f"tracked_frame_{f_idx+1:03d}.jpg")
            cv2.imwrite(snap_path, rendered)
            
        t1 = time.time()
        print(f"Frame {f_idx+1:02d}/{len(selected_paths)} [{t1-t0:.2f}s] - "
              f"Detections: {len(detections)} | Active Tracks: {len(active_tracks)} | "
              f"Singles: {stats['singles']} | Agglomerates: {stats['agglomerates']} | Overlaps: {stats['resolved_overlaps']}")
              
    video_writer.release()
    print(f"\n================ TRACKING SUMMARY ================")
    print(f"Total Unique Pellet Tracks Seen: {len(unique_tracks_seen)}")
    print(f"Total Single Pellets: {len(total_singles)}")
    print(f"Total Agglomerates: {len(total_agglomerates)}")
    print(f"Total Optical Overlaps Disentangled: {len(total_overlaps)}")
    print(f"Annotated Tracking Video Saved: {out_video_path}")
    print(f"Key Snapshots Saved to: {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-Object Tracking & Spatio-Temporal Disentanglement on Real Fluid Bed Video")
    parser.add_argument('--input_dir', type=str, default='Real-Data/T11/B4', help='Path to real dataset frames')
    parser.add_argument('--detector', type=str, default='outputs/models/pellet_detector_best.pt', help='Trained YOLO detector')
    parser.add_argument('--sequence_model', type=str, default='outputs/models/pellet_transformer_best.pt', help='Trained Spatio-Temporal Model')
    parser.add_argument('--output_dir', type=str, default='outputs/tracking_results', help='Output directory')
    parser.add_argument('--conf_high', type=float, default=0.35, help='ByteTrack high confidence threshold')
    parser.add_argument('--conf_low', type=float, default=0.12, help='ByteTrack low confidence threshold to recover occluded pellets')
    parser.add_argument('--imgsz', type=int, default=1024, help='Inference image size')
    parser.add_argument('--max_frames', type=int, default=20, help='Maximum frames to track')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Inference device')
    
    args = parser.parse_args()
    run_video_tracking(args)

