import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO
from simulation.temporal_verifier import TemporalAgglomerationVerifier

def run_temporal_verification_pipeline(
    weights_path: str = "outputs/models/pellet_detector_best.pt",
    real_data_dir: str = "Real-Data/T11/B4",
    output_dir: str = "outputs/temporal_predictions",
    conf_thresh: float = 0.28,
    num_frames: int = 12
):
    """
    Runs multi-frame temporal reasoning on consecutive real fluid-bed camera frames:
    1. Detects raw candidates on each frame using spatial YOLO model.
    2. Runs 3-frame sliding window temporal verification.
    3. Saves side-by-side visualizations comparing raw single-frame vs temporal-verified output.
    """
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(weights_path):
        weights_path = "yolo11n.pt"
        
    print(f"Loading detector weights: {weights_path}")
    model = YOLO(weights_path)
    verifier = TemporalAgglomerationVerifier(iou_threshold=0.20, max_drift_px=35.0)
    
    # Load consecutive frames
    frame_files = sorted(glob.glob(os.path.join(real_data_dir, "*.bmp")))[:num_frames]
    print(f"Loaded {len(frame_files)} consecutive frames from {real_data_dir}.")
    
    if len(frame_files) < 3:
        print("Need at least 3 frames for multi-frame temporal reasoning.")
        return
        
    # Step 1: Detect raw candidates across all frames
    all_frame_detections = []
    all_raw_images = []
    
    print("Running initial frame-by-frame detection...")
    for idx, fpath in enumerate(frame_files):
        img = cv2.imread(fpath)
        all_raw_images.append(img)
        
        preds = model.predict(source=img, conf=conf_thresh, imgsz=1024, verbose=False)
        frame_dets = []
        for r in preds:
            for b in r.boxes:
                xyxy = b.xyxy[0].cpu().numpy().astype(int).tolist()
                cid = int(b.cls[0].cpu().numpy())
                conf = float(b.conf[0].cpu().numpy())
                frame_dets.append({
                    'bbox_xyxy': xyxy,
                    'class_id': cid,
                    'class_name': 'agglomerate' if cid > 0 else 'single_pellet',
                    'conf': conf
                })
        all_frame_detections.append(frame_dets)
        
    # Step 2: Run 3-frame sliding window temporal reasoning
    print("Applying 3-frame temporal verification...")
    verified_results = []
    
    for t in range(len(frame_files)):
        if t == 0:
            window = [all_frame_detections[0], all_frame_detections[0], all_frame_detections[1]]
        elif t == len(frame_files) - 1:
            window = [all_frame_detections[t-1], all_frame_detections[t], all_frame_detections[t]]
        else:
            window = [all_frame_detections[t-1], all_frame_detections[t], all_frame_detections[t+1]]
            
        verified_t = verifier.verify_window_3_frames(window)
        verified_results.append(verified_t)
        
    # Step 3: Draw comparison overlays
    print("Exporting verified comparison frames...")
    colors = {
        'single_pellet': (0, 230, 0),       # Green
        'agglomerate': (0, 0, 255),         # Red
        'resolved_overlap': (255, 180, 0)   # Cyan/Orange for resolved overlap
    }
    
    for t in range(len(frame_files)):
        raw_img = all_raw_images[t].copy()
        raw_overlay = raw_img.copy()
        verified_overlay = raw_img.copy()
        
        # Draw Raw 1-Frame detections
        for det in all_frame_detections[t]:
            x1, y1, x2, y2 = det['bbox_xyxy']
            col = (0, 0, 255) if det['class_id'] > 0 else (0, 230, 0)
            cv2.rectangle(raw_overlay, (x1, y1), (x2, y2), col, 2)
            cv2.putText(raw_overlay, f"{det['class_name']} {det['conf']:.2f}",
                        (x1, max(18, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
                        
        # Draw 3-Frame Verified detections
        for det in verified_results[t]:
            x1, y1, x2, y2 = det['bbox_xyxy']
            status = det.get('temporal_status', 'confirmed')
            if det.get('is_optical_overlap', False):
                col = colors['resolved_overlap']
                tag = "Single (Overlap Resolved)"
            elif det['class_id'] > 0:
                col = colors['agglomerate']
                tag = "Agglomerate (Confirmed)"
            else:
                col = colors['single_pellet']
                tag = "Single Pellet"
                
            cv2.rectangle(verified_overlay, (x1, y1), (x2, y2), col, 2)
            cv2.putText(verified_overlay, tag, (x1, max(18, y1-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
                        
        # Side-by-side comparison (resized to fit display)
        half_w = int(raw_img.shape[1] * 0.5)
        half_h = int(raw_img.shape[0] * 0.5)
        
        raw_small = cv2.resize(raw_overlay, (half_w, half_h))
        ver_small = cv2.resize(verified_overlay, (half_w, half_h))
        
        cv2.putText(raw_small, "1-Frame Raw YOLO", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(ver_small, "3-Frame Temporal Verified", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        side_by_side = np.hstack([raw_small, ver_small])
        out_name = f"temporal_comparison_{t:03d}.jpg"
        cv2.imwrite(os.path.join(output_dir, out_name), side_by_side)
        
    print(f"Temporal verification completed! Comparisons saved in: {output_dir}")

if __name__ == '__main__':
    run_temporal_verification_pipeline()

