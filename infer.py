import os
import cv2
import torch
import numpy as np
import glob
from PIL import Image
from ultralytics import YOLO
from train_sequence_model import SequenceClassifier
from config import CROP_SIZE, SEQUENCE_LENGTH, IMG_WIDTH, IMG_HEIGHT

def infer(conf_threshold=0.15, suspicion_aspect_ratio=1.22):
    # Detect GPU / CPU device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running inference engine on: {device}")
    
    # Dynamically locate the most recent YOLO training run weights
    yolo_model_paths = sorted(glob.glob('runs/detect/train*/weights/best.pt'), key=os.path.getmtime)
    if not yolo_model_paths:
        print("No trained YOLO weights found in runs/detect/! Make sure to run the training step first.")
        return
        
    yolo_model_path = yolo_model_paths[-1]
    print(f"Loading latest YOLO model: {yolo_model_path}")
    yolo_model = YOLO(yolo_model_path)
    
    seq_model_path = 'sequence_model.pt'
    if not os.path.exists(seq_model_path):
        print("sequence_model.pt not found! Make sure to run the Sequence training step first.")
        return
        
    print(f"Loading ResNet-BiLSTM-Attention classifier from: {seq_model_path}")
    seq_model = SequenceClassifier().to(device)
    seq_model.load_state_dict(torch.load(seq_model_path, map_location=device))
    seq_model.eval()
    
    # Permanent classification memory & active temporal buffers
    classified = {}          # track_id -> "single" or "agglomerate"
    suspicious_buffers = {}  # track_id -> list of (3, CROP_SIZE, CROP_SIZE) float32 arrays
    
    os.makedirs('output_frames', exist_ok=True)
    
    # Sort simulation frames numerically
    frames = sorted(
        glob.glob('frames/*.jpg'), 
        key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]) if '_' in os.path.basename(x) else 0
    )
    
    if not frames:
        print("No frames found in frames/! Generate simulation frames first.")
        return
        
    print(f"Processing {len(frames)} frames through the Spatial-Temporal Detection Pipeline...")
    
    for i, frame_path in enumerate(frames):
        img_bgr = cv2.imread(frame_path)
        if img_bgr is None:
            continue
            
        frame_h, frame_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Track objects across frames with ByteTrack
        results = yolo_model.track(
            img_rgb, 
            tracker="bytetrack.yaml", 
            persist=True, 
            conf=conf_threshold, 
            verbose=False
        )
        
        for result in results:
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                continue
                
            for box, track_id_t in zip(boxes, boxes.id):
                track_id = int(track_id_t.item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w = max(1.0, x2 - x1)
                h = max(1.0, y2 - y1)
                
                # Base status: Tracked Pellet
                label = f"ID:{track_id} Pellet"
                color = (255, 120, 0) # Cyan/Blue in BGR
                
                # Check if this pellet's identity has already been definitively locked
                if track_id in classified:
                    if classified[track_id] == "agglomerate":
                        label = f"ID:{track_id} Agglomerate"
                        color = (0, 0, 255) # Red in BGR
                    else:
                        label = f"ID:{track_id} Single"
                        color = (0, 255, 0) # Green in BGR
                else:
                    # Compute geometric elongation aspect ratio
                    aspect_ratio = max(w / h, h / w)
                    
                    # If elongated or unusually large, trigger the Suspicion Pipeline
                    if aspect_ratio >= suspicion_aspect_ratio:
                        if track_id not in suspicious_buffers:
                            suspicious_buffers[track_id] = []
                            
                        # Contextual square crop for the ResNet backbone
                        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        size = max(w, h) * 1.25
                        left = int(max(0, cx - size / 2.0))
                        right = int(min(frame_w, cx + size / 2.0))
                        top = int(max(0, cy - size / 2.0))
                        bottom = int(min(frame_h, cy + size / 2.0))
                        
                        crop = img_rgb[top:bottom, left:right]
                        if crop.size > 0 and (right - left) >= 4 and (bottom - top) >= 4:
                            crop_pil = Image.fromarray(crop).resize((CROP_SIZE, CROP_SIZE), Image.Resampling.BILINEAR)
                            crop_arr = np.array(crop_pil, dtype=np.float32).transpose(2, 0, 1) / 255.0
                            suspicious_buffers[track_id].append(crop_arr)
                            
                            buffer_len = len(suspicious_buffers[track_id])
                            label = f"ID:{track_id} Suspicious [{buffer_len}/{SEQUENCE_LENGTH}]"
                            color = (0, 165, 255) # Orange in BGR
                            
                            # Once SEQUENCE_LENGTH consecutive tumbling frames are captured, classify with AI
                            if buffer_len == SEQUENCE_LENGTH:
                                seq_tensor = torch.tensor(
                                    np.stack(suspicious_buffers[track_id]), 
                                    dtype=torch.float32
                                ).unsqueeze(0).to(device) # (1, Seq_len, 3, CROP_SIZE, CROP_SIZE)
                                
                                with torch.no_grad():
                                    logits = seq_model(seq_tensor)
                                    probs = torch.softmax(logits, dim=1)
                                    pred = torch.argmax(probs, dim=1).item()
                                    confidence = probs[0, pred].item()
                                    
                                if pred == 1:
                                    classified[track_id] = "agglomerate"
                                    label = f"ID:{track_id} Agglomerate ({confidence*100:.0f}%)"
                                    color = (0, 0, 255)
                                else:
                                    classified[track_id] = "single"
                                    label = f"ID:{track_id} Single ({confidence*100:.0f}%)"
                                    color = (0, 255, 0)
                                    
                                # Release buffer memory; permanently locked!
                                del suspicious_buffers[track_id]
                                
                # Draw visually appealing styled bounding box & text
                cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(img_bgr, (int(x1), int(y1) - th - 6), (int(x1) + tw + 4, int(y1)), color, -1)
                cv2.putText(img_bgr, label, (int(x1) + 2, int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                
        # Save output annotated frame
        out_path = os.path.join('output_frames', os.path.basename(frame_path))
        cv2.imwrite(out_path, img_bgr)
        
    print(f"Inference complete! {len(frames)} frames annotated and saved in output_frames/ folder.")

if __name__ == '__main__':
    infer()
