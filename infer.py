import os
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from train_sequence_model import SequenceClassifier

def infer():
    # Load Models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    import glob
    yolo_model_paths = sorted(glob.glob('runs/detect/train*/weights/best.pt'), key=os.path.getmtime)
    if not yolo_model_paths:
        print("YOLO model not trained yet. Make sure to run the training step first.")
        return
        
    yolo_model_path = yolo_model_paths[-1]
    print(f"Loading YOLO model from: {yolo_model_path}")
        
    yolo_model = YOLO(yolo_model_path)
    
    seq_model_path = 'sequence_model.pt'
    if not os.path.exists(seq_model_path):
        print("Sequence model not trained yet. Make sure to run train_sequence_model.py first.")
        return
        
    seq_model = SequenceClassifier().to(device)
    seq_model.load_state_dict(torch.load(seq_model_path, map_location=device))
    seq_model.eval()
    
    # Memory buffers
    classified = {} # track_id -> "single" or "agglomerate"
    suspicious_buffers = {} # track_id -> list of numpy arrays
    
    os.makedirs('output_frames', exist_ok=True)
    
    # Run tracking on frames
    # Sort the frames numerically
    import glob
    frames = sorted(glob.glob('frames/*.jpg'), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    print(f"Running inference on {len(frames)} frames...")
    
    for i, frame_path in enumerate(frames):
        img_bgr = cv2.imread(frame_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Run YOLO with ByteTrack
        results = yolo_model.track(img_rgb, tracker="bytetrack.yaml", persist=True, verbose=False)
        
        for result in results:
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                continue
                
            for box, track_id_t in zip(boxes, boxes.id):
                track_id = int(track_id_t.item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w = x2 - x1
                h = y2 - y1
                
                # Default label and color
                label = f"ID: {track_id} Pellet"
                color = (255, 0, 0) # Blue (BGR format in OpenCV!)
                
                if track_id in classified:
                    if classified[track_id] == "agglomerate":
                        label = f"ID: {track_id} Agglomerate!"
                        color = (0, 0, 255) # Red
                    else:
                        label = f"ID: {track_id} Single"
                        color = (0, 255, 0) # Green
                else:
                    # Check suspicion
                    aspect_ratio = max(w/h, h/w) if h > 0 and w > 0 else 1.0
                    
                    if aspect_ratio > 1.2:
                        if track_id not in suspicious_buffers:
                            suspicious_buffers[track_id] = []
                            
                        # Crop and process
                        crop = img_rgb[int(y1):int(y2), int(x1):int(x2)]
                        if crop.size > 0:
                            crop_pil = Image.fromarray(crop).resize((64, 64))
                            crop_arr = np.array(crop_pil).transpose(2, 0, 1).astype(np.float32) / 255.0
                            suspicious_buffers[track_id].append(crop_arr)
                            
                            label = f"ID: {track_id} Suspicious ({len(suspicious_buffers[track_id])}/10)"
                            color = (0, 165, 255) # Orange
                            
                            if len(suspicious_buffers[track_id]) == 10:
                                # Classify!
                                seq_tensor = torch.tensor(np.stack(suspicious_buffers[track_id]), dtype=torch.float32).unsqueeze(0).to(device)
                                with torch.no_grad():
                                    outputs = seq_model(seq_tensor)
                                    pred = outputs.argmax(dim=1).item()
                                    
                                if pred == 1:
                                    classified[track_id] = "agglomerate"
                                    label = f"ID: {track_id} Agglomerate!"
                                    color = (0, 0, 255)
                                else:
                                    classified[track_id] = "single"
                                    label = f"ID: {track_id} Single"
                                    color = (0, 255, 0)
                                    
                                # Free up memory
                                del suspicious_buffers[track_id]
                                
                # Draw bounding box
                cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(img_bgr, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
        # Save output
        out_path = os.path.join('output_frames', os.path.basename(frame_path))
        cv2.imwrite(out_path, img_bgr)
        
    print("Inference complete! Check the output_frames/ folder.")
    
if __name__ == '__main__':
    infer()
