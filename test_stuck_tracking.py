import glob
import os
import cv2
from ultralytics import YOLO
from simulation.pellet_tracker import PelletTracker

model = YOLO('outputs/models/pellet_detector_best.pt')
tracker = PelletTracker(high_conf_thresh=0.30, low_conf_thresh=0.15, iou_thresh=0.20, max_lost_frames=8)

bmps = sorted(glob.glob('Real-Data/T12/T12-B6Aggolmeration/*.bmp'))[:25]
print(f"Running tracking across {len(bmps)} consecutive frames...")

stuck_candidates = {}

for f_idx, bmp in enumerate(bmps):
    im = cv2.imread(bmp)
    res = model.predict(im, conf=0.15, imgsz=1024, verbose=False)
    dets = []
    for b in res[0].boxes:
        xyxy = b.xyxy[0].cpu().numpy().tolist()
        conf = float(b.conf[0])
        cls = int(b.cls[0])
        dets.append({'bbox_xyxy': xyxy, 'confidence': conf, 'class_id': cls})
        
    active = tracker.update(dets, image=im)
    
    for trk in active:
        vx, vy = trk.get_velocity()
        speed = (vx**2 + vy**2)**0.5
        box = trk.get_state_xyxy()
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        
        if trk.hits >= 5 and speed < 3.5:
            if trk.id not in stuck_candidates:
                stuck_candidates[trk.id] = {
                    'first_frame': f_idx,
                    'hits': trk.hits,
                    'pos': (cx, cy),
                    'speed': speed
                }
            else:
                stuck_candidates[trk.id]['hits'] = trk.hits
                stuck_candidates[trk.id]['last_frame'] = f_idx
                stuck_candidates[trk.id]['pos'] = (cx, cy)
                stuck_candidates[trk.id]['speed'] = speed

print("\n=== TRACKS PERSISTING >= 7 FRAMES WITH LOW VELOCITY (< 3.5 px/frame) ===")
for tid, info in stuck_candidates.items():
    if info['hits'] >= 7:
        print(f"Track ID #{tid}: Hits={info['hits']} frames | Center=({info['pos'][0]:.1f}, {info['pos'][1]:.1f}) | Speed={info['speed']:.2f} px/frame")
