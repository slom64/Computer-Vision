import os
import sys
import torch
from pathlib import Path
from ultralytics import YOLO

# Ensure UTF-8 stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def train_yolo_real(epochs=60, imgsz=1024, batch=4):
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    yaml_path = Path("dataset_real.yaml").resolve().as_posix()
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Dataset config not found at: {yaml_path}")
        
    print(f"[INFO] Initializing YOLO11s from pretrained weights: yolo11s.pt")
    model = YOLO("yolo11s.pt")
    
    # Save directly to runs/detect/yolo11s_real
    project_dir = Path("runs/detect").resolve().as_posix()
    print(f"[INFO] Training output directory: {project_dir}/yolo11s_real")
    
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        batch=batch,
        workers=2,
        project=project_dir,
        name="yolo11s_real",
        exist_ok=True,
        save=True,
        box=7.5,
        cls=1.5,
        mosaic=0.5,
        close_mosaic=15,
        patience=40,
        verbose=True
    )
    
    best_weights = Path("runs/detect/yolo11s_real/weights/best.pt")
    if best_weights.exists():
        print(f"\n[SUCCESS] Training finished! Best weights saved to: {best_weights.resolve()}")
    else:
        print(f"\n[INFO] Check weights under runs/detect/yolo11s_real/weights/")
        
    return results

if __name__ == '__main__':
    train_yolo_real(epochs=50, imgsz=1024, batch=4)

