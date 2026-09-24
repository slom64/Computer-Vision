import torch
from ultralytics import YOLO
from config import IMG_WIDTH

def main():
    # Detect GPU availability
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Loading YOLO11 Small model on device: {device}...")
    
    # We upgrade to YOLO11s (Small, 9.4M params) or YOLO11m (Medium, 20M params)
    # for significantly deeper feature extraction and accurate small-object localization
    model = YOLO("yolo11s.pt")

    print("Starting high-resolution training on pellet dataset...")
    results = model.train(
        data="dataset.yaml",
        epochs=60,
        imgsz=IMG_WIDTH,      # 1024x1024 high resolution
        device=device,
        batch=8,              # Optimized for RTX 3070 8GB VRAM
        workers=4,
        box=8.5,              # Emphasize tight, exact bounding box borders
        close_mosaic=10,      # Turn off aggressive augmentation in final epochs for realistic boundaries
        patience=25,
        save=True,
        verbose=True
    )

if __name__ == '__main__':
    main()
