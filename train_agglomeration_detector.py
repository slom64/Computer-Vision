import os
import argparse
import glob
import cv2
import numpy as np
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO on synthetic fluid-bed pellet dataset & evaluate on real images.")
    parser.add_argument("--data", type=str, default="outputs/synthetic_dataset/dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Base model weights (e.g. yolo11n.pt, yolo11s.pt)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size for training")
    parser.add_argument("--device", type=str, default="0", help="CUDA device ID or 'cpu'")
    parser.add_argument("--eval_real", action="store_true", default=True, help="Run prediction on real dataset after training")
    parser.add_argument("--real_data_dir", type=str, default="Real-Data/T11/B4", help="Path to real test images")
    parser.add_argument("--output_pred_dir", type=str, default="outputs/real_predictions", help="Directory to save predictions")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs(args.output_pred_dir, exist_ok=True)
    
    print(f"=== Training Agglomeration Detector on Synthetic Data ===")
    print(f"Dataset YAML: {args.data}")
    print(f"Base Model: {args.model}")
    print(f"Epochs: {args.epochs}, Image Size: {args.imgsz}, Device: {args.device}")
    
    # 1. Initialize YOLO model
    model = YOLO(args.model)
    
    # 2. Train model
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project="outputs/train_runs",
        name="pellet_yolo11",
        exist_ok=True,
        workers=2,
        val=True
    )
    
    # 3. Save copy of best model
    best_weights = os.path.join("outputs/train_runs", "pellet_yolo11", "weights", "best.pt")
    target_weights = "outputs/models/pellet_detector_best.pt"
    if os.path.exists(best_weights):
        import shutil
        shutil.copy(best_weights, target_weights)
        print(f"\nBest model successfully saved to: {target_weights}")
    else:
        target_weights = args.model
        
    # 4. Predict on real images if requested
    if args.eval_real:
        print(f"\nRunning inference on real images in {args.real_data_dir}...")
        eval_model = YOLO(target_weights)
        
        real_images = sorted(glob.glob(os.path.join(args.real_data_dir, "*.bmp")))[:10]
        if not real_images:
            real_images = sorted(glob.glob(os.path.join(args.real_data_dir, "**", "*.bmp"), recursive=True))[:10]
            
        print(f"Found {len(real_images)} real images for testing.")
        
        for img_path in real_images:
            base_name = os.path.basename(img_path)
            res = eval_model.predict(
                source=img_path,
                conf=0.30,
                imgsz=args.imgsz,
                device=args.device,
                save=False
            )
            
            # Plot and save overlay
            for r in res:
                annotated_img = r.plot()
                out_path = os.path.join(args.output_pred_dir, f"pred_{base_name}.jpg")
                cv2.imwrite(out_path, annotated_img)
                print(f"Saved real prediction: {out_path} ({len(r.boxes)} detections)")

if __name__ == '__main__':
    main()

