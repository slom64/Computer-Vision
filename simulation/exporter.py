import os
import json
import csv
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from .config import SimulationConfig

class DatasetExporter:
    """
    Exports synthetic simulation frames into machine learning training datasets.
    Supports:
    - YOLOv8 / YOLOv11 Detection (bounding boxes)
    - YOLOv8 / YOLOv11 Segmentation (instance masks)
    - LabelMe JSON (native format matching existing human annotations)
    - COCO Dataset JSON
    - Sequence Tracking CSV & cropped sequences for LSTM/temporal models
    """
    def __init__(self, config: SimulationConfig, output_dir: str = 'synthetic_dataset'):
        self.config = config
        self.output_dir = output_dir
        self.classes = config.get_class_names()
        
        # Directories
        self.images_dir = os.path.join(output_dir, 'images')
        self.labels_dir = os.path.join(output_dir, 'labels')
        self.labelme_dir = os.path.join(output_dir, 'labelme')
        self.tracking_dir = os.path.join(output_dir, 'tracking')
        
        self.tracking_csv_path = os.path.join(self.tracking_dir, 'dataset.csv')
        self.tracking_file = None
        self.csv_writer = None
        
    def setup_directories(self, splits: List[str] = ['train', 'val']):
        for split in splits:
            os.makedirs(os.path.join(self.images_dir, split), exist_ok=True)
            os.makedirs(os.path.join(self.labels_dir, split), exist_ok=True)
        os.makedirs(self.labelme_dir, exist_ok=True)
        os.makedirs(self.tracking_dir, exist_ok=True)
        
        # Write dataset.yaml for YOLO
        yaml_path = os.path.join(self.output_dir, 'dataset.yaml')
        abs_output = os.path.abspath(self.output_dir).replace('\\', '/')
        
        names_dict = "\n".join([f"  {k}: {v}" for k, v in self.classes.items()])
        yaml_content = f"""path: {abs_output}
train: images/train
val: images/val

names:
{names_dict}
"""
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)

    def open_tracking_logger(self):
        os.makedirs(self.tracking_dir, exist_ok=True)
        self.tracking_file = open(self.tracking_csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.tracking_file)
        self.csv_writer.writerow([
            'frame_id', 'pellet_id', 'is_agglomerate', 'pellet_count',
            'class_id', 'x_center', 'y_center', 'width', 'height'
        ])

    def close_tracking_logger(self):
        if self.tracking_file:
            self.tracking_file.close()
            self.tracking_file = None

    def save_frame(
        self,
        frame_idx: int,
        image: np.ndarray,
        annotations: List[Dict[str, Any]],
        split: str = 'train',
        export_labelme: bool = True,
        export_yolo_seg: bool = False
    ):
        base_name = f"sim_pic_{frame_idx:06d}"
        img_filename = f"{base_name}.jpg"
        
        # 1. Save Image
        img_save_path = os.path.join(self.images_dir, split, img_filename)
        cv2.imwrite(img_save_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        
        # 2. Save YOLO Label
        txt_filename = f"{base_name}.txt"
        txt_save_path = os.path.join(self.labels_dir, split, txt_filename)
        yolo_lines = []
        
        for annot in annotations:
            cid = annot['class_id']
            if export_yolo_seg and 'yolo_seg' in annot and len(annot['yolo_seg']) >= 6:
                poly_coords = " ".join([f"{coord:.6f}" for coord in annot['yolo_seg']])
                yolo_lines.append(f"{cid} {poly_coords}\n")
            else:
                xc, yc, w, h = annot['bbox_yolo']
                yolo_lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                
            # Log to tracking CSV if active
            if self.csv_writer:
                is_agg = 1 if annot['count'] > 1 else 0
                self.csv_writer.writerow([
                    frame_idx, annot['id'], is_agg, annot['count'],
                    cid, annot['bbox_yolo'][0], annot['bbox_yolo'][1],
                    annot['bbox_yolo'][2], annot['bbox_yolo'][3]
                ])
                
        with open(txt_save_path, 'w') as f:
            f.writelines(yolo_lines)
            
        # 3. Save LabelMe JSON (if requested)
        if export_labelme:
            labelme_shapes = []
            for annot in annotations:
                x1, y1, x2, y2 = annot['bbox_xyxy']
                if 'polygon' in annot and len(annot['polygon']) >= 3:
                    pts = [[float(p[0]), float(p[1])] for p in annot['polygon']]
                    shape_type = "polygon"
                else:
                    pts = [[float(x1), float(y1)], [float(x2), float(y2)]]
                    shape_type = "rectangle"
                    
                labelme_shapes.append({
                    "label": "Agglomerate" if annot['count'] > 1 else "Pellet",
                    "text": f"count={annot['count']}",
                    "points": pts,
                    "group_id": None,
                    "shape_type": shape_type,
                    "flags": {
                        "count": annot['count'],
                        "type": annot['type'],
                        "class_id": annot['class_id']
                    }
                })
                
            labelme_data = {
                "version": "0.4.43",
                "flags": {},
                "shapes": labelme_shapes,
                "imagePath": img_filename,
                "imageData": None,
                "imageHeight": self.config.image_height,
                "imageWidth": self.config.image_width
            }
            labelme_path = os.path.join(self.labelme_dir, f"{base_name}.json")
            with open(labelme_path, 'w') as f:
                json.dump(labelme_data, f, indent=2)

