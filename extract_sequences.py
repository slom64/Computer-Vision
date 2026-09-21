import os
import csv
from collections import defaultdict
import numpy as np
from PIL import Image

SEQUENCE_LENGTH = 10
CROP_SIZE = 64
IMG_WIDTH = 512
IMG_HEIGHT = 512

def extract_sequences():
    os.makedirs('sequence_dataset/single', exist_ok=True)
    os.makedirs('sequence_dataset/agglomerate', exist_ok=True)
    
    # Read CSV
    pellet_data = defaultdict(list)
    with open('dataset.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pellet_data[int(row['pellet_id'])].append({
                'frame': int(row['frame']),
                'is_agglomerate': row['is_agglomerate'] == 'True',
                'x': float(row['x_center']),
                'y': float(row['y_center']),
                'w': float(row['width']),
                'h': float(row['height'])
            })
            
    print(f"Loaded data for {len(pellet_data)} unique pellets.")
    
    seq_count = 0
    for pid, records in pellet_data.items():
        # Sort by frame
        records.sort(key=lambda x: x['frame'])
        
        # We need continuous sequences of length SEQUENCE_LENGTH
        for i in range(len(records) - SEQUENCE_LENGTH + 1):
            seq_records = records[i:i+SEQUENCE_LENGTH]
            
            # Verify they are consecutive frames
            is_consecutive = True
            for j in range(1, SEQUENCE_LENGTH):
                if seq_records[j]['frame'] != seq_records[j-1]['frame'] + 1:
                    is_consecutive = False
                    break
                    
            if not is_consecutive:
                continue
                
            # Extract crops
            seq_tensors = []
            is_agg = seq_records[0]['is_agglomerate']
            
            for r in seq_records:
                img_path = f"frames/frame_{r['frame']}.jpg"
                if not os.path.exists(img_path):
                    break
                    
                img = Image.open(img_path).convert('RGB')
                
                # De-normalize coordinates
                x_center = r['x'] * IMG_WIDTH
                y_center = r['y'] * IMG_HEIGHT
                w = r['w'] * IMG_WIDTH
                h = r['h'] * IMG_HEIGHT
                
                # Crop bounding box (with a little padding to see context)
                padding = 0.2
                w = w * (1 + padding)
                h = h * (1 + padding)
                
                left = int(max(0, x_center - w/2))
                right = int(min(IMG_WIDTH, x_center + w/2))
                top = int(max(0, y_center - h/2))
                bottom = int(min(IMG_HEIGHT, y_center + h/2))
                
                # If bounding box is invalid, skip
                if right <= left or bottom <= top:
                    break
                    
                crop = img.crop((left, top, right, bottom))
                crop = crop.resize((CROP_SIZE, CROP_SIZE))
                
                # Convert to numpy array (H, W, C) -> (C, H, W)
                crop_arr = np.array(crop).transpose(2, 0, 1)
                # Normalize to 0-1
                crop_arr = crop_arr.astype(np.float32) / 255.0
                
                seq_tensors.append(crop_arr)
                
            if len(seq_tensors) == SEQUENCE_LENGTH:
                # Shape: (10, 3, 64, 64)
                seq_np = np.stack(seq_tensors)
                
                folder = 'agglomerate' if is_agg else 'single'
                out_path = os.path.join('sequence_dataset', folder, f'pid_{pid}_seq_{seq_count}.npy')
                np.save(out_path, seq_np)
                seq_count += 1
                
    print(f"Extracted {seq_count} sequences.")

if __name__ == '__main__':
    extract_sequences()
