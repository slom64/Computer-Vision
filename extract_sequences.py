import os
import csv
from collections import defaultdict
import numpy as np
from PIL import Image
from config import IMG_WIDTH, IMG_HEIGHT, SEQUENCE_LENGTH, CROP_SIZE

def extract_sequences():
    os.makedirs('sequence_dataset/single', exist_ok=True)
    os.makedirs('sequence_dataset/agglomerate', exist_ok=True)
    
    # Read CSV
    pellet_data = defaultdict(list)
    if not os.path.exists('dataset.csv'):
        print("dataset.csv not found! Run the simulation first to generate data.")
        return
        
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
            
    print(f"Loaded records for {len(pellet_data)} unique pellets.")
    
    seq_count = 0
    # Cache recently loaded PIL images to speed up consecutive sequence extraction
    cached_frame_id = None
    cached_img = None
    
    for pid, records in pellet_data.items():
        records.sort(key=lambda x: x['frame'])
        
        # Need continuous sequences of length SEQUENCE_LENGTH
        for i in range(len(records) - SEQUENCE_LENGTH + 1):
            seq_records = records[i:i + SEQUENCE_LENGTH]
            
            # Verify they are strictly consecutive frames
            is_consecutive = True
            for j in range(1, SEQUENCE_LENGTH):
                if seq_records[j]['frame'] != seq_records[j-1]['frame'] + 1:
                    is_consecutive = False
                    break
                    
            if not is_consecutive:
                continue
                
            seq_tensors = []
            is_agg = seq_records[0]['is_agglomerate']
            valid_sequence = True
            
            for r in seq_records:
                img_path = f"frames/frame_{r['frame']}.jpg"
                if not os.path.exists(img_path):
                    valid_sequence = False
                    break
                    
                if cached_frame_id == r['frame'] and cached_img is not None:
                    img = cached_img
                else:
                    try:
                        img = Image.open(img_path).convert('RGB')
                        cached_frame_id = r['frame']
                        cached_img = img
                    except Exception:
                        valid_sequence = False
                        break
                
                # De-normalize coordinates
                x_center = r['x'] * IMG_WIDTH
                y_center = r['y'] * IMG_HEIGHT
                w = r['w'] * IMG_WIDTH
                h = r['h'] * IMG_HEIGHT
                
                # Dynamic square crop with 25% contextual padding
                size = max(w, h) * 1.25
                left = int(max(0, x_center - size / 2))
                right = int(min(IMG_WIDTH, x_center + size / 2))
                top = int(max(0, y_center - size / 2))
                bottom = int(min(IMG_HEIGHT, y_center + size / 2))
                
                if (right - left) < 4 or (bottom - top) < 4:
                    valid_sequence = False
                    break
                    
                crop = img.crop((left, top, right, bottom))
                crop = crop.resize((CROP_SIZE, CROP_SIZE), Image.Resampling.BILINEAR)
                
                # Shape (C, H, W), normalized to [0, 1]
                crop_arr = np.array(crop, dtype=np.float32).transpose(2, 0, 1) / 255.0
                seq_tensors.append(crop_arr)
                
            if valid_sequence and len(seq_tensors) == SEQUENCE_LENGTH:
                # Shape: (SEQUENCE_LENGTH, 3, CROP_SIZE, CROP_SIZE)
                seq_np = np.stack(seq_tensors)
                
                folder = 'agglomerate' if is_agg else 'single'
                out_path = os.path.join('sequence_dataset', folder, f'pid_{pid}_seq_{seq_count}.npy')
                np.save(out_path, seq_np)
                seq_count += 1
                
    print(f"Successfully extracted {seq_count} sequences (CROP_SIZE={CROP_SIZE}x{CROP_SIZE}, SEQUENCE_LENGTH={SEQUENCE_LENGTH}).")

if __name__ == '__main__':
    extract_sequences()
