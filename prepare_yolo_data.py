import os
import shutil
import random
from glob import glob

def prepare_yolo_dataset():
    src_dir = 'frames'
    dataset_dir = 'yolo_dataset'
    
    # Create YOLO directory structure
    for split in ['train', 'val']:
        os.makedirs(os.path.join(dataset_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, 'labels', split), exist_ok=True)
        
    # Get all jpg files that have a corresponding txt file
    all_jpgs = glob(os.path.join(src_dir, '*.jpg'))
    valid_pairs = []
    
    for jpg_path in all_jpgs:
        txt_path = jpg_path.replace('.jpg', '.txt')
        if os.path.exists(txt_path):
            valid_pairs.append((jpg_path, txt_path))
            
    print(f"Found {len(valid_pairs)} valid image-label pairs.")
    
    # Shuffle and split (80% train, 20% val)
    random.shuffle(valid_pairs)
    split_idx = int(len(valid_pairs) * 0.8)
    
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]
    
    def copy_pairs(pairs, split):
        for jpg, txt in pairs:
            basename = os.path.basename(jpg)
            shutil.copy(jpg, os.path.join(dataset_dir, 'images', split, basename))
            shutil.copy(txt, os.path.join(dataset_dir, 'labels', split, basename.replace('.jpg', '.txt')))
            
    copy_pairs(train_pairs, 'train')
    copy_pairs(val_pairs, 'val')
    print(f"Copied {len(train_pairs)} to train and {len(val_pairs)} to val.")
    
    # Create dataset.yaml
    abs_path = os.path.abspath(dataset_dir).replace('\\', '/')
    yaml_content = f"""path: {abs_path}
train: images/train
val: images/val

names:
  0: pellet
"""
    with open('dataset.yaml', 'w') as f:
        f.write(yaml_content)
    print("Created dataset.yaml.")

if __name__ == '__main__':
    prepare_yolo_dataset()
