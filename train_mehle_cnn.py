import os
import sys
import glob
import time
import math
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Ensure UTF-8 stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# Helper Functions: Stratified Split & Metrics (Pure NumPy / PyTorch)
# ══════════════════════════════════════════════════════════════════════════════
def stratified_split_indices(labels, test_ratio=0.2, val_ratio=0.2, seed=42):
    np.random.seed(seed)
    labels = np.array(labels)
    train_idx, val_idx, test_idx = [], [], []

    for c in np.unique(labels):
        c_indices = np.where(labels == c)[0]
        np.random.shuffle(c_indices)
        n = len(c_indices)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))

        test_idx.extend(c_indices[:n_test])
        val_idx.extend(c_indices[n_test:n_test + n_val])
        train_idx.extend(c_indices[n_test + n_val:])

    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)

def compute_confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def compute_classification_report(y_true, y_pred, class_names):
    num_classes = len(class_names)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    lines = [f"{'Class':<25} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}"]
    lines.append("-" * 65)

    precisions, recalls, f1s, supports = [], [], [], []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(support)
        lines.append(f"{class_names[i]:<25} {prec*100:8.2f}% {rec*100:8.2f}% {f1*100:8.2f}% {support:<8}")

    lines.append("-" * 65)
    total_support = sum(supports)
    macro_f1 = np.mean(f1s)
    weighted_f1 = np.average(f1s, weights=supports)
    lines.append(f"{'Macro Avg':<25} {np.mean(precisions)*100:8.2f}% {np.mean(recalls)*100:8.2f}% {macro_f1*100:8.2f}% {total_support:<8}")
    lines.append(f"{'Weighted Avg':<25} {np.average(precisions, weights=supports)*100:8.2f}% {np.average(recalls, weights=supports)*100:8.2f}% {weighted_f1*100:8.2f}% {total_support:<8}")
    return "\n".join(lines), cm

def compute_roc_curve(y_true, y_scores):
    # Sort descending by score
    desc_idx = np.argsort(y_scores)[::-1]
    y_true = np.array(y_true)[desc_idx]
    y_scores = np.array(y_scores)[desc_idx]

    distinct_indices = np.where(np.diff(y_scores))[0]
    threshold_idxs = np.r_[distinct_indices, y_true.size - 1]

    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    total_pos = y_true.sum()
    total_neg = len(y_true) - total_pos

    tpr = np.r_[0, tps / max(1, total_pos)]
    fpr = np.r_[0, fps / max(1, total_neg)]

    # Trapezoidal rule for AUC
    roc_auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, roc_auc

# ══════════════════════════════════════════════════════════════════════════════
# 1. Dataset & Augmentations (Mehle et al. 2017)
# ══════════════════════════════════════════════════════════════════════════════
class PelletCropDataset(Dataset):
    """
    Dataset loader for 96x96 candidate pellet crops.
    Implements the exact augmentations described in Mehle et al. (2017):
    - Random rotation (0 to 360 degrees)
    - Random Gaussian noise (std in range [0, 0.01])
    - Horizontal and vertical flips
    - Z-score normalization: (x - mean) / std per candidate image
    """
    def __init__(self, file_paths, labels, areas=None, is_train=True):
        self.file_paths = file_paths
        self.labels = labels
        self.areas = areas if areas is not None else [0] * len(labels)
        self.is_train = is_train

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        fpath = str(self.file_paths[idx])
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((96, 96), dtype=np.uint8)
        else:
            if img.shape != (96, 96):
                img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_LINEAR)

        img_float = img.astype(np.float32) / 255.0

        if self.is_train:
            # 1. Random rotation 0 to 360 degrees
            angle = random.uniform(0, 360)
            center = (48, 48)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            img_float = cv2.warpAffine(img_float, rot_mat, (96, 96), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

            # 2. Random horizontal and vertical flips
            if random.random() > 0.5:
                img_float = np.fliplr(img_float)
            if random.random() > 0.5:
                img_float = np.flipud(img_float)

            # 3. Random Gaussian noise (std 0 to 0.01 per Mehle 2017)
            if random.random() > 0.5:
                noise_std = random.uniform(0.001, 0.01)
                noise = np.random.normal(0, noise_std, img_float.shape).astype(np.float32)
                img_float = np.clip(img_float + noise, 0.0, 1.0)

        # Standardize: subtract mean and divide by std (Mehle et al. 2017 Section 3.3)
        mean_val = float(img_float.mean())
        std_val = float(img_float.std()) + 1e-6
        img_norm = (img_float - mean_val) / std_val

        # Tensor shape (1, 96, 96)
        tensor_x = torch.tensor(img_norm, dtype=torch.float32).unsqueeze(0)
        label_y = torch.tensor(self.labels[idx], dtype=torch.long)
        area_val = torch.tensor(self.areas[idx], dtype=torch.float32)

        return tensor_x, label_y, area_val

# ══════════════════════════════════════════════════════════════════════════════
# 2. MehleCNN Architecture (VGG-Inspired Multi-Stage ConvNet)
# ══════════════════════════════════════════════════════════════════════════════
class MehleCNN(nn.Module):
    """
    VGG-inspired multi-stage Convolutional Neural Network architecture 
    for in-line pellet agglomeration classification (Mehle et al. 2017).
    """
    def __init__(self, in_channels=1, num_classes=3):
        super().__init__()
        # Stage 1: 96x96 -> 48x48
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Stage 2: 48x48 -> 24x24
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Stage 3: 24x24 -> 12x12
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Stage 4: 12x12 -> 6x6
        self.stage4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Classifier Head
        self.pool = nn.AdaptiveAvgPool2d((3, 3))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

# ══════════════════════════════════════════════════════════════════════════════
# 3. Model Training & Evaluation Engine
# ══════════════════════════════════════════════════════════════════════════════
def train_model(epochs=30, batch_size=50, lr=1e-3, num_classes=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("[INFO] Training Mehle et al. (2017) CNN Crop Classifier")
    print("=" * 80)
    print(f"* Hardware Backend : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"* Target Classes   : {num_classes} ({'Single, Connected, Overlapped' if num_classes==3 else 'Single vs Agglomerate'})")
    print(f"* Batch Size       : {batch_size} (Mehle et al. 2017)")
    print(f"* Max Epochs       : {epochs}")

    # Load dataset manifest
    metadata_path = Path('preprocessed_checkpoint/particles_metadata.csv')
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {metadata_path}! Run run_export_crops.py first.")

    df = pd.read_csv(metadata_path)
    crops_dir = Path('preprocessed_checkpoint/crops')

    class_map_3 = {'single': 0, 'connected': 1, 'overlapped': 2}
    class_names_3 = ['Single Pellet', 'Connected Agglomerate', 'Overlapped Pellets']

    valid_paths = []
    valid_labels = []
    valid_areas = []

    for _, row in df.iterrows():
        lbl_str = row['initial_label']
        fname = row['crop_filename']
        p = crops_dir / lbl_str / fname
        if p.exists():
            valid_paths.append(p)
            if num_classes == 3:
                valid_labels.append(class_map_3[lbl_str])
            else:
                valid_labels.append(0 if lbl_str == 'single' else 1)
            valid_areas.append(float(row['area_px']))

    print(f"* Total Valid Crops Loaded : {len(valid_paths)}")

    # Stratified Split: 60% Train, 20% Val, 20% Test (per Mehle et al. 2017 Section 3.4)
    train_idx, val_idx, test_idx = stratified_split_indices(valid_labels, test_ratio=0.20, val_ratio=0.20, seed=42)

    train_paths = [valid_paths[i] for i in train_idx]
    train_lbls  = [valid_labels[i] for i in train_idx]
    train_areas = [valid_areas[i] for i in train_idx]

    val_paths   = [valid_paths[i] for i in val_idx]
    val_lbls    = [valid_labels[i] for i in val_idx]
    val_areas   = [valid_areas[i] for i in val_idx]

    test_paths  = [valid_paths[i] for i in test_idx]
    test_lbls   = [valid_labels[i] for i in test_idx]
    test_areas  = [valid_areas[i] for i in test_idx]

    print(f"* Split Breakdown  : Train={len(train_paths)} (60%), Val={len(val_paths)} (20%), Test={len(test_paths)} (20%)")

    # Dataloaders
    train_dataset = PelletCropDataset(train_paths, train_lbls, train_areas, is_train=True)
    val_dataset = PelletCropDataset(val_paths, val_lbls, val_areas, is_train=False)
    test_dataset = PelletCropDataset(test_paths, test_lbls, test_areas, is_train=False)

    num_workers = 0 if os.name == 'nt' else 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Model & Loss
    model = MehleCNN(in_channels=1, num_classes=num_classes).to(device)

    # Calculate class weights to balance classes
    class_counts = np.bincount(train_lbls)
    weights = torch.tensor([len(train_lbls) / (len(class_counts) * max(1, c)) for c in class_counts], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Optimizer: AdamW with Cosine Annealing
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    save_path = Path('mehle_cnn_best.pt')

    print("\nStarting training loop...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)

        scheduler.step()

        train_loss = running_loss / total_samples
        train_acc = running_corrects / total_samples

        # Validation
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_corrects += torch.sum(preds == labels.data).item()
                val_total += inputs.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_corrects / val_total

        history['train_loss'].append(train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(epoch_val_acc)

        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:5.1f}% | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc*100:5.1f}%")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': best_val_acc,
                'num_classes': num_classes,
                'class_names': class_names_3 if num_classes == 3 else ['Single', 'Agglomerate']
            }, str(save_path))

    print(f"\n[DONE] Training complete. Best Validation Accuracy: {best_val_acc*100:.2f}%. Saved to {save_path.resolve()}")

    # ══════════════════════════════════════════════════════════════════════════
    # 4. Final Evaluation on Independent Test Set
    # ══════════════════════════════════════════════════════════════════════════
    checkpoint = torch.load(str(save_path), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []
    all_test_areas = []

    with torch.no_grad():
        for inputs, labels, areas in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)

            preds = torch.argmax(probs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_test_areas.extend(areas.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    all_test_areas = np.array(all_test_areas)

    test_acc = np.mean(all_preds == all_targets)
    print("=" * 80)
    print(f"[TEST] Independent Test Set Evaluation (N = {len(all_targets)} crops)")
    print("=" * 80)
    print(f"* Overall Test Accuracy: {test_acc*100:.2f}%\n")

    target_names = class_names_3 if num_classes == 3 else ['Single Pellet', 'Agglomerate']
    report_text, cm = compute_classification_report(all_targets, all_preds, target_names)
    print(report_text)
    print("\nConfusion Matrix:")
    print(cm)
    print("=" * 80)

    # ══════════════════════════════════════════════════════════════════════════
    # 5. ROC Curve: CNN Classifier vs. Area Threshold (Replicating Mehle Fig. 7)
    # ══════════════════════════════════════════════════════════════════════════
    binary_targets = (all_targets > 0).astype(int)
    if num_classes == 3:
        cnn_agg_prob = all_probs[:, 1] + all_probs[:, 2]
    else:
        cnn_agg_prob = all_probs[:, 1]

    fpr_cnn, tpr_cnn, roc_auc_cnn = compute_roc_curve(binary_targets, cnn_agg_prob)
    fpr_area, tpr_area, roc_auc_area = compute_roc_curve(binary_targets, all_test_areas)

    print(f"* ROC AUC - Mehle CNN Classifier : {roc_auc_cnn:.4f}")
    print(f"* ROC AUC - Baseline Area Method: {roc_auc_area:.4f}")

    # Plot Figures
    plt.figure(figsize=(18, 5))

    # 1. Training Curves
    plt.subplot(1, 3, 1)
    plt.plot(history['train_acc'], label='Train Acc', color='#1f77b4', lw=2)
    plt.plot(history['val_acc'], label='Val Acc', color='#2ca02c', lw=2)
    plt.title('Training & Validation Accuracy', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    # 2. Confusion Matrix Heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix (Test Set)', fontweight='bold')
    plt.xticks(range(num_classes), target_names, rotation=20, ha='right')
    plt.yticks(range(num_classes), target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max()/2 else 'black', fontweight='bold')

    # 3. ROC Curve Comparison (Mehle Fig. 7 replication)
    plt.subplot(1, 3, 3)
    plt.plot(fpr_cnn, tpr_cnn, color='#2ca02c', lw=2.5, label=f'Mehle CNN (AUC = {roc_auc_cnn:.3f})')
    plt.plot(fpr_area, tpr_area, color='#d62728', lw=1.8, linestyle='--', label=f'Area Method (AUC = {roc_auc_area:.3f})')
    plt.plot([0, 1], [0, 1], color='#7f7f7f', linestyle=':', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curve: CNN vs. Area Threshold (Mehle Fig. 7)', fontweight='bold')
    plt.legend(loc='lower right', frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig('mehle_cnn_evaluation.png', dpi=150)
    print("Saved evaluation figure to mehle_cnn_evaluation.png")

if __name__ == '__main__':
    train_model(epochs=30, batch_size=50, lr=1e-3, num_classes=3)

