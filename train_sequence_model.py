import os
import glob
import argparse
import time
import json
from typing import Dict, Any, Tuple
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from simulation.sequence_model import PelletSequenceModel

class PelletSequenceDataset(Dataset):
    """
    PyTorch Dataset for multi-frame pellet crop sequences and kinematics.
    Loads .npz files containing (T, H, W, 3) crops, (T, 7) kinematics,
    is_agglomerate label, and transient optical overlap flag.
    """
    def __init__(self, data_dir: str, augment: bool = True):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        self.augment = augment
        if len(self.files) == 0:
            raise ValueError(f"No .npz sequence files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        data = np.load(self.files[idx])
        crops = data['crops'].astype(np.float32) / 255.0  # (T, H, W, 3)
        kinematics = data['kinematics'].astype(np.float32) # (T, 7)
        is_agg = int(data['is_agglomerate'])
        has_overlap = int(data['has_overlap'])
        
        # Consistent spatio-temporal data augmentations
        if self.augment:
            # 1. Random horizontal flip across all frames
            if np.random.rand() > 0.5:
                crops = np.flip(crops, axis=2).copy()
                kinematics[:, 0] *= -1.0 # Invert vx
                
            # 2. Random vertical flip across all frames
            if np.random.rand() > 0.5:
                crops = np.flip(crops, axis=1).copy()
                kinematics[:, 1] *= -1.0 # Invert vy
                
            # 3. Random subtle contrast/brightness jitter
            alpha = np.random.uniform(0.92, 1.08)
            beta = np.random.uniform(-0.04, 0.04)
            crops = np.clip(crops * alpha + beta, 0.0, 1.0)
            
        # Standardize: (T, H, W, 3) -> (T, 3, H, W)
        crops = np.transpose(crops, (0, 3, 1, 2))
        
        # Normalize with ImageNet mean/std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
        crops = (crops - mean) / std
        
        return (
            torch.from_numpy(crops).float(),
            torch.from_numpy(kinematics).float(),
            torch.tensor(is_agg, dtype=torch.long),
            torch.tensor(has_overlap, dtype=torch.float32)
        )


class FocalLoss(nn.Module):
    """Focal loss to balance hard tumbling orientations and optical overlaps."""
    def __init__(self, alpha: float = 0.60, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def train_epoch(model, loader, optimizer, scaler, criterion_class, criterion_overlap, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for crops, kin, targets, overlaps in loader:
        crops = crops.to(device, non_blocking=True)
        kin = kin.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        overlaps = overlaps.to(device, non_blocking=True).unsqueeze(1)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            out = model(crops, kin)
            loss_c = criterion_class(out['class_logits'], targets)
            loss_o = criterion_overlap(out['overlap_logits'], overlaps)
            loss = loss_c + 0.35 * loss_o
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * targets.size(0)
        preds = torch.argmax(out['class_logits'], dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        
    return total_loss / max(1, total), correct / max(1, total)


def eval_epoch(model, loader, criterion_class, criterion_overlap, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    tp, fp, fn, tn = 0, 0, 0, 0
    overlap_correct = 0
    overlap_total = 0
    
    with torch.no_grad():
        for crops, kin, targets, overlaps in loader:
            crops = crops.to(device, non_blocking=True)
            kin = kin.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            overlaps = overlaps.to(device, non_blocking=True).unsqueeze(1)
            
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                out = model(crops, kin)
                loss_c = criterion_class(out['class_logits'], targets)
                loss_o = criterion_overlap(out['overlap_logits'], overlaps)
                loss = loss_c + 0.35 * loss_o
                
            total_loss += loss.item() * targets.size(0)
            preds = torch.argmax(out['class_logits'], dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            # Binary classification metrics for Agglomerate (class 1)
            for p, t in zip(preds, targets):
                if p == 1 and t == 1:
                    tp += 1
                elif p == 1 and t == 0:
                    fp += 1
                elif p == 0 and t == 1:
                    fn += 1
                else:
                    tn += 1
                    
            overlap_preds = (out['overlap_prob'] > 0.50).float()
            overlap_correct += (overlap_preds == overlaps).sum().item()
            overlap_total += overlaps.size(0)
            
    acc = correct / max(1, total)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-6, prec + rec)
    overlap_acc = overlap_correct / max(1, overlap_total)
    
    metrics = {
        'loss': total_loss / max(1, total),
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'overlap_acc': overlap_acc,
        'confusion': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train Deep Spatio-Temporal Pellet Sequence Model (Transformer/LSTM)")
    parser.add_argument('--arch', type=str, default='transformer', choices=['transformer', 'lstm'], help='Sequence architecture')
    parser.add_argument('--data_dir', type=str, default='outputs/sequence_dataset', help='Dataset root directory')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (scale to 64-128 on RTX 5090)')
    parser.add_argument('--lr', type=float, default=3e-4, help='Initial learning rate')
    parser.add_argument('--d_model', type=int, default=256, help='Model feature dimension (scale to 512 on RTX 5090)')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of temporal layers (scale to 6-8 on RTX 5090)')
    parser.add_argument('--nhead', type=int, default=8, help='Attention heads for Transformer')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda or cpu)')
    parser.add_argument('--output_dir', type=str, default='outputs/models', help='Model output directory')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('outputs/previews', exist_ok=True)
    
    device = torch.device(args.device)
    print(f"============================================================")
    print(f" Training Spatio-Temporal Pellet Classifier ({args.arch.upper()})")
    print(f" Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f" d_model: {args.d_model} | Layers: {args.num_layers} | Heads: {args.nhead}")
    print(f" Batch size: {args.batch_size} | Epochs: {args.epochs} | LR: {args.lr}")
    print(f"============================================================")
    
    # 1. Datasets & Loaders
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    
    train_dataset = PelletSequenceDataset(train_dir, augment=True)
    val_dataset = PelletSequenceDataset(val_dir, augment=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == 'cuda')
    )
    
    print(f"Dataset loaded: {len(train_dataset)} train sequences, {len(val_dataset)} val sequences.")
    
    # 2. Instantiate Model
    model = PelletSequenceModel(
        arch=args.arch,
        num_classes=2,
        d_model=args.d_model,
        num_layers=args.num_layers,
        nhead=args.nhead,
        crop_size=96,
        in_channels=3,
        kinematic_dim=7,
        dropout=0.15
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} trainable weights")
    
    # 3. Loss & Optimizer
    criterion_class = FocalLoss(alpha=0.60, gamma=2.0)
    criterion_overlap = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    best_f1 = -1.0
    best_acc = -1.0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    # 4. Training Loop
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scaler, criterion_class, criterion_overlap, device)
        val_metrics = eval_epoch(model, val_loader, criterion_class, criterion_overlap, device)
        scheduler.step()
        t1 = time.time()
        
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        
        print(f"Epoch {epoch:02d}/{args.epochs:02d} [{t1-t0:.1f}s] - "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.1f}% | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']*100:.1f}% | "
              f"F1: {val_metrics['f1']:.3f} Prec: {val_metrics['precision']:.3f} Rec: {val_metrics['recall']:.3f} | "
              f"Overlap Disentanglement Acc: {val_metrics['overlap_acc']*100:.1f}%")
              
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_acc = val_metrics['accuracy']
            save_path = os.path.join(args.output_dir, f"pellet_{args.arch}_best.pt")
            torch.save({
                'epoch': epoch,
                'arch': args.arch,
                'd_model': args.d_model,
                'num_layers': args.num_layers,
                'nhead': args.nhead,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
                'total_params': total_params
            }, save_path)
            
    print(f"\nTraining Complete in {(time.time() - t_start)/60.0:.2f} mins!")
    print(f"Best Model Saved to: outputs/models/pellet_{args.arch}_best.pt (Best F1: {best_f1:.3f}, Acc: {best_acc*100:.1f}%)")
    
    # Save training curves plot
    plot_training_curves(history, f"outputs/previews/sequence_{args.arch}_training_curves.png")

def plot_training_curves(history: Dict[str, list], out_path: str):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r--', label='Val Loss')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(epochs, [a * 100 for a in history['val_acc']], 'g-', label='Val Accuracy (%)')
    ax2.plot(epochs, [f * 100 for f in history['val_f1']], 'm--', label='Val F1 Score (%)')
    ax2.set_title('Validation Performance')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score (%)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved training curves plot to: {out_path}")

if __name__ == '__main__':
    main()

