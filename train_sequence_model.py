import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from config import SEQUENCE_LENGTH, CROP_SIZE

def get_device():
    """Auto-detect optimal hardware backend across any computer (NVIDIA GPU, Apple Silicon, or CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

class ResidualBlock(nn.Module):
    """Deep 2D Residual Block with BatchNorm, GELU activations, and identity/projection shortcuts."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + residual)
        return out

class TemporalAttention(nn.Module):
    """Multi-Head Temporal Self-Attention over tumbling sequence frames."""
    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, lstm_outputs):
        # lstm_outputs: (Batch, Seq_len, Feature_dim)
        scores = self.attn_net(lstm_outputs) # (Batch, Seq_len, 1)
        weights = F.softmax(scores, dim=1)    # (Batch, Seq_len, 1)
        context = torch.sum(lstm_outputs * weights, dim=1) # (Batch, Feature_dim)
        return context, weights

class SequenceClassifier(nn.Module):
    """
    Advanced Spatio-Temporal Classifier:
    1. Multi-Stage ResNet Spatial Backbone (extracts shape contours & neck joints per frame)
    2. Bidirectional 2-Layer LSTM (models rigid rotation & translation over time)
    3. Temporal Attention Pooling (focuses on critical tumbling angles where agglomeration is exposed)
    4. Deep MLP Classification Head with LayerNorm & Dropout.
    """
    def __init__(self, feature_dim=256, lstm_hidden=128, num_classes=2):
        super().__init__()
        
        # Spatial Feature Extractor
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        self.stage1 = ResidualBlock(32, 64, stride=2)   # 96 -> 48
        self.stage2 = ResidualBlock(64, 128, stride=2)  # 48 -> 24
        self.stage3 = ResidualBlock(128, 256, stride=2) # 24 -> 12
        self.stage4 = ResidualBlock(256, feature_dim, stride=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.spatial_norm = nn.LayerNorm(feature_dim)
        self.spatial_dropout = nn.Dropout(0.15)
        
        # Temporal Modeling (2-layer Bidirectional LSTM)
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Temporal Attention Mechanism
        self.temporal_attn = TemporalAttention(feature_dim=lstm_hidden * 2)
        
        # Classification Head
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Input x: (Batch, Seq_len, Channels, Height, Width)
        B, S, C, H, W = x.shape
        x_flat = x.view(B * S, C, H, W)
        
        # Extract spatial feature map per frame
        feat = self.stem(x_flat)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        feat = self.stage4(feat)
        feat = self.pool(feat).squeeze(-1).squeeze(-1) # (B*S, feature_dim)
        feat = self.spatial_dropout(self.spatial_norm(feat))
        
        # Reshape to sequence representation
        spatial_seq = feat.view(B, S, -1) # (Batch, Seq_len, feature_dim)
        
        # Bidirectional Temporal Modeling
        lstm_out, _ = self.lstm(spatial_seq) # (Batch, Seq_len, lstm_hidden*2)
        
        # Attention-weighted Temporal Aggregation
        context, attn_weights = self.temporal_attn(lstm_out) # (Batch, lstm_hidden*2)
        
        # Final logits
        logits = self.head(context)
        return logits

class PelletSequenceDataset(Dataset):
    def __init__(self, data_dir, is_train=True):
        self.files = []
        self.labels = []
        self.is_train = is_train
        
        agg_files = glob.glob(os.path.join(data_dir, 'agglomerate', '*.npy'))
        for f in agg_files:
            self.files.append(f)
            self.labels.append(1)
            
        single_files = glob.glob(os.path.join(data_dir, 'single', '*.npy'))
        for f in single_files:
            self.files.append(f)
            self.labels.append(0)
            
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        seq = np.load(self.files[idx]) # Shape: (Seq_len, 3, H, W)
        
        # Safety guard: ensure spatial shape strictly matches (CROP_SIZE, CROP_SIZE)
        if seq.shape[-1] != CROP_SIZE or seq.shape[-2] != CROP_SIZE:
            import cv2
            resized_frames = []
            for t in range(seq.shape[0]):
                frame_hwc = seq[t].transpose(1, 2, 0)
                frame_resized = cv2.resize(frame_hwc, (CROP_SIZE, CROP_SIZE))
                resized_frames.append(frame_resized.transpose(2, 0, 1))
            seq = np.stack(resized_frames)
        
        # Training Augmentations (temporally consistent across the 10 frames)
        if self.is_train:
            if random.random() > 0.5:
                seq = np.flip(seq, axis=-1).copy()
            if random.random() > 0.5:
                seq = np.flip(seq, axis=-2).copy()
            if random.random() > 0.5:
                scale = random.uniform(0.85, 1.15)
                seq = np.clip(seq * scale, 0.0, 1.0)
                
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

def train(epochs=20, batch_size=32, lr=5e-4):
    device = get_device()
    print(f"Training ResNet-BiLSTM-Attention Model on device: {device}")
    
    dataset = PelletSequenceDataset('sequence_dataset', is_train=True)
    if len(dataset) == 0:
        print("❌ No sequences found in sequence_dataset/! Run Step 3 (extract_sequences.py) first.")
        return
        
    print(f"Loaded {len(dataset)} total training sequences (Single & Agglomerates).")
    
    # Stratified split: 85% train, 15% validation
    generator = torch.Generator().manual_seed(42)
    train_size = int(len(dataset) * 0.85)
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    
    num_workers = 0 if os.name == 'nt' else 2
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=(len(train_set) > batch_size), num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    model = SequenceClassifier().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == batch_y).sum().item()
            total_train += batch_x.size(0)
            
        scheduler.step()
        
        # Validation pass
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_x.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == batch_y).sum().item()
                total_val += batch_x.size(0)
                
        train_acc = train_correct / max(1, total_train)
        val_acc = val_correct / max(1, total_val)
        
        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] "
              f"Train Loss: {train_loss/total_train:.4f} Acc: {train_acc*100:.1f}% | "
              f"Val Loss: {val_loss/max(1, total_val):.4f} Acc: {val_acc*100:.1f}%")
              
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_payload = {
                'model_state_dict': model.state_dict(),
                'architecture': 'ResNet-BiLSTM-Attention-v2',
                'crop_size': CROP_SIZE,
                'seq_length': SEQUENCE_LENGTH,
                'val_acc': best_val_acc
            }
            torch.save(save_payload, 'sequence_model_resnet.pt')
            torch.save(model.state_dict(), 'sequence_model.pt')
            
    print(f"[Done] Training Complete! Best Validation Accuracy: {best_val_acc*100:.2f}%. Model saved to sequence_model_resnet.pt")

if __name__ == '__main__':
    train()
