import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class PelletSequenceDataset(Dataset):
    def __init__(self, data_dir):
        self.files = []
        self.labels = []
        
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
        seq = np.load(self.files[idx])
        # shape (10, 3, 64, 64)
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

class SequenceClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple CNN feature extractor for 64x64 images
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 32x32
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 8x8
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 2)
        
    def forward(self, x):
        # x is (B, Seq, C, H, W)
        B, Seq, C, H, W = x.shape
        # Flatten batch and seq to run through CNN
        x = x.view(B * Seq, C, H, W)
        features = self.cnn(x)
        # Reshape for LSTM
        features = features.view(B, Seq, -1)
        lstm_out, (hn, cn) = self.lstm(features)
        # Take the output of the last time step
        out = self.fc(lstm_out[:, -1, :])
        return out

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = PelletSequenceDataset('sequence_dataset')
    print(f"Loaded {len(dataset)} sequences for training.")
    
    if len(dataset) == 0:
        print("No data found! Generate more simulation data.")
        return
        
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = SequenceClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(10): # Short epochs for rapid prototyping
        model.train()
        total_loss = 0
        correct = 0
        
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            
        acc = correct / len(dataset)
        print(f"Epoch {epoch+1}/10 - Loss: {total_loss/len(loader):.4f} - Acc: {acc:.4f}")
        
    torch.save(model.state_dict(), 'sequence_model.pt')
    print("Saved model to sequence_model.pt")

if __name__ == '__main__':
    train()
