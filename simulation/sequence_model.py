import math
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

class SqueezeExcitation(nn.Module):
    """Channel-wise attention block to emphasize contact-crevice and edge features."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = F.adaptive_avg_pool2d(x, 1).view(b, c)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w)).view(b, c, 1, 1)
        return x * w

class ResidualConvBlock(nn.Module):
    """Residual building block with GroupNorm and SE attention."""
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8 if out_c % 8 == 0 else 4, out_c)
        self.act1 = nn.GELU()
        
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8 if out_c % 8 == 0 else 4, out_c)
        self.se = SqueezeExcitation(out_c)
        
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(8 if out_c % 8 == 0 else 4, out_c)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act1(self.gn1(self.conv1(x)))
        out = self.se(self.gn2(self.conv2(out)))
        return F.gelu(out + res)

class ResidualCropBackbone(nn.Module):
    """
    High-resolution spatial feature extractor tailored for fluid-bed pellet crops.
    Extracts geometric neck contours, boundary curvature, and surface reflections.
    """
    def __init__(self, in_channels: int = 3, d_model: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.MaxPool2d(2, 2)  # 96 -> 48
        )
        self.stage1 = ResidualConvBlock(32, 64, stride=2)   # 48 -> 24
        self.stage2 = ResidualConvBlock(64, 128, stride=2)  # 24 -> 12
        self.stage3 = ResidualConvBlock(128, 256, stride=2) # 12 -> 6
        self.stage4 = ResidualConvBlock(256, d_model, stride=1)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.pool(out).flatten(1)
        return self.proj(out)


class TemporalMultiHeadAttentionPooling(nn.Module):
    """
    Learned attention pooling over the temporal dimension.
    Learns to dynamically weight frames where agglomerate lobes or neck crevices
    are most clearly visible (e.g. when tumbling reveals all lobes).
    """
    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    def forward(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # seq shape: (B, T, D)
        B = seq.size(0)
        q = self.query.expand(B, -1, -1)
        pooled, weights = self.attn(q, seq, seq)
        return pooled.squeeze(1), weights.squeeze(1)


class PelletSpatioTemporalTransformer(nn.Module):
    """
    Spatio-Temporal Transformer for multi-frame pellet sequence classification.
    Disentangles transient 2D optical overlaps from true physical agglomerates
    by attending across past and future frames in the tracklet trajectory.
    """
    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 4,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 32,
        kinematic_dim: int = 7
    ):
        super().__init__()
        self.d_model = d_model
        
        # Kinematic projection layer
        self.kinematic_proj = nn.Sequential(
            nn.Linear(kinematic_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # Learnable temporal positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Stack of Transformer Encoder layers with Pre-LN
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        
        # Learned Temporal Attention Pooling
        self.temporal_pooling = TemporalMultiHeadAttentionPooling(d_model, num_heads=nhead // 2)

    def forward(
        self,
        spatial_feats: torch.Tensor,
        kinematics: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        spatial_feats: (B, T, D)
        kinematics: (B, T, K) optional
        Returns:
            context_vector: (B, D)
            attn_weights: (B, T)
        """
        B, T, D = spatial_feats.shape
        x = spatial_feats
        
        if kinematics is not None:
            k_feat = self.kinematic_proj(kinematics)
            x = x + k_feat
            
        # Add temporal positional embedding
        x = x + self.pos_embed[:, :T, :]
        
        # Multi-layer Transformer Encoder
        out = self.transformer_encoder(x)
        out = self.norm(out)
        
        # Temporal attention pooling
        context, attn_weights = self.temporal_pooling(out)
        return context, attn_weights


class PelletCNNBiLSTM(nn.Module):
    """
    Bidirectional LSTM with Temporal Attention for multi-frame pellet sequence classification.
    Processes sequential frame embeddings with forward and backward temporal context.
    """
    def __init__(
        self,
        d_model: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        kinematic_dim: int = 7
    ):
        super().__init__()
        self.d_model = d_model
        
        self.kinematic_proj = nn.Sequential(
            nn.Linear(kinematic_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.proj = nn.Linear(hidden_dim * 2, d_model)
        self.temporal_pooling = TemporalMultiHeadAttentionPooling(d_model, num_heads=4)

    def forward(
        self,
        spatial_feats: torch.Tensor,
        kinematics: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = spatial_feats
        if kinematics is not None:
            x = x + self.kinematic_proj(kinematics)
            
        lstm_out, _ = self.lstm(x)
        projected = self.proj(lstm_out)
        context, attn_weights = self.temporal_pooling(projected)
        return context, attn_weights


class PelletSequenceModel(nn.Module):
    """
    Unified Spatio-Temporal Model for Pellet Tracking & Agglomeration Disentanglement.
    
    Supports:
    - Architecture: 'transformer' (default) or 'lstm'
    - Spatial backbone: Custom high-res Residual ConvNet or ResNet-18
    - Multi-task heads:
      1. Agglomeration Classification (Single vs Agglomerate / Multi-class)
      2. Transient Optical Overlap Detection (Auxiliary task)
      3. Confidence calibration score
    
    Scalable: Run on RTX 3070 locally, and seamlessly scale to RTX 5090 (32GB VRAM)
    with higher d_model, more layers, larger crop size, and batch sizes up to 128+.
    """
    def __init__(
        self,
        arch: str = 'transformer',
        num_classes: int = 2,
        d_model: int = 256,
        num_layers: int = 4,
        nhead: int = 8,
        crop_size: int = 96,
        in_channels: int = 3,
        kinematic_dim: int = 7,
        backbone_type: str = 'custom', # 'custom' or 'resnet18'
        dropout: float = 0.15
    ):
        super().__init__()
        self.arch = arch.lower()
        self.num_classes = num_classes
        self.d_model = d_model
        self.crop_size = crop_size
        self.backbone_type = backbone_type
        
        # 1. Spatial Backbone
        if backbone_type == 'resnet18':
            resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
            resnet.fc = nn.Linear(resnet.fc.in_features, d_model)
            self.backbone = resnet
        else:
            self.backbone = ResidualCropBackbone(in_channels=in_channels, d_model=d_model)
            
        # 2. Temporal Module
        if self.arch == 'transformer':
            self.temporal = PelletSpatioTemporalTransformer(
                d_model=d_model,
                num_layers=num_layers,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                kinematic_dim=kinematic_dim
            )
        elif self.arch == 'lstm':
            self.temporal = PelletCNNBiLSTM(
                d_model=d_model,
                hidden_dim=d_model,
                num_layers=min(3, num_layers),
                dropout=dropout,
                kinematic_dim=kinematic_dim
            )
        else:
            raise ValueError(f"Unknown architecture '{arch}'. Choose 'transformer' or 'lstm'.")
            
        # 3. Multi-Task Heads
        # Main classification head: Single vs Agglomerate
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Auxiliary head: Transient optical overlap probability
        self.overlap_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1)
        )

    def extract_spatial_features(self, crops_seq: torch.Tensor) -> torch.Tensor:
        """
        Processes (B, T, C, H, W) through spatial backbone into (B, T, D).
        """
        B, T, C, H, W = crops_seq.shape
        # Flatten B and T to batch through 2D CNN
        x_flat = crops_seq.view(B * T, C, H, W)
        feats = self.backbone(x_flat)
        return feats.view(B, T, self.d_model)

    def forward(
        self,
        crops_seq: torch.Tensor,
        kinematics: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for tracklet crop sequence.
        crops_seq: (B, T, C, H, W)
        kinematics: (B, T, K) optional normalized trajectory features
        
        Returns dict:
            'class_logits': (B, num_classes)
            'class_probs': (B, num_classes)
            'overlap_logits': (B, 1)
            'overlap_prob': (B, 1)
            'temporal_attn': (B, T)
        """
        # 1. Spatial feature extraction per frame
        spatial_feats = self.extract_spatial_features(crops_seq)
        
        # 2. Temporal reasoning across frames
        context, attn_weights = self.temporal(spatial_feats, kinematics)
        
        # 3. Classification & Overlap prediction
        logits = self.classifier(context)
        probs = F.softmax(logits, dim=-1)
        
        overlap_logits = self.overlap_head(context)
        overlap_prob = torch.sigmoid(overlap_logits)
        
        return {
            'class_logits': logits,
            'class_probs': probs,
            'overlap_logits': overlap_logits,
            'overlap_prob': overlap_prob,
            'temporal_attn': attn_weights
        }

