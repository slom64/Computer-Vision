import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
import torch

@dataclass
class HardwareProfile:
    tier: str                      # 'extreme' (RTX 5090/4090/A100), 'mid' (RTX 3070/4070), 'entry'
    device_name: str
    vram_gb: float
    cuda_available: bool
    recommended_yolo_model: str    # 'yolo11x.pt', 'yolo11l.pt', 'yolo11s.pt'
    recommended_imgsz: int         # 2048, 1536, 1024, 640
    recommended_batch_size: int    # 16, 8, 4
    recommended_dataset_frames: int# 500-1000, 200, 60
    recommended_workers: int
    transformer_d_model: int       # 512, 256
    transformer_num_layers: int    # 8, 4
    transformer_batch_size: int    # 128, 32
    description: str

def profile_hardware() -> HardwareProfile:
    """
    Intelligently inspects the host hardware (GPU, VRAM, CUDA capability)
    and selects the optimal model complexity and dataset parameters.
    Seamlessly adapts when moving from a local dev machine (e.g. RTX 3070 8GB)
    to a production high-capacity workstation (e.g. RTX 5090 32GB).
    """
    cuda_avail = torch.cuda.is_available()
    
    if not cuda_avail:
        return HardwareProfile(
            tier='entry',
            device_name='CPU Fallback',
            vram_gb=0.0,
            cuda_available=False,
            recommended_yolo_model='yolo11n.pt',
            recommended_imgsz=640,
            recommended_batch_size=2,
            recommended_dataset_frames=30,
            recommended_workers=2,
            transformer_d_model=128,
            transformer_num_layers=2,
            transformer_batch_size=16,
            description="CPU detected. Configured for lightweight prototyping."
        )
        
    dev_props = torch.cuda.get_device_properties(0)
    dev_name = torch.cuda.get_device_name(0)
    vram_gb = dev_props.total_memory / (1024.0 ** 3)
    
    # Tier 1: Heavy / Production High-Capacity (RTX 5090, RTX 4090, RTX 3090, A100, H100)
    if vram_gb >= 22.0 or '5090' in dev_name or '4090' in dev_name or '3090' in dev_name or 'A100' in dev_name:
        return HardwareProfile(
            tier='extreme',
            device_name=dev_name,
            vram_gb=vram_gb,
            cuda_available=True,
            recommended_yolo_model='yolo11m.pt', # Balanced high-accuracy & fast YOLO11 model (20M params)
            recommended_imgsz=1024,              # Fast 1024x1024 resolution (~15s/epoch on RTX 5090)
            recommended_batch_size=16,           # Saturated tensor core throughput
            recommended_dataset_frames=600,      # Massive synthetic dataset (~150,000+ labeled pellets)
            recommended_workers=min(4, os.cpu_count() or 2),
            transformer_d_model=512,             # Heavy spatio-temporal representation
            transformer_num_layers=8,            # 8-layer deep transformer encoder
            transformer_batch_size=128,          # Fast massive sequence parallel processing
            description=f"High-Performance Flagship ({dev_name}, {vram_gb:.1f} GB VRAM). Accelerated training profile (YOLO11m @ 1024px & batch 16, ~15s per epoch)."
        )
    # Tier 2: Mid-Range Workstation / Laptop (RTX 3070, RTX 4070, RTX 3080, T4)
    elif vram_gb >= 7.5:
        return HardwareProfile(
            tier='mid',
            device_name=dev_name,
            vram_gb=vram_gb,
            cuda_available=True,
            recommended_yolo_model='yolo11s.pt',
            recommended_imgsz=1024,
            recommended_batch_size=4,
            recommended_dataset_frames=150,
            recommended_workers=min(4, os.cpu_count() or 2),
            transformer_d_model=256,
            transformer_num_layers=4,
            transformer_batch_size=32,
            description=f"Standard GPU ({dev_name}, {vram_gb:.1f} GB VRAM). Balanced high-speed training profile."
        )
    # Tier 3: Compact GPU
    else:
        return HardwareProfile(
            tier='entry',
            device_name=dev_name,
            vram_gb=vram_gb,
            cuda_available=True,
            recommended_yolo_model='yolo11n.pt',
            recommended_imgsz=1024,
            recommended_batch_size=2,
            recommended_dataset_frames=80,
            recommended_workers=2,
            transformer_d_model=192,
            transformer_num_layers=3,
            transformer_batch_size=16,
            description=f"Entry GPU ({dev_name}, {vram_gb:.1f} GB VRAM). Memory-conserving profile."
        )

def print_hardware_summary(profile: Optional[HardwareProfile] = None):
    if profile is None:
        profile = profile_hardware()
        
    print("=" * 70)
    print(" [HARDWARE ACCELERATION & MODEL COMPLEXITY AUTO-PROFILER]")
    print("=" * 70)
    print(f" Detected Device     : {profile.device_name}")
    print(f" Dedicated VRAM      : {profile.vram_gb:.1f} GB" if profile.cuda_available else " Dedicated VRAM      : None (CPU)")
    print(f" Performance Tier    : {profile.tier.upper()}")
    print(f" Status Summary      : {profile.description}")
    print("-" * 70)
    print(" [AUTO-SELECTED CONFIGURATION]:")
    print(f"  * Vision Detector Model : {profile.recommended_yolo_model} (Base Pretrained Weights)")
    print(f"  * Native Image Size     : {profile.recommended_imgsz} px ({'Zero Downsampling Loss' if profile.recommended_imgsz >= 1536 else 'Balanced Scale'})")
    print(f"  * Training Batch Size   : {profile.recommended_batch_size}")
    print(f"  * Dataset Frame Volume  : {profile.recommended_dataset_frames} frames (~{profile.recommended_dataset_frames * 240:,} labeled pellet instances)")
    print(f"  * Sequence Transformer  : d_model={profile.transformer_d_model}, layers={profile.transformer_num_layers}, batch={profile.transformer_batch_size}")
    print("=" * 70)

